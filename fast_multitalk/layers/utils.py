import torch

from xfuser.core.distributed import (
    get_sp_group,
)
import numpy as np
from skimage import color


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def split_token_counts_and_frame_ids(T, token_frame, world_size, rank):
    S = T * token_frame
    split_sizes = [
        S // world_size + (1 if i < S % world_size else 0) for i in range(world_size)
    ]
    start = sum(split_sizes[:rank])
    end = start + split_sizes[rank]
    counts = [0] * T
    for idx in range(start, end):
        t = idx // token_frame
        counts[t] += 1

    counts_filtered = []
    frame_ids = []
    for t, c in enumerate(counts):
        if c > 0:
            counts_filtered.append(c)
            frame_ids.append(t)
    return counts_filtered, frame_ids


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range

    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


@torch.compile
def calculate_x_ref_attn_map(
    visual_q, ref_k, ref_target_masks, mode="mean", attn_bias=None
):
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)  # B, H, x_seqlens, ref_seqlens

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        torch_gc()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = (
            x_ref_attnmap.sum(-1) / ref_target_mask.sum()
        )  # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)  # B, x_seqlens, H

        if mode == "mean":
            x_ref_attnmap = x_ref_attnmap.mean(-1)  # B, x_seqlens
        elif mode == "max":
            x_ref_attnmap = x_ref_attnmap.max(-1)  # B, x_seqlens

        x_ref_attn_maps.append(x_ref_attnmap)

    del attn
    del x_ref_attn_map_source
    torch_gc()

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(
    visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False
):
    """Args:
    query (torch.tensor): B M H K
    key (torch.tensor): B M H K
    shape (tuple): (N_t, N_h, N_w)
    ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape
    if enable_sp:
        ref_k = get_sp_group().all_gather(ref_k, dim=1)

    x_seqlens = N_h * N_w
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = (
        torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)
    )

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_target_masks,
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, T, H, W]
    v1: torch.Tensor,  # [B, C, T, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    diff: torch.Tensor,  # [B, C, T, H, W]
    pred_cond: torch.Tensor,  # [B, C, T, H, W]
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 55,
):
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
        print(f"diff_norm: {diff_norm}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return normalized_update


def match_and_blend_colors(
    source_chunk: torch.Tensor, reference_image: torch.Tensor, strength: float
) -> torch.Tensor:
    """
    Matches the color of a source video chunk to a reference image and blends with the original.

    Args:
        source_chunk (torch.Tensor): The video chunk to be color-corrected (B, C, T, H, W) in range [-1, 1].
                                     Assumes B=1 (batch size of 1).
        reference_image (torch.Tensor): The reference image (B, C, 1, H, W) in range [-1, 1].
                                        Assumes B=1 and T=1 (single reference frame).
        strength (float): The strength of the color correction (0.0 to 1.0).
                          0.0 means no correction, 1.0 means full correction.

    Returns:
        torch.Tensor: The color-corrected and blended video chunk.
    """
    # print(f"[match_and_blend_colors] Input source_chunk shape: {source_chunk.shape}, reference_image shape: {reference_image.shape}, strength: {strength}")

    if strength == 0.0:
        # print(f"[match_and_blend_colors] Strength is 0, returning original source_chunk.")
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_chunk.device
    dtype = source_chunk.dtype

    # Squeeze batch dimension, permute to T, H, W, C for skimage
    # Source: (1, C, T, H, W) -> (T, H, W, C)
    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # Reference: (1, C, 1, H, W) -> (H, W, C)
    ref_np = (
        reference_image.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()
    )  # Squeeze T dimension as well

    # Normalize from [-1, 1] to [0, 1] for skimage
    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0

    # Clip to ensure values are strictly in [0, 1] after potential float precision issues
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    # Convert reference to Lab
    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError as e:
        # Handle potential errors if image data is not valid for conversion
        print(
            f"Warning: Could not convert reference image to Lab: {e}. Skipping color correction for this chunk."
        )
        return source_chunk

    corrected_frames_np_01 = []
    for i in range(source_np_01.shape[0]):  # Iterate over time (T)
        source_frame_rgb_01 = source_np_01[i]

        try:
            source_lab = color.rgb2lab(source_frame_rgb_01)
        except ValueError as e:
            print(
                f"Warning: Could not convert source frame {i} to Lab: {e}. Using original frame."
            )
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        corrected_lab_frame = source_lab.copy()

        # Perform color transfer for L, a, b channels
        for j in range(3):  # L, a, b
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()

            # Avoid division by zero if std_src is 0
            if std_src == 0:
                # If source channel has no variation, keep it as is, but shift by reference mean
                # This case is debatable, could also just copy source or target mean.
                # Shifting by target mean helps if source is flat but target isn't.
                corrected_lab_frame[:, :, j] = mean_ref
            else:
                corrected_lab_frame[:, :, j] = (
                    corrected_lab_frame[:, :, j] - mean_src
                ) * (std_ref / std_src) + mean_ref

        try:
            fully_corrected_frame_rgb_01 = color.lab2rgb(corrected_lab_frame)
        except ValueError as e:
            print(
                f"Warning: Could not convert corrected frame {i} back to RGB: {e}. Using original frame."
            )
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        # Clip again after lab2rgb as it can go slightly out of [0,1]
        fully_corrected_frame_rgb_01 = np.clip(fully_corrected_frame_rgb_01, 0.0, 1.0)

        # Blend with original source frame (in [0,1] RGB)
        blended_frame_rgb_01 = (
            1 - strength
        ) * source_frame_rgb_01 + strength * fully_corrected_frame_rgb_01
        corrected_frames_np_01.append(blended_frame_rgb_01)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    # Convert back to [-1, 1]
    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0

    # Permute back to (C, T, H, W), add batch dim, and convert to original torch.Tensor type and device
    # (T, H, W, C) -> (C, T, H, W)
    corrected_chunk_tensor = (
        torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    )
    corrected_chunk_tensor = (
        corrected_chunk_tensor.contiguous()
    )  # Ensure contiguous memory layout
    output_tensor = corrected_chunk_tensor.to(device=device, dtype=dtype)
    # print(f"[match_and_blend_colors] Output tensor shape: {output_tensor.shape}")
    return output_tensor
