import math
from typing import Tuple, List, Optional, Literal

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from ..ops.attn import attention_varlen
from ..layers.attention import SingleStreamMutiAttention
from ..layers.utils import get_attn_map_with_target
from ..utils import load_state_dict, to_param_dtype


def sinusoidal_embedding_1d(dim, position) -> torch.Tensor:
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False, device_type="cuda")
def rope_params(max_seq_len, dim, theta=10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile()
    def forward(self, x) -> torch.Tensor:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, ref_target_masks=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = attention_varlen(
            q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size
        ).type_as(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(
                q.type_as(x),
                k.type_as(x),
                grid_sizes[0],
                ref_target_masks=ref_target_masks,
            )

        return x, x_ref_attn_map


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = attention_varlen(q, k_img, v_img, k_lens=None)
        # compute attention
        x = attention_varlen(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        output_dim=768,
        norm_input_visual=True,
        class_range=24,
        class_interval=4,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
            class_range=class_range,
            class_interval=class_interval,
        )
        self.norm_x = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if norm_input_visual
            else nn.Identity()
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
    ):
        dtype = x.dtype
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y, x_ref_attn_map = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x),
            seq_lens,
            grid_sizes,
            freqs,
            ref_target_masks=ref_target_masks,
        )
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # cross attn of audio
        x_a = self.audio_cross_attn(
            self.norm_x(x),
            encoder_hidden_states=audio_embedding,
            shape=grid_sizes[0],
            x_ref_attn_map=x_ref_attn_map,
            human_num=human_num,
        )
        x = x + x_a

        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5]

        x = x.to(dtype)

        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(
            batch_size_vf, window_size_vf * blocks_vf * channels_vf
        )

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(
            batch_size_c * N_t, self.context_tokens, self.output_dim
        )

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


class WanModel(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        # audio params
        audio_window: int = 5,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,  # vae timedownsample scale
        norm_input_visual: bool = True,
        norm_output_audio: bool = True,
        weight_init: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model_type = "i2v"

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    output_dim=output_dim,
                    norm_input_visual=norm_input_visual,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.img_emb = MLPProj(1280, dim)

        # init audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # initialize weights
        if weight_init:
            self.init_weights()

        self.enable_teacache = False

    def init_freqs(self):
        d = self.dim // self.num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to("cuda")

    def teacache_init(
        self,
        use_ret_steps=True,
        teacache_thresh=0.2,
        sample_steps=40,
        model_scale="multitalk-480",
    ):
        self.enable_teacache = True

        self.__class__.cnt = 0
        self.__class__.num_steps = sample_steps * 3
        self.__class__.teacache_thresh = teacache_thresh
        self.__class__.accumulated_rel_l1_distance_even = 0
        self.__class__.accumulated_rel_l1_distance_odd = 0
        self.__class__.previous_e0_even = None
        self.__class__.previous_e0_odd = None
        self.__class__.previous_residual_even = None
        self.__class__.previous_residual_odd = None
        self.__class__.use_ret_steps = use_ret_steps

        if use_ret_steps:
            if model_scale == "multitalk-480":
                self.__class__.coefficients = [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ]
            if model_scale == "multitalk-720":
                self.__class__.coefficients = [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ]
            self.__class__.ret_steps = 5 * 3
            self.__class__.cutoff_steps = sample_steps * 3
        else:
            if model_scale == "multitalk-480":
                self.__class__.coefficients = [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ]

            if model_scale == "multitalk-720":
                self.__class__.coefficients = [
                    -114.36346466,
                    65.26524496,
                    -18.82220707,
                    4.91518089,
                    -0.23412683,
                ]
            self.__class__.ret_steps = 1 * 3
            self.__class__.cutoff_steps = sample_steps * 3 - 3

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
    ):
        assert clip_fea is not None and y is not None

        _, T, H, W = x[0].shape
        _N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale
        )
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[
            :, :, :1, : middle_index + 1, ...
        ]
        latter_first_frame_audio_emb = rearrange(
            latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_last_frame_audio_emb = latter_frame_audio_emb[
            :, :, -1:, middle_index:, ...
        ]
        latter_last_frame_audio_emb = rearrange(
            latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_middle_frame_audio_emb = latter_frame_audio_emb[
            :, :, 1:-1, middle_index : middle_index + 1, ...
        ]
        latter_middle_frame_audio_emb = rearrange(
            latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_frame_audio_emb_s = torch.concat(
            [
                latter_first_frame_audio_emb,
                latter_middle_frame_audio_emb,
                latter_last_frame_audio_emb,
            ],
            dim=2,
        )
        audio_embedding = self.audio_proj(
            first_frame_audio_emb_s, latter_frame_audio_emb_s
        )
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

        # convert ref_target_masks to token_ref_target_masks
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
            token_ref_target_masks = nn.functional.interpolate(
                ref_target_masks, size=(N_h, N_w), mode="nearest"
            )
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = token_ref_target_masks > 0
            token_ref_target_masks = token_ref_target_masks.view(
                token_ref_target_masks.shape[0], -1
            )
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)

        # teacache
        if self.enable_teacache:
            modulated_inp = e0 if self.use_ret_steps else e
            if self.cnt % 3 == 0:  # cond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance_cond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_cond += rescale_func(
                        (
                            (modulated_inp - self.previous_e0_cond).abs().mean()
                            / self.previous_e0_cond.abs().mean()
                        )
                        .cpu()
                        .item()
                    )
                    if self.accumulated_rel_l1_distance_cond < self.teacache_thresh:
                        should_calc_cond = False
                    else:
                        should_calc_cond = True
                        self.accumulated_rel_l1_distance_cond = 0
                self.previous_e0_cond = modulated_inp.clone()
            elif self.cnt % 3 == 1:  # drop_text
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_drop_text = True
                    self.accumulated_rel_l1_distance_drop_text = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_drop_text += rescale_func(
                        (
                            (modulated_inp - self.previous_e0_drop_text).abs().mean()
                            / self.previous_e0_drop_text.abs().mean()
                        )
                        .cpu()
                        .item()
                    )
                    if (
                        self.accumulated_rel_l1_distance_drop_text
                        < self.teacache_thresh
                    ):
                        should_calc_drop_text = False
                    else:
                        should_calc_drop_text = True
                        self.accumulated_rel_l1_distance_drop_text = 0
                self.previous_e0_drop_text = modulated_inp.clone()
            else:  # uncond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_uncond = True
                    self.accumulated_rel_l1_distance_uncond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_uncond += rescale_func(
                        (
                            (modulated_inp - self.previous_e0_uncond).abs().mean()
                            / self.previous_e0_uncond.abs().mean()
                        )
                        .cpu()
                        .item()
                    )
                    if self.accumulated_rel_l1_distance_uncond < self.teacache_thresh:
                        should_calc_uncond = False
                    else:
                        should_calc_uncond = True
                        self.accumulated_rel_l1_distance_uncond = 0
                self.previous_e0_uncond = modulated_inp.clone()

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
        )
        if self.enable_teacache:
            if self.cnt % 3 == 0:
                if not should_calc_cond:
                    x += self.previous_residual_cond
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_cond = x - ori_x
            elif self.cnt % 3 == 1:
                if not should_calc_drop_text:
                    x += self.previous_residual_drop_text
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_drop_text = x - ori_x
            else:
                if not should_calc_uncond:
                    x += self.previous_residual_uncond
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_uncond = x - ori_x
        else:
            for block in self.blocks:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        if self.enable_teacache:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0

        return torch.stack(x).float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def get_example_input(self, device: str = "cpu"):
        x = [torch.randn(16, 18, 104, 56, dtype=torch.float32, device=device)]
        t = torch.tensor([1], dtype=torch.float32, device=device)
        context = [torch.randn(138, 4096, dtype=torch.bfloat16, device=device)]
        clip_fea = torch.randn(1, 257, 1280, dtype=torch.bfloat16, device=device)
        seq_len = 26208
        y = torch.randn(1, 20, 18, 104, 56, dtype=torch.bfloat16, device=device)
        audio = torch.randn(1, 69, 5, 12, 768, dtype=torch.bfloat16, device=device)
        ref_target_masks = torch.randn(3, 104, 5, dtype=torch.float32, device=device)
        example_inputs = {}
        example_inputs["x"] = x
        example_inputs["t"] = t
        example_inputs["context"] = context
        example_inputs["clip_fea"] = clip_fea
        example_inputs["seq_len"] = seq_len
        example_inputs["y"] = y
        example_inputs["audio"] = audio
        example_inputs["ref_target_masks"] = ref_target_masks
        return example_inputs


def load_wan_multitalk_model(
    wan_path: str,
    multitalk_path: str,
    param_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
) -> WanModel:
    logger.info(
        f"Loading Wan model from {wan_path} and multitalk model from {multitalk_path}"
    )
    states = load_state_dict(wan_path)
    multitalk_states = load_state_dict(multitalk_path)
    for key in multitalk_states:
        states[key] = multitalk_states[key]
    with torch.device("meta"):
        wan_model = WanModel(**kwargs)
    wan_model.load_state_dict(states, assign=True)
    wan_model.init_freqs()
    wan_model.eval().requires_grad_(False)
    to_param_dtype(wan_model, param_dtype)
    return wan_model


def load_quant_wan_multitalk_model(
    quant_dir: str,
    quant: str = "int8",
    distill: Literal["lightx2v", "fusionx"] = "fusionx",
    **kwargs,
):
    from optimum.quanto import requantize
    from pathlib import Path
    import json

    with torch.device("meta"):
        wan_model = WanModel(**kwargs)
    model_path = Path(quant_dir) / f"quant_model_{quant}_{distill}.safetensors"
    states = load_state_dict(model_path)
    map_path = Path(quant_dir) / f"quant_map_{quant}_{distill}.json"
    with open(map_path, "r") as f:
        map_dict = json.load(f)
    requantize(wan_model, states, map_dict, device="cpu")
    wan_model.eval().requires_grad_(False)
    return wan_model
