import json
import math
import random
import time
from contextlib import contextmanager
from functools import partial
from PIL import Image
from typing import Literal, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import load_file
from optimum.quanto import requantize
import optimum.quanto.nn.qlinear as qlinear
from loguru import logger

from .models.clip import CLIPModel
from .models.dit import WanModel, WanLayerNorm, WanRMSNorm
from .models.vae import WanVAE
from .layers.utils import (
    MomentumBuffer,
    adaptive_projected_guidance,
    match_and_blend_colors,
)
from .pipes.audio_encoder import AudioEncoder
from .pipes.text_encoder import TextEncoder
from .vram_management import (
    AutoWrappedQLinear,
    AutoWrappedLinear,
    AutoWrappedModule,
    enable_vram_management,
)
from .config import MultiTalkConfig
from .utils import (
    torch_gc,
    resize_and_centercrop,
    to_param_dtype_fp32only,
    timestep_transform,
    seed_everything,
)
from .data import save_video_ffmpeg


class MultiTalkPipeline:
    def __init__(
        self,
        base_dir: str,
        config: MultiTalkConfig = MultiTalkConfig(),
        device_id: int = 0,
        init_on_cpu: bool = True,
        num_timesteps=1000,
        use_timestep_transform: bool = True,
        quant: Literal["int8", "fp8"] | None = "int8",
        distill_model: Literal["lightx2v", "FusionX"] = "lightx2v",
        low_vram_mode: bool = True,
        num_persistent_param_in_dit: int = 10_000_000_000,
    ):
        """
        Initializes the image-to-video generation model components.
        """
        self.base_dir = Path(base_dir)
        assert self.base_dir.exists(), f"Base directory not exists: {self.base_dir}"
        self.quant_dir = self.base_dir / "quant_models"
        assert self.quant_dir.exists(), f"Quant directory not exists: {self.quant_dir}"

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.param_dtype = config.param_dtype

        audio_encoder_dir = str(Path(base_dir) / config.wav2vec_model)
        self.audio_encoder = AudioEncoder(
            checkpoint_dir=audio_encoder_dir, device=self.device
        )

        t5_path = Path(base_dir) / config.t5_checkpoint
        t5_tokenizer_path = Path(base_dir) / config.t5_tokenizer
        self.text_encoder = TextEncoder(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=t5_path,
            tokenizer_path=str(t5_tokenizer_path),
            shard_fn=None,
            quant=quant,
            quant_dir=self.quant_dir,
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=str(Path(base_dir) / config.clip_checkpoint),
            tokenizer_path=str(Path(base_dir) / config.clip_tokenizer),
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        vae_path = Path(base_dir) / config.vae_checkpoint
        self.vae = WanVAE(
            vae_pth=str(vae_path),
            device=self.device,
        )

        # init dit model
        with torch.device("meta"):
            wan_config = json.load(open(Path(base_dir) / "config.json"))
            self.model = WanModel(weight_init=False, **wan_config)
            torch_gc()

        # load quantized distill model
        logger.info(
            f"Loading {distill_model} distilled model with {quant} quantization from {self.quant_dir}"
        )
        distill_model_path = (
            self.quant_dir / f"quant_model_{quant}_{distill_model}.safetensors"
        )
        distill_model_state_dict = load_file(distill_model_path)
        map_json_path = (
            self.quant_dir / f"quantization_map_{quant}_{distill_model}.json"
        )
        with open(map_json_path, "r") as f:
            distill_model_quantization_map = json.load(f)
        requantize(
            self.model,
            distill_model_state_dict,
            distill_model_quantization_map,
            device="cpu",
        )

        self.model.init_freqs()
        self.model.eval().requires_grad_(False)

        to_param_dtype_fp32only(self.model, self.param_dtype)

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.model_names = ["model"]
        self.cpu_offload = False
        self.vram_management = False
        if low_vram_mode:
            logger.info(
                f"Enable low vram mode with num_persistent_param_in_dit: {num_persistent_param_in_dit}"
            )
            self.enable_vram_management(
                num_persistent_param_in_dit=num_persistent_param_in_dit
            )
            self.enable_cpu_offload()

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=0):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()
        self.vram_management = True

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        start_time = time.time()
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)

                if not isinstance(model, nn.Module):
                    model = model.model

                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        end_time = time.time()
        logger.info(f"Load models to device time: {end_time - start_time} seconds")
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def generate(
        self,
        input_data: str | dict,
        video_save_path: str = None,
        motion_frame: int = 25,
        frame_num: int = 81,
        shift: float = 5.0,
        sample_steps: int = 10,
        text_guide_scale: float = 1.0,
        audio_guide_scale: float = 2.0,
        n_prompt: str = "",
        seed: int = 42,
        max_frames_num: int = 1000,
        face_scale: float = 0.05,
        progress: bool = True,
        color_correction_strength: float = 0.0,
        use_teacache: bool = True,
        teacache_thresh: float = 0.2,
        use_apg: bool = False,
        apg_momentum: float = -0.75,
        apg_norm_threshold: float = 55,
        **kwargs: Any,
    ):
        """
        Args:
            input_data (str | dict): The input data for video generation. It can be a path to a json file.
            video_save_path (str, optional): The path to save the generated video. Defaults to None.
            motion_frame (int, optional): The number of motion frames. Defaults to 25.
            frame_num (int, optional): The number of frames to generate. Defaults to 81.
            shift (float, optional): The shift value for the timestep transform. Defaults to 5.0.
            sample_steps (int, optional): The number of sampling steps. Defaults to 10.
            text_guide_scale (float, optional): The text guide scale. Defaults to 1.0.
            audio_guide_scale (float, optional): The audio guide scale. Defaults to 2.0.
            n_prompt (str, optional): The negative prompt. Defaults to "".
            seed (int, optional): The seed for the video generation. Defaults to 42.
            max_frames_num (int, optional): The maximum number of frames to generate. Defaults to 1000.
            face_scale (float, optional): The face scale. Defaults to 0.05.
            progress (bool, optional): Whether to show the progress bar. Defaults to True.
            color_correction_strength (float, optional): The color correction strength. Defaults to 0.0.
            use_teacache (bool, optional): Whether to use teacache. Defaults to True.
            teacache_thresh (float, optional): The teacache threshold. Defaults to 0.2.
            use_apg (bool, optional): Whether to use APG. Defaults to False.
            apg_momentum (float, optional): The APG momentum. Defaults to -0.75.
            apg_norm_threshold (float, optional): The APG norm threshold. Defaults to 55.
        """
        generation_start_time = time.perf_counter()
        seed_everything(seed if seed >= 0 else random.randint(0, 99999999))
        # init teacache
        if use_teacache:
            logger.info(
                f"Initializing teacache with sample steps: {sample_steps}, teacache threshold: {teacache_thresh}"
            )
            self.model.teacache_init(
                sample_steps=sample_steps,
                teacache_thresh=teacache_thresh,
            )
            logger.info(f"Teacache initialized")
        else:
            self.model.disable_teacache()

        input_data = self.audio_encoder.process_input_data(input_data)
        logger.info(f"Input prompt: {input_data['prompt']}")

        input_prompt = input_data["prompt"]
        cond_file_path = input_data["cond_image"]
        cond_image = Image.open(cond_file_path).convert("RGB")

        bucket_config = ASPECT_RATIO_627
        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        closest_bucket = sorted(
            list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio)
        )[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        cond_image = resize_and_centercrop(cond_image, (target_h, target_w))

        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2  # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()

        # read audio embeddings
        HUMAN_NUMBER = len(input_data["cond_audio"])

        full_audio_embs = []
        if HUMAN_NUMBER == 1:
            person1_audio_embedding = input_data["cond_audio"]["person1"]
            if person1_audio_embedding.shape[0] <= frame_num:
                ## frame_num必须是4n+1，如果小于frame_num，则需要截取
                new_frame_num = person1_audio_embedding.shape[0] // 4 * 4 + 1
                logger.warning(
                    f"Audio embedding length not satisfies frame nums: {person1_audio_embedding.shape[0]} <= {frame_num}, set frame_num to {new_frame_num}"
                )
                frame_num = new_frame_num
            full_audio_embs.append(person1_audio_embedding)

        elif HUMAN_NUMBER == 2:
            person1_audio_embedding = input_data["cond_audio"]["person1"]
            person2_audio_embedding = input_data["cond_audio"]["person2"]
            if (
                person1_audio_embedding.shape[0] <= frame_num
                or person2_audio_embedding.shape[0] <= frame_num
            ):
                raise ValueError(
                    f"Audio embedding length not satisfies frame nums: {person1_audio_embedding.shape[0]} > {frame_num} and {person2_audio_embedding.shape[0]} > {frame_num}"
                )
            full_audio_embs.append(person1_audio_embedding)
            full_audio_embs.append(person2_audio_embedding)

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        t5_io_time = 0
        t5_infer_time = 0
        t5_io_start = time.perf_counter()
        self.text_encoder.model.to(self.device)
        t5_io_time += time.perf_counter() - t5_io_start
        logger.info(f"Text encoder onload time: {t5_io_time} seconds")
        t5_infer_start = time.perf_counter()
        context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
        t5_infer_time += time.perf_counter() - t5_infer_start
        logger.info(f"Text encoder inference time: {t5_infer_time} seconds")
        t5_io_start = time.perf_counter()
        self.text_encoder.model.cpu()
        t5_io_time += time.perf_counter() - t5_io_start
        logger.info(f"Text encoder offload time: {t5_io_time} seconds")
        torch_gc()

        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        torch_gc()

        # start video generation iteratively
        while True:
            audio_embs = []
            # split audio with window size
            for human_idx in range(HUMAN_NUMBER):
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(
                    center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1
                )
                audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(
                    self.device
                )
                audio_embs.append(audio_emb)
            audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
            torch_gc()

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = (
                ((frame_num - 1) // self.vae_stride[0] + 1)
                * lat_h
                * lat_w
                // (self.patch_size[1] * self.patch_size[2])
            )

            noise = torch.randn(
                16,
                (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                device=self.device,
            )

            # get mask
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, cur_motion_frames_num:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(self.param_dtype)  # B 4 T H W

            with torch.no_grad():
                # get clip embedding
                clip_io_time = 0
                clip_io_start = time.perf_counter()
                self.clip.model.to(self.device)
                clip_io_time += time.perf_counter() - clip_io_start
                logger.info(f"Clip onload time: {clip_io_time} seconds")

                clip_infer_time = 0
                clip_infer_start = time.perf_counter()
                clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(
                    self.param_dtype
                )
                clip_infer_time += time.perf_counter() - clip_infer_start
                logger.info(f"Clip infer time: {clip_infer_time} seconds")
                clip_io_start = time.perf_counter()
                self.clip.model.cpu()
                clip_io_time += time.perf_counter() - clip_io_start
                logger.info(f"Clip offload time: {clip_io_time} seconds")
                torch_gc()

                # zero padding and vae encode
                video_frames = torch.zeros(
                    1,
                    cond_image.shape[1],
                    frame_num - cond_image.shape[2],
                    target_h,
                    target_w,
                ).to(self.device)
                padding_frames_pixels_values = torch.concat(
                    [cond_image, video_frames], dim=2
                )
                vae_encode_time = 0
                vae_encode_start = time.perf_counter()
                y = self.vae.encode(padding_frames_pixels_values)
                vae_encode_time += time.perf_counter() - vae_encode_start
                logger.info(f"Vae encode time: {vae_encode_time} seconds")
                y = torch.stack(y).to(self.param_dtype)  # B C T H W
                cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
                latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][
                    0
                ]  # C T H W
                y = torch.concat([msk, y], dim=1)  # B 4+C T H W
                torch_gc()

            # construct human mask
            human_masks = []
            if HUMAN_NUMBER == 1:
                background_mask = torch.ones([src_h, src_w])
                human_mask1 = torch.ones([src_h, src_w])
                human_mask2 = torch.ones([src_h, src_w])
                human_masks = [human_mask1, human_mask2, background_mask]
            elif HUMAN_NUMBER == 2:
                if "bbox" in input_data:
                    assert len(input_data["bbox"]) == len(input_data["cond_audio"]), (
                        "The number of target bbox should be the same with cond_audio"
                    )
                    background_mask = torch.zeros([src_h, src_w])
                    for _, person_bbox in input_data["bbox"].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = torch.zeros([src_h, src_w])
                        human_mask[int(x_min) : int(x_max), int(y_min) : int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min, x_max = (
                        int(src_h * face_scale),
                        int(src_h * (1 - face_scale)),
                    )
                    background_mask = torch.zeros([src_h, src_w])
                    background_mask = torch.zeros([src_h, src_w])
                    human_mask1 = torch.zeros([src_h, src_w])
                    human_mask2 = torch.zeros([src_h, src_w])
                    lefty_min, lefty_max = (
                        int((src_w // 2) * face_scale),
                        int((src_w // 2) * (1 - face_scale)),
                    )
                    righty_min, righty_max = (
                        int((src_w // 2) * face_scale + (src_w // 2)),
                        int((src_w // 2) * (1 - face_scale) + (src_w // 2)),
                    )
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask += human_mask1
                    background_mask += human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = torch.where(
                    background_mask > 0, torch.tensor(0), torch.tensor(1)
                )
                human_masks.append(background_mask)

            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            # resize and centercrop for ref_target_masks
            ref_target_masks = resize_and_centercrop(
                ref_target_masks, (target_h, target_w)
            )

            _, _, _, lat_h, lat_w = y.shape
            ref_target_masks = F.interpolate(
                ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode="nearest"
            ).squeeze()
            ref_target_masks = ref_target_masks > 0
            ref_target_masks = ref_target_masks.float().to(self.device)

            torch_gc()

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, "no_sync", noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():
                # prepare timesteps
                timesteps = list(
                    np.linspace(self.num_timesteps, 1, sample_steps, dtype=np.float32)
                )
                timesteps.append(0.0)
                timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                if self.use_timestep_transform:
                    timesteps = [
                        timestep_transform(
                            t, shift=shift, num_timesteps=self.num_timesteps
                        )
                        for t in timesteps
                    ]

                # sample videos
                latent = noise

                # prepare condition and uncondition configs
                arg_c = {
                    "context": [context],
                    "clip_fea": clip_context,
                    "seq_len": max_seq_len,
                    "y": y,
                    "audio": audio_embs,
                    "ref_target_masks": ref_target_masks,
                }

                arg_null_text = {
                    "context": [context_null],
                    "clip_fea": clip_context,
                    "seq_len": max_seq_len,
                    "y": y,
                    "audio": audio_embs,
                    "ref_target_masks": ref_target_masks,
                }

                arg_null_audio = {
                    "context": [context],
                    "clip_fea": clip_context,
                    "seq_len": max_seq_len,
                    "y": y,
                    "audio": torch.zeros_like(audio_embs)[-1:],
                    "ref_target_masks": ref_target_masks,
                }

                arg_null = {
                    "context": [context_null],
                    "clip_fea": clip_context,
                    "seq_len": max_seq_len,
                    "y": y,
                    "audio": torch.zeros_like(audio_embs)[-1:],
                    "ref_target_masks": ref_target_masks,
                }

                torch_gc()
                if not self.vram_management:
                    self.model.to(self.device)
                else:
                    self.load_models_to_device(["model"])

                # injecting motion frames
                if not is_first_clip:
                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(
                        self.device
                    )
                    motion_add_noise = torch.randn_like(
                        latent_motion_frames
                    ).contiguous()
                    add_latent = self.add_noise(
                        latent_motion_frames, motion_add_noise, timesteps[0]
                    )
                    _, T_m, _, _ = add_latent.shape
                    latent[:, :T_m] = add_latent

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416
                if use_apg:
                    text_momentumbuffer = MomentumBuffer(apg_momentum)
                    audio_momentumbuffer = MomentumBuffer(apg_momentum)

                progress_wrap = (
                    partial(tqdm, total=len(timesteps) - 1)
                    if progress
                    else (lambda x: x)
                )
                for i in progress_wrap(range(len(timesteps) - 1)):
                    timestep = timesteps[i]
                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c
                    )[0]
                    torch_gc()

                    if math.isclose(text_guide_scale, 1.0):
                        noise_pred_drop_audio = self.model(
                            latent_model_input, t=timestep, **arg_null_audio
                        )[0]
                        torch_gc()
                    else:
                        noise_pred_drop_text = self.model(
                            latent_model_input, t=timestep, **arg_null_text
                        )[0]
                        torch_gc()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null
                        )[0]
                        torch_gc()

                    if use_apg:
                        # correct update direction
                        if math.isclose(text_guide_scale, 1.0):
                            diff_uncond_audio = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (
                                audio_guide_scale - 1
                            ) * adaptive_projected_guidance(
                                diff_uncond_audio,
                                noise_pred_cond,
                                momentum_buffer=audio_momentumbuffer,
                                norm_threshold=apg_norm_threshold,
                            )
                        else:
                            diff_uncond_text = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = (
                                noise_pred_cond
                                + (text_guide_scale - 1)
                                * adaptive_projected_guidance(
                                    diff_uncond_text,
                                    noise_pred_cond,
                                    momentum_buffer=text_momentumbuffer,
                                    norm_threshold=apg_norm_threshold,
                                )
                                + (audio_guide_scale - 1)
                                * adaptive_projected_guidance(
                                    diff_uncond_audio,
                                    noise_pred_cond,
                                    momentum_buffer=audio_momentumbuffer,
                                    norm_threshold=apg_norm_threshold,
                                )
                            )
                    else:
                        # vanilla CFG strategy
                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guide_scale * (
                                noise_pred_cond - noise_pred_drop_audio
                            )
                        else:
                            noise_pred = (
                                noise_pred_uncond
                                + text_guide_scale
                                * (noise_pred_cond - noise_pred_drop_text)
                                + audio_guide_scale
                                * (noise_pred_drop_text - noise_pred_uncond)
                            )
                    noise_pred = -noise_pred

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    # injecting motion frames
                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(
                            self.device
                        )
                        motion_add_noise = torch.randn_like(
                            latent_motion_frames
                        ).contiguous()
                        add_latent = self.add_noise(
                            latent_motion_frames, motion_add_noise, timesteps[i + 1]
                        )
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                if not self.vram_management:
                    self.model.cpu()
                torch_gc()
                vae_decode_time = 0
                vae_decode_start = time.perf_counter()
                videos = self.vae.decode(x0)
                vae_decode_time += time.perf_counter() - vae_decode_start
                logger.info(f"Vae decode time: {vae_decode_time} seconds")

            # cache generated samples
            videos = torch.stack(videos).cpu()  # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(
                    videos, original_color_reference, color_correction_strength
                )
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame:
                break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_image = (
                videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            )
            audio_start_idx += frame_num - cur_motion_frames_num
            audio_end_idx = audio_start_idx + clip_length

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length = (
                            audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        )
                        add_audio_emb = torch.flip(
                            full_audio_embs[human_inx][-1 * miss_length :], dims=[0]
                        )
                        full_audio_embs[human_inx] = torch.cat(
                            [full_audio_embs[human_inx], add_audio_emb], dim=0
                        )
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            if max_frames_num <= frame_num:
                break

            torch_gc()
            torch.cuda.synchronize()

        gen_video_samples = torch.cat(gen_video_list, dim=2)[
            :, :, : int(max_frames_num)
        ]
        gen_video_samples = gen_video_samples.to(torch.float32)
        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            # split video frames
            gen_video_samples = gen_video_samples[:, :, : -1 * miss_lengths[0]]

        del noise, latent
        torch_gc()

        if video_save_path is None:
            video_save_path = f"multitalk_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        save_video_ffmpeg(
            save_path=video_save_path,
            gen_video_samples=gen_video_samples[0],
            audio_samples=input_data["video_audio"],
            audio_sample_rate=self.audio_encoder.sample_rate,
        )
        logger.info(f"Video saved to {video_save_path}")

        generation_end_time = time.perf_counter()
        logger.info(
            f"Generation time: {generation_end_time - generation_start_time} seconds"
        )
        return video_save_path


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ASPECT_RATIO_627 = {
    "0.26": ([320, 1216], 1),
    "0.38": ([384, 1024], 1),
    "0.50": ([448, 896], 1),
    "0.67": ([512, 768], 1),
    "0.82": ([576, 704], 1),
    "1.00": ([640, 640], 1),
    "1.22": ([704, 576], 1),
    "1.50": ([768, 512], 1),
    "1.86": ([832, 448], 1),
    "2.00": ([896, 448], 1),
    "2.50": ([960, 384], 1),
    "2.83": ([1088, 384], 1),
    "3.60": ([1152, 320], 1),
    "3.80": ([1216, 320], 1),
    "4.00": ([1280, 320], 1),
}


ASPECT_RATIO_960 = {
    "0.22": ([448, 2048], 1),
    "0.29": ([512, 1792], 1),
    "0.36": ([576, 1600], 1),
    "0.45": ([640, 1408], 1),
    "0.55": ([704, 1280], 1),
    "0.63": ([768, 1216], 1),
    "0.76": ([832, 1088], 1),
    "0.88": ([896, 1024], 1),
    "1.00": ([960, 960], 1),
    "1.14": ([1024, 896], 1),
    "1.31": ([1088, 832], 1),
    "1.50": ([1152, 768], 1),
    "1.58": ([1216, 768], 1),
    "1.82": ([1280, 704], 1),
    "1.91": ([1344, 704], 1),
    "2.20": ([1408, 640], 1),
    "2.30": ([1472, 640], 1),
    "2.67": ([1536, 576], 1),
    "2.89": ([1664, 576], 1),
    "3.62": ([1856, 512], 1),
    "3.75": ([1920, 512], 1),
}
