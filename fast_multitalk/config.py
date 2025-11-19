from dataclasses import dataclass

import torch


@dataclass
class MultiTalkConfig:
    # t5
    t5_model = "umt5_xxl"
    t5_dtype = torch.bfloat16
    text_len = 512
    t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer = "google/umt5-xxl"

    # clip
    clip_model = "clip_xlm_roberta_vit_h_14"
    clip_dtype = torch.float16
    clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    clip_tokenizer = "xlm-roberta-large"

    # wav2vec
    wav2vec_model = "chinese-wav2vec2-base"

    # vae
    vae_checkpoint = "Wan2.1_VAE.pth"
    vae_stride = (4, 8, 8)

    # dit
    param_dtype = torch.bfloat16
    patch_size = (1, 2, 2)
    dim = 5120
    ffn_dim = 13824
    freq_dim = 256
    num_heads = 40
    num_layers = 40
    window_size = (-1, -1)
    qk_norm = True
    cross_attn_norm = True
    eps = 1e-6

    # inference
    num_train_timesteps = 1000
    sample_fps = 16
    sample_neg_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
