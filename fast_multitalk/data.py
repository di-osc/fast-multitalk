import subprocess
import os
import tempfile
from pathlib import Path

import torch
import imageio
from tqdm import tqdm
import numpy as np
import soundfile as sf

from .utils import cache_video


def save_video_ffmpeg(
    save_path: str,
    gen_video_samples: torch.Tensor,
    audio_samples: np.ndarray,
    audio_sample_rate: int = 16000,
    fps: int = 25,
    quality: int = 9,
    high_quality_save: bool = False,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path_tmp = Path(temp_dir) / "video-temp.mp4"
        if high_quality_save:
            cache_video(
                tensor=gen_video_samples.unsqueeze(0),
                save_file=save_path_tmp,
                fps=fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            video_audio = (gen_video_samples + 1) / 2  # C T H W
            video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
            video_audio = np.clip(video_audio * 255, 0, 255).astype(
                np.uint8
            )  # to [0, 255]
            save_video(video_audio, save_path_tmp, fps=fps, quality=quality)

        # random name for audio
        audio_save_path = Path(temp_dir) / "audio-temp.wav"
        save_audio(audio_samples, audio_save_path, audio_sample_rate)

        # crop audio according to video length
        _, T, _, _ = gen_video_samples.shape
        duration = T / fps
        save_path_crop_audio = Path(temp_dir) / "audio-crop-temp.wav"
        final_command = [
            "ffmpeg",
            "-i",
            audio_save_path,
            "-t",
            f"{duration}",
            save_path_crop_audio,
        ]
        subprocess.run(
            final_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if high_quality_save:
            final_command = [
                "ffmpeg",
                "-y",
                "-i",
                save_path_tmp,
                "-i",
                save_path_crop_audio,
                "-c:v",
                "libopenh264",
                "-crf",
                "0",
                "-preset",
                "veryslow",
                "-c:a",
                "aac",
                "-shortest",
                save_path,
            ]
            subprocess.run(
                final_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            final_command = [
                "ffmpeg",
                "-y",
                "-i",
                save_path_tmp,
                "-i",
                save_path_crop_audio,
                "-c:v",
                "libopenh264",
                "-c:a",
                "aac",
                "-shortest",
                save_path,
            ]
            subprocess.run(
                final_command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)
        os.remove(audio_save_path)


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def save_audio(audio_array: np.ndarray, save_path: str, sample_rate: int):
    sf.write(save_path, audio_array, sample_rate)
