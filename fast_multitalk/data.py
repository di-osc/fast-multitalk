import subprocess
import os
import tempfile
from pathlib import Path

import torch
import imageio
import numpy as np
import soundfile as sf
from loguru import logger


def save_video_ffmpeg(
    save_path: str | Path,
    gen_video_samples: torch.Tensor,
    audio_samples: np.ndarray,
    audio_sample_rate: int = 16000,
    fps: int = 25,
    quality: int = 9,
    encoders: list[str] = ["libx264", "libopenh264"],
    merge_video_audio: bool = True,
    force_9_16: bool = False,
):
    """使用ffmpeg命令保存视频和音频到指定路径,视频保存为mp4格式,音频保存为wav格式,并使用指定的编码器进行编码.

    Args:
        save_path (str): 保存路径
        gen_video_samples (torch.Tensor): 视频样本
        audio_samples (np.ndarray): 音频样本
        audio_sample_rate (int, optional): 音频采样率. Defaults to 16000.
        fps (int, optional): 视频帧率. Defaults to 25.
        quality (int, optional): 视频质量. Defaults to 9.
        encoders (list[str], optional): 视频编码器. Defaults to ["libopenh264", "libx264"]. 可选择多个编码器,先尝试第一个,如果失败则尝试第二个.
        force_9_16 (bool, optional): Whether to force 9:16 aspect ratio. Defaults to False.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 找出ffmpeg支持的编码器
    encoders = [encoder for encoder in encoders if ffmpeg_has_encoder(encoder)]
    if len(encoders) == 0:
        logger.error("No supported encoders found")
        raise RuntimeError("No supported encoders found")
    encoder = encoders[0]
    logger.info(f"Using encoder: {encoder}")
    with tempfile.TemporaryDirectory() as temp_dir:
        video_audio = (gen_video_samples + 1) / 2  # C T H W
        video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
        video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
        save_path_tmp = Path(temp_dir) / "video-temp.mp4"
        logger.info(f"Saving video to {save_path_tmp}")
        save_video(video_audio, save_path_tmp, fps=fps, quality=quality)
        if force_9_16:
            logger.info("Forcing 9:16 aspect ratio")
            output_path = Path(temp_dir) / "video-temp-9-16.mp4"
            # ffmpeg强制9：16尺寸
            cmd = [
                "ffmpeg",
                "-i",
                str(save_path_tmp),
                "-vf",
                "scale=468:832:flags=lanczos",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                encoder,
                "-b:v",
                "6M",
                "-c:a",
                "copy",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            save_path_tmp = output_path
            logger.info(f"9:16 video saved to {save_path_tmp}")
        if not merge_video_audio:
            logger.info("Skipping video and audio merge")
            # 仅ffmpeg保存视频到指定路径,使用encoder进行编码
            result = None
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                str(save_path_tmp),
                "-c:v",
                encoder,
                str(save_path),
            ]
            try:
                result = subprocess.run(
                    ffmpeg_command, check=True, stderr=subprocess.PIPE
                )
                logger.info(
                    f"Successfully saved video to {save_path} with encoder {encoder}"
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Failed to save video to {save_path} with encoder {encoder}: {e.stderr.decode('utf-8')}"
                )
            if result is None or result.returncode != 0:
                logger.error("Failed to save video to {save_path} with any encoder")
                raise RuntimeError("Failed to save video to {save_path}")
            os.remove(save_path_tmp)
            return

        # random name for audio
        audio_save_path = Path(temp_dir) / "audio-temp.wav"
        logger.info(f"Saving audio to {audio_save_path}")
        save_audio(audio_samples, audio_save_path, audio_sample_rate)

        # crop audio according to video length
        _, T, _, _ = gen_video_samples.shape
        duration = T / fps
        save_path_crop_audio = Path(temp_dir) / f"audio-crop-temp-{duration}.wav"
        logger.info(f"Cropping audio to {save_path_crop_audio}")
        crop_audio_command = [
            "ffmpeg",
            "-i",
            audio_save_path,
            "-t",
            f"{duration}",
            save_path_crop_audio,
        ]
        subprocess.run(
            crop_audio_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info(f"Merging video and audio with encoder: {encoder}")
        merge_video_audio_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(save_path_tmp),
            "-i",
            str(save_path_crop_audio),
            "-c:v",
            encoder,
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(save_path),
        ]
        # 合并视频和音频
        subprocess.run(merge_video_audio_command, check=True, stderr=subprocess.PIPE)
        logger.info(f"Successfully merged video and audio to {save_path}")
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)
        os.remove(audio_save_path)
        logger.info("Removed all temporary files")


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def save_audio(audio_array: np.ndarray, save_path: str, sample_rate: int):
    sf.write(save_path, audio_array, sample_rate)


def ffmpeg_has_encoder(name: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return f" {name} " in proc.stdout
