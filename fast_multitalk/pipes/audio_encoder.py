from einops import rearrange
import os
import subprocess
from typing import Literal
import json

import librosa
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import torch
import pyloudnorm as pyln

from ..models.wav2vec import Wav2Vec2Model


class AudioEncoder:
    def __init__(
        self, checkpoint_dir: str, device: str = "cpu", sample_rate: int = 16000
    ):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            checkpoint_dir, local_files_only=True
        )
        self.model.feature_extractor._freeze_parameters()
        self.model.eval().requires_grad_(False)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            checkpoint_dir, local_files_only=True
        )
        self.model.to(device)
        self.sample_rate = sample_rate
        self.device = device

    @torch.inference_mode()
    def get_embedding(self, speech_array: np.ndarray):
        sr = self.sample_rate
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25  # Assume the video fps is 25

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        embeddings = self.model(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )
        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()
        return audio_emb

    def process_input_data(self, input_data: str | dict) -> dict:
        if isinstance(input_data, str):
            with open(input_data, "r") as f:
                input_data = json.load(f)
        if len(input_data["cond_audio"]) == 2:
            new_human_speech1, new_human_speech2, sum_human_speechs = (
                audio_prepare_multi(
                    input_data["cond_audio"]["person1"],
                    input_data["cond_audio"]["person2"],
                    input_data["audio_type"],
                    sample_rate=self.sample_rate,
                )
            )
            audio_embedding_1 = self.get_embedding(new_human_speech1)
            audio_embedding_2 = self.get_embedding(new_human_speech2)
            input_data["cond_audio"]["person1"] = audio_embedding_1
            input_data["cond_audio"]["person2"] = audio_embedding_2
            input_data["video_audio"] = sum_human_speechs
        elif len(input_data["cond_audio"]) == 1:
            human_speech = audio_prepare_single(
                input_data["cond_audio"]["person1"], sample_rate=self.sample_rate
            )
            audio_embedding = self.get_embedding(human_speech)
            input_data["cond_audio"]["person1"] = audio_embedding
            input_data["video_audio"] = human_speech
        return input_data


def audio_prepare_single(audio_path: str, sample_rate: int) -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def audio_prepare_multi(
    left_path: str,
    right_path: str,
    audio_type: Literal["para", "add"] = "add",
    sample_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (left_path == "None" or right_path == "None"):
        human_speech_array1 = audio_prepare_single(left_path, sample_rate)
        human_speech_array2 = audio_prepare_single(right_path, sample_rate)
    elif left_path == "None":
        human_speech_array2 = audio_prepare_single(right_path, sample_rate)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == "None":
        human_speech_array1 = audio_prepare_single(left_path, sample_rate)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type == "para":
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == "add":
        new_human_speech1 = np.concatenate(
            [
                human_speech_array1[: human_speech_array1.shape[0]],
                np.zeros(human_speech_array2.shape[0]),
            ]
        )
        new_human_speech2 = np.concatenate(
            [
                np.zeros(human_speech_array1.shape[0]),
                human_speech_array2[: human_speech_array2.shape[0]],
            ]
        )
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split("/")[-1].split(".")[0] + ".wav"
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array
