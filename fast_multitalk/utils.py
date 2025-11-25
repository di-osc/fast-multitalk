from contextlib import contextmanager
import math
import argparse
import os.path as osp
import binascii
import os
import random
import requests
import uuid

import torch
import torchvision
import imageio
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32:
                module._buffers[name] = buf.to(param_dtype)


def resize_and_centercrop(cond_image, target_size):
    """
    Resize image or tensor to the target size without padding.
    """

    # Get the original size
    if isinstance(cond_image, torch.Tensor):
        _, orig_h, orig_w = cond_image.shape
    else:
        orig_h, orig_w = cond_image.height, cond_image.width

    target_h, target_w = target_size

    # Calculate the scaling factor for resizing
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    # Compute the final size
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    # Resize
    if isinstance(cond_image, torch.Tensor):
        if len(cond_image.shape) == 3:
            cond_image = cond_image[None]
        resized_tensor = nn.functional.interpolate(
            cond_image, size=(final_h, final_w), mode="nearest"
        ).contiguous()
        # crop
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor.squeeze(0)
    else:
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)
        # tensor and crop
        resized_tensor = (
            torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        )
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor[:, :, None, :, :]

    return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def cache_video(
    tensor,
    save_file=None,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5,
):
    # cache file
    cache_file = (
        osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file
    )

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack(
                [
                    torchvision.utils.make_grid(
                        u, nrow=nrow, normalize=normalize, value_range=value_range
                    )
                    for u in tensor.unbind(2)
                ],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec="libopenh264", quality=8
            )
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f"cache_video failed, error: {error}", flush=True)
        return None


def cache_image(
    tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1), retry=5
):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".gif", ".webp"]:
        suffix = ".png"

    # save to cache
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range,
            )
            return save_file
        except Exception:
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")


def download_file(url: str, work_dir: str, filename: str = str(uuid.uuid4())) -> str:
    local_filename = filename
    try:
        local_filename = f"{work_dir}/{local_filename}"
        logger.info(f"输入文件：{url}，下载到本地文件：{local_filename}")
        with requests.get(url, stream=True, timeout=30000) as response:
            response.raise_for_status()  # 检查请求是否成功（状态码200）
            # 获取文件总大小（用于进度显示）
            total_size = int(response.headers.get("content-length", 0))
            # 以二进制写模式打开本地文件
            with open(local_filename, "wb") as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉保持连接的空chunk
                        file.write(chunk)
                        downloaded += len(chunk)
                        # 可选：显示下载进度
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(
                                f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                                end="",
                                flush=True,
                            )

            logger.info(f"\n文件已成功下载并保存为: {local_filename}")
            return local_filename

    except requests.exceptions.RequestException as e:
        logger.error(f"下载失败: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"保存文件时出错: {str(e)}")
        return None
