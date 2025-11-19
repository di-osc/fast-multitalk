import json
import os

import torch
from loguru import logger
from optimum.quanto import requantize
from safetensors.torch import load_file

from ..models.t5 import umt5_xxl
from ..tokenizer import HuggingfaceTokenizer


class TextEncoder:
    def __init__(
        self,
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.cuda.current_device(),
        checkpoint_path: str = None,
        tokenizer_path: str = None,
        shard_fn: callable = None,
        quant: str = None,
        quant_dir: str = None,
    ):
        assert quant is None or quant in ("int8", "fp8")
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        if quant is not None:
            with torch.device("meta"):
                model = umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=dtype,
                    device=torch.device("meta"),
                )
            logger.info(
                f"Loading quantized T5 from {os.path.join(quant_dir, f't5_{quant}.safetensors')}"
            )
            model_state_dict = load_file(
                os.path.join(quant_dir, f"t5_{quant}.safetensors")
            )
            with open(os.path.join(quant_dir, f"t5_map_{quant}.json"), "r") as f:
                quantization_map = json.load(f)
            requantize(model, model_state_dict, quantization_map, device="cpu")
        else:
            model = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=dtype,
                    device=device,
                )
                .eval()
                .requires_grad_(False)
            )
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
        self.model = model
        self.model.eval().requires_grad_(False)
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean="whitespace"
        )

    def __call__(self, texts: list[str], device: torch.device) -> list[torch.Tensor]:
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
