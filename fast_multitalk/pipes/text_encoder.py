import json
import os
from typing import List, Dict
import time

import torch
from loguru import logger
from optimum.quanto import requantize
from safetensors.torch import load_file

from ..models.t5 import umt5_xxl
from ..tokenizer import HuggingfaceTokenizer
from ..utils import torch_gc


class TextEncoder:
    def __init__(
        self,
        quant_dir: str,
        text_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_path: str = None,
        tokenizer_path: str = None,
        quant: str = "int8",
        cache_contexts: List[str] = [],
        **kwargs,
    ):
        assert quant is None or quant in ("int8", "fp8")
        self.text_len = text_len
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
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
        model_state_dict = load_file(os.path.join(quant_dir, f"t5_{quant}.safetensors"))
        with open(os.path.join(quant_dir, f"t5_map_{quant}.json"), "r") as f:
            quantization_map = json.load(f)
        requantize(model, model_state_dict, quantization_map, device="cpu")
        self.model = model
        self.model.eval().requires_grad_(False)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean="whitespace"
        )
        self.device = "cpu"
        self.cache_contexts: Dict[str, torch.Tensor] = {}
        for context in cache_contexts:
            self.model.to("cuda")
            self.cache_contexts[context] = self.predict(context).to("cpu")
            self.model.to("cpu")
        torch_gc()

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        results = []
        for text in texts:
            if text in self.cache_contexts:
                logger.info(f"Using cached context for {text}")
                results.append(self.cache_contexts[text].to("cuda"))
            else:
                self.onload()
                results.append(self.predict(text))
        self.offload()
        torch_gc()
        return results

    def offload(self):
        if self.device == "cpu":
            return
        logger.info("Offloading text encoder to CPU")
        start_time = time.time()
        self.model.to("cpu")
        end_time = time.time()
        logger.info(f"Text encoder offload time: {end_time - start_time} seconds")
        self.device = "cpu"

    def onload(self):
        if self.device == "cuda":
            return
        logger.info("Loading text encoder to cuda")
        start_time = time.time()
        self.model.to("cuda")
        end_time = time.time()
        logger.info(f"Text encoder onload time: {end_time - start_time} seconds")
        self.device = "cuda"

    def predict(self, text: str) -> torch.Tensor:
        start_time = time.time()
        device = "cuda"
        ids, mask = self.tokenizer([text], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        outputs = [u[:v] for u, v in zip(context, seq_lens)][0]
        end_time = time.time()
        logger.info(f"Text encoder predict time: {end_time - start_time} seconds")
        return outputs
