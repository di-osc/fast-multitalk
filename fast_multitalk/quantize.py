import time
from typing import Literal

from torchao.quantization import (
    autoquant,
    quantize_,
    FPXWeightOnlyConfig,
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
)
from torchao.quantization.quant_api import PerRow
from torchao.sparsity import sparsify_
from torchao.dtypes import SemiSparseLayout
from optimum.quanto import quantize, qint8, freeze, qfloat8
import torch
from loguru import logger


def quantize_model(
    model: torch.nn.Module,
    quant: Literal[
        "qint8",
        "qfloat8",
        "fp8wo",
        "fp8dq",
        "fp8dqrow",
        "fp6_e3m2",
        "fp6_e2m3",
        "fp5_e2m2",
        "fp4_e2m1",
        "int8wo",
        "int8dq",
        "int4dq",
        "int4wo",
        "autoquant",
        "sparsify",
    ],
    device: str = "cpu",
    exclude: list[str] = [],
) -> None:
    logger.info(f"Quantizing model to {quant}")
    start_time = time.perf_counter()
    match quant:
        case "qint8":
            quantize(model=model, weights=qint8, exclude=exclude)
            freeze(model)
        case "qfloat8":
            quantize(model=model, weights=qfloat8, exclude=exclude)
            freeze(model)
        case "fp8wo":
            quantize_(model, Float8WeightOnlyConfig(), device=device)
        case "fp8dq":
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(),
                device=device,
            )
        case "fp8dqrow":
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
                device=device,
            )
        case "fp6_e3m2":
            quantize_(
                model,
                FPXWeightOnlyConfig(3, 2),
                device=device,
            )
        case "fp6_e2m3":
            quantize_(
                model,
                FPXWeightOnlyConfig(2, 3),
                device=device,
            )
        case "fp5_e2m2":
            quantize_(
                model,
                FPXWeightOnlyConfig(2, 2),
                device=device,
            )
        case "fp4_e2m1":
            quantize_(
                model,
                FPXWeightOnlyConfig(2, 1),
                device=device,
            )
        case "int8wo":
            quantize_(model, Int8WeightOnlyConfig(), device=device)
        case "int8dq":
            quantize_(
                model,
                Int8DynamicActivationInt8WeightConfig(),
                device=device,
            )
        case "int4dq":
            quantize_(
                model,
                Int8DynamicActivationInt4WeightConfig(),
                device=device,
            )
        case "int4wo":
            quantize_(model, Int4WeightOnlyConfig(), device=device)
        case "autoquant":
            autoquant(
                model,
                error_on_unseen=False,
                device=device,
            )
        case "sparsify":
            sparsify_(
                model,
                Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()),
            )
        case _:
            raise ValueError(f"Invalid quant: {quant}")
    end_time = time.perf_counter()
    logger.info(f"Quantization time: {end_time - start_time} seconds")
