import copy
import time

import torch
import optimum.quanto.nn.qlinear as qlinear

from .utils import init_weights_on_device


IO_SPENT = {
    "other": 0,
    "cast_to": 0,
    "cast_to_device": 0,
    "layer_infer": 0,
    "model_infer": 0,
}


def reset_io_spent():
    global IO_SPENT
    IO_SPENT["other"] = 0
    IO_SPENT["cast_to"] = 0
    IO_SPENT["cast_to_device"] = 0
    IO_SPENT["layer_infer"] = 0


def cast_to(weight, dtype, device):
    global IO_SPENT
    start = time.perf_counter()
    # Handle quantized tensors from torchao
    if hasattr(weight, "__class__") and "torchao.dtypes" in str(weight.__class__):
        r = weight.to(device=device)
    elif hasattr(weight, "__class__") and "torchao.quantization" in str(
        weight.__class__
    ):
        r = weight.to(dtype=dtype, device=device)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight)
    end = time.perf_counter()
    IO_SPENT["cast_to"] += end - start
    return r


def cast_to_device(weight, device):
    global IO_SPENT
    if hasattr(weight, "__class__") and "optimum.quanto" in str(weight.__class__):
        start = time.perf_counter()
        r = weight.to(device)
        end = time.perf_counter()

        IO_SPENT["cast_to_device"] += end - start
        return r
    else:
        start = time.perf_counter()
        r = torch.empty_like(weight, device=device)
        r.copy_(weight)
        end = time.perf_counter()
        IO_SPENT["cast_to_device"] += end - start
        return r


class AutoWrappedModule(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (
            self.offload_dtype != self.onload_dtype
            or self.offload_device != self.onload_device
        ):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (
            self.offload_dtype != self.onload_dtype
            or self.offload_device != self.onload_device
        ):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if (
            self.onload_dtype == self.computation_dtype
            and self.onload_device == self.computation_device
        ):
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(
                dtype=self.computation_dtype, device=self.computation_device
            )
        r = module(*args, **kwargs)
        return r


class AutoWrappedQLinear(qlinear.QLinear):
    def __init__(
        self,
        module: qlinear.QLinear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_device = offload_device

        self.onload_device = onload_device
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_device != self.onload_device):
            self.to(device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_device != self.onload_device):
            self.to(device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_device == self.computation_device:
            r = torch.nn.functional.linear(x, self.weight, bias=self.bias)
            return r
        else:
            qweight = cast_to_device(self.weight, self.computation_device)
            bias = (
                None
                if self.bias is None
                else cast_to_device(self.bias, self.computation_device)
            )
            r = torch.nn.functional.linear(x, qweight, bias)
            return r


class AutoWrappedLinear(torch.nn.Linear):
    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (
            self.offload_dtype != self.onload_dtype
            or self.offload_device != self.onload_device
        ):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (
            self.offload_dtype != self.onload_dtype
            or self.offload_device != self.onload_device
        ):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if (
            self.onload_dtype == self.computation_dtype
            and self.onload_device == self.computation_device
        ):
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(
                self.weight, self.computation_dtype, self.computation_device
            )
            bias = (
                None
                if self.bias is None
                else cast_to(self.bias, self.computation_dtype, self.computation_device)
            )
        r = torch.nn.functional.linear(x, weight, bias)
        return r


def enable_vram_management_recursively(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,
    total_num_param=0,
):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if (
                    max_num_param is not None
                    and total_num_param + num_param > max_num_param
                ):
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(
                module,
                module_map,
                module_config,
                max_num_param,
                overflow_module_config,
                total_num_param,
            )
    return total_num_param


def enable_vram_management(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,
):
    enable_vram_management_recursively(
        model,
        module_map,
        module_config,
        max_num_param,
        overflow_module_config,
        total_num_param=0,
    )
    model.vram_management_enabled = True
