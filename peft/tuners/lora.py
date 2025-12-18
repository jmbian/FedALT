# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].
    """
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    asymmetric: bool = field(
        default=True,
        metadata={"help": "If True, use 1 A and multiple B matrices. If False, use multiple A and B matrices."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.
    """
    def __init__(self, config: LoraConfig, model: torch.nn.Module):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_8bit or loaded_in_4bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 4-bit/8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "asymmetric": self.peft_config.asymmetric,
        }

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                # Handle 8-bit quantization
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update({
                        "has_fp16_weights": target.state.has_fp16_weights,
                        "memory_efficient_backward": target.state.memory_efficient_backward,
                        "threshold": target.state.threshold,
                        "index": target.index,
                    })
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )

                elif loaded_in_4bit and isinstance(target, bnb.nn.Linear4bit):
                    kwargs.update({
                        "compute_dtype": target.compute_dtype,
                        "compress_statistics": target.weight.compress_statistics,
                        "quant_type": target.weight.quant_type,
                    })
                    if self.peft_config.enable_lora is None:
                        new_module = Linear4bit(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear4bit(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )

                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(
                        in_features=target.in_features,
                        out_features=target.out_features,
                        r=self.peft_config.r,
                        lora_alpha=self.peft_config.lora_alpha,
                        lora_nums=self.peft_config.lora_nums,
                        lora_dropout=self.peft_config.lora_dropout,
                        fan_in_fan_out=self.peft_config.fan_in_fan_out,
                        merge_weights=self.peft_config.merge_weights,
                        asymmetric=self.peft_config.asymmetric,
                    )

                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    """LoRA implemented in a dense layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        asymmetric: bool = True,
        **kwargs,
    ):
        self.r = r  
        self.lora_alpha = lora_alpha
        self.lora_num = lora_nums
        self.asymmetric = asymmetric
        self.fan_in_fan_out = fan_in_fan_out
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            if self.asymmetric:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                for i in range(self.lora_num):
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            else:
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)  
        
        if hasattr(self, 'r') and self.r > 0: 
            if self.asymmetric:
                if hasattr(self, "lora_A"):
                    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()
            else:
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_A{i}"):
                        nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()

            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, "lora_route"):
            self.lora_route.train(mode)
        if self.asymmetric:
            if hasattr(self, "lora_A"):
                self.lora_A.train(mode)
            for i in range(self.lora_num):
                getattr(self, f"lora_B{i}").train(mode)
        else:
            for i in range(self.lora_num):
                getattr(self, f"lora_A{i}").train(mode)
                getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        if hasattr(self, "lora_route"):
            self.lora_route.eval()
        if self.asymmetric:
            if hasattr(self, "lora_A"):
                self.lora_A.eval()
            for i in range(self.lora_num):
                getattr(self, f"lora_B{i}").eval()
        else:
            for i in range(self.lora_num):
                getattr(self, f"lora_A{i}").eval()
                getattr(self, f"lora_B{i}").eval()

    def forward(self, x: torch.Tensor):
        weight = self.weight

        if self.disable_adapters:
            result = F.linear(x, transpose(weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(weight, self.fan_in_fan_out), bias=self.bias)

            if self.r > 0:
                route_logits = self.lora_route(x)
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)

                if self.asymmetric:
                    lora_A_output = self.lora_A(self.lora_dropout(x))
                    for i in range(self.lora_num):
                        lora_B = getattr(self, f"lora_B{i}")
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling
                else:
                    dropped_x = self.lora_dropout(x)
                    for i in range(self.lora_num):
                        lora_A = getattr(self, f"lora_A{i}")
                        lora_B = getattr(self, f"lora_B{i}")
                        lora_A_output = lora_A(dropped_x)
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling

        blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
        return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
        

class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    def __init__(
        self,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        asymmetric: bool = True,
        **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        
        self.lora_num = lora_nums
        self.asymmetric = asymmetric
        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            if self.asymmetric:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                for i in range(self.lora_num):
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            else:
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'r') and self.r > 0:
            if self.asymmetric:
                if hasattr(self, "lora_A"):
                    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()
            else:
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_A{i}"):
                        nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()
            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
                    
                route_logits = self.lora_route(x)
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(expected_dtype)
                
                if self.asymmetric:
                    lora_A_output = self.lora_A(self.lora_dropout(x))
                    for i in range(self.lora_num):
                        lora_B = getattr(self, f"lora_B{i}")
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling
                else:
                    dropped_x = self.lora_dropout(x)
                    for i in range(self.lora_num):
                        lora_A = getattr(self, f"lora_A{i}")
                        lora_B = getattr(self, f"lora_B{i}")
                        lora_A_output = lora_A(dropped_x)
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling
                result = result.to(expected_dtype)
            else:
                route_logits = self.lora_route(x)
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
                
                if self.asymmetric:
                    lora_A_output = self.lora_A(self.lora_dropout(x))
                    for i in range(self.lora_num):
                        lora_B = getattr(self, f"lora_B{i}")
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling
                else:
                    dropped_x = self.lora_dropout(x)
                    for i in range(self.lora_num):
                        lora_A = getattr(self, f"lora_A{i}")
                        lora_B = getattr(self, f"lora_B{i}")
                        lora_A_output = lora_A(dropped_x)
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        lora_output = lora_B(lora_A_output)
                        result = result + scaled_route * lora_output * self.scaling
        
        blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
        return result


class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        asymmetric: bool = True,
        **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
            
        self.enable_lora = enable_lora
        self.lora_num = lora_nums
        self.asymmetric = asymmetric
        
        if r > 0 and any(enable_lora):
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            if self.asymmetric:
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                for i in range(self.lora_num):
                    setattr(self, f"lora_B{i}", nn.Conv1d(
                        r * sum(enable_lora),
                        out_features // len(enable_lora) * sum(enable_lora),
                        kernel_size=1,
                        groups=sum(enable_lora),
                        bias=False
                    ))
            else:
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, r * sum(enable_lora), bias=False))
                    setattr(self, f"lora_B{i}", nn.Conv1d(
                        r * sum(enable_lora),
                        out_features // len(enable_lora) * sum(enable_lora),
                        kernel_size=1,
                        groups=sum(enable_lora),
                        bias=False
                    ))
            
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'r') and self.r > 0:
            if self.asymmetric:
                if hasattr(self, "lora_A"):
                    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()
            else:
                for i in range(self.lora_num):
                    if hasattr(self, f"lora_A{i}"):
                        nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                    if hasattr(self, f"lora_B{i}"):
                        getattr(self, f"lora_B{i}").weight.data.zero_()
            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, task_types=None):
        result = super().forward(x)
        
        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
                
                route_logits = self.lora_route(x)
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(expected_dtype)
                
                if self.asymmetric:
                    lora_A_output = self.lora_A(self.lora_dropout(x))
                    after_A = lora_A_output.transpose(-2, -1)
                    for i in range(self.lora_num):
                        lora_B = getattr(self, f"lora_B{i}")
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        after_B = lora_B(after_A).transpose(-2, -1)
                        output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                        result = result + scaled_route * output
                else:
                    dropped_x = self.lora_dropout(x)
                    for i in range(self.lora_num):
                        lora_A = getattr(self, f"lora_A{i}")
                        lora_B = getattr(self, f"lora_B{i}")
                        after_A = lora_A(dropped_x).transpose(-2, -1)
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        after_B = lora_B(after_A).transpose(-2, -1)
                        output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                        result = result + scaled_route * output
                result = result.to(expected_dtype)
            else:
                route_logits = self.lora_route(x) 
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
                
                if self.asymmetric:
                    lora_A_output = self.lora_A(self.lora_dropout(x))
                    after_A = lora_A_output.transpose(-2, -1)
                    for i in range(self.lora_num):
                        lora_B = getattr(self, f"lora_B{i}")
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        after_B = lora_B(after_A).transpose(-2, -1)
                        output = self.zero_pad(after_B) * self.scaling
                        result = result + scaled_route * output
                else:
                    dropped_x = self.lora_dropout(x)
                    for i in range(self.lora_num):
                        lora_A = getattr(self, f"lora_A{i}")
                        lora_B = getattr(self, f"lora_B{i}")
                        after_A = lora_A(dropped_x).transpose(-2, -1)
                        scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                        after_B = lora_B(after_A).transpose(-2, -1)
                        output = self.zero_pad(after_B) * self.scaling
                        result = result + scaled_route * output
                        
        blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
        return result