from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Sequence
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from .tensor_utils import factor_int_balanced
from .tt_linear import TTLinear


DEFAULT_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass
class TTDecompositionSummary:
    layer_idx: int
    module_name: str
    dense_params: int
    tt_params: int
    compression_ratio: float
    in_modes: List[int]
    out_modes: List[int]
    tt_ranks: List[int]


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module: nn.Module = model
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _get_llama_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Expected a Hugging Face LlamaForCausalLM-like model with model.layers")


def projection_mode_map(
    layer_mlp: nn.Module,
    *,
    order: int,
    hidden_modes: Sequence[int] | None = None,
    intermediate_modes: Sequence[int] | None = None,
) -> Dict[str, tuple[List[int], List[int]]]:
    hidden_size = int(layer_mlp.gate_proj.in_features)
    intermediate_size = int(layer_mlp.gate_proj.out_features)

    if hidden_modes is None:
        hidden_modes = factor_int_balanced(hidden_size, order=order)
    if intermediate_modes is None:
        intermediate_modes = factor_int_balanced(intermediate_size, order=order)

    hidden_modes = list(hidden_modes)
    intermediate_modes = list(intermediate_modes)
    return {
        "gate_proj": (hidden_modes, intermediate_modes),
        "up_proj": (hidden_modes, intermediate_modes),
        "down_proj": (intermediate_modes, hidden_modes),
    }


@torch.no_grad()
def replace_llama_ffn_with_tt(
    model: nn.Module,
    layer_indices: Sequence[int],
    tt_rank: int | Sequence[int],
    *,
    order: int = 12,
    hidden_modes: Sequence[int] | None = None,
    intermediate_modes: Sequence[int] | None = None,
    projections: Sequence[str] = DEFAULT_MLP_PROJECTIONS,
    decompose_dtype: torch.dtype = torch.float64,
    decompose_device: torch.device | str = "cpu",
    token_chunk_size: int | None = None,
    clear_cuda_cache: bool = True,
    algorithm: str = "svd",
    keep_interleaved_tt_tensor: bool = False,
) -> List[TTDecompositionSummary]:
    """
    Replace selected Llama MLP projections with TTLinear.
    """
    layers = _get_llama_layers(model)
    summaries: List[TTDecompositionSummary] = []

    for layer_idx in tqdm(layer_indices, desc='Decomposing layers'):
        mlp = layers[layer_idx].mlp
        mode_map = projection_mode_map(
            mlp,
            order=order,
            hidden_modes=hidden_modes,
            intermediate_modes=intermediate_modes,
        )

        for proj_name in projections:
            dense_linear = getattr(mlp, proj_name)
            in_modes, out_modes = mode_map[proj_name]
            tt_linear = TTLinear.from_linear(
                dense_linear,
                tt_rank=tt_rank,
                in_modes=in_modes,
                out_modes=out_modes,
                order=order,
                decompose_dtype=decompose_dtype,
                decompose_device=decompose_device,
                output_device=dense_linear.weight.device,
                token_chunk_size=token_chunk_size,
                algorithm=algorithm,
                keep_interleaved_tt_tensor=keep_interleaved_tt_tensor,
            )
            setattr(mlp, proj_name, tt_linear)

            dense_params = dense_linear.weight.numel()
            if dense_linear.bias is not None:
                dense_params += dense_linear.bias.numel()

            summaries.append(
                TTDecompositionSummary(
                    layer_idx=int(layer_idx),
                    module_name=f"model.layers.{layer_idx}.mlp.{proj_name}",
                    dense_params=int(dense_params),
                    tt_params=tt_linear.num_tt_parameters(),
                    compression_ratio=tt_linear.compression_ratio(),
                    in_modes=list(in_modes),
                    out_modes=list(out_modes),
                    tt_ranks=tt_linear.tt_ranks,
                )
            )

            del dense_linear
            gc.collect()
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return summaries


@torch.no_grad()
def replace_tt_with_dense_reconstruction(
    model: nn.Module,
    module_names: Sequence[str] | None = None,
) -> List[str]:
    """
    In-place replace TTLinear with dense nn.Linear reconstructions.
    """
    if module_names is None:
        module_names = [
            name for name, module in model.named_modules() if isinstance(module, TTLinear)
        ]

    replaced: List[str] = []
    for module_name in module_names:
        module = get_module_by_name(model, module_name)
        if not isinstance(module, TTLinear):
            continue
        set_module_by_name(model, module_name, module.to_linear())
        replaced.append(module_name)
        del module

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced


@torch.no_grad()
def collect_tt_module_stats(model: nn.Module) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, module in model.named_modules():
        if isinstance(module, TTLinear):
            rows.append(
                {
                    "module": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "in_modes": list(module.in_modes),
                    "out_modes": list(module.out_modes),
                    "tt_ranks": list(module.tt_ranks),
                    "dense_params": module.dense_parameter_count(),
                    "tt_params": module.num_tt_parameters(),
                    "compression_ratio": module.compression_ratio(),
                    "token_chunk_size": module.token_chunk_size,
                }
            )
    return rows
