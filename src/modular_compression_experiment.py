from __future__ import annotations

import copy
import gc
import json
import math
import random
import re
import time
import warnings
import logging
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import platform
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import tntorch as tn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import tensorly as tl
from tensorly.decomposition import partial_tucker
from src.tt_llm import factor_int_balanced, get_module_by_name, infer_input_device, set_module_by_name
from src.td_moe import td_moe_compress
from src.utils.eval import eval_ppl
from src.head_decomp_methods import *

# Model registry

@dataclass(frozen=True)
class ModelArchitectureSpec:
    key: str
    default_model_name: str
    family: str
    block_prefix: str
    module_groups: dict[str, list[str]]
    attention_module_path: str
    mlp_module_path: str
    tensorllm_qkvo_order: tuple[str, str, str, str]
    tensorllm_qkvo_modules: dict[str, str]
    output_projection_names: tuple[str, ...]
    embedding_prefixes: tuple[str, ...]
    supports_tensorllm_4d_equal_qkvo: bool = True
    notes: str = ""
    # Path of the MoE router/gate Linear inside the block; "" if model has no MoE router.
    router_module_path: str = ""


MODEL_ARCH_REGISTRY: dict[str, ModelArchitectureSpec] = {
    "gptj": ModelArchitectureSpec(
        key="gptj",
        default_model_name="EleutherAI/gpt-j-6B",
        family="gptj",
        block_prefix="transformer.h",
        attention_module_path="attn",
        mlp_module_path="mlp",
        module_groups={
            "mha": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"],
            "mlp": ["mlp.fc_in", "mlp.fc_out"],
        },
        tensorllm_qkvo_order=("q", "k", "v", "o"),
        tensorllm_qkvo_modules={
            "q": "attn.q_proj",
            "k": "attn.k_proj",
            "v": "attn.v_proj",
            "o": "attn.out_proj",
        },
        output_projection_names=("attn.out_proj",),
        embedding_prefixes=("transformer.wte",),
        notes="GPT-J uses equal-size Q/K/V/O projections, so TensorLLM 4D QKVO is supported.",
    ),
    "llama": ModelArchitectureSpec(
        key="llama",
        default_model_name="meta-llama/Meta-Llama-3-8B",
        family="llama",
        block_prefix="model.layers",
        attention_module_path="self_attn",
        mlp_module_path="mlp",
        module_groups={
            "mha": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
            "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
        tensorllm_qkvo_order=("q", "k", "v", "o"),
        tensorllm_qkvo_modules={
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
        },
        output_projection_names=("self_attn.o_proj",),
        embedding_prefixes=("model.embed_tokens",),
        notes=(
            "Works for Llama-style decoder models. If K/V are smaller than Q/O due to GQA, "
            "automatic Tucker-on-MHA falls back to per-projection 3D Tucker unless disabled."
        ),
    ),
    "qwen3_30b_a3b": ModelArchitectureSpec(
        key="qwen3_30b_a3b",
        default_model_name="Qwen/Qwen3-30B-A3B",
        family="qwen3_moe",
        block_prefix="model.layers",
        module_groups={
            "moe_experts": ["mlp.experts"],
        },
        attention_module_path="self_attn",
        mlp_module_path="mlp",
        tensorllm_qkvo_order=("q", "k", "v", "o"),
        tensorllm_qkvo_modules={
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
        },
        output_projection_names=("self_attn.o_proj",),
        embedding_prefixes=("model.embed_tokens",),
        supports_tensorllm_4d_equal_qkvo=False,
        notes="Qwen3-30B-A3B text MoE.",
        router_module_path="mlp.gate",
    ),
    "gpt_oss_20b": ModelArchitectureSpec(
        key="gpt_oss_20b",
        default_model_name="openai/gpt-oss-20b",
        family="gpt_oss_moe",
        block_prefix="model.layers",
        module_groups={
            "moe_experts": ["mlp.experts"],
        },
        attention_module_path="self_attn",
        mlp_module_path="mlp",
        tensorllm_qkvo_order=("q", "k", "v", "o"),
        tensorllm_qkvo_modules={
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
        },
        output_projection_names=("self_attn.o_proj",),
        embedding_prefixes=("model.embed_tokens",),
        supports_tensorllm_4d_equal_qkvo=False,
        notes="OpenAI GPT-OSS 20B MoE: 32 experts, 4 active per token, 24 layers, GQA (64 Q / 8 KV heads).",
        router_module_path="mlp.router",
    ),
    "llama2": ModelArchitectureSpec(
        key="llama",
        default_model_name="meta-llama/Llama-2-7b-hf",
        family="llama",
        block_prefix="model.layers",
        attention_module_path="self_attn",
        mlp_module_path="mlp",
        module_groups={
            "mha": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
            "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
        tensorllm_qkvo_order=("q", "k", "v", "o"),
        tensorllm_qkvo_modules={
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
        },
        output_projection_names=("self_attn.o_proj",),
        embedding_prefixes=("model.embed_tokens",),
        notes=(
            "Works for Llama-style decoder models. If K/V are smaller than Q/O due to GQA, "
            "automatic Tucker-on-MHA falls back to per-projection 3D Tucker unless disabled."
        ),
    ),
}


def get_model_architecture(key: str) -> ModelArchitectureSpec:
    key = str(key).lower()
    if key not in MODEL_ARCH_REGISTRY:
        raise KeyError(f"Unknown model architecture {key!r}. Available: {sorted(MODEL_ARCH_REGISTRY)}")
    return MODEL_ARCH_REGISTRY[key]


def add_model_architecture(spec: ModelArchitectureSpec):
    MODEL_ARCH_REGISTRY[spec.key] = spec


def full_module_name_for(model_spec: ModelArchitectureSpec, layer_idx: int, module_name: str) -> str:
    module_name = str(module_name)
    if module_name.startswith(model_spec.block_prefix + "."):
        return module_name
    return f"{model_spec.block_prefix}.{int(layer_idx)}.{module_name}"


def group_modules(model_spec: ModelArchitectureSpec, group_or_module: str) -> list[str]:
    name = str(group_or_module)
    if name == "all":
        seen = []
        for group in model_spec.module_groups:
            for m in model_spec.module_groups.get(group, []):
                if m not in seen:
                    seen.append(m)
        return seen
    if name in model_spec.module_groups:
        return list(model_spec.module_groups[name])
    return [name]


def module_group_for_relative_name(model_spec: ModelArchitectureSpec, module_name: str) -> str:
    for group, names in model_spec.module_groups.items():
        if module_name in names:
            return group
    return "custom"


def unique_layer_indices(target_specs: Sequence[dict]) -> list[int]:
    return sorted({int(s["layer_idx"]) for s in target_specs})


def get_transformer_layers(model: nn.Module, model_spec: ModelArchitectureSpec):
    return get_module_by_name(model, model_spec.block_prefix)


def get_attention_module(block: nn.Module, model_spec: ModelArchitectureSpec):
    return get_module_by_name(block, model_spec.attention_module_path)


def get_mlp_module(block: nn.Module, model_spec: ModelArchitectureSpec):
    return get_module_by_name(block, model_spec.mlp_module_path)


# Config dataclasses

@dataclass(frozen=True)
class ModulePlan:
    target: str
    decomposition_method: str  # "tt", "dense_sparse", "tensorllm_tucker_mha", "none"
    rank: Optional[int] = None
    quant_method: str = "none"  # "none", "rtn_symmetric", "gptq_tt", "tensorllm_pre_reconstruct_rtn"
    quant_bits: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeSpec:
    label: str
    plans: list[ModulePlan]
    with_lora: bool = False
    lora_targets: Optional[list[str]] = None  # None means all planned targets.
    lora_params: dict[str, Any] = field(default_factory=dict)
    lora_quant_bits: Optional[int] = None
    lora_quant_method: str = "rtn_symmetric"
    benchmark_with_dense_reconstruction: bool = True


@dataclass
class RunnerConfig:
    model_arch_key: str
    model_name: str
    model_dtype: torch.dtype = torch.float16
    model_storage_bits: int = 16
    scale_storage_bits: int = 16
    lora_storage_bits: int = 16
    device_map: str | dict | None = "auto"
    output_dir: Path | str = Path("results_modular")
    decompose_dtype: torch.dtype = torch.float64
    decompose_device: str = "cuda"
    tt_order: int = 12
    token_chunk_size: Optional[int] = 16
    dense_sparse_outlier_fraction: float = 5e-7
    tensorllm_stack_rank: int = 2
    tensorllm_head_dim_rank: int = 4
    tensorllm_tucker_type: str = "partial_tucker_v5"
    tensorllm_allow_gqa_per_projection_fallback: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_lr: float = 2e-4
    lora_weight_decay: float = 0.0
    lora_max_steps: int = 100
    lora_grad_accum_steps: int = 8
    lora_train_batch_size: int = 1
    lora_train_seq_len: int = 256
    lora_num_train_sequences: int = 512
    run_ppl: bool = True
    ppl_datasets: list[str] = field(default_factory=lambda: ["wikitext2"])
    ppl_seqlen: int = 2048
    run_lm_eval: bool = False
    lm_eval_tasks: list[str] = field(default_factory=list)
    lm_eval_num_fewshot: int = 0
    lm_eval_batch_size: int | str = 1
    lm_eval_limit: Optional[int | float] = None
    lm_eval_log_samples: bool = False
    lm_eval_strict: bool = True
    run_hotpot: bool = False
    hotpot_max_examples: Optional[int] = None
    hotpot_batch_size: int = 16
    hotpot_max_new_tokens: int = 15
    hotpot_beam: int = 1
    hotpot_use_llama_tokenizer_filter: bool = True
    hotpot_filter_tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    run_generation_examples: bool = False
    generation_prompts: list[str] = field(default_factory=list)
    generation_max_new_tokens: int = 80
    run_activation_geometry: bool = False
    geometry_num_samples: int = 20
    geometry_seq_len: int = 512
    geometry_seed: int = 42
    run_sd_moe_diagnostics: bool = False
    sd_moe_top1pct_floor: int = 8
    sd_moe_head_band: tuple[float, float] = (0.0, 0.01)
    sd_moe_tail_band: tuple[float, float] = (0.01, 0.10)
    sd_moe_per_layer_dump_dir: Optional[str] = None
    sd_moe_topk_for_intervention: Optional[int] = None  # None -> use model's official top-k
    topk_outliers: int = 64
    random_error_samples: int = 256
    # Optional benchmark acceleration.
    use_torch_compile_for_benchmarks: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: Optional[str] = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune", or None
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: Optional[bool] = None
    torch_compile_warmup: bool = True
    torch_compile_warmup_steps: int = 2
    torch_compile_strict: bool = False
    quiet_torch_compile_warnings: bool = True
    quiet_transformers_loading_warnings: bool = True
    enable_tf32_for_benchmarks: bool = False
    save_partial_results: bool = True
    partial_results_name: str = "partial_results"

    @property
    def model_spec(self) -> ModelArchitectureSpec:
        return get_model_architecture(self.model_arch_key)


# Basic utilities

def _package_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def runtime_version_summary() -> dict:
    return {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "lm_eval": _package_version("lm_eval", "lm-eval", "lm-evaluation-harness"),
        "tensorly": _package_version("tensorly"),
        "tntorch": _package_version("tntorch"),
    }


def _root_exception(exc: BaseException) -> BaseException:
    root = exc
    seen = set()
    while True:
        if id(root) in seen:
            return root
        seen.add(id(root))
        nxt = getattr(root, "__cause__", None) or getattr(root, "__context__", None)
        if nxt is None:
            return root
        root = nxt


def exception_to_status_fields(exc: BaseException) -> dict:
    root = _root_exception(exc)
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_repr": repr(exc),
        "error_traceback": traceback.format_exc(),
        "error_root_type": type(root).__name__,
        "error_root_message": str(root),
        "runtime_versions": runtime_version_summary(),
    }


def exception_to_metric_fields(prefix: str, exc: BaseException) -> dict:
    root = _root_exception(exc)
    out = {
        f"{prefix}_error": f"{type(exc).__name__}: {exc}",
        f"{prefix}_error_repr": repr(exc),
        f"{prefix}_error_traceback": traceback.format_exc(),
        f"{prefix}_error_root_type": type(root).__name__,
        f"{prefix}_error_root_message": str(root),
        f"{prefix}_runtime_versions": runtime_version_summary(),
    }
    msg = f"{type(root).__name__}: {root}"
    if "Feature type 'List' not found" in msg:
        out[f"{prefix}_error_hint"] = (
            "Likely lm_eval/datasets task-cache or version incompatibility. "
            "Clear the Hugging Face datasets cache for lm_eval tasks and use a clean, compatible lm_eval/datasets install."
        )
    return out


def clean_mem():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def normalize_torch_device_name(device: str | torch.device) -> str:
    name = str(device).lower()
    if name == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_string(name: str):
    return getattr(torch, name.replace("torch.", ""))

def configure_runtime_noise_controls(cfg: RunnerConfig):
    if getattr(cfg, "enable_tf32_for_benchmarks", True):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if getattr(cfg, "quiet_transformers_loading_warnings", True):
        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

    if getattr(cfg, "quiet_torch_compile_warnings", True):
        warnings.filterwarnings(
            "ignore",
            message=r".*cudagraph partition.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*",
        )

        try:
            import torch._logging as torch_logging

            torch_logging.set_logs(
                dynamo=logging.ERROR,
                inductor=logging.ERROR,
                perf_hints=False,
                cudagraphs=False,
                graph_breaks=False,
                recompiles=False,
            )
        except Exception:
            pass

        for logger_name in [
            "torch._dynamo",
            "torch._inductor",
            "torch._inductor.compile_fx",
            "torch._inductor.cudagraph_trees",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)


def _model_device_summary(model: nn.Module) -> dict:
    param_devices = sorted({str(p.device) for p in model.parameters()})
    buffer_devices = sorted({str(b.device) for b in model.buffers() if torch.is_tensor(b) and b.numel() > 0})
    hf_device_map = getattr(model, "hf_device_map", None)
    return {
        "param_devices": param_devices,
        "buffer_devices": buffer_devices,
        "hf_device_map": None if hf_device_map is None else {str(k): str(v) for k, v in hf_device_map.items()},
    }


def _model_is_single_cuda_for_compile(model: nn.Module) -> tuple[bool, str]:
    summary = _model_device_summary(model)
    param_devices = summary["param_devices"]
    buffer_devices = summary["buffer_devices"]

    if not torch.cuda.is_available():
        return False, "CUDA is not available."
    if not param_devices:
        return False, "Model has no parameters."
    if any(d == "meta" for d in param_devices + buffer_devices):
        return False, f"Model has meta tensors: {summary}."
    if any(not d.startswith("cuda") for d in param_devices):
        return False, f"Model parameters are not all on CUDA: {summary}."
    if any((not d.startswith("cuda")) and d != "cpu" for d in buffer_devices):
        return False, f"Model buffers include unsupported devices: {summary}."
    cuda_param_devices = {d for d in param_devices if d.startswith("cuda")}
    if len(cuda_param_devices) != 1:
        return False, f"Model spans multiple CUDA devices; skipping torch.compile: {summary}."

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is not None:
        values = {str(v).lower() for v in hf_device_map.values()}
        if any(v in {"cpu", "disk", "meta"} for v in values):
            return False, f"Model uses CPU/disk/meta offload; skipping torch.compile: {summary}."
        cuda_like = all(v.isdigit() or v.startswith("cuda") for v in values)
        if not cuda_like:
            return False, f"Model device_map is not a single CUDA placement: {summary}."

    return True, ""


@torch.no_grad()
def maybe_compile_model_for_benchmarks(model, tokenizer, cfg: RunnerConfig):
    import torch
    configure_runtime_noise_controls(cfg)
    stats = {
        "torch_compile_requested": bool(cfg.use_torch_compile_for_benchmarks),
        "torch_compile_applied": False,
        "torch_compile_backend": None,
        "torch_compile_mode": None,
        "torch_compile_time_s": 0.0,
        "torch_compile_error": None,
        "torch_compile_error_traceback": None,
        "torch_compile_skip_reason": None,
        "torch_compile_device_summary": _model_device_summary(model),
    }

    if not cfg.use_torch_compile_for_benchmarks:
        return model, stats

    if not hasattr(torch, "compile"):
        stats["torch_compile_error"] = "torch.compile is not available in this PyTorch build."
        if cfg.torch_compile_strict:
            raise RuntimeError(stats["torch_compile_error"])
        return model, stats

    ok_to_compile, skip_reason = _model_is_single_cuda_for_compile(model)
    if not ok_to_compile:
        stats["torch_compile_skip_reason"] = skip_reason
        if cfg.torch_compile_strict:
            raise RuntimeError(skip_reason)
        return model, stats

    # torch._dynamo mis-reconstructs the Python 3.13 LOAD_GLOBAL bytecode, which
    # produces a NameError for globals (e.g. "torch") inside traced functions.
    # PyTorch 2.7+ fixes this; skip compilation on 3.13+ with older builds.
    import sys as _sys
    _py_ver = _sys.version_info[:2]
    _torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    if _py_ver >= (3, 13) and _torch_ver < (2, 7):
        skip_msg = (
            f"torch.compile skipped: Python {_py_ver[0]}.{_py_ver[1]} + "
            f"PyTorch {torch.__version__} has a known dynamo LOAD_GLOBAL "
            f"incompatibility (fixed in PyTorch 2.7+)."
        )
        stats["torch_compile_skip_reason"] = skip_msg
        if cfg.torch_compile_strict:
            raise RuntimeError(skip_msg)
        return model, stats

    t0 = time.time()
    try:
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "fullgraph": bool(cfg.torch_compile_fullgraph),
        }
        if cfg.torch_compile_mode is not None:
            compile_kwargs["mode"] = cfg.torch_compile_mode
        if cfg.torch_compile_dynamic is not None:
            compile_kwargs["dynamic"] = cfg.torch_compile_dynamic

        compiled_model = torch.compile(model, **compile_kwargs)

        if cfg.torch_compile_warmup:
            device = infer_input_device(model)
            token_id = tokenizer.eos_token_id
            if token_id is None:
                token_id = tokenizer.pad_token_id
            if token_id is None:
                token_id = 0

            max_pos = getattr(model.config, "max_position_embeddings", cfg.ppl_seqlen)
            warmup_len = int(min(int(cfg.ppl_seqlen), int(max_pos)))
            dummy = torch.full(
                (1, warmup_len),
                int(token_id),
                dtype=torch.long,
                device=device,
            )

            for _ in range(int(cfg.torch_compile_warmup_steps)):
                _ = compiled_model(input_ids=dummy, use_cache=False)

            torch.cuda.synchronize()

        stats.update({
            "torch_compile_applied": True,
            "torch_compile_backend": cfg.torch_compile_backend,
            "torch_compile_mode": cfg.torch_compile_mode,
            "torch_compile_time_s": round(time.time() - t0, 3),
        })
        return compiled_model, stats

    except Exception as exc:
        stats["torch_compile_error"] = f"{type(exc).__name__}: {exc}"
        stats["torch_compile_error_traceback"] = traceback.format_exc(limit=8)
        stats["torch_compile_time_s"] = round(time.time() - t0, 3)
        if cfg.torch_compile_strict:
            raise
        warnings.warn(f"torch.compile failed; continuing with eager model. Error: {type(exc).__name__}: {exc}", RuntimeWarning)
        return model, stats

def _torch_fp8_dtypes() -> set[torch.dtype]:
    return {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dtype is not None
    }


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _torch_fp8_dtypes()


def _fp8_scale_param_name(weight_name: str) -> str:
    if weight_name == "weight":
        return "weight_scale_inv"
    return f"{weight_name}_scale_inv"


def _get_fp8_scale_param(module: nn.Module, weight_name: str) -> Optional[torch.Tensor]:
    return getattr(module, _fp8_scale_param_name(weight_name), None)


def _infer_fp8_block_size(weight: torch.Tensor, scale: torch.Tensor, block_size) -> tuple[int, int]:
    rows, cols = weight.shape[-2:]
    if block_size is not None:
        block_m, block_n = (int(block_size[0]), int(block_size[1]))
    elif scale.numel() == 1:
        block_m, block_n = rows, cols
    else:
        scale_rows, scale_cols = scale.shape[-2:]
        if rows % scale_rows != 0 or cols % scale_cols != 0:
            raise ValueError(
                f"Cannot infer FP8 block size for weight shape {tuple(weight.shape)} "
                f"and scale shape {tuple(scale.shape)}"
            )
        block_m, block_n = rows // scale_rows, cols // scale_cols

    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"FP8 weight shape {tuple(weight.shape)} is not divisible by block size "
            f"({block_m}, {block_n})"
        )
    return block_m, block_n


def _dequantize_finegrained_fp8_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if scale_inv.numel() == 1:
        return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(dtype)

    rows, cols = weight.shape[-2:]
    block_m, block_n = _infer_fp8_block_size(weight, scale_inv, block_size)
    leading_shape = weight.shape[:-2]
    rows_tiles = rows // block_m
    cols_tiles = cols // block_n

    reshaped = weight.to(torch.float32).reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)
    expanded_scales = scale_inv.to(torch.float32).reshape(*leading_shape, rows_tiles, cols_tiles)
    expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(-3)
    return (reshaped * expanded_scales).reshape(weight.shape).to(dtype)


def _quantize_finegrained_fp8_weight(
    weight: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
    block_size=None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape[-2:]
    if block_size is None:
        block_size = (rows, cols)
    block_m, block_n = (int(block_size[0]), int(block_size[1]))
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"FP8 quantization requires weight shape {tuple(weight.shape)} to be divisible "
            f"by block size ({block_m}, {block_n})"
        )

    leading_shape = weight.shape[:-2]
    rows_tiles = rows // block_m
    cols_tiles = cols // block_n
    fp8_info = torch.finfo(fp8_dtype)

    weight_f32 = weight.to(torch.float32)
    reshaped = weight_f32.reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)
    max_abs = reshaped.abs().amax(dim=(-3, -1))
    safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = fp8_info.max / safe_max_abs
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
    scaled = reshaped * scales.unsqueeze(-1).unsqueeze(-3)
    quantized = torch.clamp(scaled, min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)
    scale_inv = (1.0 / scales).to(scale_dtype)
    return quantized.reshape(weight.shape), scale_inv.reshape(*leading_shape, rows_tiles, cols_tiles)


def _module_weight_for_decomposition(
    module: nn.Module,
    weight_name: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = getattr(module, weight_name)
    scale_inv = _get_fp8_scale_param(module, weight_name)
    if scale_inv is not None and _is_fp8_dtype(weight.dtype):
        weight_on_device = weight.detach().to(device=device)
        scale_on_device = scale_inv.detach().to(device=device)
        return _dequantize_finegrained_fp8_weight(
            weight_on_device,
            scale_on_device,
            block_size=getattr(module, "block_size", None),
            dtype=dtype,
        )
    return weight.detach().to(device=device, dtype=dtype)


def _assign_module_weight_from_decomposition(module: nn.Module, weight_name: str, value: torch.Tensor) -> bool:
    weight = getattr(module, weight_name)
    scale_inv = _get_fp8_scale_param(module, weight_name)
    if scale_inv is not None and _is_fp8_dtype(weight.dtype):
        quantized, new_scale_inv = _quantize_finegrained_fp8_weight(
            value.to(device=weight.device),
            fp8_dtype=weight.dtype,
            block_size=getattr(module, "block_size", None),
            scale_dtype=scale_inv.dtype,
        )
        weight.data.copy_(quantized)
        scale_inv.data.copy_(new_scale_inv.to(device=scale_inv.device, dtype=scale_inv.dtype))
        return True

    weight.data.copy_(value.to(device=weight.device, dtype=weight.dtype))
    return False


def _should_cast_float32_param_to_model_dtype(name: str) -> bool:
    return not (
        name.endswith("_scale_inv")
        or name.endswith(".weight_scale_inv")
        or name.endswith(".activation_scale")
    )


def load_model_and_tokenizer(cfg: RunnerConfig):
    configure_runtime_noise_controls(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    extra_load_kwargs: dict = {}
    torch_dtype = cfg.model_dtype
    if cfg.model_spec.family == "gpt_oss_moe":
        from transformers import Mxfp4Config
        extra_load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    elif cfg.model_spec.family == "qwen3_moe" and "fp8" in cfg.model_name.lower():
        from transformers import FineGrainedFP8Config
        extra_load_kwargs["quantization_config"] = FineGrainedFP8Config()
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=cfg.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **extra_load_kwargs,
    )

    if cfg.device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    if isinstance(torch_dtype, torch.dtype) and torch_dtype != torch.float32:
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and _should_cast_float32_param_to_model_dtype(name):
                param.data = param.data.to(torch_dtype)

    if cfg.model_spec.family == "qwen3_moe" and "fp8" in cfg.model_name.lower():
        config = getattr(model.config, "text_config", model.config)
        config._experts_implementation = "grouped_mm"

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def make_target_specs(target_set: dict, model_spec: ModelArchitectureSpec, selectors: Optional[Sequence[str]] = None) -> list[dict]:
    """Expand a target set into concrete module specs.

    target_set can contain either:
      - groups/module_groups: ["mha", "mlp"]
      - modules: explicit relative names such as ["attn.q_proj"]
    selectors narrows this to a recipe plan target list.
    """
    known = target_set.get("known_superweights", {}) or {}
    layer_indices = [int(x) for x in target_set["layer_indices"]]

    if selectors is None:
        selectors = list(target_set.get("groups", target_set.get("module_groups", []))) + list(target_set.get("modules", []))
        if not selectors:
            selectors = ["mha"]

    specs = []
    seen = set()
    for layer_idx in layer_indices:
        for selector in selectors:
            for module_name in group_modules(model_spec, selector):
                full_name = full_module_name_for(model_spec, layer_idx, module_name)
                key = (layer_idx, module_name, full_name)
                if key in seen:
                    continue
                seen.add(key)
                coords = []
                if (layer_idx, module_name) in known:
                    coords = list(known[(layer_idx, module_name)])
                elif full_name in known:
                    coords = list(known[full_name])
                specs.append({
                    "target_set": target_set["label"],
                    "layer_idx": int(layer_idx),
                    "module_name": module_name,
                    "module_group": module_group_for_relative_name(model_spec, module_name),
                    "full_name": full_name,
                    "selector": selector,
                    "superweight_coords": coords,
                })
    return specs


def recipe_selectors(recipe: RecipeSpec) -> list[str]:
    selectors = []
    for p in recipe.plans:
        if p.target not in selectors:
            selectors.append(p.target)
    return selectors


def get_shapes_for_linear(linear: nn.Linear, order: int):
    in_modes = factor_int_balanced(int(linear.in_features), order=order)
    out_modes = factor_int_balanced(int(linear.out_features), order=order)
    return in_modes, out_modes


def is_embedding_param(name: str, model_spec: ModelArchitectureSpec) -> bool:
    return any(name.startswith(prefix) for prefix in model_spec.embedding_prefixes)


def get_model_dense_bits(model: nn.Module, cfg: RunnerConfig, *, include_embeddings: bool = True) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not include_embeddings and is_embedding_param(name, cfg.model_spec):
            continue
        total += int(p.numel()) * int(cfg.model_storage_bits)
    return int(total)


def linear_dense_bits(module: nn.Module, cfg: RunnerConfig) -> int:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
    bits = int(module.weight.numel()) * int(cfg.model_storage_bits)
    if module.bias is not None:
        bits += int(module.bias.numel()) * int(cfg.model_storage_bits)
    return int(bits)


def moe_experts_gate_up_dense_bits(module: nn.Module, cfg: RunnerConfig) -> int:
    return int(module.gate_up_proj.numel() * cfg.model_storage_bits)


def dense_bits_for_target_module(module: nn.Module, cfg: RunnerConfig) -> int:
    if isinstance(module, nn.Linear):
        return linear_dense_bits(module, cfg)
    if hasattr(module, "gate_up_proj") and torch.is_tensor(module.gate_up_proj):
        return moe_experts_gate_up_dense_bits(module, cfg)
    raise TypeError(f"Unsupported module for dense-bit accounting: {type(module).__name__}")


def baseline_bits_for_specs(model: nn.Module, specs: Sequence[dict], cfg: RunnerConfig) -> int:
    return int(sum(dense_bits_for_target_module(get_module_by_name(model, s["full_name"]), cfg) for s in specs))


def named_params_under_block(model: nn.Module, model_spec: ModelArchitectureSpec, layer_idx: int) -> list[tuple[str, nn.Parameter]]:
    prefix = f"{model_spec.block_prefix}.{int(layer_idx)}."
    return [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]


def block_dense_bits(model: nn.Module, model_spec: ModelArchitectureSpec, cfg: RunnerConfig, layer_indices: Sequence[int]) -> int:
    total = 0
    for layer_idx in layer_indices:
        for _, p in named_params_under_block(model, model_spec, layer_idx):
            total += int(p.numel()) * int(cfg.model_storage_bits)
    return int(total)


def enumerate_group_specs(target_set: dict, model_spec: ModelArchitectureSpec, group: str) -> list[dict]:
    return make_target_specs({**target_set, "groups": [group], "modules": []}, model_spec, selectors=[group])


# Storage records

@dataclass
class StorageRecord:
    full_name: str
    layer_idx: int
    module_name: str
    module_group: str
    method: str
    dense_bits: int
    storage_bits: int
    dense_params: int
    storage_params: int
    quant_method: str = "none"
    quant_bits: Optional[int] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def attach_storage_record(module: nn.Module, record: StorageRecord):
    module._compression_storage_record = record.to_dict()
    return module


def storage_record_from_module(module: nn.Module) -> Optional[dict]:
    if isinstance(module, LoRAOnFrozenModule):
        base = storage_record_from_module(module.base_module)
        if base is None and isinstance(module.base_module, nn.Linear):
            return None
        if base is None:
            return None
        rec = dict(base)
        lora_bits = lora_storage_bits(module)
        rec["storage_bits"] = int(rec["storage_bits"]) + int(lora_bits)
        rec["details"] = dict(rec.get("details", {})) | {"lora_bits": int(lora_bits), "lora_quant_bits": getattr(module, "lora_storage_quant_bits", None)}
        return rec
    return getattr(module, "_compression_storage_record", None)


# New TT and dense-sparse TT modules

def boundary_ranks_from_cores(cores):
    ranks = [int(cores[0].shape[0])]
    ranks.extend(int(core.shape[-1]) for core in cores)
    return ranks


class NewTTLayer(nn.Module):
    def __init__(self, *, in_modes, out_modes, cores, bias=None, token_chunk_size: Optional[int] = 16):
        super().__init__()
        self.in_modes = [int(x) for x in in_modes]
        self.out_modes = [int(x) for x in out_modes]
        self.in_features = int(np.prod(self.in_modes))
        self.out_features = int(np.prod(self.out_modes))
        self.d = len(self.in_modes)
        self.cores = nn.ParameterList([nn.Parameter(c) for c in cores])
        self.token_chunk_size = token_chunk_size
        self.storage_quant_bits = None
        self.storage_scale_bits = 16
        self.storage_quant_method = None
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias)

    @staticmethod
    def reshape_for_tt(matrix, in_shapes, out_shapes):
        d = len(out_shapes)
        matrix = matrix.T.reshape(list(in_shapes) + list(out_shapes))
        perm = [x for i in range(d) for x in (i, i + d)]
        matrix = torch.permute(matrix, perm)
        all_dims = [int(in_shapes[o]) * int(out_shapes[o]) for o in range(d)]
        return matrix.reshape(all_dims)

    @staticmethod
    def dense_to_tt_cores(matrix, in_shapes, out_shapes, maxrank=None):
        tensor = NewTTLayer.reshape_for_tt(matrix=matrix, in_shapes=in_shapes, out_shapes=out_shapes)
        tt = tn.Tensor(tensor)
        if maxrank is not None:
            tt.round_tt(rmax=maxrank)
        else:
            tt.round_tt()
        cores = []
        for i, core in enumerate(tt.cores):
            r_prev, _, r_next = core.shape
            cores.append(core.view(r_prev, int(in_shapes[i]), int(out_shapes[i]), r_next).contiguous())
        return cores

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, in_modes, out_modes, maxrank, decompose_dtype, decompose_device, output_device=None, token_chunk_size=16):
        dense_weight = linear.weight.detach().to(device=decompose_device, dtype=decompose_dtype)
        cores = cls.dense_to_tt_cores(dense_weight, in_shapes=in_modes, out_shapes=out_modes, maxrank=maxrank)
        target_device = linear.weight.device if output_device is None else output_device
        moved_cores = [core.to(device=target_device, dtype=linear.weight.dtype).contiguous() for core in cores]
        bias = None if linear.bias is None else linear.bias.detach().clone().to(device=target_device, dtype=linear.weight.dtype)
        return cls(in_modes=in_modes, out_modes=out_modes, cores=moved_cores, bias=bias, token_chunk_size=token_chunk_size)

    @property
    def tt_ranks(self):
        return boundary_ranks_from_cores(self.cores)

    def num_tt_parameters(self):
        return int(sum(c.numel() for c in self.cores) + (0 if self.bias is None else self.bias.numel()))

    @torch.no_grad()
    def to_dense_weight(self):
        flat = self.cores[0].reshape(self.cores[0].shape[0], -1, self.cores[0].shape[-1]).squeeze(0)
        for core in self.cores[1:]:
            core_flat = core.reshape(core.shape[0], -1, core.shape[-1])
            flat = torch.einsum("mr,rps->mps", flat, core_flat).reshape(flat.shape[0] * core_flat.shape[1], core_flat.shape[2])
        interleaved_shape = []
        for in_dim, out_dim in zip(self.in_modes, self.out_modes):
            interleaved_shape.extend([int(in_dim), int(out_dim)])
        interleaved = flat.squeeze(-1).reshape(*interleaved_shape)
        in_pos = list(range(0, 2 * self.d, 2))
        out_pos = list(range(1, 2 * self.d, 2))
        tensor_in_out = interleaved.permute(*in_pos, *out_pos).contiguous()
        return tensor_in_out.reshape(self.in_features, self.out_features).T.contiguous()

    def _forward_chunk(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, *self.in_modes)
        result = x.unsqueeze(-1)
        D = x.dim()
        d = self.d
        for core in self.cores:
            result = torch.tensordot(result, core, dims=([D - d, -1], [1, 0]))
        result = result.squeeze(-1).reshape(*batch_shape, self.out_features)
        if self.bias is not None:
            result = result + self.bias
        return result

    def forward(self, x):
        if self.token_chunk_size is None:
            return self._forward_chunk(x)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        outs = []
        for start in range(0, x_flat.shape[0], self.token_chunk_size):
            outs.append(self._forward_chunk(x_flat[start:start + self.token_chunk_size]))
        return torch.cat(outs, dim=0).reshape(*orig_shape, self.out_features)


class SparseCorrection(nn.Module):
    def __init__(self, *, out_features: int, in_features: int, rows, cols, values, storage_value_bits: int = 16):
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        value_tensor = torch.as_tensor(values)
        device = value_tensor.device
        dtype = value_tensor.dtype
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long, device=device))
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long, device=device))
        self.register_buffer("values", value_tensor.to(device=device, dtype=dtype))
        self.storage_value_bits = int(storage_value_bits)

    def forward(self, x):
        if self.values.numel() == 0:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, x.shape[-1])
        rows = self.rows.to(device=x_flat.device)
        cols = self.cols.to(device=x_flat.device)
        values = self.values.to(device=x_flat.device, dtype=x_flat.dtype)
        gathered = x_flat[:, cols] * values
        corr = x_flat.new_zeros(x_flat.shape[0], self.out_features)
        corr.index_add_(1, rows, gathered)
        return corr.reshape(*x.shape[:-1], self.out_features)

    @torch.no_grad()
    def to_dense_weight(self):
        w = self.values.new_zeros((self.out_features, self.in_features))
        if self.values.numel() > 0:
            w[self.rows, self.cols] = self.values
        return w

    def storage_bits(self):
        if self.values.numel() == 0:
            return 0
        row_bits = max(1, math.ceil(math.log2(max(self.out_features, 2))))
        col_bits = max(1, math.ceil(math.log2(max(self.in_features, 2))))
        return int(self.values.numel()) * (int(self.storage_value_bits) + row_bits + col_bits)


class DenseSparseNewTTLayer(nn.Module):
    def __init__(self, *, tt_part: NewTTLayer, sparse_part: SparseCorrection):
        super().__init__()
        self.tt_part = tt_part
        self.sparse_part = sparse_part
        self.in_features = int(tt_part.in_features)
        self.out_features = int(tt_part.out_features)

    def forward(self, x):
        y = self.tt_part(x)
        return y + self.sparse_part(x).to(device=y.device, dtype=y.dtype)

    @torch.no_grad()
    def to_dense_weight(self):
        return self.tt_part.to_dense_weight() + self.sparse_part.to_dense_weight()


def split_dense_inliers_and_outliers(weight: torch.Tensor, outlier_fraction: float, include_coords=None):
    total = weight.numel()
    k = max(1, int(total * float(outlier_fraction)))
    flat_abs = weight.abs().reshape(-1)
    top_idx = torch.topk(flat_abs, k=min(k, flat_abs.numel())).indices.tolist()
    H, W = weight.shape
    keep_coords = {(int(idx // W), int(idx % W)) for idx in top_idx}
    if include_coords is not None:
        for row, col in include_coords:
            keep_coords.add((int(row), int(col)))
    sparse = torch.zeros_like(weight)
    mask = torch.zeros_like(weight, dtype=torch.bool)
    for row, col in keep_coords:
        sparse[row, col] = weight[row, col]
        mask[row, col] = True
    return weight - sparse, sparse, mask


def dense_sparse_new_tt_from_linear(linear: nn.Linear, *, tt_rank_inliers, outlier_fraction, include_coords, in_modes, out_modes, decompose_dtype, decompose_device, output_device, token_chunk_size, cfg: RunnerConfig):
    dense_weight = linear.weight.detach().to(device=decompose_device, dtype=decompose_dtype)
    inlier_weight, sparse_weight, sparse_mask = split_dense_inliers_and_outliers(dense_weight, outlier_fraction, include_coords=include_coords)
    inlier_cores = NewTTLayer.dense_to_tt_cores(inlier_weight, in_shapes=in_modes, out_shapes=out_modes, maxrank=tt_rank_inliers)
    target_device = linear.weight.device if output_device is None else output_device
    target_dtype = linear.weight.dtype
    moved_cores = [c.to(device=target_device, dtype=target_dtype).contiguous() for c in inlier_cores]
    bias = None if linear.bias is None else linear.bias.detach().clone().to(device=target_device, dtype=target_dtype)
    tt_part = NewTTLayer(in_modes=in_modes, out_modes=out_modes, cores=moved_cores, bias=bias, token_chunk_size=token_chunk_size)
    coords = sparse_mask.nonzero(as_tuple=False).cpu().tolist()
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    values = [float(sparse_weight[int(r), int(c)].item()) for r, c in coords]
    sparse_values = torch.tensor(values, device=target_device, dtype=target_dtype) if values else torch.empty(0, device=target_device, dtype=target_dtype)
    sparse_part = SparseCorrection(out_features=linear.out_features, in_features=linear.in_features, rows=rows, cols=cols, values=sparse_values, storage_value_bits=cfg.model_storage_bits).to(target_device)
    module = DenseSparseNewTTLayer(tt_part=tt_part, sparse_part=sparse_part).to(target_device)
    return module, {
        "kept_outlier_count": int(sparse_mask.sum().item()),
        "kept_fraction_actual": float(sparse_mask.sum().item() / max(linear.weight.numel(), 1)),
        "inlier_tt_ranks": boundary_ranks_from_cores(moved_cores),
        "sparse_storage_bits": int(sparse_part.storage_bits()),
        "inlier_tt_params": int(sum(c.numel() for c in moved_cores) + (0 if bias is None else bias.numel())),
    }


# Quantization

def quantize_tensor_symmetric(x: torch.Tensor, bits: int):
    qmax = (1 << (int(bits) - 1)) - 1
    max_abs = float(x.detach().abs().max().item()) if x.numel() else 0.0
    if max_abs == 0.0:
        return torch.zeros_like(x, dtype=torch.int32), 1.0
    scale = max_abs / max(qmax, 1)
    q = torch.clamp(torch.round(x.float() / scale), -qmax, qmax).to(torch.int32)
    return q, float(scale)


def fake_quant_tensor_symmetric(x: torch.Tensor, bits: int):
    q, scale = quantize_tensor_symmetric(x, bits)
    return (q.to(device=x.device, dtype=torch.float32) * float(scale)).to(dtype=x.dtype), scale


@torch.no_grad()
def fake_quantize_newtt_inplace(layer: NewTTLayer, bits: int, cfg: RunnerConfig, method: str = "rtn_symmetric"):
    for core in layer.cores:
        x_hat, _ = fake_quant_tensor_symmetric(core.data, bits)
        core.data.copy_(x_hat)
    layer.storage_quant_bits = int(bits)
    layer.storage_scale_bits = int(cfg.scale_storage_bits)
    layer.storage_quant_method = method
    return layer


@torch.no_grad()
def fake_quantize_dense_sparse_newtt_inplace(layer: DenseSparseNewTTLayer, bits: int, cfg: RunnerConfig):
    fake_quantize_newtt_inplace(layer.tt_part, bits=bits, cfg=cfg)
    return layer


@torch.no_grad()
def fake_quantize_linear_inplace(layer: nn.Linear, bits: int, cfg: RunnerConfig, method: str = "rtn_symmetric_dense"):
    x_hat, _ = fake_quant_tensor_symmetric(layer.weight.data, bits)
    layer.weight.data.copy_(x_hat)
    layer.storage_quant_bits = int(bits)
    layer.storage_scale_bits = int(cfg.scale_storage_bits)
    layer.storage_quant_method = method
    return layer


# LoRA

class LoRAOnFrozenModule(nn.Module):
    def __init__(self, base_module: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        if not hasattr(base_module, "in_features") or not hasattr(base_module, "out_features"):
            raise TypeError(f"Base module must expose in_features/out_features, got {type(base_module).__name__}")
        self.base_module = base_module
        self.in_features = int(base_module.in_features)
        self.out_features = int(base_module.out_features)
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(max(r, 1))
        self.dropout = nn.Dropout(dropout)
        for p in self.base_module.parameters():
            p.requires_grad = False
        try:
            base_param = next(base_module.parameters())
            base_device = base_param.device
        except StopIteration:
            base_device = torch.device("cpu")
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=base_device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=base_device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_storage_quant_bits = None
        self.lora_storage_scale_bits = 16
        self.lora_quant_method = "none"

    def forward(self, x):
        base = self.base_module(x)
        x_lora = self.dropout(x).to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        lora = (x_lora @ self.lora_A.t()) @ self.lora_B.t()
        return base + (lora * self.scaling).to(device=base.device, dtype=base.dtype)

    @torch.no_grad()
    def merged_dense_weight(self):
        base_weight = get_effective_weight_for_module(self.base_module).detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        delta = (
            self.lora_B.detach().to(device="cpu", dtype=torch.float32)
            @ self.lora_A.detach().to(device="cpu", dtype=torch.float32)
        ) * float(self.scaling)
        return base_weight + delta

    @torch.no_grad()
    def merged_dense_bias(self):
        if isinstance(self.base_module, DenseSparseNewTTLayer):
            if self.base_module.tt_part.bias is not None:
                return self.base_module.tt_part.bias.detach().clone()

        if isinstance(self.base_module, NewTTLayer):
            if self.base_module.bias is not None:
                return self.base_module.bias.detach().clone()

        if hasattr(self.base_module, "bias") and self.base_module.bias is not None:
            return self.base_module.bias.detach().clone()

        return None


def lora_storage_bits(module: LoRAOnFrozenModule, cfg: Optional[RunnerConfig] = None) -> int:
    qbits = getattr(module, "lora_storage_quant_bits", None)
    base_bits = int(cfg.lora_storage_bits) if cfg is not None else 16
    scale_bits = int(getattr(module, "lora_storage_scale_bits", (cfg.scale_storage_bits if cfg is not None else 16)))
    if qbits is None:
        return int((module.lora_A.numel() + module.lora_B.numel()) * base_bits)
    return int((module.lora_A.numel() + module.lora_B.numel()) * int(qbits) + 2 * scale_bits)


@torch.no_grad()
def fake_quantize_lora_inplace(module: LoRAOnFrozenModule, bits: int, cfg: RunnerConfig):
    for p in (module.lora_A, module.lora_B):
        x_hat, _ = fake_quant_tensor_symmetric(p.data, bits)
        p.data.copy_(x_hat)
    module.lora_storage_quant_bits = int(bits)
    module.lora_storage_scale_bits = int(cfg.scale_storage_bits)
    module.lora_quant_method = "rtn_symmetric"
    return module


def freeze_all_parameters(model):
    for p in model.parameters():
        p.requires_grad = False


def count_trainable_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def attach_lora_to_specs(model: nn.Module, specs: Sequence[dict], cfg: RunnerConfig, recipe: RecipeSpec):
    params = {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout}
    params.update(recipe.lora_params or {})
    for spec in specs:
        full_name = spec["full_name"]
        base = get_module_by_name(model, full_name)
        if isinstance(base, LoRAOnFrozenModule):
            continue
        set_module_by_name(model, full_name, LoRAOnFrozenModule(base, **params))


def build_wikitext2_train_loader(
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    num_sequences: int,
    seed: int = 0,
    shuffle: bool = False,
):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    ids = encoded.input_ids[0]
    n_blocks = ids.numel() // seq_len
    ids = ids[: n_blocks * seq_len].view(n_blocks, seq_len)

    subset_gen = torch.Generator().manual_seed(seed)
    loader_gen = torch.Generator().manual_seed(seed + 1)

    if ids.shape[0] > num_sequences:
        ids = ids[torch.randperm(ids.shape[0], generator=subset_gen)[:num_sequences]]

    return DataLoader(
        TensorDataset(ids),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=loader_gen if shuffle else None,
    )


def run_lora_training(model, tokenizer, cfg: RunnerConfig, *, seed: int = 0):
    freeze_all_parameters(model)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lora_lr, weight_decay=cfg.lora_weight_decay)
    loader = build_wikitext2_train_loader(tokenizer, seq_len=cfg.lora_train_seq_len, batch_size=cfg.lora_train_batch_size, num_sequences=cfg.lora_num_train_sequences, seed=seed, shuffle=True)
    model.train()
    device = infer_input_device(model)
    history = []
    step_count = 0
    micro_step = 0
    tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=cfg.lora_max_steps, desc="LoRA training", leave=True)
    loader_iter = iter(loader)
    while step_count < cfg.lora_max_steps:
        try:
            (input_ids,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (input_ids,) = next(loader_iter)
        micro_step += 1
        input_ids = input_ids.to(device)
        tokens_seen += int(input_ids.numel())
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss / int(cfg.lora_grad_accum_steps)
        loss.backward()
        if micro_step % int(cfg.lora_grad_accum_steps) == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_count += 1
            current_loss = float(outputs.loss.detach().float().cpu().item())
            history.append({"step": step_count, "tokens_seen": tokens_seen, "loss": current_loss})
            pbar.update(1)
            pbar.set_postfix(loss=f"{current_loss:.4f}", tokens=tokens_seen)
    pbar.close()
    model.eval()
    return pd.DataFrame(history)


# TensorLLM-style Tucker on MHA

def _require_tensorly():
    if tl is None or partial_tucker is None:
        raise ImportError(f"tensorly/partial_tucker unavailable: {_TENSORLY_IMPORT_ERROR!r}")
    tl.set_backend("pytorch")


def _linear_weight_for_tucker(module: nn.Linear, *, is_output_projection: bool):
    w = module.weight.detach()
    return w.T if is_output_projection else w


def _assign_linear_weight_from_tucker(module: nn.Linear, reconstructed: torch.Tensor, *, is_output_projection: bool):
    w = reconstructed.T if is_output_projection else reconstructed
    module.weight = nn.Parameter(w.to(device=module.weight.device, dtype=module.weight.dtype).contiguous())


def _head_shape_for_weight(weight_for_tucker: torch.Tensor, model) -> tuple[int, int, int]:
    hidden_in = int(weight_for_tucker.shape[0])
    out_dim = int(weight_for_tucker.shape[1])
    head_dim = int(getattr(model.config, "hidden_size", hidden_in)) // int(getattr(model.config, "num_attention_heads"))
    if out_dim % head_dim != 0:
        raise ValueError(f"Cannot infer attention heads: out_dim={out_dim}, head_dim={head_dim}")
    return hidden_in, out_dim // head_dim, head_dim


def _tucker_storage_bits(core, factors, *, dense_bits, quant_bits, cfg: RunnerConfig, quantized: bool) -> tuple[int, dict]:
    value_bits = int(quant_bits) if quantized else int(cfg.model_storage_bits)
    components = {"core_numel": int(core.numel()), "factor_numels": [int(f.numel()) for f in factors]}
    n_components = 1 + len(factors)
    storage_bits = (int(core.numel()) + sum(int(f.numel()) for f in factors)) * value_bits
    if quantized:
        storage_bits += n_components * int(cfg.scale_storage_bits)
    return int(storage_bits), components | {"dense_bits": int(dense_bits), "value_bits": value_bits, "scale_bits_per_component": int(cfg.scale_storage_bits) if quantized else 0}


def _maybe_quantize_components(core, factors, plan: ModulePlan, cfg: RunnerConfig):
    qbits = plan.quant_bits
    quantized = plan.quant_method in {"tensorllm_pre_reconstruct_rtn", "rtn_symmetric"} and qbits is not None
    if not quantized:
        return core, factors, False
    qcore, _ = fake_quant_tensor_symmetric(core, int(qbits))
    qfactors = []
    for f in factors:
        qf, _ = fake_quant_tensor_symmetric(f, int(qbits))
        qfactors.append(qf)
    return qcore, qfactors, True


def _partial_tucker_reconstruct(tensor, *, modes, rank, plan: ModulePlan, cfg: RunnerConfig):
    work = tensor.detach().to(device=normalize_torch_device_name(cfg.decompose_device), dtype=torch.float32).contiguous()
    (core, factors), rec_errors = partial_tucker(
        work,
        modes=modes,
        rank=rank,
        init="svd",
        svd="randomized_svd",
        random_state=0,
        tol=1e-5,
        verbose=False,
    )
    core, factors, quantized = _maybe_quantize_components(core, factors, plan, cfg)
    reconstructed = tl.tenalg.multi_mode_dot(core, factors, modes=modes)
    rel_error = float((torch.norm(reconstructed - work) / torch.norm(work).clamp_min(1e-12)).detach().float().cpu().item())
    return reconstructed, core, factors, quantized, rel_error


def apply_tensorllm_tucker_mha_to_layers(model: nn.Module, target_specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    _require_tensorly()
    model_spec = cfg.model_spec
    rank = int(plan.rank or global_rank)
    params = dict(plan.params or {})
    stack_rank = int(params.get("stack_rank", cfg.tensorllm_stack_rank))
    head_dim_rank = int(params.get("head_dim_rank", cfg.tensorllm_head_dim_rank))
    tucker_type = str(params.get("tucker_type", cfg.tensorllm_tucker_type))
    allow_fallback = bool(params.get("allow_gqa_per_projection_fallback", cfg.tensorllm_allow_gqa_per_projection_fallback))

    records = []
    for layer_idx in unique_layer_indices(target_specs):
        modules = {name: get_module_by_name(model, full_module_name_for(model_spec, layer_idx, rel)) for name, rel in model_spec.tensorllm_qkvo_modules.items()}
        weights = {}
        shapes = {}
        dense_bits = 0
        for name, module in modules.items():
            rel = model_spec.tensorllm_qkvo_modules[name]
            is_o = rel in model_spec.output_projection_names
            w = _linear_weight_for_tucker(module, is_output_projection=is_o).detach().to(
                device=normalize_torch_device_name(cfg.decompose_device),
                dtype=torch.float32,
            ).contiguous()
            weights[name] = w
            shapes[name] = tuple(w.shape)
            dense_bits += linear_dense_bits(module, cfg)

        equal_shapes = len(set(shapes.values())) == 1
        if equal_shapes:
            stacked = []
            for name in model_spec.tensorllm_qkvo_order:
                hidden_in, n_heads_proj, head_dim = _head_shape_for_weight(weights[name], model)
                stacked.append(weights[name].view(hidden_in, n_heads_proj, head_dim))
            tensor = torch.stack(stacked, dim=3)
            if tucker_type == "partial_tucker_v5":
                modes = [0, 2, 3]
                ranks = [min(rank, tensor.shape[0]), min(head_dim_rank, tensor.shape[2]), min(stack_rank, tensor.shape[3])]
            else:
                modes = [0, 2, 3]
                ranks = [min(rank * tensor.shape[1], tensor.shape[0]), min(rank, tensor.shape[2]), min(stack_rank, tensor.shape[3])]
            reconstructed, core, factors, quantized, rel_error = _partial_tucker_reconstruct(tensor, modes=modes, rank=ranks, plan=plan, cfg=cfg)
            storage_bits, comp_details = _tucker_storage_bits(core, factors, dense_bits=dense_bits, quant_bits=plan.quant_bits, cfg=cfg, quantized=quantized)
            for idx, name in enumerate(model_spec.tensorllm_qkvo_order):
                rel = model_spec.tensorllm_qkvo_modules[name]
                full = full_module_name_for(model_spec, layer_idx, rel)
                module = modules[name]
                is_o = rel in model_spec.output_projection_names
                _assign_linear_weight_from_tucker(module, reconstructed[:, :, :, idx].reshape_as(weights[name]), is_output_projection=is_o)

                rec = StorageRecord(
                    full_name=full,
                    layer_idx=int(layer_idx),
                    module_name=rel,
                    module_group="mha",
                    method="tensorllm_tucker4d_qkvo",
                    dense_bits=linear_dense_bits(module, cfg),
                    storage_bits=int(storage_bits / 4),
                    dense_params=int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
                    storage_params=int((core.numel() + sum(f.numel() for f in factors)) / 4),
                    quant_method=plan.quant_method if quantized else "none",
                    quant_bits=plan.quant_bits if quantized else None,
                    details={"shared_qkvo_record": True, "layer_storage_bits_total": storage_bits, "relative_reconstruction_error": rel_error, "ranks": ranks, **comp_details},
                )
                attach_storage_record(module, rec)
                records.append(rec.to_dict())
        else:
            if not allow_fallback:
                raise ValueError(
                    f"Layer {layer_idx} Q/K/V/O shapes differ ({shapes}). 4D QKVO TensorLLM requires equal shapes. "
                    "Set allow_gqa_per_projection_fallback=True to use per-projection 3D Tucker."
                )
            warnings.warn(
                f"Layer {layer_idx}: Q/K/V/O shapes differ ({shapes}); using per-projection 3D Tucker fallback.",
                RuntimeWarning,
            )
            for name in model_spec.tensorllm_qkvo_order:
                rel = model_spec.tensorllm_qkvo_modules[name]
                full = full_module_name_for(model_spec, layer_idx, rel)
                module = modules[name]
                is_o = rel in model_spec.output_projection_names
                w = weights[name]
                hidden_in, n_heads_proj, head_dim = _head_shape_for_weight(w, model)
                tensor = w.view(hidden_in, n_heads_proj, head_dim)
                modes = [0, 2]
                ranks = [min(rank, tensor.shape[0]), min(head_dim_rank, tensor.shape[2])]
                reconstructed, core, factors, quantized, rel_error = _partial_tucker_reconstruct(tensor, modes=modes, rank=ranks, plan=plan, cfg=cfg)
                _assign_linear_weight_from_tucker(module, reconstructed.reshape_as(w), is_output_projection=is_o)
                dense_bits_i = linear_dense_bits(module, cfg)
                storage_bits, comp_details = _tucker_storage_bits(core, factors, dense_bits=dense_bits_i, quant_bits=plan.quant_bits, cfg=cfg, quantized=quantized)
                rec = StorageRecord(
                    full_name=full,
                    layer_idx=int(layer_idx),
                    module_name=rel,
                    module_group="mha",
                    method="tensorllm_tucker3d_per_projection_fallback",
                    dense_bits=dense_bits_i,
                    storage_bits=storage_bits,
                    dense_params=int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
                    storage_params=int(core.numel() + sum(f.numel() for f in factors)),
                    quant_method=plan.quant_method if quantized else "none",
                    quant_bits=plan.quant_bits if quantized else None,
                    details={"relative_reconstruction_error": rel_error, "ranks": ranks, "projection": name, **comp_details},
                )
                attach_storage_record(module, rec)
                records.append(rec.to_dict())
    return {"decomposition_per_module": records, "tensorllm_records": records}


def apply_td_moe_to_layers(
    model: nn.Module,
    target_specs: Sequence[dict],
    plan: ModulePlan,
    cfg: RunnerConfig,
    *,
    global_rank: int,
) -> dict:
    params = dict(plan.params or {})
    compression_ratio = float(params.get("compression_ratio", 0.2))
    expert_mode = str(params.get("expert_mode", "preserve"))
    svd_backend = str(params.get("svd_backend", "gram_eigh"))

    records = []
    diagnostic_cache: dict[int, dict] = {}
    keep_diagnostic_cache = bool(getattr(cfg, "run_sd_moe_diagnostics", False))
    for spec in target_specs:
        full_name = spec["full_name"]
        experts_module = get_module_by_name(model, full_name)

        gate_up = _module_weight_for_decomposition(
            experts_module,
            "gate_up_proj",
            device=normalize_torch_device_name(cfg.decompose_device),
            dtype=torch.float32,
        )
        # GPT-OSS stores gate_up_proj as [E, d_in, 2*d_out] with interleaved gate/up output columns.
        # Qwen3 stores gate_up_proj as [E, 2*d_out, d_in] with sequential gate/up output rows.
        family = cfg.model_spec.family
        if family == "gpt_oss_moe":
            num_experts, d_in, two_d_out = gate_up.shape
            gate = gate_up[..., ::2]
            up = gate_up[..., 1::2]
        elif family == "qwen3_moe":
            num_experts, two_d_out, d_in = gate_up.shape
            d_out = two_d_out // 2
            gate = gate_up[:, :d_out, :]
            up = gate_up[:, d_out:, :]
        else:
            raise NotImplementedError(f"gate_up_proj layout unknown for model family {family!r}")

        # gpt_oss has expert axis last for inputs (E, d_in, d_out_*); transpose
        # into the (E, d_out, d_in) layout that diagnostics math expects.
        if family == "gpt_oss_moe":
            gate_canonical = gate.permute(0, 2, 1).contiguous()
            up_canonical = up.permute(0, 2, 1).contiguous()
        else:
            gate_canonical = gate
            up_canonical = up

        if keep_diagnostic_cache:
            cache_entry = {
                "gate_base": gate_canonical.detach().to(device="cpu", dtype=torch.float32).clone(),
                "up_base": up_canonical.detach().to(device="cpu", dtype=torch.float32).clone(),
            }
            router_path = getattr(cfg.model_spec, "router_module_path", "") or ""
            if router_path:
                try:
                    block = get_transformer_layers(model, cfg.model_spec)[int(spec["layer_idx"])]
                    router_module = get_module_by_name(block, router_path)
                    W_router = getattr(router_module, "weight", None)
                    if torch.is_tensor(W_router):
                        cache_entry["W_router"] = W_router.detach().to(device="cpu", dtype=torch.float32).clone()
                except Exception as exc:
                    warnings.warn(f"sd_moe_diag: failed to read router weight for layer {spec['layer_idx']}: {exc!r}", RuntimeWarning)
            diagnostic_cache[int(spec["layer_idx"])] = cache_entry

        gate_result = td_moe_compress(
            gate,
            compression_ratio=compression_ratio,
            expert_mode=expert_mode,
            whitening_mode="none",
            svd_backend=svd_backend,
            return_reconstructed=True,
        )
        up_result = td_moe_compress(
            up,
            compression_ratio=compression_ratio,
            expert_mode=expert_mode,
            whitening_mode="none",
            svd_backend=svd_backend,
            return_reconstructed=True,
        )

        gate_rec = gate_result["reconstructed"].to(device=gate_up.device, dtype=gate_up.dtype)
        up_rec = up_result["reconstructed"].to(device=gate_up.device, dtype=gate_up.dtype)
        if family == "gpt_oss_moe":
            reconstructed = torch.empty_like(gate_up)
            reconstructed[..., ::2] = gate_rec
            reconstructed[..., 1::2] = up_rec
        else:
            reconstructed = torch.cat([gate_rec, up_rec], dim=1)

        requantized_fp8 = _assign_module_weight_from_decomposition(experts_module, "gate_up_proj", reconstructed)

        gate_stats = gate_result["compression_stats"]
        up_stats = up_result["compression_stats"]
        dense_params = int(gate_up.numel())
        storage_params = int(gate_stats["compressed_params"] + up_stats["compressed_params"])

        if keep_diagnostic_cache:
            entry = diagnostic_cache[int(spec["layer_idx"])]
            if family == "gpt_oss_moe":
                entry["gate_post"] = gate_rec.detach().permute(0, 2, 1).contiguous().to(device="cpu", dtype=torch.float32).clone()
                entry["up_post"] = up_rec.detach().permute(0, 2, 1).contiguous().to(device="cpu", dtype=torch.float32).clone()
            else:
                entry["gate_post"] = gate_rec.detach().to(device="cpu", dtype=torch.float32).clone()
                entry["up_post"] = up_rec.detach().to(device="cpu", dtype=torch.float32).clone()
            entry["gate_ranks"] = tuple(int(r) for r in gate_result["ranks"])
            entry["up_ranks"] = tuple(int(r) for r in up_result["ranks"])
            for key, src in (("gate_factors", gate_result["factors"]), ("up_factors", up_result["factors"])):
                entry[key] = {
                    name: (val.detach().to(device="cpu", dtype=torch.float32).clone() if val is not None else None)
                    for name, val in src.items()
                }
            entry["full_name"] = full_name
            entry["module_name"] = spec["module_name"]
            entry["compression_ratio"] = compression_ratio
            entry["expert_mode"] = expert_mode

        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group="moe_experts",
            method="td_moe",
            dense_bits=int(dense_params * cfg.model_storage_bits),
            storage_bits=int(storage_params * cfg.model_storage_bits),
            dense_params=dense_params,
            storage_params=storage_params,
            details={
                "gate_ranks": gate_result["ranks"],
                "up_ranks": up_result["ranks"],
                "gate_compression_stats": gate_stats,
                "up_compression_stats": up_stats,
                "compression_ratio": compression_ratio,
                "expert_mode": expert_mode,
                "svd_backend": svd_backend,
                "runtime_weight_dtype": str(experts_module.gate_up_proj.dtype),
                "runtime_requantized_fp8": bool(requantized_fp8),
            },
        )
        attach_storage_record(experts_module, rec)
        records.append(rec.to_dict())

        del gate, up, gate_up, gate_rec, up_rec, reconstructed, gate_canonical, up_canonical
        clean_mem()

    out = {"decomposition_per_module": records}
    if keep_diagnostic_cache:
        out["sd_moe_diagnostic_cache"] = diagnostic_cache
    return out


# Decomposition and quantization application

def apply_tt_to_specs(model: nn.Module, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    records = []
    rank = int(plan.rank or global_rank)
    for spec in specs:
        full_name = spec["full_name"]
        dense_module = get_module_by_name(model, full_name)
        if not isinstance(dense_module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before TT at {full_name}, got {type(dense_module).__name__}")
        in_modes, out_modes = get_shapes_for_linear(dense_module, cfg.tt_order)
        tt_module = NewTTLayer.from_linear(
            dense_module,
            in_modes=in_modes,
            out_modes=out_modes,
            maxrank=rank,
            decompose_dtype=cfg.decompose_dtype,
            decompose_device=normalize_torch_device_name(cfg.decompose_device),
            output_device=dense_module.weight.device,
            token_chunk_size=cfg.token_chunk_size,
        )
        dense_bits = linear_dense_bits(dense_module, cfg)
        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group=spec.get("module_group", "custom"),
            method="tt",
            dense_bits=dense_bits,
            storage_bits=tt_layer_storage_bits(tt_module, cfg),
            dense_params=int(dense_module.weight.numel() + (0 if dense_module.bias is None else dense_module.bias.numel())),
            storage_params=tt_module.num_tt_parameters(),
            details={"tt_ranks": list(tt_module.tt_ranks), "tt_order": int(cfg.tt_order)},
        )
        attach_storage_record(tt_module, rec)
        set_module_by_name(model, full_name, tt_module)
        records.append(rec.to_dict())
    return {"decomposition_per_module": records}


def apply_dense_sparse_to_specs(model: nn.Module, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    records = []
    rank = int(plan.rank or global_rank)
    params = dict(plan.params or {})
    outlier_fraction = float(params.get("outlier_fraction", cfg.dense_sparse_outlier_fraction))
    for spec in specs:
        full_name = spec["full_name"]
        dense_module = get_module_by_name(model, full_name)
        if not isinstance(dense_module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before dense-sparse TT at {full_name}, got {type(dense_module).__name__}")
        in_modes, out_modes = get_shapes_for_linear(dense_module, cfg.tt_order)
        ds_module, summary = dense_sparse_new_tt_from_linear(
            dense_module,
            tt_rank_inliers=rank,
            outlier_fraction=outlier_fraction,
            include_coords=spec.get("superweight_coords", []),
            in_modes=in_modes,
            out_modes=out_modes,
            decompose_dtype=cfg.decompose_dtype,
            decompose_device=normalize_torch_device_name(cfg.decompose_device),
            output_device=dense_module.weight.device,
            token_chunk_size=cfg.token_chunk_size,
            cfg=cfg,
        )
        dense_bits = linear_dense_bits(dense_module, cfg)
        storage_bits = int(tt_layer_storage_bits(ds_module.tt_part, cfg) + ds_module.sparse_part.storage_bits())
        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group=spec.get("module_group", "custom"),
            method="dense_sparse_tt",
            dense_bits=dense_bits,
            storage_bits=storage_bits,
            dense_params=int(dense_module.weight.numel() + (0 if dense_module.bias is None else dense_module.bias.numel())),
            storage_params=int(summary["inlier_tt_params"] + summary["kept_outlier_count"]),
            details=summary | {"outlier_fraction": outlier_fraction},
        )
        attach_storage_record(ds_module, rec)
        set_module_by_name(model, full_name, ds_module)
        records.append(rec.to_dict())
    return {"decomposition_per_module": records}


def apply_laser_to_specs(model: nn.Module, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    records = []
    rank = int(plan.rank or global_rank)
    for spec in specs:
        full_name = spec["full_name"]
        module = get_module_by_name(model, full_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before Laser at {full_name}, got {type(module).__name__}")
        dense_bits = linear_dense_bits(module, cfg)
        dense_params = int(module.weight.numel() + (0 if module.bias is None else module.bias.numel()))
        module, summary = laser_svd_apply(module, rank, name=full_name)
        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group=spec.get("module_group", "custom"),
            method="laser",
            dense_bits=dense_bits,
            storage_bits=dense_bits,
            dense_params=dense_params,
            storage_params=dense_params,
            details={
                "effective_rank": summary.effective_rank,
                "compression_ratio": summary.compression_ratio,
                "reconstruction_error": summary.reconstruction_error,
            },
        )
        attach_storage_record(module, rec)
        records.append(rec.to_dict())
    return {"decomposition_per_module": records}


def apply_per_head_tt_to_mha_layers(model: nn.Module, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    records = []
    tt_rank = int(plan.rank or global_rank)
    params = dict(plan.params or {})
    order = int(params.get("tt_order", cfg.tt_order))
    model_spec = cfg.model_spec
    for spec in specs:
        full_name = spec["full_name"]
        module = get_module_by_name(model, full_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before per-head TT at {full_name}, got {type(module).__name__}")
        is_output = spec["module_name"] in model_spec.output_projection_names
        head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        n_heads = (module.in_features if is_output else module.out_features) // head_dim
        dense_bits = linear_dense_bits(module, cfg)
        dense_params = int(module.weight.numel() + (0 if module.bias is None else module.bias.numel()))
        per_head_tt_apply(
            module,
            n_heads=n_heads,
            tt_rank=tt_rank,
            is_output_projection=is_output,
            order=order,
            decompose_dtype=cfg.decompose_dtype,
            decompose_device=normalize_torch_device_name(cfg.decompose_device),
            name=full_name,
        )
        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group=spec.get("module_group", "custom"),
            method="per_head_tt",
            dense_bits=dense_bits,
            storage_bits=dense_bits,
            dense_params=dense_params,
            storage_params=dense_params,
            details={"n_heads": n_heads, "tt_rank": tt_rank, "order": order},
        )
        attach_storage_record(module, rec)
        records.append(rec.to_dict())
    return {"decomposition_per_module": records}


def apply_per_head_tucker_to_mha_layers(model: nn.Module, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    records = []
    params = dict(plan.params or {})
    hidden_rank = int(params.get("hidden_rank", plan.rank or global_rank))
    head_dim_rank = int(params.get("head_dim_rank", cfg.tensorllm_head_dim_rank))
    model_spec = cfg.model_spec
    for spec in specs:
        full_name = spec["full_name"]
        module = get_module_by_name(model, full_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before per-head Tucker at {full_name}, got {type(module).__name__}")
        is_output = spec["module_name"] in model_spec.output_projection_names
        head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        n_heads = (module.in_features if is_output else module.out_features) // head_dim
        dense_bits = linear_dense_bits(module, cfg)
        dense_params = int(module.weight.numel() + (0 if module.bias is None else module.bias.numel()))
        per_head_tucker_apply(
            module,
            n_heads=n_heads,
            hidden_rank=hidden_rank,
            head_dim_rank=head_dim_rank,
            is_output_projection=is_output,
            decompose_dtype=cfg.decompose_dtype,
            name=full_name,
        )
        rec = StorageRecord(
            full_name=full_name,
            layer_idx=int(spec["layer_idx"]),
            module_name=spec["module_name"],
            module_group=spec.get("module_group", "custom"),
            method="per_head_tucker",
            dense_bits=dense_bits,
            storage_bits=dense_bits,
            dense_params=dense_params,
            storage_params=dense_params,
            details={"n_heads": n_heads, "hidden_rank": hidden_rank, "head_dim_rank": head_dim_rank},
        )
        attach_storage_record(module, rec)
        records.append(rec.to_dict())
    return {"decomposition_per_module": records}



def apply_tensorllm_tucker_mlp_to_layers(model: nn.Module, target_specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    """TensorLLM-style Tucker for FFN/MLP layers.

    Same-shaped projections (e.g. gate_proj + up_proj in LLaMA SwiGLU, both
    [intermediate, hidden]) are stacked into [out, in, n] and Tucker is applied
    on modes [0, 1] only, yielding shared row-factor F0[out, rank] and column-
    factor F1[in, in_rank] while each projection retains its own core slice
    core[:, :, i].  Projections whose shape is unique in the layer (e.g.
    down_proj) get independent Tucker-2D (modes [0, 1]).
    """
    _require_tensorly()
    tl.set_backend("pytorch")
    model_spec = cfg.model_spec
    rank = int(plan.rank or global_rank)
    params = dict(plan.params or {})
    in_rank = int(params.get("ffn_in_rank", rank))
    decompose_device = normalize_torch_device_name(cfg.decompose_device)

    records = []
    for layer_idx in unique_layer_indices(target_specs):
        layer_specs = [s for s in target_specs if int(s["layer_idx"]) == layer_idx]

        # Collect modules for this layer
        mod_data = {}   # module_name -> {"module", "weight", "full_name", "spec"}
        for spec in layer_specs:
            full_name = spec["full_name"]
            module = get_module_by_name(model, full_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {full_name}, got {type(module).__name__}")
            w = module.weight.detach().to(device=decompose_device, dtype=torch.float32).contiguous()
            mod_data[spec["module_name"]] = {
                "module": module,
                "weight": w,
                "full_name": full_name,
                "spec": spec,
                "orig_device": module.weight.device,
                "orig_dtype": module.weight.dtype,
            }

        # Group by weight shape so same-shaped modules share Tucker factors
        from collections import defaultdict
        shape_groups: dict[tuple, list[str]] = defaultdict(list)
        for mod_name, info in mod_data.items():
            shape_groups[tuple(info["weight"].shape)].append(mod_name)

        for shape, group_names in shape_groups.items():
            if len(group_names) >= 2:
                # ── Shared Tucker across same-shaped projections ──────────────
                weights = [mod_data[n]["weight"] for n in group_names]
                stacked = torch.stack(weights, dim=2)   # [out, in, n_group]
                out_f, in_f, n_g = stacked.shape
                r_out = min(rank, out_f)
                r_in  = min(in_rank, in_f)

                # Tucker on modes [0,1] → shared F0, F1; per-proj cores via slices
                reconstructed, core, factors, quantized, rel_error = _partial_tucker_reconstruct(
                    stacked, modes=[0, 1], rank=[r_out, r_in], plan=plan, cfg=cfg
                )
                # reconstructed: [out, in, n_g]
                # core:          [r_out, r_in, n_g]
                # factors[0]:    [out, r_out],  factors[1]: [in, r_in]

                shared_factor_params = sum(int(f.numel()) for f in factors)

                for i, mod_name in enumerate(group_names):
                    info = mod_data[mod_name]
                    module = info["module"]
                    dense_bits_i = linear_dense_bits(module, cfg)
                    dense_params_i = int(module.weight.numel() + (0 if module.bias is None else module.bias.numel()))

                    rec_w = reconstructed[:, :, i].to(device=info["orig_device"], dtype=info["orig_dtype"]).contiguous()
                    module.weight = nn.Parameter(rec_w)

                    # Storage: per-proj core slice + equal share of shared factors
                    core_params_i = int(core[:, :, i].numel())
                    storage_params_i = core_params_i + shared_factor_params // n_g
                    value_bits = int(plan.quant_bits) if quantized else int(cfg.model_storage_bits)
                    storage_bits_i = storage_params_i * value_bits

                    rec_record = StorageRecord(
                        full_name=info["full_name"],
                        layer_idx=int(layer_idx),
                        module_name=info["spec"]["module_name"],
                        module_group="mlp",
                        method="tensorllm_tucker_mlp",
                        dense_bits=dense_bits_i,
                        storage_bits=storage_bits_i,
                        dense_params=dense_params_i,
                        storage_params=storage_params_i,
                        quant_method=plan.quant_method if quantized else "none",
                        quant_bits=plan.quant_bits if quantized else None,
                        details={
                            "shared_group": group_names,
                            "out_rank": r_out,
                            "in_rank": r_in,
                            "relative_reconstruction_error": rel_error,
                        },
                    )
                    attach_storage_record(module, rec_record)
                    records.append(rec_record.to_dict())
            else:
                # ── Independent Tucker-2D for singleton projections ───────────
                mod_name = group_names[0]
                info = mod_data[mod_name]
                module = info["module"]
                w = info["weight"]
                out_f, in_f = w.shape
                r_out = min(rank, out_f)
                r_in  = min(in_rank, in_f)

                reconstructed, core, factors, quantized, rel_error = _partial_tucker_reconstruct(
                    w, modes=[0, 1], rank=[r_out, r_in], plan=plan, cfg=cfg
                )

                dense_bits_i = linear_dense_bits(module, cfg)
                storage_bits_i, comp_details = _tucker_storage_bits(
                    core, factors, dense_bits=dense_bits_i, quant_bits=plan.quant_bits, cfg=cfg, quantized=quantized
                )
                dense_params_i = int(module.weight.numel() + (0 if module.bias is None else module.bias.numel()))

                module.weight = nn.Parameter(
                    reconstructed.to(device=info["orig_device"], dtype=info["orig_dtype"]).contiguous()
                )

                rec_record = StorageRecord(
                    full_name=info["full_name"],
                    layer_idx=int(layer_idx),
                    module_name=info["spec"]["module_name"],
                    module_group="mlp",
                    method="tensorllm_tucker_mlp",
                    dense_bits=dense_bits_i,
                    storage_bits=storage_bits_i,
                    dense_params=dense_params_i,
                    storage_params=int(core.numel() + sum(f.numel() for f in factors)),
                    quant_method=plan.quant_method if quantized else "none",
                    quant_bits=plan.quant_bits if quantized else None,
                    details={"out_rank": r_out, "in_rank": r_in, "relative_reconstruction_error": rel_error, **comp_details},
                )
                attach_storage_record(module, rec_record)
                records.append(rec_record.to_dict())

    return {"decomposition_per_module": records}

def apply_tensorllm_tucker_mha_sep_last_to_layers(model: nn.Module, target_specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> dict:
    """Tucker on QKVO with shared hidden-dim factor but separate cores per projection.

    Standard 4D Tucker stacks Q/K/V/O and compresses the QKVO dimension (stack_rank<4),
    finding a shared subspace across all four projections. This variant runs 4D Tucker
    only to derive the shared factor F0[hidden, rank] for the hidden dimension, then 
    discards the QKVO mixing factor. The head_dim factor is NOT shared - each projection
    maintains its own head-specific subspaces. Each projection gets its own 3D core 
    computed by projecting the original weight through the shared F0 (optimal given 
    that factor), then reconstructed back to a dense nn.Linear.

    Storage per projection (amortising the shared factor equally):
        core[rank, n_heads, head_dim]  +  F0.numel / n_proj
    """
    _require_tensorly()
    tl.set_backend("pytorch")
    model_spec = cfg.model_spec
    rank = int(plan.rank or global_rank)
    params = dict(plan.params or {})
    # head_dim_rank is no longer needed since we don't compress head_dim
    stack_rank = int(params.get("stack_rank", cfg.tensorllm_stack_rank))
    allow_fallback = bool(params.get("allow_gqa_per_projection_fallback", cfg.tensorllm_allow_gqa_per_projection_fallback))
    decompose_device = normalize_torch_device_name(cfg.decompose_device)

    records = []
    for layer_idx in unique_layer_indices(target_specs):
        modules = {
            name: get_module_by_name(model, full_module_name_for(model_spec, layer_idx, rel))
            for name, rel in model_spec.tensorllm_qkvo_modules.items()
        }
        weights = {}
        shapes = {}
        dense_bits = 0
        for name, module in modules.items():
            rel = model_spec.tensorllm_qkvo_modules[name]
            is_o = rel in model_spec.output_projection_names
            w = _linear_weight_for_tucker(module, is_output_projection=is_o).detach().to(
                device=decompose_device, dtype=torch.float32
            ).contiguous()
            weights[name] = w
            shapes[name] = tuple(w.shape)
            dense_bits += linear_dense_bits(module, cfg)

        equal_shapes = len(set(shapes.values())) == 1
        if equal_shapes:
            # Step 1: build 4D stacked tensor to obtain shared factor
            stacked = []
            for name in model_spec.tensorllm_qkvo_order:
                hidden_in, n_heads_proj, head_dim = _head_shape_for_weight(weights[name], model)
                stacked.append(weights[name].view(hidden_in, n_heads_proj, head_dim))
            tensor4d = torch.stack(stacked, dim=3)          # [hidden, n_heads, head_dim, 4]

            # Only decompose modes [0, 3]: hidden and QKVO
            # Mode 1 (n_heads) and Mode 2 (head_dim) stay in the core
            modes = [0, 3]
            ranks = [
                min(rank, tensor4d.shape[0]),
                min(stack_rank, tensor4d.shape[3]),
            ]
            work = tensor4d.detach().to(device=decompose_device, dtype=torch.float32).contiguous()
            (_, shared_factors), _ = partial_tucker(
                work, modes=modes, rank=ranks,
                init="svd", svd="randomized_svd", random_state=0, tol=1e-5, verbose=False,
            )
            # shared_factors[0]: [hidden, rank]             (mode-0 hidden factor, kept)
            # shared_factors[1]: QKVO mixing factor         (mode-3, discarded -- "last factor")
            F0 = shared_factors[0]   # [hidden, rank]
            # Discard shared_factors[1] (QKVO factor)
            n_proj = len(model_spec.tensorllm_qkvo_order)

            # Step 2: per-projection optimal core + reconstruction
            for idx, name in enumerate(model_spec.tensorllm_qkvo_order):
                rel = model_spec.tensorllm_qkvo_modules[name]
                full = full_module_name_for(model_spec, layer_idx, rel)
                module = modules[name]
                is_o = rel in model_spec.output_projection_names
                w = weights[name]
                hidden_in, n_heads_proj, head_dim = _head_shape_for_weight(w, model)
                w_3d = w.view(hidden_in, n_heads_proj, head_dim)   # [hidden, n_heads, head_dim]

                # Optimal per-projection core given shared factor F0:
                #   core_proj = W x_0 F0.T   -> [rank, n_heads, head_dim]
                # Only project along mode-0 (hidden), keeping head_dim intact
                core_proj = tl.tenalg.multi_mode_dot(w_3d, [F0.T], modes=[0])

                # Reconstruct: W_rec = core_proj x_0 F0  -> [hidden, n_heads, head_dim]
                rec_3d = tl.tenalg.multi_mode_dot(core_proj, [F0], modes=[0])

                rel_error = float(
                    (torch.norm(rec_3d - w_3d) / torch.norm(w_3d).clamp_min(1e-12))
                    .detach().float().cpu().item()
                )
                _assign_linear_weight_from_tucker(module, rec_3d.reshape_as(w), is_output_projection=is_o)

                # Storage: per-proj core + equal share of the shared factor F0
                core_params = int(core_proj.numel())
                shared_params_per_proj = int(F0.numel() / n_proj)
                storage_params = core_params + shared_params_per_proj
                storage_bits = storage_params * int(cfg.model_storage_bits)
                dense_bits_i = linear_dense_bits(module, cfg)

                rec_record = StorageRecord(
                    full_name=full,
                    layer_idx=int(layer_idx),
                    module_name=rel,
                    module_group="mha",
                    method="tensorllm_tucker_mha_sep_last",
                    dense_bits=dense_bits_i,
                    storage_bits=storage_bits,
                    dense_params=int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
                    storage_params=storage_params,
                    details={
                        "shared_hidden_rank": int(F0.shape[1]),
                        "per_proj_core_shape": list(core_proj.shape),
                        "relative_reconstruction_error": rel_error,
                        "projection": name,
                    },
                )
                attach_storage_record(module, rec_record)
                records.append(rec_record.to_dict())
        else:
            # Fallback for GQA (unequal Q/K/V/O shapes): per-projection 2D Tucker
            if not allow_fallback:
                raise ValueError(
                    f"Layer {layer_idx}: Q/K/V/O shapes differ ({shapes}). "
                    "Set allow_gqa_per_projection_fallback=True to use per-projection Tucker fallback."
                )
            warnings.warn(
                f"Layer {layer_idx}: Q/K/V/O shapes differ ({shapes}); using per-projection 2D Tucker fallback.",
                RuntimeWarning,
            )
            for name in model_spec.tensorllm_qkvo_order:
                rel = model_spec.tensorllm_qkvo_modules[name]
                full = full_module_name_for(model_spec, layer_idx, rel)
                module = modules[name]
                is_o = rel in model_spec.output_projection_names
                w = weights[name]
                hidden_in, n_heads_proj, head_dim = _head_shape_for_weight(w, model)
                tensor = w.view(hidden_in, n_heads_proj, head_dim)
                # Only decompose mode 0 (hidden), keeping n_heads and head_dim in core
                modes = [0]
                ranks = [min(rank, tensor.shape[0])]
                reconstructed, core, factors, quantized, rel_error = _partial_tucker_reconstruct(
                    tensor, modes=modes, rank=ranks, plan=plan, cfg=cfg
                )
                _assign_linear_weight_from_tucker(module, reconstructed.reshape_as(w), is_output_projection=is_o)
                dense_bits_i = linear_dense_bits(module, cfg)
                storage_bits, comp_details = _tucker_storage_bits(
                    core, factors, dense_bits=dense_bits_i, quant_bits=plan.quant_bits, cfg=cfg, quantized=quantized
                )
                rec_record = StorageRecord(
                    full_name=full,
                    layer_idx=int(layer_idx),
                    module_name=rel,
                    module_group="mha",
                    method="tensorllm_tucker_mha_sep_last_gqa_fallback",
                    dense_bits=dense_bits_i,
                    storage_bits=storage_bits,
                    dense_params=int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
                    storage_params=int(core.numel() + sum(f.numel() for f in factors)),
                    quant_method=plan.quant_method if quantized else "none",
                    quant_bits=plan.quant_bits if quantized else None,
                    details={"relative_reconstruction_error": rel_error, "ranks": ranks, "projection": name, **comp_details},
                )
                attach_storage_record(module, rec_record)
                records.append(rec_record.to_dict())
    return {"decomposition_per_module": records, "tensorllm_records": records}


def apply_decomposition_plan(model: nn.Module, target_set: dict, plan: ModulePlan, cfg: RunnerConfig, *, global_rank: int) -> tuple[list[dict], dict]:
    specs = make_target_specs(target_set, cfg.model_spec, selectors=[plan.target])
    t0 = time.time()
    if plan.decomposition_method == "none":
        stats = {"decomposition_per_module": []}
    elif plan.decomposition_method == "tt":
        stats = apply_tt_to_specs(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "dense_sparse":
        stats = apply_dense_sparse_to_specs(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method in {"tensorllm_tucker_mha", "tensorllm_tucker4d"}:
        stats = apply_tensorllm_tucker_mha_to_layers(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "td_moe":
        stats = apply_td_moe_to_layers(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "tensorllm_tucker_mlp":
        stats = apply_tensorllm_tucker_mlp_to_layers(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "laser":
        stats = apply_laser_to_specs(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "tt_per_head":
        stats = apply_per_head_tt_to_mha_layers(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "tucker_per_head":
        stats = apply_per_head_tucker_to_mha_layers(model, specs, plan, cfg, global_rank=global_rank)
    elif plan.decomposition_method == "tensorllm_tucker_mha_sep_last":
        stats = apply_tensorllm_tucker_mha_sep_last_to_layers(model, specs, plan, cfg, global_rank=global_rank)
    else:
        raise ValueError(f"Unknown decomposition method: {plan.decomposition_method}")
    stats["decomposition_apply_time_s"] = round(time.time() - t0, 3)
    stats["plan"] = asdict(plan)
    return specs, stats


def collect_target_module_inputs(model, tokenizer, specs, *, num_sequences, seq_len, batch_size=1, seed=0, max_tokens=None):
    captured = {s["full_name"]: [] for s in specs}
    handles = []
    def make_hook(full_name):
        def hook_fn(mod, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captured[full_name].append(x.detach().float().cpu())
        return hook_fn
    for s in specs:
        handles.append(get_module_by_name(model, s["full_name"]).register_forward_pre_hook(make_hook(s["full_name"])))
    loader = build_wikitext2_train_loader(tokenizer, seq_len=seq_len, batch_size=batch_size, num_sequences=num_sequences, seed=seed)
    device = infer_input_device(model)
    model.eval()
    for seen, (input_ids,) in enumerate(loader, start=1):
        with torch.no_grad():
            _ = model(input_ids=input_ids.to(device))
        if seen >= num_sequences:
            break
    for h in handles:
        h.remove()
    out = {}
    for full_name, chunks in captured.items():
        if not chunks:
            raise RuntimeError(f"No calibration inputs captured for {full_name}")
        x = torch.cat([t.reshape(-1, t.shape[-1]) for t in chunks], dim=0)
        if max_tokens is not None:
            x = x[:max_tokens]
        out[full_name] = x.contiguous()
    return out


def apply_rtn_to_specs(model: nn.Module, specs: Sequence[dict], cfg: RunnerConfig, bits: int):
    for spec in specs:
        module = get_module_by_name(model, spec["full_name"])
        if isinstance(module, NewTTLayer):
            fake_quantize_newtt_inplace(module, bits=bits, cfg=cfg)
        elif isinstance(module, DenseSparseNewTTLayer):
            fake_quantize_dense_sparse_newtt_inplace(module, bits=bits, cfg=cfg)
        elif isinstance(module, LoRAOnFrozenModule):
            base = module.base_module
            if isinstance(base, NewTTLayer):
                fake_quantize_newtt_inplace(base, bits=bits, cfg=cfg)
            elif isinstance(base, DenseSparseNewTTLayer):
                fake_quantize_dense_sparse_newtt_inplace(base, bits=bits, cfg=cfg)
            elif isinstance(base, nn.Linear):
                fake_quantize_linear_inplace(base, bits=bits, cfg=cfg)
            else:
                raise TypeError(type(base).__name__)
        elif isinstance(module, nn.Linear):
            fake_quantize_linear_inplace(module, bits=bits, cfg=cfg)
        else:
            raise TypeError(f"Unsupported module for RTN at {spec['full_name']}: {type(module).__name__}")


def apply_quantization_plan(model, tokenizer, specs: Sequence[dict], plan: ModulePlan, cfg: RunnerConfig, *, seed: int) -> dict:
    if plan.quant_method in {"none", "tensorllm_pre_reconstruct_rtn"}:
        return {"quantization_method": plan.quant_method, "quantization_bits": plan.quant_bits, "quantization_apply_time_s": 0.0}
    t0 = time.time()
    if plan.quant_method == "rtn_symmetric":
        apply_rtn_to_specs(model, specs, cfg, bits=int(plan.quant_bits))
    elif plan.quant_method == "gptq_tt":
        raise NotImplementedError("GPTQ-TT support can be plugged in here; use rtn_symmetric for this modular runner.")
    else:
        raise ValueError(f"Unknown quantization method: {plan.quant_method}")
    return {"quantization_method": plan.quant_method, "quantization_bits": plan.quant_bits, "quantization_apply_time_s": round(time.time() - t0, 3)}


def quantize_lora_modules(model: nn.Module, specs: Sequence[dict], cfg: RunnerConfig, recipe: RecipeSpec):
    if recipe.lora_quant_bits is None:
        return {"lora_quant_method": "none", "lora_quant_bits": None, "lora_quantized_modules": 0}
    count = 0
    for spec in specs:
        module = get_module_by_name(model, spec["full_name"])
        if isinstance(module, LoRAOnFrozenModule):
            fake_quantize_lora_inplace(module, bits=int(recipe.lora_quant_bits), cfg=cfg)
            count += 1
    return {"lora_quant_method": recipe.lora_quant_method, "lora_quant_bits": int(recipe.lora_quant_bits), "lora_quantized_modules": count}


# Storage accounting

def tt_layer_storage_bits(layer: NewTTLayer, cfg: RunnerConfig) -> int:
    qbits = getattr(layer, "storage_quant_bits", None)
    scale_bits = int(getattr(layer, "storage_scale_bits", cfg.scale_storage_bits))
    bits = 0
    for core in layer.cores:
        if qbits is None:
            bits += int(core.numel()) * int(cfg.model_storage_bits)
        else:
            bits += int(core.numel()) * int(qbits) + scale_bits
    if layer.bias is not None:
        bits += int(layer.bias.numel()) * int(cfg.model_storage_bits)
    return int(bits)


def target_module_storage_bits(module: nn.Module, cfg: RunnerConfig) -> int:
    rec = storage_record_from_module(module)
    if rec is not None:
        if isinstance(module, NewTTLayer):
            return tt_layer_storage_bits(module, cfg)
        if isinstance(module, DenseSparseNewTTLayer):
            return int(tt_layer_storage_bits(module.tt_part, cfg) + module.sparse_part.storage_bits())
        if isinstance(module, LoRAOnFrozenModule):
            base_bits = target_module_storage_bits(module.base_module, cfg)
            return int(base_bits + lora_storage_bits(module, cfg))
        return int(rec["storage_bits"])
    if isinstance(module, nn.Linear):
        qbits = getattr(module, "storage_quant_bits", None)
        if qbits is None:
            return int(module.weight.numel() * cfg.model_storage_bits + (0 if module.bias is None else module.bias.numel() * cfg.model_storage_bits))
        return int(module.weight.numel() * int(qbits) + int(cfg.scale_storage_bits) + (0 if module.bias is None else module.bias.numel() * cfg.model_storage_bits))
    if isinstance(module, NewTTLayer):
        return tt_layer_storage_bits(module, cfg)
    if isinstance(module, DenseSparseNewTTLayer):
        return int(tt_layer_storage_bits(module.tt_part, cfg) + module.sparse_part.storage_bits())
    if isinstance(module, LoRAOnFrozenModule):
        return int(target_module_storage_bits(module.base_module, cfg) + lora_storage_bits(module, cfg))
    if hasattr(module, "gate_up_proj") and torch.is_tensor(module.gate_up_proj):
        return moe_experts_gate_up_dense_bits(module, cfg)
    raise TypeError(f"Unsupported module for storage accounting: {type(module).__name__}")


def current_storage_records(model: nn.Module, specs: Sequence[dict], cfg: RunnerConfig) -> list[dict]:
    out = []
    for spec in specs:
        module = get_module_by_name(model, spec["full_name"])
        rec = storage_record_from_module(module)
        if rec is None:
            dense_bits = dense_bits_for_target_module(module, cfg)
            rec = StorageRecord(
                full_name=spec["full_name"], layer_idx=int(spec["layer_idx"]), module_name=spec["module_name"], module_group=spec.get("module_group", "custom"),
                method="dense_or_unrecorded", dense_bits=dense_bits, storage_bits=target_module_storage_bits(module, cfg), dense_params=0, storage_params=0,
            ).to_dict()
        else:
            rec = dict(rec)
            rec["storage_bits"] = int(target_module_storage_bits(module, cfg))
        out.append(rec)
    return out


def compute_storage_summary(model: nn.Module, baseline_model: nn.Module, target_set: dict, affected_specs: Sequence[dict], all_candidate_specs: Sequence[dict], cfg: RunnerConfig) -> dict:
    model_spec = cfg.model_spec
    layer_indices = target_set["layer_indices"]
    affected_full = {s["full_name"] for s in affected_specs}
    affected_dense_bits = baseline_bits_for_specs(baseline_model, affected_specs, cfg)
    affected_storage_bits = sum(target_module_storage_bits(get_module_by_name(model, s["full_name"]), cfg) for s in affected_specs)

    def group_bits(group: str):
        specs = enumerate_group_specs(target_set, model_spec, group)
        dense = baseline_bits_for_specs(baseline_model, specs, cfg)
        storage = 0
        for s in specs:
            if s["full_name"] in affected_full:
                storage += target_module_storage_bits(get_module_by_name(model, s["full_name"]), cfg)
            else:
                storage += dense_bits_for_target_module(get_module_by_name(baseline_model, s["full_name"]), cfg)
        return int(dense), int(storage)

    group_storage = {group: group_bits(group) for group in model_spec.module_groups}
    block_dense = block_dense_bits(baseline_model, model_spec, cfg, layer_indices)
    block_storage = int(block_dense - affected_dense_bits + affected_storage_bits)
    total_dense = get_model_dense_bits(baseline_model, cfg, include_embeddings=True)
    total_storage = int(total_dense - affected_dense_bits + affected_storage_bits)
    noemb_dense = get_model_dense_bits(baseline_model, cfg, include_embeddings=False)
    noemb_storage = int(noemb_dense - affected_dense_bits + affected_storage_bits)
    records = current_storage_records(model, affected_specs, cfg)

    summary = {
        "storage_records": records,
        "affected_modules_dense_bits": int(affected_dense_bits),
        "affected_modules_storage_bits": int(affected_storage_bits),
        "affected_modules_compression_ratio": float(affected_dense_bits / max(affected_storage_bits, 1)),
        "affected_blocks_dense_bits": int(block_dense),
        "affected_blocks_storage_bits": int(block_storage),
        "affected_blocks_compression_ratio": float(block_dense / max(block_storage, 1)),
        "overall_no_embeddings_dense_bits": int(noemb_dense),
        "overall_no_embeddings_storage_bits": int(noemb_storage),
        "overall_no_embeddings_compression_ratio": float(noemb_dense / max(noemb_storage, 1)),
        "overall_with_embeddings_dense_bits": int(total_dense),
        "overall_with_embeddings_storage_bits": int(total_storage),
        "overall_with_embeddings_compression_ratio": float(total_dense / max(total_storage, 1)),
        "target_layer_bits": int(affected_storage_bits),
        "overall_model_bits": int(total_storage),
        "target_layer_compression_ratio": float(affected_dense_bits / max(affected_storage_bits, 1)),
        "overall_model_compression_ratio": float(total_dense / max(total_storage, 1)),
    }
    for group, (dense, storage) in group_storage.items():
        prefix = f"{group}_in_affected_blocks"
        summary[f"{prefix}_dense_bits"] = int(dense)
        summary[f"{prefix}_storage_bits"] = int(storage)
        summary[f"{prefix}_compression_ratio"] = float(dense / max(storage, 1))
    return summary


# Dense reconstruction for benchmarks

@torch.no_grad()
def get_effective_weight_for_module(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if isinstance(module, LoRAOnFrozenModule):
        return module.merged_dense_weight().detach().to(device="cpu", dtype=torch.float32).contiguous()
    if isinstance(module, NewTTLayer):
        return module.to_dense_weight().detach().to(device="cpu", dtype=torch.float32).contiguous()
    if isinstance(module, DenseSparseNewTTLayer):
        return module.to_dense_weight().detach().to(device="cpu", dtype=torch.float32).contiguous()
    if hasattr(module, "gate_up_proj") and torch.is_tensor(module.gate_up_proj):
        gate_up = _module_weight_for_decomposition(
            module,
            "gate_up_proj",
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        return gate_up.reshape(-1, gate_up.shape[-1]).contiguous()
    raise TypeError(f"Unsupported module type for effective weight: {type(module).__name__}")


def get_effective_target_weights(model: nn.Module, specs: Sequence[dict]) -> dict[str, torch.Tensor]:
    return {s["full_name"]: get_effective_weight_for_module(get_module_by_name(model, s["full_name"])) for s in specs}


def _module_device_dtype(module: nn.Module, cfg: RunnerConfig) -> tuple[torch.device, torch.dtype]:
    if isinstance(module, LoRAOnFrozenModule):
        return _module_device_dtype(module.base_module, cfg)

    if isinstance(module, nn.Linear):
        return module.weight.device, module.weight.dtype

    if isinstance(module, NewTTLayer):
        p = next(module.parameters(), None)
        if p is not None:
            return p.device, p.dtype

    if isinstance(module, DenseSparseNewTTLayer):
        p = next(module.tt_part.parameters(), None)
        if p is not None:
            return p.device, p.dtype

    for p in module.parameters(recurse=True):
        return p.device, p.dtype

    for b in module.buffers(recurse=True):
        if torch.is_tensor(b):
            return b.device, b.dtype

    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), cfg.model_dtype


@torch.no_grad()
def module_to_dense_linear(module: nn.Module, *, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    weight_cpu = get_effective_weight_for_module(module).detach().to(
        device="cpu",
        dtype=dtype,
    ).contiguous()

    if isinstance(module, LoRAOnFrozenModule):
        bias = module.merged_dense_bias()
    elif hasattr(module, "bias"):
        bias = None if module.bias is None else module.bias.detach().clone()
    else:
        bias = None

    out_features, in_features = weight_cpu.shape

    dense = nn.Linear(
        int(in_features),
        int(out_features),
        bias=bias is not None,
        device="cpu",
        dtype=dtype,
    )
    dense.weight.data.copy_(weight_cpu)

    if bias is not None:
        dense.bias.data.copy_(bias.detach().to(device="cpu", dtype=dtype))

    rec = storage_record_from_module(module)
    if rec is not None:
        dense._compression_storage_record = rec

    dense = dense.to(device=device, dtype=dtype)

    del weight_cpu, bias
    return dense


@torch.no_grad()
def reconstruct_specs_to_dense_for_benchmark(
    model: nn.Module,
    specs: Sequence[dict],
    cfg: RunnerConfig,
) -> list[dict]:
    records = []

    for spec in specs:
        full_name = spec["full_name"]
        old_module = get_module_by_name(model, full_name)
        old_type = type(old_module).__name__
        old_device, old_dtype = _module_device_dtype(old_module, cfg)

        if isinstance(old_module, nn.Linear):
            records.append({
                "module": full_name,
                "old_type": old_type,
                "new_type": old_type,
                "reconstructed": False,
                "device": str(old_device),
                "dtype": str(old_dtype),
            })
            continue

        rec = storage_record_from_module(old_module)
        if rec is not None and rec.get("method") == "td_moe":
            records.append({
                "module": full_name,
                "old_type": old_type,
                "new_type": old_type,
                "reconstructed": False,
                "device": str(old_device),
                "dtype": str(old_dtype),
            })
            continue

        dense = module_to_dense_linear(old_module, device=old_device, dtype=old_dtype)
        set_module_by_name(model, full_name, dense)

        del old_module, dense
        clean_mem()

        new_module = get_module_by_name(model, full_name)
        new_device, new_dtype = _module_device_dtype(new_module, cfg)

        records.append({
            "module": full_name,
            "old_type": old_type,
            "new_type": type(new_module).__name__,
            "reconstructed": True,
            "device": str(new_device),
            "dtype": str(new_dtype),
        })

    clean_mem()
    return records


# Diagnostics, metrics, and evaluation

def summarize_superweight_error_multi(current_weights: dict, baseline_weights: dict, specs: Sequence[dict]):
    rows = []
    for spec in specs:
        full_name = spec["full_name"]
        for row, col in spec.get("superweight_coords", []):
            orig = float(baseline_weights[full_name][row, col].item())
            cur = float(current_weights[full_name][row, col].item())
            abs_error = abs(cur - orig)
            rel_error = abs_error / max(abs(orig), 1e-12)
            rows.append({"module": full_name, "row": int(row), "col": int(col), "original_value": orig, "current_value": cur, "abs_error": abs_error, "rel_error": rel_error})
    if not rows:
        return {"superweight_abs_error_mean": float("nan"), "superweight_abs_error_max": float("nan"), "superweight_rel_error_mean": float("nan"), "superweight_rel_error_max": float("nan"), "superweight_error_detail": []}
    df = pd.DataFrame(rows)
    return {"superweight_abs_error_mean": float(df.abs_error.mean()), "superweight_abs_error_max": float(df.abs_error.max()), "superweight_rel_error_mean": float(df.rel_error.mean()), "superweight_rel_error_max": float(df.rel_error.max()), "superweight_error_detail": rows}


def summarize_layer_error_multi(current_weights: dict, baseline_weights: dict):
    rows = []
    total_sq_num = 0.0
    total_sq_den = 0.0
    global_max_abs = 0.0
    for full_name in baseline_weights:
        cur = current_weights[full_name]
        base = baseline_weights[full_name]
        diff = cur - base
        sq_num = float(torch.sum(diff * diff).item())
        sq_den = float(torch.sum(base * base).item())
        fro_rel = math.sqrt(sq_num / max(sq_den, 1e-12))
        max_abs = float(diff.abs().max().item())
        rows.append({"module": full_name, "fro_rel_error": fro_rel, "max_abs_error": max_abs})
        total_sq_num += sq_num
        total_sq_den += sq_den
        global_max_abs = max(global_max_abs, max_abs)
    return {"target_layer_fro_rel_error": math.sqrt(total_sq_num / max(total_sq_den, 1e-12)), "target_layer_max_abs_error": global_max_abs, "target_layer_error_detail": rows}


def sample_dense_entry_error_summary_multi(
    current_weights: dict,
    baseline_weights: dict,
    specs: Sequence[dict],
    *,
    topk_outliers=64,
    random_samples=256,
    seed=0,
):
    rows = []
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for spec in specs:
        full_name = spec["full_name"]
        base = baseline_weights[full_name]
        cur = current_weights[full_name]
        H, W = base.shape
        total = H * W

        seen = set()
        for row, col in spec.get("superweight_coords", []):
            seen.add((int(row), int(col)))

        flat_abs = base.abs().reshape(-1)
        k = min(int(topk_outliers) + len(seen), flat_abs.numel())
        if k > 0:
            top_idx = torch.topk(flat_abs, k=k).indices.cpu()
            picked = 0
            for idx_t in top_idx:
                idx = int(idx_t.item())
                row, col = divmod(idx, W)
                key = (int(row), int(col))
                if key in seen:
                    continue
                seen.add(key)

                orig = float(base[row, col].item())
                curv = float(cur[row, col].item())
                ae = abs(curv - orig)
                rows.append({
                    "module": full_name,
                    "tag": "top_outlier",
                    "row": int(row),
                    "col": int(col),
                    "original_abs": abs(orig),
                    "abs_error": ae,
                    "rel_error": ae / max(abs(orig), 1e-12),
                })

                picked += 1
                if picked >= topk_outliers:
                    break

        picked = 0
        attempts = 0
        max_attempts = max(1000, int(random_samples) * 20)

        while picked < int(random_samples) and attempts < max_attempts:
            attempts += 1
            idx = int(torch.randint(0, total, (1,), generator=gen).item())
            row, col = divmod(idx, W)
            key = (int(row), int(col))
            if key in seen:
                continue

            seen.add(key)
            orig = float(base[row, col].item())
            curv = float(cur[row, col].item())
            ae = abs(curv - orig)

            rows.append({
                "module": full_name,
                "tag": "random",
                "row": int(row),
                "col": int(col),
                "original_abs": abs(orig),
                "abs_error": ae,
                "rel_error": ae / max(abs(orig), 1e-12),
            })
            picked += 1

    return pd.DataFrame(rows)


def evaluate_ppl_datasets(model, tokenizer, cfg: RunnerConfig) -> dict:
    if not cfg.run_ppl:
        return {}
    if eval_ppl is None:
        return {"ppl_error": "src.utils.eval.eval_ppl is not available"}
    ppl = eval_ppl(model, tokenizer, datasets=list(cfg.ppl_datasets), seqlen=int(cfg.ppl_seqlen))
    out = {}
    for ds, value in ppl.items():
        out[f"ppl_{ds}"] = float(value)
    if "wikitext2" in ppl:
        out["wikitext2_ppl"] = float(ppl["wikitext2"])
    return out


@torch.no_grad()
def generate_examples_for_prompts(model, tokenizer, cfg: RunnerConfig):
    if not cfg.run_generation_examples or not cfg.generation_prompts:
        return None
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    device = infer_input_device(model)
    outputs = {}
    for prompt in cfg.generation_prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=cfg.generation_max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
        outputs[prompt] = tokenizer.decode(out[0], skip_special_tokens=True)
    return outputs



def evaluate_lm_eval_harness(model, tokenizer, cfg: RunnerConfig) -> dict:
    out = {
        "lm_eval_requested": bool(cfg.run_lm_eval),
        "lm_eval_attempted": False,
        "lm_eval_tasks_configured": list(cfg.lm_eval_tasks),
        "lm_eval_skipped_reason": None,
    }

    if not cfg.run_lm_eval:
        out["lm_eval_skipped_reason"] = "cfg.run_lm_eval is False"
        return out

    if not cfg.lm_eval_tasks:
        msg = "cfg.run_lm_eval is True, but cfg.lm_eval_tasks is empty"
        out["lm_eval_skipped_reason"] = msg
        if cfg.lm_eval_strict:
            raise ValueError(msg)
        return out

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except Exception as exc:
        if cfg.lm_eval_strict:
            raise
        out["lm_eval_error"] = f"lm-evaluation-harness unavailable: {exc!r}"
        return out

    out["lm_eval_attempted"] = True
    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=cfg.lm_eval_batch_size)
        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=list(cfg.lm_eval_tasks),
            num_fewshot=int(cfg.lm_eval_num_fewshot),
            batch_size=cfg.lm_eval_batch_size,
            limit=cfg.lm_eval_limit,
            log_samples=cfg.lm_eval_log_samples,
        )
    except Exception as exc:
        if cfg.lm_eval_strict:
            raise
        out["lm_eval_error"] = repr(exc)
        return out

    flat = {}
    for task, metrics in res.get("results", {}).items():
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                flat[f"lm_eval/{task}/{key}"] = float(value)
    out.update({"lm_eval_results": res.get("results", {}), "lm_eval_versions": res.get("versions", {}), **flat})
    return out

def normalize_text_for_metric(text, *, case_sensitive=False, strip=True):
    text = text.strip() if strip else text
    return text if case_sensitive else text.lower()


def simple_tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def f1_precision_recall(generation, answer, *, case_sensitive=False, strip=True):
    generation = normalize_text_for_metric(generation, case_sensitive=case_sensitive, strip=strip)
    answer = normalize_text_for_metric(answer, case_sensitive=case_sensitive, strip=strip)
    gen_set = set(simple_tokenize_words(generation))
    ans_set = set(simple_tokenize_words(answer))
    inter = gen_set.intersection(ans_set)
    precision = float(len(inter)) / float(max(1, len(gen_set)))
    recall = float(len(inter)) / float(max(1, len(ans_set)))
    return {"f1": (2 * precision * recall) / float(max(1e-12, precision + recall)), "precision": precision, "recall": recall}


def generation_match(generation, answer, *, case_sensitive=False, strip=True):
    generation = normalize_text_for_metric(generation, case_sensitive=case_sensitive, strip=strip)
    answer = normalize_text_for_metric(answer, case_sensitive=case_sensitive, strip=strip)
    return answer in generation


def find_answer_len(question_answer_token_ids, answer, tokenizer):
    answer_stripped = answer.strip()
    length = int(question_answer_token_ids.shape[0])
    for i in range(length - 1, -1, -1):
        pad = tokenizer.decode(question_answer_token_ids[i:], clean_up_tokenization_spaces=False)
        if pad.strip() == answer_stripped:
            return length - i
    return max(1, len(tokenizer(answer, add_special_tokens=False).input_ids))


def answer_log_prob(log_prob, question_answer_token_ids, answer, tokenizer):
    answer_len = find_answer_len(question_answer_token_ids, answer, tokenizer)
    selected_log_prob = log_prob[:-1, :]
    indices = question_answer_token_ids[1:].unsqueeze(1)
    selected = torch.gather(selected_log_prob, index=indices, dim=1)
    return {"total_logprob": float(selected.sum().item()), "answer_logprob": float(selected[-answer_len:, 0].sum().item()), "answer_length": int(answer_len)}


def load_hotpot_dataset_tensorllm_style(tokenizer, cfg: RunnerConfig, *, filter_tokenizer=None):
    filter_tokenizer = filter_tokenizer or tokenizer
    full_dataset = load_dataset("hotpot_qa", "fullwiki")
    num_val = len(full_dataset["validation"])
    train = []
    for ctr, dp in enumerate(full_dataset["train"]):
        if ctr >= num_val:
            break
        q, a = dp["question"].strip(), dp["answer"].strip()
        if len(filter_tokenizer(a).input_ids) <= 15:
            train.append({"question": q, "answer": a})
    validation = []
    for dp in full_dataset["validation"]:
        q, a = dp["question"].strip(), dp["answer"].strip()
        if len(filter_tokenizer(a).input_ids) <= 15:
            validation.append({"question": q, "answer": a})
    dataset = train + validation
    if cfg.hotpot_max_examples is not None:
        dataset = dataset[:int(cfg.hotpot_max_examples)]
    return dataset


@torch.no_grad()
def evaluate_hotpot_tensorllm_style(model, tokenizer, cfg: RunnerConfig):
    if not cfg.run_hotpot:
        return {}
    filter_tokenizer = None
    if cfg.hotpot_use_llama_tokenizer_filter:
        filter_tokenizer = AutoTokenizer.from_pretrained(cfg.hotpot_filter_tokenizer_name, use_fast=True, trust_remote_code=True)
    dataset = load_hotpot_dataset_tensorllm_style(tokenizer, cfg, filter_tokenizer=filter_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = infer_input_device(model)
    model.eval()
    preds = []
    num_correct = sum_f1 = sum_total_lp = sum_mean_lp = total_answer_tokens = 0.0
    num_logprob_examples = 0
    for i in tqdm(range(0, len(dataset), cfg.hotpot_batch_size), desc="HotpotQA eval"):
        batch = dataset[i:i + cfg.hotpot_batch_size]
        questions = [dp["question"].strip() for dp in batch]
        answers = [dp["answer"].strip() for dp in batch]
        prompted = [f"{q}? The answer is" if not q.endswith("?") and not q.endswith(".") else f"{q} The answer is" for q in questions]
        inputs = tokenizer(prompted, return_tensors="pt", padding=True, truncation=True).to(device)
        qa = tokenizer([f"{pq} {a}" for pq, a in zip(prompted, answers)], return_tensors="pt", padding=True, truncation=True).to(device)
        gen_kwargs = dict(max_new_tokens=cfg.hotpot_max_new_tokens, min_new_tokens=1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if cfg.hotpot_beam > 1:
            gen_kwargs.update(num_beams=cfg.hotpot_beam, do_sample=False)
        gen_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **gen_kwargs)
        generations = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logits = model(qa.input_ids, attention_mask=qa.attention_mask).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        for j in range(len(batch)):
            lp = answer_log_prob(log_probs[j], qa.input_ids[j], answers[j], tokenizer)
            correct = generation_match(generations[j], answers[j])
            f1pr = f1_precision_recall(generations[j], answers[j])
            num_correct += 1.0 if correct else 0.0
            sum_f1 += f1pr["f1"]
            num_logprob_examples += 1
            sum_total_lp += lp["answer_logprob"]
            sum_mean_lp += lp["answer_logprob"] / float(max(1, lp["answer_length"]))
            total_answer_tokens += lp["answer_length"]
            preds.append({"correct": bool(correct), "f1_score": f1pr["f1"], **lp})
    n = max(1, len(preds))
    val_size = int(0.2 * len(preds))
    def _acc_log_loss(ps):
        if not ps:
            return float("nan"), float("nan")
        return float(np.mean([1.0 if p["correct"] else 0.0 for p in ps]) * 100.0), float(np.mean([-p["answer_logprob"] / max(1, p["answer_length"]) for p in ps]))
    val_acc, val_loss = _acc_log_loss(preds[:val_size])
    test_acc, test_loss = _acc_log_loss(preds[val_size:])
    return {
        "hotpot_0_1_correctness": (num_correct * 100.0) / float(n),
        "hotpot_avg_f1_score": sum_f1 / float(n),
        "hotpot_mean_log_prob": sum_mean_lp / float(max(1, num_logprob_examples)),
        "hotpot_perplexity": math.exp(-sum_total_lp / float(max(1, total_answer_tokens))),
        "hotpot_dataset_size": int(n),
        "hotpot_val_acc": val_acc,
        "hotpot_val_logloss": val_loss,
        "hotpot_test_acc": test_acc,
        "hotpot_test_logloss": test_loss,
    }


# Activation geometry

def _first_tensor_from_hook_args(args, kwargs=None, *, preferred_keys=("hidden_states", "input", "inputs_embeds")):
    kwargs = kwargs or {}
    for key in preferred_keys:
        value = kwargs.get(key, None)
        if torch.is_tensor(value):
            return value
    if isinstance(args, (tuple, list)):
        for value in args:
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(args):
        return args
    raise RuntimeError(f"Could not find a tensor in hook args/kwargs. args_type={type(args).__name__}, kwargs_keys={list(kwargs.keys())}")


def _first_tensor_from_output(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for value in out:
            if torch.is_tensor(value):
                return value
    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        return out.last_hidden_state
    raise RuntimeError(f"Could not find tensor in hook output of type {type(out).__name__}")


def _detach_hidden(x):
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


@torch.no_grad()
def collect_layer_activations(model, tokenizer, cfg: RunnerConfig, *, layer_indices: Sequence[int]):
    layers = get_transformer_layers(model, cfg.model_spec)
    data = {int(idx): [] for idx in layer_indices}
    current = {int(idx): {} for idx in layer_indices}
    handles = []
    def make_before_hook(layer_idx):
        def hook(module, args, kwargs):
            current[layer_idx]["before_layer"] = _detach_hidden(_first_tensor_from_hook_args(args, kwargs, preferred_keys=("hidden_states",)))
        return hook
    def make_after_mha_hook(layer_idx):
        def hook(module, args, kwargs, out):
            current[layer_idx]["after_mha"] = _detach_hidden(_first_tensor_from_output(out))
        return hook
    def make_after_mlp_hook(layer_idx):
        def hook(module, args, kwargs, out):
            current[layer_idx]["after_mlp"] = _detach_hidden(_first_tensor_from_output(out))
        return hook
    def make_after_layer_hook(layer_idx):
        def hook(module, args, kwargs, out):
            current[layer_idx]["after_layer"] = _detach_hidden(_first_tensor_from_output(out))
        return hook
    def make_after_mha_residual_hook(layer_idx):
        def hook(module, args, kwargs):
            current[layer_idx]["after_mha_residual"] = _detach_hidden(_first_tensor_from_hook_args(args, kwargs))
        return hook
    def make_mlp_input_post_ln_hook(layer_idx):
        def hook(module, args, kwargs, out):
            current[layer_idx]["mlp_input_post_ln"] = _detach_hidden(_first_tensor_from_output(out))
        return hook
    def make_router_logits_hook(layer_idx):
        def hook(module, args, kwargs, out):
            current[layer_idx]["router_logits"] = _detach_hidden(_first_tensor_from_output(out))
        return hook
    capture_sd_moe = bool(getattr(cfg, "run_sd_moe_diagnostics", False))
    router_path = getattr(cfg.model_spec, "router_module_path", "") or ""
    for layer_idx in layer_indices:
        block = layers[int(layer_idx)]
        handles.append(block.register_forward_pre_hook(make_before_hook(int(layer_idx)), with_kwargs=True))
        handles.append(get_attention_module(block, cfg.model_spec).register_forward_hook(make_after_mha_hook(int(layer_idx)), with_kwargs=True))
        handles.append(get_mlp_module(block, cfg.model_spec).register_forward_hook(make_after_mlp_hook(int(layer_idx)), with_kwargs=True))
        handles.append(block.register_forward_hook(make_after_layer_hook(int(layer_idx)), with_kwargs=True))
        if hasattr(block, "post_attention_layernorm"):
            handles.append(block.post_attention_layernorm.register_forward_pre_hook(make_after_mha_residual_hook(int(layer_idx)), with_kwargs=True))
            if capture_sd_moe:
                handles.append(block.post_attention_layernorm.register_forward_hook(make_mlp_input_post_ln_hook(int(layer_idx)), with_kwargs=True))
        if capture_sd_moe and router_path:
            try:
                router_module = get_module_by_name(block, router_path)
            except Exception:
                router_module = None
            if router_module is not None:
                handles.append(router_module.register_forward_hook(make_router_logits_hook(int(layer_idx)), with_kwargs=True))
    loader = build_wikitext2_train_loader(tokenizer, seq_len=cfg.geometry_seq_len, batch_size=1, num_sequences=cfg.geometry_num_samples, seed=cfg.geometry_seed, shuffle=False)
    device = infer_input_device(model)
    model.eval()
    try:
        for (input_ids,) in loader:
            for layer_idx in layer_indices:
                current[int(layer_idx)] = {}
            _ = model(input_ids=input_ids.to(device))
            for layer_idx in layer_indices:
                sample = current[int(layer_idx)]
                if "after_mha_residual" not in sample and "before_layer" in sample and "after_mha" in sample:
                    sample["after_mha_residual"] = sample["before_layer"] + sample["after_mha"]
                squeezed = {k: (v.squeeze(0) if torch.is_tensor(v) and v.ndim == 3 and v.shape[0] == 1 else v) for k, v in sample.items()}
                data[int(layer_idx)].append(squeezed)
    finally:
        for h in handles:
            h.remove()
        clean_mem()
    return data


def _safe_cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-12)


def _angle_deg_from_cos(c):
    return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0)))


def _cat_valid(tensors):
    valid = [x.reshape(-1).float() for x in tensors if x is not None and x.numel() > 0]
    return None if not valid else torch.cat(valid, dim=0)


def _stats_from_values(values):
    if values is None or values.numel() == 0:
        return {"mean": None, "median": None, "std": None, "num_tokens": 0}
    v = values.detach().float().cpu().numpy().astype(np.float32)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "num_tokens": int(len(v))}


def measure_activation_geometry(og_data: dict, method_data: dict, *, method_name: str):
    hook_points = ["before_layer", "after_mha", "after_mha_residual", "after_mlp", "after_layer"]
    stats_values = defaultdict(list)
    per_layer_stats = {}
    def add(per_layer_values, key, values):
        values = values.detach().float().cpu()
        stats_values[key].append(values)
        per_layer_values[key].append(values)
    for layer_idx in sorted(og_data.keys()):
        per_layer_values = defaultdict(list)
        for og_sample, cp_sample in zip(og_data[layer_idx], method_data[layer_idx]):
            for point in hook_points:
                if point in og_sample and point in cp_sample:
                    og, cp = og_sample[point], cp_sample[point]
                    cosv = _safe_cos(og, cp)
                    angle = _angle_deg_from_cos(cosv)
                    og_norm, cp_norm = og.norm(p=2, dim=-1), cp.norm(p=2, dim=-1)
                    diff = (og - cp).norm(p=2, dim=-1)
                    for key, val in [(f"cos_{point}_og_vs_method", cosv), (f"angle_deg_{point}_og_vs_method", angle), (f"norm_{point}_og", og_norm), (f"norm_{point}_method", cp_norm), (f"norm_ratio_{point}_method_over_og", cp_norm / og_norm.clamp_min(1e-12)), (f"diff_norm_{point}_og_vs_method", diff), (f"rel_diff_norm_{point}_og_vs_method", diff / og_norm.clamp_min(1e-12))]:
                        add(per_layer_values, key, val)
            for delta_name, (start_key, end_key) in {"delta_layer": ("before_layer", "after_layer"), "delta_mha": ("before_layer", "after_mha_residual"), "delta_mlp_to_layer": ("after_mha_residual", "after_layer")}.items():
                if start_key in og_sample and end_key in og_sample and start_key in cp_sample and end_key in cp_sample:
                    og_delta = og_sample[end_key] - og_sample[start_key]
                    cp_delta = cp_sample[end_key] - cp_sample[start_key]
                    cosv = _safe_cos(og_delta, cp_delta)
                    angle = _angle_deg_from_cos(cosv)
                    og_norm, cp_norm = og_delta.norm(p=2, dim=-1), cp_delta.norm(p=2, dim=-1)
                    diff = (og_delta - cp_delta).norm(p=2, dim=-1)
                    for key, val in [(f"cos_{delta_name}_og_vs_method", cosv), (f"angle_deg_{delta_name}_og_vs_method", angle), (f"norm_{delta_name}_og", og_norm), (f"norm_{delta_name}_method", cp_norm), (f"norm_ratio_{delta_name}_method_over_og", cp_norm / og_norm.clamp_min(1e-12)), (f"diff_norm_{delta_name}_og_vs_method", diff), (f"rel_diff_norm_{delta_name}_og_vs_method", diff / og_norm.clamp_min(1e-12))]:
                        add(per_layer_values, key, val)
        per_layer_stats[layer_idx] = {k: _stats_from_values(_cat_valid(vs)) for k, vs in per_layer_values.items()}
    summary = {k: _stats_from_values(_cat_valid(vs)) for k, vs in stats_values.items()}
    flat = {}
    for key, st in summary.items():
        flat[f"{key}_mean"] = st["mean"] if st["mean"] is not None else float("nan")
        flat[f"{key}_median"] = st["median"] if st["median"] is not None else float("nan")
        flat[f"{key}_std"] = st["std"] if st["std"] is not None else float("nan")
        flat[f"{key}_num_tokens"] = st["num_tokens"]
    return {"method_name": method_name, "stats": summary, "per_layer_stats": per_layer_stats, "flat": flat}


def _flatten_layer_activation_samples(samples: Sequence[dict], key: str) -> Optional[torch.Tensor]:
    """Concat per-sample 2D tensors at ``samples[i][key]`` into one (N_total, d)."""
    rows = []
    for s in samples:
        v = s.get(key)
        if not torch.is_tensor(v):
            continue
        if v.ndim == 3:
            v = v.reshape(-1, v.shape[-1])
        elif v.ndim == 2:
            pass
        elif v.ndim == 1:
            v = v.unsqueeze(0)
        else:
            continue
        rows.append(v.to(dtype=torch.float32, device="cpu"))
    if not rows:
        return None
    return torch.cat(rows, dim=0)


def _safe_layer_aggregate(values: Sequence[float]) -> dict:
    arr = np.asarray([float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}


def _tensor_to_jsonable(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return t.detach().to(dtype=torch.float32, device="cpu").tolist()


@torch.no_grad()
def compute_sd_moe_diagnostics(
    *,
    cfg: RunnerConfig,
    recipe_label: str,
    target_set: dict,
    affected_specs: Sequence[dict],
    diagnostic_cache: dict[int, dict],
    baseline_activations: dict,
    method_activations: dict,
    output_dir: Optional[Path],
) -> dict:
    """Run all SD-MoE-style diagnostics for one recipe.

    See ``src/sd_moe_diagnostics.py`` for the underlying math.
    """
    from src import sd_moe_diagnostics as sdm

    layer_indices = sorted(diagnostic_cache.keys())
    if not layer_indices:
        return {"flat": {}, "per_layer": {}, "skipped_reason": "no diagnostic cache"}

    # Resolve dump dir (None disables disk writes; in-memory metrics still returned).
    dump_dir = None
    if output_dir is not None:
        dump_dir = Path(output_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    # Resolve top-k for routing intervention. None -> read from model config (Qwen3 = 8, gpt-oss = 4).
    topk_intervention = cfg.sd_moe_topk_for_intervention
    if topk_intervention is None:
        # Best-effort: read num_experts_per_tok from model config; fallback 8.
        try:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
            topk_intervention = int(getattr(hf_cfg, "num_experts_per_tok", 8))
        except Exception:
            topk_intervention = 8

    head_band = tuple(cfg.sd_moe_head_band)
    tail_band = tuple(cfg.sd_moe_tail_band)
    top1pct_floor = int(cfg.sd_moe_top1pct_floor)
    cdf_k_grid = (1, 2, 4, 8, 16, 32)

    diag_device = torch.device(normalize_torch_device_name(cfg.decompose_device))

    layer_metrics_flat: list[dict] = []
    per_layer_dump: dict[int, dict] = {}

    for layer_idx in tqdm(layer_indices, desc=f"sd_moe_diag {recipe_label}"):
        cache = diagnostic_cache[layer_idx]
        layer_payload: dict = {
            "layer_idx": int(layer_idx),
            "model_arch_key": cfg.model_arch_key,
            "model_name": cfg.model_name,
            "recipe_label": recipe_label,
            "target_set_label": target_set["label"],
            "compression_ratio": cache.get("compression_ratio"),
            "expert_mode": cache.get("expert_mode"),
            "tucker_ranks": {
                "gate": list(cache.get("gate_ranks", ())),
                "up": list(cache.get("up_ranks", ())),
            },
        }
        flat_layer: dict = {"layer_idx": int(layer_idx)}

        # 1. Expert parameter spectral overlap (gate, up).
        param_overlap_payload: dict = {}
        for projection in ("gate", "up"):
            W_base = cache[f"{projection}_base"]
            W_post = cache.get(f"{projection}_post")
            res = sdm.expert_param_overlap(
                W_base,
                W_post,
                top1pct_floor=top1pct_floor,
                head_band=head_band,
                tail_band=tail_band,
                cdf_k_grid=cdf_k_grid,
                device=diag_device,
            )
            proj_payload = {
                "k_top1pct": int(res["k_top1pct"]),
                "head_band_indices": list(res["head_band_indices"]),
                "tail_band_indices": list(res["tail_band_indices"]),
                "cdf_k_grid": list(res["cdf_k_grid"]),
                "head_energy_cdf_base_mean": _tensor_to_jsonable(
                    res["head_energy_cdf_base"].mean(dim=0) if res["head_energy_cdf_base"].numel() else None
                ),
                "head_energy_cdf_post_mean": _tensor_to_jsonable(
                    res["head_energy_cdf_post"].mean(dim=0) if res["head_energy_cdf_post"] is not None else None
                ),
                "pair_top1pct_offdiag_mean_base": sdm.offdiag_mean(res["pair_top1pct_base"]),
                "pair_top1pct_offdiag_mean_post": sdm.offdiag_mean(res["pair_top1pct_post"]),
                "pair_tail_offdiag_mean_base": sdm.offdiag_mean(res["pair_tail_base"]),
                "pair_tail_offdiag_mean_post": sdm.offdiag_mean(res["pair_tail_post"]),
                "drift_per_expert_top1pct_mean": float(res["drift_per_expert_top1pct"].mean().item()) if res["drift_per_expert_top1pct"] is not None else float("nan"),
                "drift_per_expert_tail_mean": float(res["drift_per_expert_tail"].mean().item()) if res["drift_per_expert_tail"] is not None else float("nan"),
                "drift_per_expert_top1pct": _tensor_to_jsonable(res["drift_per_expert_top1pct"]),
                "drift_per_expert_tail": _tensor_to_jsonable(res["drift_per_expert_tail"]),
            }
            param_overlap_payload[projection] = proj_payload
            for stat_key in (
                "pair_top1pct_offdiag_mean_base",
                "pair_top1pct_offdiag_mean_post",
                "pair_tail_offdiag_mean_base",
                "pair_tail_offdiag_mean_post",
                "drift_per_expert_top1pct_mean",
                "drift_per_expert_tail_mean",
            ):
                flat_layer[f"sd_moe_param__{projection}__{stat_key}"] = float(proj_payload[stat_key])
            # Stash V_top sets for downstream analyses (kept on CPU).
            cache[f"{projection}_V_base_head"] = res["V_base_head"]
            cache[f"{projection}_V_post_head"] = res["V_post_head"]
            cache[f"{projection}_V_base_tail"] = res["V_base_tail"]
            cache[f"{projection}_V_post_tail"] = res["V_post_tail"]
        layer_payload["expert_param_overlap"] = param_overlap_payload

        # Activation tensors (router input, router logits) for this layer.
        base_samples = baseline_activations.get(layer_idx, [])
        post_samples = method_activations.get(layer_idx, [])
        X_base = _flatten_layer_activation_samples(base_samples, "mlp_input_post_ln")
        X_post = _flatten_layer_activation_samples(post_samples, "mlp_input_post_ln")
        R_base = _flatten_layer_activation_samples(base_samples, "router_logits")
        R_post = _flatten_layer_activation_samples(post_samples, "router_logits")

        # 2a. Router/gating: parameter-space alignment.
        router_alignment_payload = {}
        E_router = None
        # Find the router's W: try the live model... but we don't have model here.
        # Instead, use baseline router weights inferred from R_base (logits) -> approximate via least-squares?
        # Simpler: capture router weights inside collect (we do not). For now skip parameter row alignment unless we
        # have access to W_router - we will pass it separately via cache.
        W_router = cache.get("W_router")
        if W_router is not None and W_router.numel() > 0:
            E_router = int(W_router.shape[0])
            for projection in ("gate", "up"):
                V_base_head = cache[f"{projection}_V_base_head"]
                V_post_head = cache[f"{projection}_V_post_head"]
                # Pad if number of expert subspaces doesn't match router dim.
                if len(V_base_head) == E_router:
                    align_base = sdm.router_param_alignment(W_router, V_base_head, device=diag_device)
                    align_post = sdm.router_param_alignment(W_router, V_post_head, device=diag_device) if V_post_head is not None else None
                    router_alignment_payload[f"router_vs_expert_{projection}_base"] = _tensor_to_jsonable(align_base)
                    router_alignment_payload[f"router_vs_expert_{projection}_post"] = _tensor_to_jsonable(align_post)
                    flat_layer[f"sd_moe_router__router_vs_expert_{projection}_base_mean"] = float(align_base.mean().item())
                    if align_post is not None:
                        flat_layer[f"sd_moe_router__router_vs_expert_{projection}_post_mean"] = float(align_post.mean().item())
        # Activation top-1% subspace (used for router-vs-activation alignment + Analysis 4).
        U_act_base_head = U_act_post_head = None
        if X_base is not None:
            d_in = X_base.shape[1]
            k_top_act = sdm.percentile_rank(d_in, head_band[1], floor=top1pct_floor)
            U_act_base_head, _ = sdm.activation_top_subspace(X_base, rank=k_top_act, device=diag_device)
            if W_router is not None:
                align_base_act = sdm.router_vs_activation_alignment(W_router, U_act_base_head, device=diag_device)
                router_alignment_payload["router_vs_activation_base"] = _tensor_to_jsonable(align_base_act)
                flat_layer["sd_moe_router__router_vs_activation_base_mean"] = float(align_base_act.mean().item())
        if X_post is not None:
            d_in_p = X_post.shape[1]
            k_top_act_p = sdm.percentile_rank(d_in_p, head_band[1], floor=top1pct_floor)
            U_act_post_head, _ = sdm.activation_top_subspace(X_post, rank=k_top_act_p, device=diag_device)
            if W_router is not None:
                align_post_act = sdm.router_vs_activation_alignment(W_router, U_act_post_head, device=diag_device)
                router_alignment_payload["router_vs_activation_post"] = _tensor_to_jsonable(align_post_act)
                flat_layer["sd_moe_router__router_vs_activation_post_mean"] = float(align_post_act.mean().item())
        layer_payload["router_param_alignment"] = router_alignment_payload

        # 2b. Routing intervention (top-k agreement, KL/JSD, entropy, expert counts).
        intervention_payload: dict = {"strict_sd_moe": False, "complementary_intervention": True,
                                       "reason": "TD-MoE does not directly modify router weights; behavioral drift comes via expert weight changes."}
        if R_base is not None and R_post is not None and R_base.shape == R_post.shape:
            E_logits = int(R_base.shape[1])
            counts_base = sdm.expert_token_counts(R_base, topk_intervention)
            counts_post = sdm.expert_token_counts(R_post, topk_intervention)
            jsd_out = sdm.jsd_softmax(R_base, R_post)
            ks = {"k1": 1, "k_official": int(topk_intervention), "k_2x": int(2 * topk_intervention)}
            agreement_per_k = {}
            for tag, k_value in ks.items():
                if 1 <= k_value <= E_logits:
                    agreement_per_k[tag] = sdm.topk_set_agreement(R_base, R_post, k=k_value)
                    flat_layer[f"sd_moe_router__topk_agreement_{tag}_jaccard_mean"] = float(agreement_per_k[tag]["mean_jaccard"])
                    flat_layer[f"sd_moe_router__topk_agreement_{tag}_exact_mean"] = float(agreement_per_k[tag]["mean_exact_match"])
            intervention_payload.update({
                "topk_agreement": agreement_per_k,
                "expert_token_count_base": counts_base.tolist(),
                "expert_token_count_post": counts_post.tolist(),
                "expert_assignment_kl_base_post": sdm.kl_divergence(counts_base.float(), counts_post.float()),
                "expert_assignment_kl_post_base": sdm.kl_divergence(counts_post.float(), counts_base.float()),
                "routing_entropy_base_mean": float(jsd_out["entropy_base_mean"]),
                "routing_entropy_post_mean": float(jsd_out["entropy_post_mean"]),
                "routing_entropy_delta_mean": float(jsd_out["entropy_post_mean"] - jsd_out["entropy_base_mean"]),
                "softmax_jsd_mean": float(jsd_out["jsd_mean"]),
                "num_tokens": int(R_base.shape[0]),
            })
            flat_layer["sd_moe_router__expert_assignment_kl_base_post"] = float(intervention_payload["expert_assignment_kl_base_post"])
            flat_layer["sd_moe_router__softmax_jsd_mean"] = float(jsd_out["jsd_mean"])
            flat_layer["sd_moe_router__routing_entropy_delta_mean"] = float(intervention_payload["routing_entropy_delta_mean"])
        layer_payload["router_intervention"] = intervention_payload

        # 3. Activation projection onto expert spectral directions.
        projection_payload: dict = {}
        for projection in ("gate", "up"):
            V_base_head = cache[f"{projection}_V_base_head"]
            V_post_head = cache[f"{projection}_V_post_head"]
            V_base_tail = cache[f"{projection}_V_base_tail"]
            V_post_tail = cache[f"{projection}_V_post_tail"]
            energies_base_head = []
            energies_post_head = []
            energies_base_tail = []
            energies_post_tail = []
            rates_base_head = []
            rates_post_head = []
            taus_base_head = []
            taus_post_head = []
            E_local = len(V_base_head)
            if X_base is not None:
                for e in range(E_local):
                    energies_base_head.append(sdm.activation_projection_energy(X_base, V_base_head[e], device=diag_device))
                    energies_base_tail.append(sdm.activation_projection_energy(X_base, V_base_tail[e], device=diag_device))
                    rate_b, tau_b = sdm.activation_rate_top1pct(X_base, V_base_head[e], device=diag_device)
                    rates_base_head.append(rate_b)
                    taus_base_head.append(tau_b)
            if X_post is not None and V_post_head is not None:
                for e in range(E_local):
                    energies_post_head.append(sdm.activation_projection_energy(X_post, V_post_head[e], device=diag_device))
                    energies_post_tail.append(sdm.activation_projection_energy(X_post, V_post_tail[e], device=diag_device))
                    rate_p, tau_p = sdm.activation_rate_top1pct(X_post, V_post_head[e], device=diag_device)
                    rates_post_head.append(rate_p)
                    taus_post_head.append(tau_p)
            proj_payload = {
                "top1pct_proj_energy_base": energies_base_head,
                "top1pct_proj_energy_post": energies_post_head,
                "tail_proj_energy_base": energies_base_tail,
                "tail_proj_energy_post": energies_post_tail,
                "top1pct_activation_rate_base": rates_base_head,
                "top1pct_activation_rate_post": rates_post_head,
                "top1pct_activation_tau_base": taus_base_head,
                "top1pct_activation_tau_post": taus_post_head,
            }
            projection_payload[projection] = proj_payload
            for tag, vals in (
                ("top1pct_proj_energy_base", energies_base_head),
                ("top1pct_proj_energy_post", energies_post_head),
                ("tail_proj_energy_base", energies_base_tail),
                ("tail_proj_energy_post", energies_post_tail),
                ("top1pct_activation_rate_base", rates_base_head),
                ("top1pct_activation_rate_post", rates_post_head),
            ):
                flat_layer[f"sd_moe_actproj__{projection}__{tag}_mean"] = _safe_layer_aggregate(vals)["mean"]
        layer_payload["activation_projection"] = projection_payload

        # 4. Shared activation subspace + comparison to TD-MoE U_in.
        subspace_payload: dict = {}
        for projection in ("gate", "up"):
            ranks = cache.get(f"{projection}_ranks") or (0, 0, 0)
            r3 = int(ranks[2]) if len(ranks) >= 3 else 0
            U_in = (cache.get(f"{projection}_factors") or {}).get("U_in")
            for regime, X_reg in (("base", X_base), ("post", X_post)):
                if X_reg is None or r3 < 1:
                    subspace_payload[f"act_top_r3_vs_Uin_{projection}_{regime}"] = float("nan")
                    continue
                metrics = sdm.shared_activation_subspace_metrics(
                    X_reg,
                    r3_gate=r3 if projection == "gate" else 1,
                    r3_up=r3 if projection == "up" else 1,
                    Uin_gate=U_in if projection == "gate" else None,
                    Uin_up=U_in if projection == "up" else None,
                    cross_sample_top1pct_rank=sdm.percentile_rank(X_reg.shape[1], head_band[1], floor=top1pct_floor),
                    device=diag_device,
                )
                key = f"act_top_r3_vs_Uin_{projection}_{regime}"
                subspace_payload[key] = float(metrics[f"act_top_r3_vs_Uin_{projection}"])
                flat_layer[f"sd_moe_subspace__{key}"] = subspace_payload[key]
                if projection == "gate":
                    cs_key = f"cross_sample_top1pct_similarity_{regime}"
                    if cs_key not in subspace_payload:
                        subspace_payload[cs_key] = float(metrics["cross_sample_top1pct_similarity"])
                        flat_layer[f"sd_moe_subspace__{cs_key}"] = subspace_payload[cs_key]
        subspace_payload["r3_gate"] = int(cache.get("gate_ranks", (0, 0, 0))[2]) if cache.get("gate_ranks") else 0
        subspace_payload["r3_up"] = int(cache.get("up_ranks", (0, 0, 0))[2]) if cache.get("up_ranks") else 0
        layer_payload["shared_activation_subspace"] = subspace_payload

        layer_payload["gradient_spectral_alignment"] = {
            "status": "not_implemented",
            "reason": "post-training, no backward pass; calibration-backward variant deferred",
        }

        # Free V_top caches that we no longer need.
        for projection in ("gate", "up"):
            for k in (f"{projection}_V_base_head", f"{projection}_V_post_head", f"{projection}_V_base_tail", f"{projection}_V_post_tail"):
                cache.pop(k, None)
        clean_mem()

        per_layer_dump[int(layer_idx)] = layer_payload
        layer_metrics_flat.append(flat_layer)

        if dump_dir is not None:
            slug_recipe = re.sub(r"[^A-Za-z0-9._-]+", "_", recipe_label)
            slug_target = re.sub(r"[^A-Za-z0-9._-]+", "_", target_set["label"])
            run_dir = dump_dir / f"{slug_recipe}__{slug_target}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(run_dir / f"layer_{int(layer_idx):04d}.json", "w") as f:
                    json.dump(layer_payload, f, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, (np.integer,)) else str(o)))
            except Exception as exc:
                warnings.warn(f"sd_moe_diagnostics: failed to dump layer {layer_idx}: {exc!r}", RuntimeWarning)

    # Aggregate per-layer flats into run-level scalars for the row.
    flat_summary: dict = {}
    if layer_metrics_flat:
        keys: set[str] = set()
        for fl in layer_metrics_flat:
            keys.update(k for k in fl.keys() if k != "layer_idx")
        for key in sorted(keys):
            vals = [fl.get(key) for fl in layer_metrics_flat if key in fl]
            agg = _safe_layer_aggregate(vals)
            flat_summary[f"{key}__layers_mean"] = agg["mean"]
            flat_summary[f"{key}__layers_min"] = agg["min"]
            flat_summary[f"{key}__layers_max"] = agg["max"]

    return {
        "flat": flat_summary,
        "per_layer": per_layer_dump,
        "per_layer_flat": layer_metrics_flat,
        "topk_intervention": int(topk_intervention),
    }


# Main experiment runner

def _norm_resume_value(x):
    try:
        if x is None or pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def recipe_resume_key(model_name: str, target_set: dict, recipe: RecipeSpec, rank: int):
    return (model_name, target_set["label"], recipe.label, int(rank), json.dumps([asdict(p) for p in recipe.plans], sort_keys=True), bool(recipe.with_lora), recipe.lora_quant_bits)



def baseline_resume_key(model_name: str, target_set: Optional[dict] = None):
    return (model_name, "__model_baseline__", "baseline")


def completed_keys_from_results(results: list[dict]):
    keys = set()
    for row in results:
        if row.get("method_label") == "baseline":
            if row.get("resume_key") is not None:
                keys.add(tuple(row["resume_key"]))
            else:
                keys.add((row.get("model_name"), "__model_baseline__", "baseline"))
        elif row.get("resume_key") is not None:
            keys.add(tuple(row["resume_key"]))
    return keys


def run_benchmarks_for_model(model, tokenizer, cfg: RunnerConfig) -> dict:
    benchmark_model, compile_stats = maybe_compile_model_for_benchmarks(model, tokenizer, cfg)

    metrics = {
        "ppl_requested": bool(cfg.run_ppl),
        "lm_eval_requested": bool(cfg.run_lm_eval),
        "hotpot_requested": bool(cfg.run_hotpot),
        "generation_examples_requested": bool(cfg.run_generation_examples),
    }
    metrics.update(compile_stats)

    with torch.inference_mode():
        metrics.update(evaluate_ppl_datasets(benchmark_model, tokenizer, cfg))
        metrics.update(evaluate_lm_eval_harness(benchmark_model, tokenizer, cfg))
        metrics.update(evaluate_hotpot_tensorllm_style(benchmark_model, tokenizer, cfg))

        gen_examples = generate_examples_for_prompts(benchmark_model, tokenizer, cfg)

    if gen_examples is not None:
        metrics["generation_examples"] = gen_examples
        first = cfg.generation_prompts[0]
        metrics["generation_preview_prompt"] = first
        metrics["generation_preview_text"] = gen_examples[first]

    return metrics


@torch.no_grad()
def run_model_baseline_benchmark_metrics(cfg: RunnerConfig) -> dict:
    model, tokenizer = load_model_and_tokenizer(cfg)
    try:
        return run_benchmarks_for_model(model, tokenizer, cfg)
    finally:
        del model, tokenizer
        clean_mem()


@torch.no_grad()
def run_model_baseline_row(
    cfg: RunnerConfig,
    *,
    baseline_benchmark_metrics: Optional[dict] = None,
) -> dict:
    model, tokenizer = load_model_and_tokenizer(cfg)
    try:
        total_dense = get_model_dense_bits(model, cfg, include_embeddings=True)
        noemb_dense = get_model_dense_bits(model, cfg, include_embeddings=False)

        row = {
            "target_set": "__model_baseline__",
            "model_arch_key": cfg.model_arch_key,
            "model_family": cfg.model_spec.family,
            "model_name": cfg.model_name,
            "method_label": "baseline",
            "rank": None,
            "target_modules": [],
            "target_layers": [],
            "storage_records": [],
            "affected_modules_dense_bits": 0,
            "affected_modules_storage_bits": 0,
            "affected_modules_compression_ratio": float("nan"),
            "mha_in_affected_blocks_dense_bits": 0,
            "mha_in_affected_blocks_storage_bits": 0,
            "mha_in_affected_blocks_compression_ratio": float("nan"),
            "mlp_in_affected_blocks_dense_bits": 0,
            "mlp_in_affected_blocks_storage_bits": 0,
            "mlp_in_affected_blocks_compression_ratio": float("nan"),
            "affected_blocks_dense_bits": 0,
            "affected_blocks_storage_bits": 0,
            "affected_blocks_compression_ratio": float("nan"),
            "overall_no_embeddings_dense_bits": int(noemb_dense),
            "overall_no_embeddings_storage_bits": int(noemb_dense),
            "overall_no_embeddings_compression_ratio": 1.0,
            "overall_with_embeddings_dense_bits": int(total_dense),
            "overall_with_embeddings_storage_bits": int(total_dense),
            "overall_with_embeddings_compression_ratio": 1.0,
            "target_layer_bits": 0,
            "overall_model_bits": int(total_dense),
            "target_layer_compression_ratio": float("nan"),
            "overall_model_compression_ratio": 1.0,
            "superweight_abs_error_mean": float("nan"),
            "superweight_abs_error_max": float("nan"),
            "superweight_rel_error_mean": float("nan"),
            "superweight_rel_error_max": float("nan"),
            "superweight_error_detail": [],
            "target_layer_fro_rel_error": float("nan"),
            "target_layer_max_abs_error": float("nan"),
            "target_layer_error_detail": [],
            "mean_top_outlier_rel_error": float("nan"),
            "mean_random_rel_error": float("nan"),
            "decomposition_apply_time_s": 0.0,
            "quantization_apply_time_s": 0.0,
            "baseline_benchmarks_cached_per_model": baseline_benchmark_metrics is not None,
        }

        if baseline_benchmark_metrics is not None:
            row.update(baseline_benchmark_metrics)
        else:
            row.update(run_benchmarks_for_model(model, tokenizer, cfg))

        row["resume_key"] = list(baseline_resume_key(cfg.model_name))
        return row
    finally:
        del model, tokenizer
        clean_mem()


def collect_baseline_storage_context(model: nn.Module, target_set: dict, affected_specs: Sequence[dict], cfg: RunnerConfig) -> dict:
    model_spec = cfg.model_spec
    group_specs = {group: enumerate_group_specs(target_set, model_spec, group) for group in model_spec.module_groups}
    dense_bits_by_full_name = {}
    all_group_specs = [spec for specs in group_specs.values() for spec in specs]
    for spec in list(affected_specs) + all_group_specs:
        full_name = spec["full_name"]
        if full_name not in dense_bits_by_full_name:
            dense_bits_by_full_name[full_name] = dense_bits_for_target_module(get_module_by_name(model, full_name), cfg)

    affected_dense_bits = int(sum(dense_bits_by_full_name[s["full_name"]] for s in affected_specs))
    group_dense = {
        group: int(sum(dense_bits_by_full_name[s["full_name"]] for s in specs))
        for group, specs in group_specs.items()
    }
    block_dense = block_dense_bits(model, model_spec, cfg, target_set["layer_indices"])
    total_dense = get_model_dense_bits(model, cfg, include_embeddings=True)
    noemb_dense = get_model_dense_bits(model, cfg, include_embeddings=False)

    return {
        "group_specs": group_specs,
        "dense_bits_by_full_name": dense_bits_by_full_name,
        "affected_dense_bits": int(affected_dense_bits),
        "group_dense": group_dense,
        "block_dense": int(block_dense),
        "total_dense": int(total_dense),
        "noemb_dense": int(noemb_dense),
    }


def compute_storage_summary_from_baseline_context(model: nn.Module, target_set: dict, affected_specs: Sequence[dict], cfg: RunnerConfig, baseline_context: dict) -> dict:
    affected_full = {s["full_name"] for s in affected_specs}
    affected_dense_bits = int(baseline_context["affected_dense_bits"])
    affected_storage_bits = int(sum(target_module_storage_bits(get_module_by_name(model, s["full_name"]), cfg) for s in affected_specs))
    dense_bits_by_full_name = baseline_context["dense_bits_by_full_name"]

    def group_bits(group: str):
        specs = baseline_context["group_specs"][group]
        dense = int(sum(dense_bits_by_full_name[s["full_name"]] for s in specs))
        storage = 0
        for s in specs:
            if s["full_name"] in affected_full:
                storage += target_module_storage_bits(get_module_by_name(model, s["full_name"]), cfg)
            else:
                storage += int(dense_bits_by_full_name[s["full_name"]])
        return int(dense), int(storage)

    group_storage = {group: group_bits(group) for group in baseline_context["group_specs"]}
    block_dense = int(baseline_context["block_dense"])
    block_storage = int(block_dense - affected_dense_bits + affected_storage_bits)
    total_dense = int(baseline_context["total_dense"])
    total_storage = int(total_dense - affected_dense_bits + affected_storage_bits)
    noemb_dense = int(baseline_context["noemb_dense"])
    noemb_storage = int(noemb_dense - affected_dense_bits + affected_storage_bits)
    records = current_storage_records(model, affected_specs, cfg)

    summary = {
        "storage_records": records,
        "affected_modules_dense_bits": int(affected_dense_bits),
        "affected_modules_storage_bits": int(affected_storage_bits),
        "affected_modules_compression_ratio": float(affected_dense_bits / max(affected_storage_bits, 1)),
        "affected_blocks_dense_bits": int(block_dense),
        "affected_blocks_storage_bits": int(block_storage),
        "affected_blocks_compression_ratio": float(block_dense / max(block_storage, 1)),
        "overall_no_embeddings_dense_bits": int(noemb_dense),
        "overall_no_embeddings_storage_bits": int(noemb_storage),
        "overall_no_embeddings_compression_ratio": float(noemb_dense / max(noemb_storage, 1)),
        "overall_with_embeddings_dense_bits": int(total_dense),
        "overall_with_embeddings_storage_bits": int(total_storage),
        "overall_with_embeddings_compression_ratio": float(total_dense / max(total_storage, 1)),
        "target_layer_bits": int(affected_storage_bits),
        "overall_model_bits": int(total_storage),
        "target_layer_compression_ratio": float(affected_dense_bits / max(affected_storage_bits, 1)),
        "overall_model_compression_ratio": float(total_dense / max(total_storage, 1)),
    }
    for group, (dense, storage) in group_storage.items():
        prefix = f"{group}_in_affected_blocks"
        summary[f"{prefix}_dense_bits"] = int(dense)
        summary[f"{prefix}_storage_bits"] = int(storage)
        summary[f"{prefix}_compression_ratio"] = float(dense / max(storage, 1))
    return summary


def _safe_filename_component(text_value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text_value)).strip("_") or "model"


def _maybe_save_partial_checkpoint(cfg: RunnerConfig, results: list[dict], cfgs: list[RunnerConfig], target_sets: list[dict], recipes: list[RecipeSpec], status_rows: list[dict]):
    if not getattr(cfg, "save_partial_results", True):
        return
    try:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_filename_component(cfg.model_name)}__{getattr(cfg, 'partial_results_name', 'partial_results')}.json"
        status_df = pd.DataFrame(status_rows)
        save_results(
            output_path,
            results_df=pd.DataFrame(results),
            results=results,
            cfgs=cfgs,
            target_sets=target_sets,
            recipes=recipes,
            status_df=status_df,
        )
        status_df.to_csv(output_path.with_name(output_path.stem + "__status.csv"), index=False)
    except Exception as exc:
        warnings.warn(f"Could not save partial checkpoint: {exc!r}", RuntimeWarning)


def stage_row(
    model,
    tokenizer,
    baseline_model,
    target_set,
    recipe_label,
    rank,
    affected_specs,
    all_candidate_specs,
    cfg,
    *,
    baseline_target_weights,
    method_stats,
    storage_summary,
    geometry_stats=None,
    sd_moe_diagnostics=None,
    run_benchmarks: bool = True,
    benchmark_metrics: Optional[dict] = None,
):
    current_weights = get_effective_target_weights(model, affected_specs)
    sw_stats = summarize_superweight_error_multi(current_weights, baseline_target_weights, affected_specs)
    layer_stats = summarize_layer_error_multi(current_weights, baseline_target_weights)
    cloud = sample_dense_entry_error_summary_multi(
        current_weights,
        baseline_target_weights,
        affected_specs,
        topk_outliers=cfg.topk_outliers,
        random_samples=cfg.random_error_samples,
        seed=0,
    )

    def mean_for(tag, col, min_original_abs=None):
        if len(cloud) == 0 or col not in cloud:
            return float("nan")
        sub = cloud[cloud["tag"] == tag]
        if min_original_abs is not None:
            sub = sub[sub["original_abs"] >= min_original_abs]
        return float(sub[col].mean()) if len(sub) else float("nan")

    row = {
        "target_set": target_set["label"],
        "model_arch_key": cfg.model_arch_key,
        "model_family": cfg.model_spec.family,
        "model_name": cfg.model_name,
        "method_label": recipe_label,
        "rank": int(rank) if rank is not None else None,
        "target_modules": [s["full_name"] for s in affected_specs],
        "target_layers": [int(x) for x in target_set["layer_indices"]],
        **storage_summary,
        **sw_stats,
        **layer_stats,
        "mean_top_outlier_rel_error": mean_for("top_outlier", "rel_error"),
        "mean_random_rel_error": mean_for("random", "rel_error", min_original_abs=1e-3),
    }

    row.update(method_stats)

    if benchmark_metrics is not None:
        row.update(benchmark_metrics)
    elif run_benchmarks:
        row.update(run_benchmarks_for_model(model, tokenizer, cfg))

    if geometry_stats is not None:
        row["activation_geometry_stats"] = geometry_stats.get("stats")
        row["activation_geometry_per_layer_stats"] = geometry_stats.get("per_layer_stats")
        row.update(geometry_stats.get("flat", {}))

    if sd_moe_diagnostics is not None:
        row["sd_moe_diagnostics_per_layer"] = sd_moe_diagnostics.get("per_layer_flat")
        row["sd_moe_diagnostics_topk_intervention"] = sd_moe_diagnostics.get("topk_intervention")
        row.update(sd_moe_diagnostics.get("flat", {}))

    return row


def run_recipe_for_target_set(target_set: dict, recipe: RecipeSpec, rank: int, cfg: RunnerConfig, *, seed: int = 0):
    model, tokenizer = load_model_and_tokenizer(cfg)
    try:
        all_candidate_specs = make_target_specs(target_set, cfg.model_spec)
        affected_specs_all = make_target_specs(target_set, cfg.model_spec, selectors=recipe_selectors(recipe))

        baseline_target_weights = get_effective_target_weights(model, affected_specs_all)
        baseline_storage_context = collect_baseline_storage_context(model, target_set, affected_specs_all, cfg)

        need_activations = bool(cfg.run_activation_geometry or cfg.run_sd_moe_diagnostics)
        baseline_activations = None
        if need_activations:
            baseline_activations = collect_layer_activations(model, tokenizer, cfg, layer_indices=unique_layer_indices(affected_specs_all))

        method_stats = {"recipe": {"label": recipe.label, "plans": [asdict(p) for p in recipe.plans], "with_lora": recipe.with_lora}}
        applied_specs = []
        sd_moe_diag_cache: dict[int, dict] = {}
        for plan in recipe.plans:
            specs, plan_stats = apply_decomposition_plan(model, target_set, plan, cfg, global_rank=rank)
            qstats = apply_quantization_plan(model, tokenizer, specs, plan, cfg, seed=seed)
            plan_stats.update(qstats)
            cache = plan_stats.pop("sd_moe_diagnostic_cache", None)
            if cache:
                sd_moe_diag_cache.update(cache)
            method_stats.setdefault("plan_stats", []).append(plan_stats)
            for s in specs:
                if s["full_name"] not in {x["full_name"] for x in applied_specs}:
                    applied_specs.append(s)

        loss_history = None
        if recipe.with_lora:
            lora_selectors = recipe.lora_targets if recipe.lora_targets is not None else recipe_selectors(recipe)
            lora_specs = make_target_specs(target_set, cfg.model_spec, selectors=lora_selectors)
            attach_lora_to_specs(model, lora_specs, cfg, recipe)
            print(f"[{target_set['label']} | {recipe.label}, rank={rank}] trainable parameters:", count_trainable_parameters(model))
            loss_history = run_lora_training(model, tokenizer, cfg, seed=seed)
            method_stats.update(quantize_lora_modules(model, lora_specs, cfg, recipe))
            for s in lora_specs:
                if s["full_name"] not in {x["full_name"] for x in applied_specs}:
                    applied_specs.append(s)

        storage_summary = compute_storage_summary_from_baseline_context(model, target_set, affected_specs_all, cfg, baseline_storage_context)

        if recipe.benchmark_with_dense_reconstruction:
            dense_reconstruction_records = reconstruct_specs_to_dense_for_benchmark(
                model,
                affected_specs_all,
                cfg,
            )
            method_stats["benchmarked_with_dense_reconstruction"] = True
            method_stats["dense_reconstruction_records"] = dense_reconstruction_records
        else:
            method_stats["benchmarked_with_dense_reconstruction"] = False

        geometry_stats = None
        sd_moe_diagnostics_stats = None
        method_activations = None
        if need_activations:
            method_activations = collect_layer_activations(model, tokenizer, cfg, layer_indices=unique_layer_indices(affected_specs_all))
        if cfg.run_activation_geometry and method_activations is not None:
            geometry_stats = measure_activation_geometry(baseline_activations, method_activations, method_name=recipe.label)
        if cfg.run_sd_moe_diagnostics and method_activations is not None and sd_moe_diag_cache:
            dump_dir = (
                Path(cfg.sd_moe_per_layer_dump_dir)
                if cfg.sd_moe_per_layer_dump_dir
                else (Path(cfg.output_dir) / "sd_moe_diagnostics" if cfg.output_dir else None)
            )
            sd_moe_diagnostics_stats = compute_sd_moe_diagnostics(
                cfg=cfg,
                recipe_label=recipe.label,
                target_set=target_set,
                affected_specs=affected_specs_all,
                diagnostic_cache=sd_moe_diag_cache,
                baseline_activations=baseline_activations,
                method_activations=method_activations,
                output_dir=dump_dir,
            )

        row = stage_row(
            model,
            tokenizer,
            None,
            target_set,
            recipe.label,
            rank,
            affected_specs_all,
            all_candidate_specs,
            cfg,
            baseline_target_weights=baseline_target_weights,
            method_stats=method_stats,
            storage_summary=storage_summary,
            geometry_stats=geometry_stats,
            sd_moe_diagnostics=sd_moe_diagnostics_stats,
            run_benchmarks=False,
        )

        del baseline_target_weights
        if baseline_activations is not None:
            del baseline_activations
        if method_activations is not None:
            del method_activations
        sd_moe_diag_cache.clear()

        gc.collect()
        clean_mem()

        row.update(run_benchmarks_for_model(model, tokenizer, cfg))
        row["resume_key"] = list(recipe_resume_key(cfg.model_name, target_set, recipe, rank))
        return row, loss_history
    finally:
        del model, tokenizer
        clean_mem()


def run_experiment_grid(
    target_sets: list[dict],
    recipes: list[RecipeSpec],
    ranks: list[int],
    cfgs: list[RunnerConfig],
    *,
    results: Optional[list[dict]] = None,
    loss_histories: Optional[dict] = None,
):
    results = [] if results is None else results
    loss_histories = {} if loss_histories is None else loss_histories
    completed = completed_keys_from_results(results)
    status_rows = []

    total = len(cfgs) * (1 + len(target_sets) * len(ranks) * len(recipes))
    pb = tqdm(total=total, desc="Experiment grid")

    for cfg in cfgs:
        model_baseline_key = baseline_resume_key(cfg.model_name)

        baseline_status = {
            "model_name": cfg.model_name,
            "target_set": "__model_baseline__",
            "method_label": "baseline",
            "status": None,
            "error_type": None,
            "error_message": None,
        }
        try:
            if model_baseline_key in completed:
                baseline_status["status"] = "already_finished"
            else:
                row = run_model_baseline_row(cfg)
                results.append(row)
                completed.add(tuple(row["resume_key"]))
                baseline_status["status"] = "completed_now"
        except KeyboardInterrupt:
            baseline_status["status"] = "interrupted"
            baseline_status["error_type"] = "KeyboardInterrupt"
            baseline_status["error_message"] = "Interrupted by user"
            raise
        except Exception as exc:
            baseline_status["status"] = "failed"
            baseline_status.update(exception_to_status_fields(exc))
        finally:
            status_rows.append(baseline_status)
            clean_mem()
            pb.update(1)
            _maybe_save_partial_checkpoint(cfg, results, [cfg], target_sets, recipes, status_rows)

        for target_set in target_sets:
            for rank in ranks:
                for idx, recipe in enumerate(recipes):
                    key = recipe_resume_key(cfg.model_name, target_set, recipe, int(rank))
                    status = {
                        "model_name": cfg.model_name,
                        "target_set": target_set["label"],
                        "method_label": recipe.label,
                        "rank": int(rank),
                        "status": None,
                        "error_type": None,
                        "error_message": None,
                    }

                    try:
                        if key in completed:
                            status["status"] = "already_finished"
                        else:
                            pb.set_description(f"{cfg.model_arch_key}:{target_set['label']} | {recipe.label} | rank={rank}")
                            seed = 100000 * (idx + 1) + 1000 * int(rank) + abs(hash((target_set["label"], cfg.model_name))) % 997
                            row, hist = run_recipe_for_target_set(target_set, recipe, int(rank), cfg, seed=seed)
                            results.append(row)
                            completed.add(tuple(row["resume_key"]))

                            if hist is not None:
                                loss_histories[(cfg.model_name, target_set["label"], recipe.label, int(rank))] = hist

                            status["status"] = "completed_now"

                    except KeyboardInterrupt:
                        status["status"] = "interrupted"
                        status["error_type"] = "KeyboardInterrupt"
                        status["error_message"] = "Interrupted by user"
                        raise
                    except Exception as exc:
                        status["status"] = "failed"
                        status.update(exception_to_status_fields(exc))

                    finally:
                        status_rows.append(status)
                        clean_mem()
                        pb.update(1)
                        _maybe_save_partial_checkpoint(cfg, results, [cfg], target_sets, recipes, status_rows)

    pb.close()
    return results, loss_histories, pd.DataFrame(status_rows)

# Experiment grid helpers

def middle_expanding_layer_sets(num_layers: int, *, label_prefix: str = "middle", max_fraction: float = 0.25) -> list[dict]:
    """e.g. 16, then 16+17, ..., until quarter of all layers are included."""
    start = int(num_layers // 2)
    max_count = max(1, int(math.ceil(num_layers * max_fraction)))
    out = []
    for count in range(1, max_count + 1):
        layers = list(range(start, min(start + count, num_layers)))
        if len(layers) < count:
            break
        out.append({"label": f"{label_prefix}_layers_{layers[0]}_{layers[-1]}", "layer_indices": layers, "groups": ["mha", "mlp"], "known_superweights": {}})
    return out


def middle_out_expanding_layer_sets(
    num_layers: int,
    *,
    label_prefix: str = "middle_out",
    center_layer: Optional[int] = None,
    right_warmup_fraction: float = 0.25,
    max_fraction: float = 0.5,
    groups: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Middle-out ranges: c, c-c+1, ..., then alternate left/right expansion."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    center = int(num_layers // 2 if center_layer is None else center_layer)
    if not 0 <= center < num_layers:
        raise ValueError(f"center_layer must be in [0, {num_layers}), got {center}")

    max_count = max(1, min(num_layers, int(math.ceil(num_layers * max_fraction))))
    right_warmup_count = max(1, int(math.ceil(num_layers * right_warmup_fraction)))
    right_warmup_limit = min(num_layers - 1, center + right_warmup_count - 1)
    selected_groups = list(groups) if groups is not None else ["mha", "mlp"]

    out = []
    left = right = center
    expand_left_next = True

    while len(out) < max_count:
        layers = list(range(left, right + 1))
        out.append({
            "label": f"{label_prefix}_layers_{left}_{right}",
            "layer_indices": layers,
            "groups": selected_groups,
            "known_superweights": {},
        })

        if len(layers) >= max_count or (left == 0 and right == num_layers - 1):
            break

        if right < right_warmup_limit:
            right += 1
            continue

        if expand_left_next and left > 0:
            left -= 1
            expand_left_next = False
        elif right < num_layers - 1:
            right += 1
            expand_left_next = True
        elif left > 0:
            left -= 1
            expand_left_next = False
        else:
            break

    return out


def get_num_layers_from_config(model_name: str, *, trust_remote_code=True) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError(f"Could not infer number of layers from config for {model_name}")


def default_recipes(*, bitwidth: int = 4, lora_quant_bits: Optional[int] = 4, benchmark_with_dense_reconstruction: bool = True) -> list[RecipeSpec]:
    return [
        RecipeSpec(label="tucker_mha", plans=[ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none")], with_lora=False, benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction),
        RecipeSpec(label="tucker_mha__tt_mlp", plans=[ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none"), ModulePlan("mlp", "tt", quant_method="none")], with_lora=False, benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction),
        RecipeSpec(label="tt_mlp", plans=[ModulePlan("mlp", "tt", quant_method="none")], with_lora=False, benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction),
        RecipeSpec(label="tt_all", plans=[ModulePlan("all", "tt", quant_method="none")], with_lora=False, benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction),
    ]

def load_results_checkpoint(path: str | Path) -> tuple[list[dict], pd.DataFrame, dict]:
    path = Path(path)
    with open(path, "r") as f:
        payload = json.load(f)

    results = payload.get("results", []) or []
    status = payload.get("status", []) or []

    results_df = pd.DataFrame(results)
    if len(results_df) and "resume_key" in results_df.columns:
        results_df["_resume_key_str"] = results_df["resume_key"].apply(
            lambda x: json.dumps(x, sort_keys=True, default=str)
        )
        results_df = (
            results_df
            .drop_duplicates("_resume_key_str", keep="last")
            .drop(columns=["_resume_key_str"])
        )
        results = results_df.to_dict(orient="records")

    return results, pd.DataFrame(status), payload

def save_results(output_path: str | Path, *, results_df: pd.DataFrame, results: list[dict], cfgs: list[RunnerConfig], target_sets: list[dict], recipes: list[RecipeSpec], status_df: Optional[pd.DataFrame] = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configs": [asdict(c) | {"model_dtype": dtype_name(c.model_dtype), "decompose_dtype": dtype_name(c.decompose_dtype), "output_dir": str(c.output_dir)} for c in cfgs],
        "target_sets": target_sets,
        "recipes": [asdict(r) for r in recipes],
        "results": results_df.to_dict(orient="records"),
        "status": [] if status_df is None else status_df.to_dict(orient="records"),
    }
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.dtype):
            return dtype_name(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    csv_path = output_path.with_suffix(".csv")
    results_df.to_csv(csv_path, index=False)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    return output_path, csv_path
