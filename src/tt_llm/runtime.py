from __future__ import annotations

import gc
from typing import Any

import torch


def cleanup_memory(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def format_cuda_memory() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"

    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    return f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak={peak:.2f} GB"
