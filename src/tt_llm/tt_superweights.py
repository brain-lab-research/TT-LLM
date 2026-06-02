from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .tt_linear import TTLinear


@dataclass
class TTEntryTrace:
    row: int
    col: int
    row_multi_index: List[int]
    col_multi_index: List[int]
    value: float
    slice_shapes: List[tuple[int, int]]
    slice_max_abs: List[float]
    contribution_max_abs: List[float]


@dataclass
class TTEntryError:
    row: int
    col: int
    original_value: float
    tt_value: float
    abs_error: float
    rel_error: float


@dataclass
class DenseVsTTEntry:
    row: int
    col: int
    original_value: float
    approx_value: float
    abs_error: float
    rel_error: float
    original_abs_rank: int | None = None



def mixed_radix_unravel(index: int, modes: Sequence[int]) -> List[int]:
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    dims = [int(m) for m in modes]
    total = 1
    for dim in dims:
        total *= dim
    if index >= total:
        raise ValueError(f"index {index} is out of range for modes {dims}")

    digits: List[int] = []
    remainder = int(index)
    for dim in reversed(dims):
        digits.append(remainder % dim)
        remainder //= dim
    return list(reversed(digits))


@torch.no_grad()
def tt_entry_slices(
    tt_module: TTLinear,
    row: int,
    col: int,
) -> tuple[List[int], List[int], List[torch.Tensor]]:
    row_digits = mixed_radix_unravel(int(row), tt_module.out_modes)
    col_digits = mixed_radix_unravel(int(col), tt_module.in_modes)
    slices = [
        core[:, int(row_digit), int(col_digit), :]
        for core, row_digit, col_digit in zip(tt_module.tt_cores, row_digits, col_digits)
    ]
    return row_digits, col_digits, slices


@torch.no_grad()
def tt_entry_value(tt_module: TTLinear, row: int, col: int) -> torch.Tensor:
    _, _, slices = tt_entry_slices(tt_module, row=row, col=col)
    value = slices[0]
    for current in slices[1:]:
        value = value @ current
    return value.squeeze()


@torch.no_grad()
def tt_entry_prefix_suffix(
    tt_module: TTLinear,
    row: int,
    col: int,
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    _, _, slices = tt_entry_slices(tt_module, row=row, col=col)
    device = slices[0].device
    dtype = slices[0].dtype

    prefix = [torch.ones(1, device=device, dtype=dtype)]
    for current in slices:
        prefix.append(prefix[-1] @ current)

    suffix = [torch.empty(0, device=device, dtype=dtype) for _ in range(len(slices) + 1)]
    suffix[-1] = torch.ones(1, device=device, dtype=dtype)
    for idx in reversed(range(len(slices))):
        suffix[idx] = slices[idx] @ suffix[idx + 1]

    return prefix, suffix, slices


@torch.no_grad()
def tt_entry_contribution_maps(tt_module: TTLinear, row: int, col: int) -> List[torch.Tensor]:
    prefix, suffix, slices = tt_entry_prefix_suffix(tt_module, row=row, col=col)
    maps: List[torch.Tensor] = []
    for idx, current in enumerate(slices):
        left = prefix[idx].reshape(-1, 1)
        right = suffix[idx + 1].reshape(1, -1)
        maps.append(current * left * right)
    return maps


@torch.no_grad()
def trace_tt_entry(tt_module: TTLinear, row: int, col: int) -> TTEntryTrace:
    row_digits, col_digits, slices = tt_entry_slices(tt_module, row=row, col=col)
    contribution_maps = tt_entry_contribution_maps(tt_module, row=row, col=col)
    value = float(tt_entry_value(tt_module, row=row, col=col).float().cpu().item())

    return TTEntryTrace(
        row=int(row),
        col=int(col),
        row_multi_index=row_digits,
        col_multi_index=col_digits,
        value=value,
        slice_shapes=[tuple(int(v) for v in current.shape) for current in slices],
        slice_max_abs=[float(current.abs().max().float().cpu().item()) for current in slices],
        contribution_max_abs=[float(current.abs().max().float().cpu().item()) for current in contribution_maps],
    )


@torch.no_grad()
def compare_entry_with_dense(
    tt_module: TTLinear,
    dense_weight: torch.Tensor,
    row: int,
    col: int,
) -> TTEntryError:
    original = float(dense_weight[int(row), int(col)].float().cpu().item())
    approx = float(tt_entry_value(tt_module, row=int(row), col=int(col)).float().cpu().item())
    abs_error = abs(approx - original)
    rel_error = abs_error / max(abs(original), 1e-12)
    return TTEntryError(
        row=int(row),
        col=int(col),
        original_value=original,
        tt_value=approx,
        abs_error=float(abs_error),
        rel_error=float(rel_error),
    )


@torch.no_grad()
def compare_selected_entries_with_dense(
    tt_module: TTLinear,
    dense_weight: torch.Tensor,
    coordinates: Sequence[tuple[int, int]],
) -> List[DenseVsTTEntry]:
    rows: List[DenseVsTTEntry] = []
    for row, col in coordinates:
        original = float(dense_weight[int(row), int(col)].float().cpu().item())
        approx = float(tt_entry_value(tt_module, row=int(row), col=int(col)).float().cpu().item())
        abs_error = abs(approx - original)
        rel_error = abs_error / max(abs(original), 1e-12)
        rows.append(
            DenseVsTTEntry(
                row=int(row),
                col=int(col),
                original_value=original,
                approx_value=approx,
                abs_error=float(abs_error),
                rel_error=float(rel_error),
            )
        )
    return rows


@torch.no_grad()
def topk_dense_outlier_coordinates(weight: torch.Tensor, k: int) -> List[tuple[int, int]]:
    flat = weight.detach().abs().reshape(-1)
    k = min(int(k), flat.numel())
    if k <= 0:
        return []
    _, idx = torch.topk(flat, k=k)
    cols = weight.shape[1]
    result = []
    for flat_idx in idx.tolist():
        row = int(flat_idx // cols)
        col = int(flat_idx % cols)
        result.append((row, col))
    return result


@torch.no_grad()
def sample_entry_error_summary(
    tt_module: TTLinear,
    dense_weight: torch.Tensor,
    *,
    superweight_coords: Sequence[tuple[int, int]],
    topk_outliers: int = 64,
    random_samples: int = 256,
    seed: int = 0,
):
    import pandas as pd

    seen = set()
    tagged_coords: List[tuple[int, int, str]] = []

    for row, col in superweight_coords:
        key = (int(row), int(col))
        if key not in seen:
            seen.add(key)
            tagged_coords.append((key[0], key[1], "superweight"))

    for row, col in topk_dense_outlier_coordinates(dense_weight, k=topk_outliers):
        key = (int(row), int(col))
        if key not in seen:
            seen.add(key)
            tagged_coords.append((key[0], key[1], "top_outlier"))

    generator = torch.Generator(device=dense_weight.device if dense_weight.is_cuda else "cpu")
    generator.manual_seed(int(seed))
    numel = dense_weight.numel()
    random_count = min(int(random_samples), numel)
    sampled = torch.randperm(numel, generator=generator, device=generator.device)[:random_count].cpu().tolist()
    cols = dense_weight.shape[1]
    for flat_idx in sampled:
        row = int(flat_idx // cols)
        col = int(flat_idx % cols)
        key = (row, col)
        if key not in seen:
            seen.add(key)
            tagged_coords.append((row, col, "random"))

    rows = []
    for row, col, tag in tagged_coords:
        original = float(dense_weight[row, col].float().cpu().item())
        approx = float(tt_entry_value(tt_module, row=row, col=col).float().cpu().item())
        abs_error = abs(approx - original)
        rel_error = abs_error / max(abs(original), 1e-12)
        rows.append(
            {
                "tag": tag,
                "row": row,
                "col": col,
                "original_value": original,
                "original_abs": abs(original),
                "approx_value": approx,
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def entry_trace_table(tt_module: TTLinear, row: int, col: int):
    import pandas as pd

    trace = trace_tt_entry(tt_module, row=row, col=col)
    rows = []
    for core_idx, (
        row_digit,
        col_digit,
        slice_shape,
        slice_max_abs,
        contribution_max_abs,
    ) in enumerate(
        zip(
            trace.row_multi_index,
            trace.col_multi_index,
            trace.slice_shapes,
            trace.slice_max_abs,
            trace.contribution_max_abs,
        )
    ):
        rows.append(
            {
                "core_idx": core_idx,
                "row_mode": row_digit,
                "col_mode": col_digit,
                "slice_shape": slice_shape,
                "slice_max_abs": slice_max_abs,
                "contribution_max_abs": contribution_max_abs,
            }
        )
    return pd.DataFrame(rows)
