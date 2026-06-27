from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import tntorch as tn

from src.tt_llm import TTLinear, factor_int_balanced
from src.tt_llm.tensor_utils import linear_weight_to_interleaved_tensor, normalize_modes


@dataclass
class DenseSparseTTSummary:
    outlier_fraction: float
    kept_outlier_count: int
    total_count: int
    kept_fraction_actual: float
    inlier_tt_params: int
    sparse_tt_params: int
    combined_tt_params: int
    dense_params: int
    inlier_tt_ranks: List[int]
    sparse_tt_ranks: List[int]
    combined_tt_ranks: List[int]
    compression_ratio: float



def _merge_interleaved_tt_to_matrix_tt(raw_cores: Sequence[torch.Tensor], out_modes: Sequence[int], in_modes: Sequence[int]) -> List[torch.Tensor]:
    order = len(in_modes)
    if len(raw_cores) != 2 * order:
        raise ValueError(f"Expected {2 * order} TT cores, got {len(raw_cores)}")

    matrix_cores: List[torch.Tensor] = []
    for k in range(order):
        out_core = raw_cores[2 * k]
        in_core = raw_cores[2 * k + 1]
        mpo_core = torch.einsum("aob,bic->aoic", out_core, in_core).contiguous()
        if int(mpo_core.shape[1]) != int(out_modes[k]) or int(mpo_core.shape[2]) != int(in_modes[k]):
            raise ValueError(
                f"Unexpected merged core shape {tuple(mpo_core.shape)} for modes {(out_modes[k], in_modes[k])}"
            )
        matrix_cores.append(mpo_core)
    return matrix_cores



def _boundary_ranks_from_cores(cores: Sequence[torch.Tensor]) -> List[int]:
    if len(cores) == 0:
        return [1, 1]
    ranks = [int(cores[0].shape[0])]
    ranks.extend(int(core.shape[-1]) for core in cores)
    return ranks



def build_interleaved_tntorch_tensor(
    weight: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    tt_rank: int | Sequence[int] | None,
    *,
    algorithm: str = "svd",
) -> tn.Tensor:
    out_features, in_features = weight.shape
    out_modes = normalize_modes(out_modes, out_features, "out_modes")
    in_modes = normalize_modes(in_modes, in_features, "in_modes")

    if len(out_modes) != len(in_modes):
        raise ValueError(
            f"out_modes and in_modes must have equal length, got {len(out_modes)} and {len(in_modes)}"
        )

    interleaved = linear_weight_to_interleaved_tensor(weight, out_modes=out_modes, in_modes=in_modes)
    tt_tensor = tn.Tensor(interleaved)

    if tt_rank is not None:
        n_dims = interleaved.dim()
        if isinstance(tt_rank, int):
            ranks = [int(tt_rank)] * (n_dims - 1)
        else:
            ranks = [int(r) for r in tt_rank]
            if len(ranks) != n_dims - 1:
                raise ValueError(
                    f"For interleaved decomposition, tt_rank must have length {n_dims - 1}, got {len(ranks)}"
                )
        tt_tensor.round_tt(rmax=ranks, algorithm=algorithm)

    return tt_tensor



def split_dense_inliers_and_outliers(
    weight: torch.Tensor,
    outlier_fraction: float,
    *,
    include_coords: Iterable[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if outlier_fraction < 0 or outlier_fraction > 1:
        raise ValueError(f"outlier_fraction must be in [0, 1], got {outlier_fraction}")

    flat_abs = weight.detach().abs().reshape(-1)
    total = flat_abs.numel()
    keep = int(math.ceil(total * float(outlier_fraction)))

    mask = torch.zeros_like(flat_abs, dtype=torch.bool)
    if keep > 0:
        top_idx = torch.topk(flat_abs, k=min(keep, total)).indices
        mask[top_idx] = True

    if include_coords is not None:
        n_cols = weight.shape[1]
        for row, col in include_coords:
            mask[int(row) * n_cols + int(col)] = True

    mask = mask.view_as(weight)
    sparse = torch.where(mask, weight, torch.zeros_like(weight))
    inliers = weight - sparse
    return inliers, sparse, mask

def _mixed_radix_unravel(index: int, modes: Sequence[int]) -> list[int]:
    digits = []
    rem = int(index)
    for dim in reversed(modes):
        digits.append(rem % int(dim))
        rem //= int(dim)
    return list(reversed(digits))


def _single_spike_matrix_tt_cores(
    *,
    row: int,
    col: int,
    value: float,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """
    Exact rank-1 matrix-TT for one dense entry W[row, col] = value.
    """
    row_digits = _mixed_radix_unravel(int(row), out_modes)
    col_digits = _mixed_radix_unravel(int(col), in_modes)

    cores = []
    for k, (o_dim, i_dim, o_idx, i_idx) in enumerate(zip(out_modes, in_modes, row_digits, col_digits)):
        core = torch.zeros(1, int(o_dim), int(i_dim), 1, dtype=dtype, device=device)
        core[0, int(o_idx), int(i_idx), 0] = float(value) if k == 0 else 1.0
        cores.append(core)
    return cores

def add_matrix_tt_cores(cores_a: Sequence[torch.Tensor], cores_b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if len(cores_a) != len(cores_b):
        raise ValueError(f"Core length mismatch: {len(cores_a)} vs {len(cores_b)}")
    if len(cores_a) == 0:
        return []
    if len(cores_a) == 1:
        return [cores_a[0] + cores_b[0]]

    out: List[torch.Tensor] = []
    d = len(cores_a)
    for k, (a, b) in enumerate(zip(cores_a, cores_b)):
        if a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
            raise ValueError(f"Mode mismatch at core {k}: {tuple(a.shape)} vs {tuple(b.shape)}")

        if k == 0:
            out.append(torch.cat([a, b], dim=-1))
        elif k == d - 1:
            out.append(torch.cat([a, b], dim=0))
        else:
            a_left, o_dim, i_dim, a_right = a.shape
            b_left, _, _, b_right = b.shape
            top = torch.cat(
                [a, torch.zeros(a_left, o_dim, i_dim, b_right, device=a.device, dtype=a.dtype)],
                dim=-1,
            )
            bottom = torch.cat(
                [torch.zeros(b_left, o_dim, i_dim, a_right, device=b.device, dtype=b.dtype), b],
                dim=-1,
            )
            out.append(torch.cat([top, bottom], dim=0))
    return out

def _build_exact_sparse_matrix_tt_cores(
    sparse_weight: torch.Tensor,
    *,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    sparse_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Build the exact sparse matrix-TT as a sum of rank-1 spike-TTs.
    """
    coords = sparse_mask.nonzero(as_tuple=False).cpu().tolist()
    if len(coords) == 0:
        raise ValueError("Sparse mask is empty")

    acc_cores = None
    for row, col in coords:
        value = float(sparse_weight[int(row), int(col)].item())
        spike_cores = _single_spike_matrix_tt_cores(
            row=int(row),
            col=int(col),
            value=value,
            out_modes=out_modes,
            in_modes=in_modes,
            dtype=sparse_weight.dtype,
            device=sparse_weight.device,
        )
        if acc_cores is None:
            acc_cores = spike_cores
        else:
            acc_cores = add_matrix_tt_cores(acc_cores, spike_cores)

    return acc_cores

def dense_sparse_tt_from_linear_exact_sparse(
    linear: nn.Linear,
    *,
    tt_rank_inliers: int | Sequence[int],
    outlier_fraction: float,
    include_coords: Iterable[tuple[int, int]] | None = None,
    in_modes: Sequence[int] | None = None,
    out_modes: Sequence[int] | None = None,
    order: int = 12,
    decompose_dtype: torch.dtype = torch.float64,
    decompose_device: torch.device | str = "cpu",
    output_device: torch.device | str | None = None,
    token_chunk_size: int | None = None,
    algorithm: str = "svd",
) -> tuple[TTLinear, DenseSparseTTSummary]:
    """
    Dense+sparse TT where:
      - inliers are TT-compressed with rank truncation
      - sparse outliers are represented exactly as a sum of rank-1 spike matrix-TTs
      - the two matrix-TTs are summed exactly
    """
    if in_modes is None:
        in_modes = factor_int_balanced(linear.in_features, order=order)
    if out_modes is None:
        out_modes = factor_int_balanced(linear.out_features, order=order)

    in_modes = normalize_modes(in_modes, linear.in_features, "in_modes")
    out_modes = normalize_modes(out_modes, linear.out_features, "out_modes")

    target_device = linear.weight.device if output_device is None else output_device
    target_dtype = linear.weight.dtype

    dense_weight = linear.weight.detach().to(device=decompose_device, dtype=decompose_dtype)

    inlier_weight, sparse_weight, sparse_mask = split_dense_inliers_and_outliers(
        dense_weight,
        outlier_fraction,
        include_coords=include_coords,
    )

    inlier_tt = build_interleaved_tntorch_tensor(
        inlier_weight,
        out_modes=out_modes,
        in_modes=in_modes,
        tt_rank=tt_rank_inliers,
        algorithm=algorithm,
    )
    inlier_cores = _merge_interleaved_tt_to_matrix_tt(
        inlier_tt.cores,
        out_modes=out_modes,
        in_modes=in_modes,
    )

    sparse_cores = _build_exact_sparse_matrix_tt_cores(
        sparse_weight,
        out_modes=out_modes,
        in_modes=in_modes,
        sparse_mask=sparse_mask,
    )

    combined_cores = add_matrix_tt_cores(inlier_cores, sparse_cores)
    moved_cores = [
        core.to(device=target_device, dtype=target_dtype).contiguous()
        for core in combined_cores
    ]

    bias = None
    if linear.bias is not None:
        bias = linear.bias.detach().clone().to(device=target_device, dtype=target_dtype)

    tt_module = TTLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        in_modes=in_modes,
        out_modes=out_modes,
        tt_cores=moved_cores,
        bias=bias,
        token_chunk_size=token_chunk_size,
    )

    dense_params = linear.weight.numel() + (0 if linear.bias is None else linear.bias.numel())
    inlier_tt_params = int(sum(core.numel() for core in inlier_cores))
    sparse_tt_params = int(sum(core.numel() for core in sparse_cores))
    combined_tt_params = int(sum(core.numel() for core in moved_cores)) + (0 if bias is None else bias.numel())
    kept_outlier_count = int(sparse_mask.sum().item())

    summary = DenseSparseTTSummary(
        outlier_fraction=float(outlier_fraction),
        kept_outlier_count=kept_outlier_count,
        total_count=int(linear.weight.numel()),
        kept_fraction_actual=float(kept_outlier_count / max(int(linear.weight.numel()), 1)),
        inlier_tt_params=inlier_tt_params,
        sparse_tt_params=sparse_tt_params,
        combined_tt_params=combined_tt_params,
        dense_params=int(dense_params),
        inlier_tt_ranks=_boundary_ranks_from_cores(inlier_cores),
        sparse_tt_ranks=_boundary_ranks_from_cores(sparse_cores),
        combined_tt_ranks=_boundary_ranks_from_cores(moved_cores),
        compression_ratio=float(dense_params / max(combined_tt_params, 1)),
    )
    return tt_module, summary