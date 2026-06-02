"""
Additional weight decomposition methods for transformer LLMs:

  1. Laser          – TruncatedSVD low-rank approximation. Compression ratio is
                      printed before applying. Result is stored as a dense layer
                      of the same shape (U_k @ diag(S_k) @ V_k^T).
  2. per_head_tt    – TT decomposition applied independently to each attention
                      head's weight slice (not the concatenated block).
  3. per_head_tucker– 2-D Tucker decomposition applied independently to each
                      attention head's weight slice (not the concatenated block).

All three functions take an ``nn.Linear`` and modify it **in place**, returning
the same module with updated weights. They reconstruct back to a dense layer so
the model graph is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker

def _weight_to_per_head_slices(
    weight: torch.Tensor,
    n_heads: int,
    is_output_projection: bool = False,
) -> List[torch.Tensor]:
    """
    Split a 2-D projection weight into per-head slices.

    Q / K / V  weight [n_heads*head_dim, hidden_size]
               -> n_heads tensors of shape [head_dim, hidden_size]

    O          weight [hidden_size, n_heads*head_dim]
               -> n_heads tensors of shape [hidden_size, head_dim]
    """
    if is_output_projection:
        out_f, in_f = weight.shape
        head_dim = in_f // n_heads
        return list(weight.view(out_f, n_heads, head_dim).permute(1, 0, 2).unbind(0))
    else:
        out_f, in_f = weight.shape
        head_dim = out_f // n_heads
        return list(weight.view(n_heads, head_dim, in_f).unbind(0))


def _per_head_slices_to_weight(
    head_slices: List[torch.Tensor],
    n_heads: int,
    is_output_projection: bool = False,
) -> torch.Tensor:
    """Reassemble per-head slices back into the original 2-D weight."""
    stacked = torch.stack(head_slices, dim=0)  # [n_heads, ...]
    if is_output_projection:
        n, h, d = stacked.shape
        return stacked.permute(1, 0, 2).reshape(h, n * d).contiguous()
    else:
        n, h, d = stacked.shape
        return stacked.reshape(n * h, d).contiguous()


@dataclass
class LaserSVDSummary:
    module_name: str
    original_shape: Tuple[int, int]
    effective_rank: int
    compression_ratio: float
    reconstruction_error: float


def _laser_compression_ratio(out_f: int, in_f: int, rank: int) -> float:
    """Theoretical compression: dense_params / (U_k + S_k + V_k^T params)."""
    dense = out_f * in_f
    low_rank = rank * (out_f + 1 + in_f)
    return float(dense) / max(low_rank, 1)


@torch.no_grad()
def laser_svd_apply(
    module: nn.Linear,
    rank: int,
    *,
    name: str = "",
) -> Tuple[nn.Linear, LaserSVDSummary]:
    """
    Apply TruncatedSVD-based rank-k approximation to an nn.Linear, in-place.

    Steps
    -----
    1. Compute compression ratio for the requested rank and print it.
    2. Decompose W with sklearn TruncatedSVD(n_components=rank).
    3. Reconstruct W_k = U_k @ diag(S_k) @ V_k^T  (dense, same shape as W).
    4. Replace module.weight with W_k.

    Parameters
    ----------
    module : nn.Linear
        Layer to modify.
    rank : int
        Number of singular values to keep.
    name : str
        Optional label used in the printed summary.

    Returns
    -------
    module : nn.Linear
        Same object, weight replaced in-place.
    summary : LaserSVDSummary
        Compression statistics.
    """
    from sklearn.decomposition import TruncatedSVD

    W = module.weight.data
    out_f, in_f = W.shape
    max_rank = min(out_f, in_f)
    # TruncatedSVD requires n_components < min(n_samples, n_features)
    k = max(1, min(rank, max_rank - 1))

    compression = _laser_compression_ratio(out_f, in_f, k)
    print(
        f"[Laser] {name or 'layer'}  shape={tuple(W.shape)}  "
        f"rank={k}/{max_rank}  theoretical_compression={compression:.2f}x"
    )

    W_np = W.detach().float().cpu().numpy()
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=4, random_state=42)
    W_transformed = svd.fit_transform(W_np)
    W_approx_np = W_transformed @ svd.components_

    W_approx = torch.from_numpy(W_approx_np).to(device=W.device, dtype=W.dtype).contiguous()
    rec_error = float(
        (
            torch.norm(W_approx.float() - W.float())
            / torch.norm(W.float()).clamp_min(1e-12)
        ).item()
    )
    module.weight = nn.Parameter(W_approx)

    return module, LaserSVDSummary(
        module_name=name,
        original_shape=(out_f, in_f),
        effective_rank=k,
        compression_ratio=compression,
        reconstruction_error=rec_error,
    )

@torch.no_grad()
def per_head_tt_apply(
    module: nn.Linear,
    n_heads: int,
    tt_rank: int,
    *,
    is_output_projection: bool = False,
    order: int = 4,
    decompose_dtype: torch.dtype = torch.float64,
    decompose_device: str = "cpu",
    name: str = "",
) -> nn.Linear:
    """
    Apply TT decomposition independently to each attention-head slice of a
    projection weight, then reassemble into a dense nn.Linear (same shape).

    For each head the weight slice [head_dim, hidden_size] (or [hidden_size,
    head_dim] for the O projection) is decomposed as a matrix TT with the
    given rank, then reconstructed back to a dense matrix.

    Parameters
    ----------
    module : nn.Linear
    n_heads : int
    tt_rank : int
        TT bond dimension.
    is_output_projection : bool
        True for the output (O) projection where heads are in the input dim.
    order : int
        TT order (number of matrix-TT cores per head).
    decompose_dtype : torch.dtype
        Working dtype for the decomposition.
    decompose_device : str
        Device for the decomposition (e.g. "cpu" or "cuda").
    name : str

    Returns
    -------
    module : nn.Linear  (weights replaced in-place, still dense)
    """
    from src.tt_llm.tt_linear import TTLinear
    from src.tt_llm.tensor_utils import factor_int_balanced

    W = module.weight.data
    original_device = W.device
    original_dtype = W.dtype

    head_slices = _weight_to_per_head_slices(W, n_heads, is_output_projection)

    reconstructed: List[torch.Tensor] = []
    for h_idx, head_w in enumerate(head_slices):
        out_f, in_f = head_w.shape
        in_modes = factor_int_balanced(in_f, order=order)
        out_modes = factor_int_balanced(out_f, order=order)

        dummy = nn.Linear(in_f, out_f, bias=False)
        dummy.weight = nn.Parameter(head_w.contiguous())

        tt = TTLinear.from_linear(
            dummy,
            tt_rank=tt_rank,
            in_modes=in_modes,
            out_modes=out_modes,
            order=order,
            decompose_dtype=decompose_dtype,
            decompose_device=decompose_device,
            output_device=decompose_device,
        )
        rec = tt.to_dense_weight().to(device=original_device, dtype=original_dtype)
        reconstructed.append(rec)
        del tt, dummy

    new_weight = _per_head_slices_to_weight(reconstructed, n_heads, is_output_projection)
    module.weight = nn.Parameter(new_weight.to(device=original_device, dtype=original_dtype))

    if name:
        print(
            f"[per_head_tt] {name}  n_heads={n_heads}  "
            f"tt_rank={tt_rank}  order={order}"
        )
    return module


@torch.no_grad()
def per_head_tucker_apply(
    module: nn.Linear,
    n_heads: int,
    hidden_rank: int,
    head_dim_rank: int,
    *,
    is_output_projection: bool = False,
    decompose_dtype: torch.dtype = torch.float32,
    name: str = "",
) -> nn.Linear:
    """
    Apply 2-D Tucker decomposition independently to each attention-head slice
    of a projection weight, then reassemble into a dense nn.Linear (same shape).

    For Q/K/V each head slice has shape [head_dim, hidden_size]; for O it is
    [hidden_size, head_dim]. Tucker-2D(rank=[r0, r1]) factorises the slice as:

        core [r0, r1]  +  factor0 [m, r0]  +  factor1 [n, r1]
        reconstruction: multi_mode_dot(core, [f0, f1]) -> [m, n]

    Parameters
    ----------
    module : nn.Linear
    n_heads : int
    hidden_rank : int
        Tucker rank for the dimension that corresponds to hidden_size (or the
        larger dimension).
    head_dim_rank : int
        Tucker rank for the head_dim dimension.
    is_output_projection : bool
    decompose_dtype : torch.dtype
    name : str

    Returns
    -------
    module : nn.Linear  (weights replaced in-place, still dense)
    """
    tl.set_backend("pytorch")

    W = module.weight.data
    original_device = W.device
    original_dtype = W.dtype

    head_slices = _weight_to_per_head_slices(W, n_heads, is_output_projection)

    reconstructed: List[torch.Tensor] = []
    for head_w in head_slices:
        m, n = head_w.shape
        rank = [min(hidden_rank, m), min(head_dim_rank, n)]

        work = head_w.detach().to(dtype=decompose_dtype).contiguous()
        (core, factors), _ = partial_tucker(
            work,
            modes=[0, 1],
            rank=rank,
            init="svd",
            svd="randomized_svd",
            random_state=0,
            tol=1e-5,
            verbose=False,
        )
        rec = tl.tenalg.multi_mode_dot(core, factors, modes=[0, 1])
        reconstructed.append(rec.to(dtype=original_dtype))

    new_weight = _per_head_slices_to_weight(reconstructed, n_heads, is_output_projection)
    module.weight = nn.Parameter(new_weight.to(device=original_device, dtype=original_dtype))

    if name:
        print(
            f"[per_head_tucker] {name}  n_heads={n_heads}  "
            f"hidden_rank={hidden_rank}  head_dim_rank={head_dim_rank}"
        )
    return module
