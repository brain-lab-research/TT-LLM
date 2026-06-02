"""Core TD-MoE math and budget utilities.

The tensor layout used throughout this module is:
    expert tensor shape = (num_experts, d_out, d_in)

TD-MoE supports two expert modes:
    - ``compress``: full 3-mode Tucker with an expert factor ``U_expert``
    - ``preserve``: keep the expert mode uncompressed and only factor feature dims
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


FactorDict = Dict[str, Optional[torch.Tensor]]
SVD_BACKENDS = {"gram_eigh", "full", "randomized"}


def _validate_expert_mode(expert_mode: str) -> None:
    if expert_mode not in {"compress", "preserve"}:
        raise ValueError(
            f"Unsupported expert_mode={expert_mode!r}. Expected 'compress' or 'preserve'."
        )


def _validate_svd_backend(svd_backend: str) -> None:
    if svd_backend not in SVD_BACKENDS:
        raise ValueError(
            f"Unsupported svd_backend={svd_backend!r}. "
            f"Expected one of {sorted(SVD_BACKENDS)}."
        )


def unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-n unfolding of a tensor."""
    return tensor.moveaxis(mode, 0).reshape(tensor.shape[mode], -1)


def mode_n_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """Multiply a tensor by ``matrix`` along the given mode."""
    moved = tensor.moveaxis(mode, 0)
    original_shape = moved.shape
    flat = moved.reshape(original_shape[0], -1)
    result = matrix @ flat
    new_shape = (matrix.shape[0],) + original_shape[1:]
    return result.reshape(new_shape).moveaxis(0, mode)


def truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    use_randomized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a truncated SVD."""
    if matrix.ndim != 2:
        raise ValueError(f"truncated_svd expects a 2D matrix, got {matrix.ndim}D")

    max_rank = min(matrix.shape)
    rank = max(1, min(rank, max_rank))
    matrix_f32 = matrix.to(torch.float32)

    if use_randomized and rank < max_rank // 2:
        u, s, v = torch.svd_lowrank(matrix_f32, q=rank, niter=4)
        return u[:, :rank], s[:rank], v[:, :rank].T

    u, s, vh = torch.linalg.svd(matrix_f32, full_matrices=False)
    return u[:, :rank], s[:rank], vh[:rank, :]


def leading_left_singular_vectors(
    matrix: torch.Tensor,
    rank: int,
    *,
    svd_backend: str = "gram_eigh",
) -> torch.Tensor:
    """Compute leading left singular vectors for HOSVD.

    HOSVD only needs the left singular subspace of each unfolding. For the
    tall-row TD-MoE unfoldings, eigendecomposing ``A @ A.T`` is mathematically
    equivalent to a full SVD but avoids forming the unused right singular
    vectors.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"leading_left_singular_vectors expects a 2D matrix, got {matrix.ndim}D"
        )
    _validate_svd_backend(svd_backend)

    max_rank = min(matrix.shape)
    rank = max(1, min(rank, max_rank))
    matrix_f32 = matrix.to(torch.float32)

    if svd_backend == "randomized":
        if rank >= max_rank:
            u, _, _ = torch.linalg.svd(matrix_f32, full_matrices=False)
            return u[:, :rank]
        u, _, _ = torch.svd_lowrank(matrix_f32, q=rank, niter=4)
        return u[:, :rank]

    if svd_backend == "full" or matrix_f32.shape[0] > matrix_f32.shape[1]:
        u, _, _ = torch.linalg.svd(matrix_f32, full_matrices=False)
        return u[:, :rank]

    gram = matrix_f32 @ matrix_f32.T
    _, eigenvectors = torch.linalg.eigh(gram)
    return torch.flip(eigenvectors[:, -rank:], dims=[1]).contiguous()


def tucker_decomposition(
    tensor: torch.Tensor,
    ranks: Tuple[int, int, int],
    *,
    use_randomized: bool = True,
    expert_mode: str = "compress",
    svd_backend: str = "gram_eigh",
) -> Tuple[torch.Tensor, FactorDict]:
    """HOSVD-based Tucker decomposition for expert tensors."""
    _validate_expert_mode(expert_mode)
    _validate_svd_backend(svd_backend)

    tensor_f32 = tensor.to(torch.float32)
    num_experts, d_out, d_in = tensor_f32.shape
    r1, r2, r3 = ranks

    if expert_mode == "preserve":
        if r1 != num_experts:
            raise ValueError(
                f"expert_mode='preserve' requires r1={num_experts}, got r1={r1}"
            )
        u_expert = None
    else:
        u_expert = leading_left_singular_vectors(
            unfold(tensor_f32, 0),
            r1,
            svd_backend=svd_backend,
        )

    u_out = leading_left_singular_vectors(
        unfold(tensor_f32, 1),
        r2,
        svd_backend=svd_backend,
    )
    u_in = leading_left_singular_vectors(
        unfold(tensor_f32, 2),
        r3,
        svd_backend=svd_backend,
    )

    core = tensor_f32
    if u_expert is not None:
        core = mode_n_product(core, u_expert.T, mode=0)
    core = mode_n_product(core, u_out.T, mode=1)
    core = mode_n_product(core, u_in.T, mode=2)

    factors: FactorDict = {
        "U_expert": u_expert,
        "U_out": u_out,
        "U_in": u_in,
    }
    return core, factors


def extract_factor_subspace(factors: FactorDict, mode: str) -> Optional[torch.Tensor]:
    """Return orthonormal columns of ``U_in``/``U_out``/``U_expert`` if present.

    ``mode`` is one of ``"in"``, ``"out"``, ``"expert"``.
    """
    key_map = {"in": "U_in", "out": "U_out", "expert": "U_expert"}
    if mode not in key_map:
        raise ValueError(f"Unknown factor mode {mode!r}; expected one of {sorted(key_map)}")
    U = factors.get(key_map[mode])
    return None if U is None else U


def reconstruct_from_tucker(core: torch.Tensor, factors: FactorDict) -> torch.Tensor:
    """Reconstruct a dense expert tensor from Tucker factors."""
    result = core.to(torch.float32)

    u_in = factors.get("U_in")
    u_out = factors.get("U_out")
    u_expert = factors.get("U_expert")

    if u_out is None or u_in is None:
        raise ValueError("Both U_out and U_in must be present for reconstruction")

    if u_expert is not None:
        result = mode_n_product(result, u_expert, mode=0)
    result = mode_n_product(result, u_out, mode=1)
    result = mode_n_product(result, u_in, mode=2)
    return result


def compute_tucker_params(
    ranks: Tuple[int, int, int],
    num_experts: int,
    d_out: int,
    d_in: int,
    *,
    expert_mode: str = "compress",
) -> int:
    """Compute parameter count for a Tucker representation."""
    _validate_expert_mode(expert_mode)
    r1, r2, r3 = ranks

    core_params = r1 * r2 * r3
    factor_params = d_out * r2 + d_in * r3
    if expert_mode == "compress":
        factor_params += num_experts * r1
    return core_params + factor_params


def _candidate_r3_values(raw_r3: float, upper: int, step: int = 1) -> List[int]:
    if step > upper:
        step = 1
    if step <= 1:
        floor_val = math.floor(raw_r3)
        candidates = {floor_val - 1, floor_val, floor_val + 1}
        return [v for v in sorted(candidates) if 1 <= v <= upper]
    base = round(raw_r3 / step) * step
    candidates = {base - step, base, base + step}
    return [v for v in sorted(candidates) if step <= v <= upper]


def budget_constrained_rank_search(
    num_experts: int,
    d_out: int,
    d_in: int,
    *,
    compression_ratio: Optional[float] = None,
    target_params: Optional[int] = None,
    expert_mode: str = "compress",
    rank_alignment: int = 8,
) -> Tuple[int, int, int]:
    """Search Tucker ranks that best match a target parameter budget.

    ``rank_alignment`` snaps ``r2`` and ``r3`` to multiples of this value so
    that the innermost stride of transposed core / factor tensors is 16-byte
    aligned and the fused ``grouped_mm`` kernels accept them at inference.
    Default 8 corresponds to 16 bytes in bf16; use 4 for fp32 or 1 to
    disable.
    """
    _validate_expert_mode(expert_mode)
    if rank_alignment < 1:
        raise ValueError(f"rank_alignment must be >= 1, got {rank_alignment}")

    original_params = num_experts * d_out * d_in
    if target_params is None:
        if compression_ratio is None:
            raise ValueError("Provide either target_params or compression_ratio")
        target_params = int(round((1.0 - compression_ratio) * original_params))
    target_params = max(1, min(int(target_params), original_params))

    best_err = float("inf")
    best_ranks = (1, 1, 1)

    r_step = rank_alignment if d_out >= rank_alignment and d_in >= rank_alignment else 1
    r2_start = 1 if r_step == 1 else r_step

    if expert_mode == "preserve":
        r1 = num_experts
        for r2 in range(r2_start, d_out + 1, r_step):
            denominator = num_experts * r2 + d_in
            numerator = target_params - d_out * r2
            if denominator <= 0:
                continue
            raw_r3 = numerator / denominator
            for r3 in _candidate_r3_values(raw_r3, d_in, step=r_step):
                params = compute_tucker_params(
                    (r1, r2, r3), num_experts, d_out, d_in, expert_mode=expert_mode
                )
                err = abs(params - target_params)
                if err < best_err:
                    best_err = err
                    best_ranks = (r1, r2, r3)
    else:
        for r1 in range(1, num_experts + 1):
            for r2 in range(r2_start, d_out + 1, r_step):
                denominator = r1 * r2 + d_in
                numerator = target_params - (num_experts * r1 + d_out * r2)
                if denominator <= 0:
                    continue
                raw_r3 = numerator / denominator
                for r3 in _candidate_r3_values(raw_r3, d_in, step=r_step):
                    params = compute_tucker_params(
                        (r1, r2, r3), num_experts, d_out, d_in, expert_mode=expert_mode
                    )
                    err = abs(params - target_params)
                    if err < best_err:
                        best_err = err
                        best_ranks = (r1, r2, r3)

    return best_ranks


def compute_whitening_matrix(covariance: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Compute ``(covariance + epsilon I)^(-1/2)``."""
    cov_f32 = covariance.to(torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_f32)
    clipped = eigenvalues.clamp(min=epsilon)
    return eigenvectors @ torch.diag(clipped.pow(-0.5)) @ eigenvectors.T


def compute_inverse_whitening(covariance: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Compute the inverse whitening matrix ``(covariance + epsilon I)^(1/2)``."""
    cov_f32 = covariance.to(torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_f32)
    clipped = eigenvalues.clamp(min=epsilon)
    return eigenvectors @ torch.diag(clipped.pow(0.5)) @ eigenvectors.T


def compute_whitening_pair(
    covariance: torch.Tensor,
    epsilon: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute whitening and inverse-whitening matrices from one eigensolve."""
    cov_f32 = covariance.to(torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_f32)
    clipped = eigenvalues.clamp(min=epsilon)
    whitening = eigenvectors @ torch.diag(clipped.pow(-0.5)) @ eigenvectors.T
    inverse = eigenvectors @ torch.diag(clipped.pow(0.5)) @ eigenvectors.T
    return whitening, inverse


def apply_whitening(
    tensor: torch.Tensor,
    *,
    S_in: Optional[torch.Tensor] = None,
    S_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply input/output whitening to an expert tensor."""
    result = tensor.to(torch.float32)
    if S_out is not None:
        result = mode_n_product(result, S_out, mode=1)
    if S_in is not None:
        result = mode_n_product(result, S_in, mode=2)
    return result


def recolor_factors(
    factors: FactorDict,
    *,
    S_in_inv: Optional[torch.Tensor] = None,
    S_out_inv: Optional[torch.Tensor] = None,
) -> FactorDict:
    """Absorb inverse whitening matrices into the factor matrices."""
    recolored: FactorDict = {
        "U_expert": factors.get("U_expert"),
        "U_out": factors.get("U_out"),
        "U_in": factors.get("U_in"),
    }

    if recolored["U_out"] is not None and S_out_inv is not None:
        recolored["U_out"] = S_out_inv @ recolored["U_out"]
    if recolored["U_in"] is not None and S_in_inv is not None:
        recolored["U_in"] = S_in_inv @ recolored["U_in"]
    return recolored


def reconstruction_error_stats(
    expert_weights: torch.Tensor,
    core: torch.Tensor,
    factors: FactorDict,
    *,
    chunk_size: int = 8,
) -> Tuple[float, float]:
    """Compute dense reconstruction error without materializing all experts."""
    u_out = factors.get("U_out")
    u_in = factors.get("U_in")
    u_expert = factors.get("U_expert")
    if u_out is None or u_in is None:
        raise ValueError("Both U_out and U_in must be present for reconstruction")

    num_experts = expert_weights.shape[0]
    chunk_size = max(1, int(chunk_size))
    sumsq = torch.zeros((), dtype=torch.float64, device=expert_weights.device)

    for start in range(0, num_experts, chunk_size):
        end = min(start + chunk_size, num_experts)
        if u_expert is None:
            expert_core = core[start:end]
        else:
            expert_core = torch.einsum("er,rab->eab", u_expert[start:end], core)
        reconstructed = torch.einsum("oa,eab,ib->eoi", u_out, expert_core, u_in)
        diff = reconstructed.to(torch.float32) - expert_weights[start:end].to(torch.float32)
        sumsq += diff.pow(2).sum().to(torch.float64)

    frobenius = torch.sqrt(sumsq).item()
    mse = (sumsq / expert_weights.numel()).item()
    return frobenius, mse


def td_moe_compress(
    expert_weights: torch.Tensor,
    *,
    target_params: Optional[int] = None,
    compression_ratio: Optional[float] = None,
    expert_mode: str = "compress",
    use_randomized_svd: bool = True,
    eigenvalue_clip: float = 1e-3,
    input_covariance: Optional[torch.Tensor] = None,
    output_covariance: Optional[torch.Tensor] = None,
    whitening_mode: str = "none",
    ranks_override: Optional[Tuple[int, int, int]] = None,
    rank_alignment: int = 8,
    svd_backend: str = "gram_eigh",
    return_reconstructed: bool = True,
) -> Dict[str, object]:
    """Run the full TD-MoE compression pipeline."""
    _validate_expert_mode(expert_mode)
    _validate_svd_backend(svd_backend)
    if whitening_mode not in {"none", "input", "output", "both"}:
        raise ValueError(
            f"Unsupported whitening_mode={whitening_mode!r}. Expected one of: none, input, output, both."
        )

    num_experts, d_out, d_in = expert_weights.shape
    device = expert_weights.device

    S_in = None
    S_out = None
    S_in_inv = None
    S_out_inv = None

    if whitening_mode in {"input", "both"} and input_covariance is not None:
        S_in, S_in_inv = compute_whitening_pair(input_covariance, eigenvalue_clip)
        S_in = S_in.to(device)
        S_in_inv = S_in_inv.to(device)

    if whitening_mode in {"output", "both"} and output_covariance is not None:
        S_out, S_out_inv = compute_whitening_pair(output_covariance, eigenvalue_clip)
        S_out = S_out.to(device)
        S_out_inv = S_out_inv.to(device)

    whitened = apply_whitening(expert_weights, S_in=S_in, S_out=S_out)

    if ranks_override is not None:
        ranks = ranks_override
    else:
        ranks = budget_constrained_rank_search(
            num_experts,
            d_out,
            d_in,
            compression_ratio=compression_ratio,
            target_params=target_params,
            expert_mode=expert_mode,
            rank_alignment=rank_alignment,
        )

    core, factors = tucker_decomposition(
        whitened,
        ranks,
        use_randomized=use_randomized_svd,
        expert_mode=expert_mode,
        svd_backend=svd_backend,
    )
    factors = recolor_factors(factors, S_in_inv=S_in_inv, S_out_inv=S_out_inv)

    reconstructed = reconstruct_from_tucker(core, factors) if return_reconstructed else None

    original_params = num_experts * d_out * d_in
    compressed_params = compute_tucker_params(
        ranks, num_experts, d_out, d_in, expert_mode=expert_mode
    )
    actual_ratio = 1.0 - (compressed_params / original_params)
    if target_params is None:
        target_params = compressed_params if compression_ratio is None else int(
            round((1.0 - compression_ratio) * original_params)
        )

    if reconstructed is None:
        reconstruction_frobenius, reconstruction_mse = reconstruction_error_stats(
            expert_weights,
            core,
            factors,
        )
    else:
        diff = reconstructed.to(torch.float32) - expert_weights.to(torch.float32)
        reconstruction_frobenius = torch.norm(diff).item()
        reconstruction_mse = torch.mean(diff.pow(2)).item()

    compression_stats = {
        "original_params": original_params,
        "target_params": int(target_params),
        "compressed_params": compressed_params,
        "actual_compression": actual_ratio,
        "compression_ratio": 1.0 - (int(target_params) / original_params),
        "ranks": tuple(int(v) for v in ranks),
        "reconstruction_error_frobenius": reconstruction_frobenius,
        "reconstruction_mse": reconstruction_mse,
    }

    return {
        "core": core,
        "factors": factors,
        "ranks": tuple(int(v) for v in ranks),
        "reconstructed": reconstructed,
        "compression_stats": compression_stats,
        "expert_mode": expert_mode,
        "whitening_mode": whitening_mode,
        "svd_backend": svd_backend,
    }


def allocate_layer_parameter_budgets(
    layer_indices: List[int],
    activation_counts: Dict[int, torch.Tensor],
    original_params_per_layer: int,
    global_compression_ratio: float,
    *,
    policy: str = "uniform",
    smoothing: float = 0.0,
) -> Dict[int, int]:
    """Allocate target parameter budgets across layers."""
    if policy not in {"uniform", "activation_weighted"}:
        raise ValueError(
            f"Unsupported layer budget policy {policy!r}. Expected 'uniform' or 'activation_weighted'."
        )
    if not layer_indices:
        return {}

    total_target = int(round((1.0 - global_compression_ratio) * original_params_per_layer * len(layer_indices)))

    if policy == "uniform":
        weights = torch.ones(len(layer_indices), dtype=torch.float64)
    else:
        values = []
        for layer_idx in layer_indices:
            counts = activation_counts.get(layer_idx)
            values.append(float(counts.float().mean().item()) if counts is not None else 1.0)
        weights = torch.tensor(values, dtype=torch.float64)
        if torch.all(weights == 0):
            weights = torch.ones_like(weights)

    weights = weights / weights.sum()

    if smoothing > 0:
        uniform = torch.full_like(weights, 1.0 / len(weights))
        weights = (1.0 - smoothing) * weights + smoothing * uniform
        weights = weights / weights.sum()

    budgets = {}
    assigned = 0
    for idx, layer_idx in enumerate(layer_indices):
        if idx == len(layer_indices) - 1:
            budget = total_target - assigned
        else:
            budget = int(round(weights[idx].item() * total_target))
            assigned += budget
        budgets[layer_idx] = max(1, min(budget, original_params_per_layer))
    return budgets


def allocate_layerwise_budgets(
    layer_indices: List[int],
    activation_counts: Dict[int, torch.Tensor],
    num_experts: int,
    d_out: int,
    d_in: int,
    global_compression_ratio: float,
    smoothing: float = 0.0,
) -> Dict[int, float]:
    """Backward-compatible wrapper that returns per-layer compression ratios."""
    original_params = num_experts * d_out * d_in
    budgets = allocate_layer_parameter_budgets(
        layer_indices,
        activation_counts,
        original_params,
        global_compression_ratio,
        policy="activation_weighted",
        smoothing=smoothing,
    )
    return {
        layer_idx: max(0.0, min(0.95, 1.0 - (budget / original_params)))
        for layer_idx, budget in budgets.items()
    }


def structured_tucker_pruning(
    core: torch.Tensor,
    factors: FactorDict,
    pruning_ratio: float,
) -> Tuple[torch.Tensor, FactorDict]:
    """Prune low-energy latent dimensions from a Tucker representation."""
    if pruning_ratio <= 0:
        return core, factors

    if not (0.0 <= pruning_ratio < 1.0):
        raise ValueError("pruning_ratio must be in [0, 1)")

    _, r_out, r_in = core.shape
    keep_out = max(1, int(math.floor((1.0 - pruning_ratio) * r_out)))
    keep_in = max(1, int(math.floor((1.0 - pruning_ratio) * r_in)))

    energy_out = torch.linalg.vector_norm(core, dim=(0, 2))
    energy_in = torch.linalg.vector_norm(core, dim=(0, 1))

    idx_out = torch.topk(energy_out, keep_out).indices.sort().values
    idx_in = torch.topk(energy_in, keep_in).indices.sort().values

    pruned_core = core[:, idx_out][:, :, idx_in]
    pruned_factors: FactorDict = {
        "U_expert": factors.get("U_expert"),
        "U_out": factors["U_out"][:, idx_out] if factors.get("U_out") is not None else None,
        "U_in": factors["U_in"][:, idx_in] if factors.get("U_in") is not None else None,
    }
    return pruned_core, pruned_factors
