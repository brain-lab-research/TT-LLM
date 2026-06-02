from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import tntorch as tn


def prime_factors(n: int) -> List[int]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    factors: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factor_int_balanced(n: int, order: int) -> List[int]:
    """
    Factor an integer into `order` positive mode sizes with roughly balanced products.
    """
    if order <= 0:
        raise ValueError(f"order must be >= 1, got {order}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if order == 1:
        return [n]

    bins = [1] * order
    for factor in sorted(prime_factors(n), reverse=True):
        idx = min(range(order), key=lambda i: bins[i])
        bins[idx] *= factor
    return sorted(int(x) for x in bins)


def normalize_modes(modes: Sequence[int], expected_product: int, name: str) -> List[int]:
    modes = [int(m) for m in modes]
    if any(m <= 0 for m in modes):
        raise ValueError(f"All {name} must be positive, got {modes}")

    product = math.prod(modes)
    if product != expected_product:
        raise ValueError(
            f"Product of {name} must be {expected_product}, got {modes} with product {product}"
        )
    return modes


def normalize_tt_ranks(tt_rank: int | Sequence[int], order: int) -> List[int]:
    """
    Normalize rank input into internal TT ranks of length `order - 1`.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order == 1:
        return []

    if isinstance(tt_rank, int):
        if tt_rank < 1:
            raise ValueError(f"tt_rank must be >= 1, got {tt_rank}")
        return [int(tt_rank)] * (order - 1)

    ranks = [int(r) for r in tt_rank]
    if any(r < 1 for r in ranks):
        raise ValueError(f"All TT ranks must be >= 1, got {ranks}")

    if len(ranks) == order - 1:
        return ranks

    if len(ranks) == order + 1:
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(
                "Full TT rank sequences must include boundary ranks and therefore start and end with 1"
            )
        return ranks[1:-1]

    raise ValueError(
        f"Expected tt_rank as int, length {order - 1}, or length {order + 1}; got {ranks}"
    )


def boundary_tt_ranks_from_cores(cores: Sequence[torch.Tensor]) -> List[int]:
    if len(cores) == 0:
        return [1, 1]
    ranks = [int(cores[0].shape[0])]
    ranks.extend(int(core.shape[-1]) for core in cores)
    return ranks


def count_tt_parameters(cores: Iterable[torch.Tensor]) -> int:
    return sum(int(core.numel()) for core in cores)


def _interleaving_permutation(order: int) -> List[int]:
    permutation: List[int] = []
    for k in range(order):
        permutation.extend([k, order + k])
    return permutation


def _deinterleaving_permutation(order: int) -> List[int]:
    return list(range(0, 2 * order, 2)) + list(range(1, 2 * order, 2))


def linear_weight_to_interleaved_tensor(
    weight: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    out_features, in_features = weight.shape
    out_modes = normalize_modes(out_modes, out_features, "out_modes")
    in_modes = normalize_modes(in_modes, in_features, "in_modes")

    if len(out_modes) != len(in_modes):
        raise ValueError(
            f"out_modes and in_modes must have equal length, got {len(out_modes)} and {len(in_modes)}"
        )

    order = len(in_modes)
    tensor = weight.reshape(*out_modes, *in_modes)
    return tensor.permute(*_interleaving_permutation(order)).contiguous()


def tt_cores_to_interleaved_tensor(
    tt_cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    if len(tt_cores) == 0:
        raise ValueError("tt_cores must be non-empty")

    flat = tt_cores[0].reshape(tt_cores[0].shape[0], -1, tt_cores[0].shape[-1]).squeeze(0)
    for core in tt_cores[1:]:
        core_flat = core.reshape(core.shape[0], -1, core.shape[-1])
        flat = torch.einsum("mr,rns->mns", flat, core_flat).reshape(
            flat.shape[0] * core_flat.shape[1],
            core_flat.shape[2],
        )

    interleaved_shape: List[int] = []
    for out_dim, in_dim in zip(out_modes, in_modes):
        interleaved_shape.extend([int(out_dim), int(in_dim)])
    return flat.squeeze(-1).reshape(*interleaved_shape)


def tt_cores_to_dense_weight(
    tt_cores: Sequence[torch.Tensor],
    out_modes: Sequence[int],
    in_modes: Sequence[int],
) -> torch.Tensor:
    order = len(in_modes)
    interleaved = tt_cores_to_interleaved_tensor(tt_cores, out_modes=out_modes, in_modes=in_modes)
    dense_tensor = interleaved.permute(*_deinterleaving_permutation(order)).contiguous()
    return dense_tensor.reshape(math.prod(out_modes), math.prod(in_modes))


def decompose_linear_weight_to_tt_cores(
    weight: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    tt_rank: int | Sequence[int],
    *,
    algorithm: str = "svd",
    return_interleaved_tt: bool = False,
) -> List[torch.Tensor]:
    out_features, in_features = weight.shape
    out_modes = normalize_modes(out_modes, out_features, "out_modes")
    in_modes = normalize_modes(in_modes, in_features, "in_modes")

    if len(out_modes) != len(in_modes):
        raise ValueError(
            f"out_modes and in_modes must have equal length, got {len(out_modes)} and {len(in_modes)}"
        )

    order = len(in_modes)

    interleaved = linear_weight_to_interleaved_tensor(
        weight,
        out_modes=out_modes,
        in_modes=in_modes,
    )
    # [o1, i1, o2, i2, ..., od, id]
    n_dims = interleaved.dim()

    if isinstance(tt_rank, int):
        ranks = [int(tt_rank)] * (n_dims - 1)
    else:
        ranks = [int(r) for r in tt_rank]
        if len(ranks) != n_dims - 1:
            raise ValueError(
                f"For interleaved decomposition, tt_rank must have length {n_dims - 1}, got {len(ranks)}"
            )

    tt_tensor = tn.Tensor(interleaved)
    tt_tensor.round_tt(rmax=ranks, algorithm=algorithm)

    tt_cores_4d: List[torch.Tensor] = []
    raw_cores = tt_tensor.cores
    if len(raw_cores) != 2 * order:
        raise ValueError(f"Expected {2 * order} TT cores, got {len(raw_cores)}")

    for k in range(order):
        out_core = raw_cores[2 * k]       # [r_{2k}, o_k, r_{2k+1}]
        in_core = raw_cores[2 * k + 1]    # [r_{2k+1}, i_k, r_{2k+2}]

        mpo_core = torch.einsum("aob,bic->aoic", out_core, in_core).contiguous()

        expected_shape = (mpo_core.shape[0], int(out_modes[k]), int(in_modes[k]), mpo_core.shape[-1])
        if mpo_core.shape[1] != expected_shape[1] or mpo_core.shape[2] != expected_shape[2]:
            raise ValueError(
                f"Unexpected merged core shape: got {tuple(mpo_core.shape)}, "
                f"expected second/third dims {(expected_shape[1], expected_shape[2])}"
            )

        tt_cores_4d.append(mpo_core)

    if return_interleaved_tt:
        return tt_cores_4d, tt_tensor
    return tt_cores_4d


def paired_tt_unfolding_ranks(
    weight: torch.Tensor,
    out_modes: Sequence[int],
    in_modes: Sequence[int],
    *,
    rtol: float = 1e-5,
) -> List[int]:
    """
    Compute the exact TT unfolding ranks for the paired matrix tensorization.
    """
    out_modes = normalize_modes(out_modes, weight.shape[0], "out_modes")
    in_modes = normalize_modes(in_modes, weight.shape[1], "in_modes")
    if len(out_modes) != len(in_modes):
        raise ValueError(
            f"out_modes and in_modes must have equal length, got {len(out_modes)} and {len(in_modes)}"
        )

    interleaved = linear_weight_to_interleaved_tensor(weight, out_modes=out_modes, in_modes=in_modes)
    paired_shape = [int(o) * int(i) for o, i in zip(out_modes, in_modes)]

    ranks: List[int] = []
    for k in range(1, len(paired_shape)):
        left = math.prod(paired_shape[:k])
        right = math.prod(paired_shape[k:])
        unfolding = interleaved.reshape(left, right)
        rank = int(torch.linalg.matrix_rank(unfolding, rtol=rtol).item())
        ranks.append(rank)
    return ranks
