from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_utils import (
    boundary_tt_ranks_from_cores,
    count_tt_parameters,
    decompose_linear_weight_to_tt_cores,
    factor_int_balanced,
    normalize_modes,
    tt_cores_to_dense_weight,
)


class TTLinear(nn.Module):
    """
    Matrix-TT replacement for ``nn.Linear``.

    The dense weight ``W in R^{out x in}`` is tensorized into paired modes
    ``[out_1, in_1, ..., out_d, in_d]`` and stored as TT matrix cores of shape
    ``[r_{k-1}, out_k, in_k, r_k]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_modes: Sequence[int],
        out_modes: Sequence[int],
        tt_cores: Sequence[torch.Tensor],
        bias: torch.Tensor | None = None,
        token_chunk_size: int | None = None,
        tt_tensor_interleaved: object | None = None,
    ) -> None:
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.in_modes = normalize_modes(in_modes, self.in_features, "in_modes")
        self.out_modes = normalize_modes(out_modes, self.out_features, "out_modes")
        self.tt_tensor_interleaved = tt_tensor_interleaved
        if len(self.in_modes) != len(self.out_modes):
            raise ValueError(
                f"in_modes and out_modes must have equal length, got {len(self.in_modes)} and {len(self.out_modes)}"
            )

        self.order = len(self.in_modes)
        if len(tt_cores) != self.order:
            raise ValueError(f"Expected {self.order} TT cores, got {len(tt_cores)}")

        normalized_cores: List[torch.Tensor] = []
        for core, out_dim, in_dim in zip(tt_cores, self.out_modes, self.in_modes):
            if core.dim() != 4:
                raise ValueError(f"Each TT core must be 4D [r_prev, out_dim, in_dim, r_next], got {tuple(core.shape)}")
            if int(core.shape[1]) != int(out_dim) or int(core.shape[2]) != int(in_dim):
                raise ValueError(
                    "TT core shape does not match the provided mode sizes: "
                    f"expected (*, {out_dim}, {in_dim}, *), got {tuple(core.shape)}"
                )
            normalized_cores.append(core.contiguous())

        self.tt_cores = nn.ParameterList([nn.Parameter(core) for core in normalized_cores])
        self.token_chunk_size = None if token_chunk_size is None else int(token_chunk_size)
        if self.token_chunk_size is not None and self.token_chunk_size <= 0:
            raise ValueError(f"token_chunk_size must be positive, got {self.token_chunk_size}")

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.contiguous())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        tt_rank: int | Sequence[int],
        *,
        in_modes: Sequence[int] | None = None,
        out_modes: Sequence[int] | None = None,
        order: int = 12,
        decompose_dtype: torch.dtype = torch.float64,
        decompose_device: torch.device | str = "cpu",
        output_device: torch.device | str | None = None,
        token_chunk_size: int | None = None,
        algorithm: str = "svd",
        keep_interleaved_tt_tensor: bool = False,
    ) -> "TTLinear":
        if in_modes is None:
            in_modes = factor_int_balanced(linear.in_features, order=order)
        if out_modes is None:
            out_modes = factor_int_balanced(linear.out_features, order=order)

        in_modes = normalize_modes(in_modes, linear.in_features, "in_modes")
        out_modes = normalize_modes(out_modes, linear.out_features, "out_modes")

        target_device = linear.weight.device if output_device is None else output_device
        target_dtype = linear.weight.dtype

        dense_weight = linear.weight.detach().to(device=decompose_device, dtype=decompose_dtype)
        decomp = decompose_linear_weight_to_tt_cores(
            dense_weight,
            out_modes=out_modes,
            in_modes=in_modes,
            tt_rank=tt_rank,
            algorithm=algorithm,
            return_interleaved_tt=keep_interleaved_tt_tensor,
        )
        del dense_weight

        if keep_interleaved_tt_tensor:
            tt_cores_cpu, tt_tensor_interleaved = decomp
        else:
            tt_cores_cpu = decomp
            tt_tensor_interleaved = None

        tt_cores = [
            core.to(device=target_device, dtype=target_dtype).contiguous()
            for core in tt_cores_cpu
        ]

        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().clone().to(device=target_device, dtype=target_dtype)

        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            in_modes=in_modes,
            out_modes=out_modes,
            tt_cores=tt_cores,
            bias=bias,
            token_chunk_size=token_chunk_size,
            tt_tensor_interleaved=tt_tensor_interleaved,
        )

    @property
    def tt_ranks(self) -> List[int]:
        return boundary_tt_ranks_from_cores(self.tt_cores)

    def dense_parameter_count(self) -> int:
        total = self.out_features * self.in_features
        if self.bias is not None:
            total += self.bias.numel()
        return int(total)

    def num_tt_parameters(self) -> int:
        total = count_tt_parameters(self.tt_cores)
        if self.bias is not None:
            total += self.bias.numel()
        return int(total)

    def compression_ratio(self) -> float:
        return float(self.dense_parameter_count() / self.num_tt_parameters())

    def to_dense_weight(self) -> torch.Tensor:
        return tt_cores_to_dense_weight(self.tt_cores, out_modes=self.out_modes, in_modes=self.in_modes)

    def to_linear(self) -> nn.Linear:
        dense = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.tt_cores[0].device,
            dtype=self.tt_cores[0].dtype,
        )
        with torch.no_grad():
            dense.weight.copy_(self.to_dense_weight().to(device=dense.weight.device, dtype=dense.weight.dtype))
            if self.bias is not None:
                dense.bias.copy_(self.bias)
        return dense

    def dense_reconstruction_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.to_dense_weight(), self.bias)

    def _forward_flat_tt(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D tensor [tokens, in_features], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        work = x
        core_dtype = self.tt_cores[0].dtype
        if work.dtype != core_dtype:
            work = work.to(core_dtype)

        batch = work.shape[0]
        state = work.reshape(batch, self.in_features, 1, 1)
        out_suffix = 1

        for core, in_dim in zip(reversed(self.tt_cores), reversed(self.in_modes)):
            i_prefix = state.shape[1] // int(in_dim)
            state = state.reshape(batch, i_prefix, int(in_dim), core.shape[-1], out_suffix)
            state = torch.einsum("bpirq,aoir->bpaoq", state, core)
            out_suffix *= int(core.shape[1])
            state = state.reshape(batch, i_prefix, core.shape[0], out_suffix)

        out = state.reshape(batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def true_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        flat_x = x.reshape(-1, self.in_features)
        if self.token_chunk_size is None or flat_x.shape[0] <= self.token_chunk_size:
            flat_y = self._forward_flat_tt(flat_x)
        else:
            chunks = []
            for start in range(0, flat_x.shape[0], self.token_chunk_size):
                stop = min(start + self.token_chunk_size, flat_x.shape[0])
                chunks.append(self._forward_flat_tt(flat_x[start:stop]))
            flat_y = torch.cat(chunks, dim=0)

        return flat_y.reshape(*x.shape[:-1], self.out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.true_forward(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"order={self.order}, tt_ranks={self.tt_ranks}, token_chunk_size={self.token_chunk_size}"
        )
