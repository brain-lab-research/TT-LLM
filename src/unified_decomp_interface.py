from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LASERConfig:
    """Configuration for LASER (Low-rank Approximation via SVD on weight matRices)."""
    lname: str          # "q_proj", "mlp", "all", ...
    rate: float
    intervention: str = "rank-reduction"


@dataclass
class Tucker4DConfig:
    """4-D Tucker decomposition of the stacked QKVO attention tensor."""
    qkvo_rank: int
    stack_rank: int
    head_dim_rank: int
    tucker_type: str = "partial_tucker_v5"

    # Optional quantization of Tucker core/factors before dense reconstruction.
    tucker_quant_bits: Optional[int] = None
    tucker_quant_method: str = "rtn_symmetric"
    tucker_storage_bits: int = 16
    tucker_scale_bits: int = 16


@dataclass
class Tucker3DConfig:
    """3-D Tucker decomposition of a single attention projection (ablation)."""
    qkvo_rank: int
    attention_matrix: str   # "Q" | "K" | "V" | "O"

    # Optional quantization of Tucker core/factors before dense reconstruction.
    tucker_quant_bits: Optional[int] = None
    tucker_quant_method: str = "rtn_symmetric"
    tucker_storage_bits: int = 16
    tucker_scale_bits: int = 16


@dataclass
class TTConfig:
    """Tensor-Train decomposition of linear layers via TTLinear module replacement."""
    tt_rank: int
    order: int = 12
    attn_projs: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    mlp_projs: Tuple[str, ...] = ("fc_in", "fc_out")
    decompose_attn: bool = True
    decompose_mlp: bool = True
    token_chunk_size: Optional[int] = None


DecompConfig = Union[LASERConfig, Tucker4DConfig, Tucker3DConfig, TTConfig]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TensorDecompositionBase(ABC):
    """Unified interface for all tensor decomposition methods."""

    def __init__(self, device: str, logger=None):
        self.device = device
        self.logger = logger

    @abstractmethod
    def apply(self, model: nn.Module, layer_num: int) -> nn.Module:
        """Apply decomposition to the given layer; return the modified model."""

    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        """Serializable summary of this decomposition's parameters."""

    def _timed_apply(self, model: nn.Module, layer_num: int) -> Tuple[nn.Module, float]:
        t0 = time.time()
        model = self.apply(model, layer_num)
        return model, time.time() - t0


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class LASERDecomposition(TensorDecompositionBase):
    """SVD-based rank-reduction on individual weight matrices."""

    def __init__(self, config: LASERConfig, device: str, logger=None):
        super().__init__(device, logger)
        self.config = config

    def apply(self, model: nn.Module, layer_num: int) -> nn.Module:
        from src.tucker_llm.laser.LaserWrapper import LaserWrapper

        return LaserWrapper.get_edited_model(
            model=model.to(self.device),
            lname=self.config.lname,
            lnum=layer_num,
            rate=self.config.rate,
            intervention=self.config.intervention,
            logger=self.logger,
            in_place=True,
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "method": "LASER",
            "lname": self.config.lname,
            "rate": self.config.rate,
            "intervention": self.config.intervention,
        }


class Tucker4DDecomposition(TensorDecompositionBase):
    """Tucker decomposition treating QKVO as a single 4-D tensor."""

    def __init__(self, config: Tucker4DConfig, device: str, logger=None):
        super().__init__(device, logger)
        self.config = config

    def apply(self, model: nn.Module, layer_num: int) -> nn.Module:
        from src.tucker_llm.laser.LaserWrapper import LaserWrapper

        return LaserWrapper.get_QKVO_edited_model(
            model=model.to(self.device),
            lnum=layer_num,
            device=self.device,
            qkvo_rank=self.config.qkvo_rank,
            stack_rank=self.config.stack_rank,
            head_dim_rank=self.config.head_dim_rank,
            qkvo_intervention=self.config.tucker_type,
            logger=self.logger,
            in_place=True,
            tucker_quant_bits=self.config.tucker_quant_bits,
            tucker_quant_method=self.config.tucker_quant_method,
            tucker_storage_bits=self.config.tucker_storage_bits,
            tucker_scale_bits=self.config.tucker_scale_bits,
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "method": "Tucker4D",
            "qkvo_rank": self.config.qkvo_rank,
            "stack_rank": self.config.stack_rank,
            "head_dim_rank": self.config.head_dim_rank,
            "tucker_type": self.config.tucker_type,
            "tucker_quant_bits": self.config.tucker_quant_bits,
            "tucker_quant_method": self.config.tucker_quant_method,
            "tucker_storage_bits": self.config.tucker_storage_bits,
            "tucker_scale_bits": self.config.tucker_scale_bits,
        }


class Tucker3DDecomposition(TensorDecompositionBase):
    """Tucker decomposition of a single attention projection."""

    def __init__(self, config: Tucker3DConfig, device: str, logger=None):
        super().__init__(device, logger)
        self.config = config

    def apply(self, model: nn.Module, layer_num: int) -> nn.Module:
        from src.tucker_llm.laser.LaserWrapper import LaserWrapper

        return LaserWrapper.get_3D_Tucker_edited_model(
            model=model.to(self.device),
            lnum=layer_num,
            device=self.device,
            qkvo_rank=self.config.qkvo_rank,
            attention_matrix=self.config.attention_matrix,
            logger=self.logger,
            in_place=True,
            tucker_quant_bits=self.config.tucker_quant_bits,
            tucker_quant_method=self.config.tucker_quant_method,
            tucker_storage_bits=self.config.tucker_storage_bits,
            tucker_scale_bits=self.config.tucker_scale_bits,
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "method": "Tucker3D",
            "qkvo_rank": self.config.qkvo_rank,
            "attention_matrix": self.config.attention_matrix,
            "tucker_quant_bits": self.config.tucker_quant_bits,
            "tucker_quant_method": self.config.tucker_quant_method,
            "tucker_storage_bits": self.config.tucker_storage_bits,
            "tucker_scale_bits": self.config.tucker_scale_bits,
        }


class TTDecomposition(TensorDecompositionBase):
    """Tensor-Train decomposition using TTLinear module replacement.

    Designed for GPTJ-style models:
        model.transformer.h[i].attn
        model.transformer.h[i].mlp
    """

    def __init__(self, config: TTConfig, device: str, logger=None):
        super().__init__(device, logger)
        self.config = config

    def apply(self, model: nn.Module, layer_num: int) -> nn.Module:
        from src.tt_llm.tt_linear import TTLinear

        model = model.to(self.device)
        layer = model.transformer.h[layer_num]

        if self.config.decompose_attn:
            for proj_name in self.config.attn_projs:
                linear = getattr(layer.attn, proj_name)
                setattr(
                    layer.attn,
                    proj_name,
                    TTLinear.from_linear(
                        linear,
                        tt_rank=self.config.tt_rank,
                        order=self.config.order,
                        output_device=self.device,
                        token_chunk_size=self.config.token_chunk_size,
                    ),
                )

        if self.config.decompose_mlp:
            for proj_name in self.config.mlp_projs:
                linear = getattr(layer.mlp, proj_name)
                setattr(
                    layer.mlp,
                    proj_name,
                    TTLinear.from_linear(
                        linear,
                        tt_rank=self.config.tt_rank,
                        order=self.config.order,
                        output_device=self.device,
                        token_chunk_size=self.config.token_chunk_size,
                    ),
                )

        return model

    def config_dict(self) -> Dict[str, Any]:
        return {
            "method": "TT",
            "tt_rank": self.config.tt_rank,
            "order": self.config.order,
            "attn_projs": list(self.config.attn_projs),
            "mlp_projs": list(self.config.mlp_projs),
            "decompose_attn": self.config.decompose_attn,
            "decompose_mlp": self.config.decompose_mlp,
            "token_chunk_size": self.config.token_chunk_size,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class DecompositionFactory:
    """Instantiate the correct decomposition class from a typed config."""

    _registry: Dict[type, type] = {
        LASERConfig: LASERDecomposition,
        Tucker4DConfig: Tucker4DDecomposition,
        Tucker3DConfig: Tucker3DDecomposition,
        TTConfig: TTDecomposition,
    }

    @classmethod
    def create(
        cls,
        config: DecompConfig,
        device: str,
        logger=None,
    ) -> TensorDecompositionBase:
        decomp_cls = cls._registry.get(type(config))
        if decomp_cls is None:
            raise ValueError(
                f"No decomposition registered for config type {type(config).__name__}. "
                f"Available: {[c.__name__ for c in cls._registry]}"
            )
        return decomp_cls(config, device, logger)

    @classmethod
    def register(cls, config_cls: type, decomp_cls: type) -> None:
        """Register a new decomposition method at runtime."""
        cls._registry[config_cls] = decomp_cls
