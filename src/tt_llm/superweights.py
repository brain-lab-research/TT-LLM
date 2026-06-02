from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WeightCoordinate:
    module_name: str
    layer_idx: int
    row: int
    col: int


@dataclass
class ActivationMaximum:
    layer_idx: int
    capture: str
    value: float
    abs_value: float
    batch_idx: int
    token_idx: int
    channel_idx: int
    full_index: tuple[int, ...]


@dataclass
class SuperweightCandidate:
    module_name: str
    layer_idx: int
    row: int
    col: int
    input_max: ActivationMaximum
    output_max: ActivationMaximum
    weight_value: float | None = None


_DEFAULT_LAYER_PATHS = ("model.layers", "transformer.h")


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def resolve_transformer_layers(model: nn.Module, layer_path: str | None = None):
    if layer_path is not None:
        return layer_path, resolve_module(model, layer_path)

    for candidate in _DEFAULT_LAYER_PATHS:
        try:
            return candidate, resolve_module(model, candidate)
        except AttributeError:
            continue

    raise ValueError(
        "Could not infer transformer layer path. Pass layer_path explicitly, for example 'model.layers'."
    )


def infer_input_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass

    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_weight_module(model: nn.Module, coord: WeightCoordinate) -> nn.Module:
    layer_path, _ = resolve_transformer_layers(model)
    module = resolve_module(model, f"{layer_path}.{coord.layer_idx}.{coord.module_name}")
    if not hasattr(module, "weight"):
        raise ValueError(f"Module {coord.module_name} has no weight parameter")
    return module


@torch.no_grad()
def get_weight_value(model: nn.Module, coord: WeightCoordinate) -> float:
    module = _resolve_weight_module(model, coord)
    return float(module.weight.data[coord.row, coord.col].item())


@torch.no_grad()
def set_weight_value(model: nn.Module, coord: WeightCoordinate, value: float) -> None:
    module = _resolve_weight_module(model, coord)
    module.weight.data[coord.row, coord.col] = value


@torch.no_grad()
def capture_layerwise_maxima(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    module_name: str = "mlp.down_proj",
    layer_path: str | None = None,
    capture: str = "output",
    use_abs: bool = True,
) -> List[ActivationMaximum]:
    if capture not in {"input", "output"}:
        raise ValueError(f"capture must be 'input' or 'output', got {capture!r}")

    layer_path, layers = resolve_transformer_layers(model, layer_path)
    _ = layer_path
    layerwise: Dict[int, ActivationMaximum] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs, output):
            tensor = inputs[0] if capture == "input" else output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor hook payload, got {type(tensor)!r}")

            work = tensor.detach()
            flat = work.abs().reshape(-1) if use_abs else work.reshape(-1)
            max_abs, flat_idx = torch.max(flat.float(), dim=0)
            unraveled = torch.unravel_index(flat_idx, work.shape)
            value = float(work[unraveled].float().cpu().item())
            index_tuple = tuple(int(i.item()) for i in unraveled)
            if len(index_tuple) == 2:
                batch_idx, channel_idx = index_tuple
                token_idx = 0
            elif len(index_tuple) == 3:
                batch_idx, token_idx, channel_idx = index_tuple
            else:
                raise ValueError(
                    f"Expected 2D or 3D activation tensor for hook analysis, got shape {tuple(work.shape)}"
                )

            layerwise[layer_idx] = ActivationMaximum(
                layer_idx=int(layer_idx),
                capture=capture,
                value=value,
                abs_value=float(max_abs.item()),
                batch_idx=int(batch_idx),
                token_idx=int(token_idx),
                channel_idx=int(channel_idx),
                full_index=index_tuple,
            )

        return hook

    for layer_idx, layer in enumerate(layers):
        target = layer if module_name in {"", "layer", None} else resolve_module(layer, module_name)
        hooks.append(target.register_forward_hook(make_hook(int(layer_idx))))

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(infer_input_device(model)) for k, v in encoded.items()}

    model.eval()
    model(**encoded)

    for handle in hooks:
        handle.remove()

    return [layerwise[i] for i in sorted(layerwise)]


@torch.no_grad()
def find_superweight_candidates(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    module_name: str = "mlp.down_proj",
    layer_path: str | None = None,
    min_abs_value: float = 50.0,
    require_same_token: bool = True,
) -> List[SuperweightCandidate]:
    inputs = capture_layerwise_maxima(
        model,
        tokenizer,
        prompt,
        module_name=module_name,
        layer_path=layer_path,
        capture="input",
    )
    outputs = capture_layerwise_maxima(
        model,
        tokenizer,
        prompt,
        module_name=module_name,
        layer_path=layer_path,
        capture="output",
    )

    output_by_layer = {item.layer_idx: item for item in outputs}
    candidates: List[SuperweightCandidate] = []

    for input_max in inputs:
        output_max = output_by_layer.get(input_max.layer_idx)
        if output_max is None:
            continue
        if input_max.abs_value < float(min_abs_value) or output_max.abs_value < float(min_abs_value):
            continue
        if require_same_token and (
            input_max.batch_idx != output_max.batch_idx or input_max.token_idx != output_max.token_idx
        ):
            continue

        coord = WeightCoordinate(
            module_name=module_name,
            layer_idx=int(input_max.layer_idx),
            row=int(output_max.channel_idx),
            col=int(input_max.channel_idx),
        )
        candidates.append(
            SuperweightCandidate(
                module_name=module_name,
                layer_idx=int(input_max.layer_idx),
                row=int(output_max.channel_idx),
                col=int(input_max.channel_idx),
                input_max=input_max,
                output_max=output_max,
                weight_value=get_weight_value(model, coord),
            )
        )

    return candidates


@torch.no_grad()
def find_superweights_iterative(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    module_name: str = "mlp.down_proj",
    layer_path: str | None = None,
    min_abs_value: float = 50.0,
    require_same_token: bool = True,
    max_superweights: int = 8,
    select: str = "earliest",
) -> Dict[WeightCoordinate, float]:
    """
    Iteratively identify and temporarily prune superweights.
    """
    if select not in {"earliest", "largest_input", "largest_output", "largest_joint"}:
        raise ValueError(f"Unsupported select mode: {select}")

    found: Dict[WeightCoordinate, float] = {}

    def sort_key(candidate: SuperweightCandidate):
        if select == "earliest":
            return (candidate.layer_idx, -candidate.input_max.abs_value, -candidate.output_max.abs_value)
        if select == "largest_input":
            return (-candidate.input_max.abs_value, candidate.layer_idx)
        if select == "largest_output":
            return (-candidate.output_max.abs_value, candidate.layer_idx)
        return (-max(candidate.input_max.abs_value, candidate.output_max.abs_value), candidate.layer_idx)

    try:
        for _ in range(int(max_superweights)):
            candidates = find_superweight_candidates(
                model,
                tokenizer,
                prompt,
                module_name=module_name,
                layer_path=layer_path,
                min_abs_value=min_abs_value,
                require_same_token=require_same_token,
            )
            candidates = [
                candidate
                for candidate in candidates
                if WeightCoordinate(candidate.module_name, candidate.layer_idx, candidate.row, candidate.col) not in found
            ]
            if not candidates:
                break

            chosen = sorted(candidates, key=sort_key)[0]
            coord = WeightCoordinate(chosen.module_name, chosen.layer_idx, chosen.row, chosen.col)
            value = get_weight_value(model, coord)
            found[coord] = value
            set_weight_value(model, coord, 0.0)
    finally:
        restore_superweights(model, found)

    return found


@torch.no_grad()
def prune_superweights(model: nn.Module, coords_to_values: Mapping[WeightCoordinate, float]) -> None:
    for coord in coords_to_values:
        set_weight_value(model, coord, 0.0)


@torch.no_grad()
def restore_superweights(model: nn.Module, coords_to_values: Mapping[WeightCoordinate, float]) -> None:
    for coord, value in coords_to_values.items():
        set_weight_value(model, coord, float(value))


@torch.no_grad()
def scale_superweights(
    model: nn.Module,
    coords_to_values: Mapping[WeightCoordinate, float],
    scale: float,
) -> None:
    for coord, value in coords_to_values.items():
        set_weight_value(model, coord, float(value) * float(scale))


@torch.no_grad()
def collect_topk_weight_coordinates(
    model: nn.Module,
    *,
    k: int,
    weight_suffix: str = ".weight",
) -> List[tuple[str, int, float]]:
    """
    Return the global top-k absolute-magnitude weight coordinates across the model.

    Each entry is ``(parameter_name, flat_index, value)``.
    """
    if k <= 0:
        return []

    device = None
    kept_values = torch.empty(0)
    kept_param_ids = torch.empty(0, dtype=torch.long)
    kept_flat_indices = torch.empty(0, dtype=torch.long)
    names: List[str] = []
    params: List[torch.Tensor] = []

    for param_id, (name, param) in enumerate(model.named_parameters()):
        if not name.endswith(weight_suffix):
            continue
        flat = param.data.detach().abs().reshape(-1)
        if flat.numel() == 0:
            continue
        local_k = min(int(k), flat.numel())
        values, indices = torch.topk(flat, k=local_k)
        if device is None:
            device = values.device
            kept_values = torch.empty(0, device=device)
            kept_param_ids = torch.empty(0, dtype=torch.long, device=device)
            kept_flat_indices = torch.empty(0, dtype=torch.long, device=device)
        kept_values = torch.cat([kept_values, values])
        kept_param_ids = torch.cat([kept_param_ids, torch.full_like(indices, int(param_id), dtype=torch.long)])
        kept_flat_indices = torch.cat([kept_flat_indices, indices.to(dtype=torch.long)])
        names.append(name)
        params.append(param.data)

        if kept_values.numel() > 4 * int(k):
            top_values, top_idx = torch.topk(kept_values, k=int(k))
            kept_values = top_values
            kept_param_ids = kept_param_ids[top_idx]
            kept_flat_indices = kept_flat_indices[top_idx]

    if kept_values.numel() == 0:
        return []

    final_k = min(int(k), kept_values.numel())
    top_values, top_idx = torch.topk(kept_values, k=final_k)
    param_ids = kept_param_ids[top_idx].tolist()
    flat_indices = kept_flat_indices[top_idx].tolist()

    result: List[tuple[str, int, float]] = []
    for value, param_id, flat_index in zip(top_values.tolist(), param_ids, flat_indices):
        name = [n for i, n in enumerate(names) if i == param_id]
        if not name:
            name = [n for i, (n, p) in enumerate(model.named_parameters()) if i == param_id]
        result.append((name[0], int(flat_index), float(value)))
    return result


@torch.no_grad()
def layerwise_profile_table(maxima: Sequence[ActivationMaximum]):
    import pandas as pd

    rows = []
    for item in maxima:
        rows.append(
            {
                "layer_idx": item.layer_idx,
                "capture": item.capture,
                "value": item.value,
                "abs_value": item.abs_value,
                "batch_idx": item.batch_idx,
                "token_idx": item.token_idx,
                "channel_idx": item.channel_idx,
                "full_index": item.full_index,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def candidates_table(candidates: Sequence[SuperweightCandidate]):
    import pandas as pd

    rows = []
    for item in candidates:
        rows.append(
            {
                "module_name": item.module_name,
                "layer_idx": item.layer_idx,
                "row": item.row,
                "col": item.col,
                "weight_value": item.weight_value,
                "input_abs_value": item.input_max.abs_value,
                "output_abs_value": item.output_max.abs_value,
                "token_idx": item.input_max.token_idx,
                "input_channel": item.input_max.channel_idx,
                "output_channel": item.output_max.channel_idx,
            }
        )
    return pd.DataFrame(rows)
