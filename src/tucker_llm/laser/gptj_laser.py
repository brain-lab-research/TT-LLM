import gc
from copy import deepcopy

import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker

from .abstract_laser import AbstractLaser
from .matrix_utils import do_low_rank, sorted_mat, prune


def _normalize_device(device):
    device = str(device)
    if device == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _clear_cuda_cache(device):
    device = str(device)
    if torch.cuda.is_available() and device.startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    gc.collect()


def _parse_partial_tucker_result(result):
    # TensorLy versions differ:
    #   (core, factors)
    #   ((core, factors), rec_errors)
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], tuple)
        and len(result[0]) == 2
    ):
        return result[0], result[1]
    return result, None


def _fake_quant_tensor_symmetric(x: torch.Tensor, bits: int):
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"Invalid quantization bits: {bits}")

    max_abs = x.detach().abs().max()
    if float(max_abs) == 0.0:
        return torch.zeros_like(x)

    scale = max_abs / float(qmax)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return (q * scale).to(device=x.device, dtype=x.dtype)


def _maybe_quantize_tucker_core_factors(core, factors, *, bits=None, method="rtn_symmetric"):
    if bits is None:
        return core, factors
    if method != "rtn_symmetric":
        raise ValueError(f"Unsupported Tucker quantization method: {method}")

    core = _fake_quant_tensor_symmetric(core, int(bits))
    factors = [_fake_quant_tensor_symmetric(f, int(bits)) for f in factors]
    return core, factors


def _tucker_storage_bits(core, factors, *, quant_bits=None, value_bits=16, scale_bits=16):
    tensors = [core] + list(factors)
    if quant_bits is None:
        return int(sum(t.numel() for t in tensors) * int(value_bits))

    # One scale per stored tensor: core + each factor.
    return int(sum(t.numel() * int(quant_bits) + int(scale_bits) for t in tensors))


def _attach_tensorllm_stats(model, layer, stats):
    records = list(getattr(model, "_tensorllm_decomp_stats", []))
    records.append(stats)
    setattr(model, "_tensorllm_decomp_stats", records)
    setattr(layer, "_tensorllm_decomp_stats", stats)


class GPTJLaser(AbstractLaser):
    n_heads = 16
    hidden_size = 4096
    head_dim = hidden_size // n_heads

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_name(name):
        if name == "k_proj":
            converted_names = "attn.k_proj.weight"
        elif name == "q_proj":
            converted_names = "attn.q_proj.weight"
        elif name == "v_proj":
            converted_names = "attn.v_proj.weight"
        elif name == "out_proj":
            converted_names = "attn.out_proj.weight"
        elif name == "fc_in":
            converted_names = "mlp.fc_in.weight"
        elif name == "fc_out":
            converted_names = "mlp.fc_out.weight"
        elif name == "None":
            converted_names = "None"
        elif name == "mlp":
            converted_names = ["mlp.fc_in.weight", "mlp.fc_out.weight"]
        elif name == "attn":
            converted_names = [
                "attn.k_proj.weight",
                "attn.q_proj.weight",
                "attn.v_proj.weight",
                "attn.out_proj.weight",
            ]
        elif name == "all":
            converted_names = [
                "attn.k_proj.weight",
                "attn.q_proj.weight",
                "attn.v_proj.weight",
                "attn.out_proj.weight",
                "mlp.fc_in.weight",
                "mlp.fc_out.weight",
            ]
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_names

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):
        if lnum_to_modify != -1 and not name.startswith(f"transformer.h.{lnum_to_modify}."):
            return False

        if isinstance(converted_names, list):
            return any(name.endswith(converted_name) for converted_name in converted_names)
        if isinstance(converted_names, str):
            return name.endswith(converted_names)

        raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

    @staticmethod
    def get_3D_Tucker_edited_model(
        model,
        lnum,
        device,
        qkvo_rank,
        attention_matrix,
        in_place=True,
        tucker_quant_bits=None,
        tucker_quant_method="rtn_symmetric",
        tucker_storage_bits=16,
        tucker_scale_bits=16,
    ):
        device = _normalize_device(device)
        model_edit = model if in_place else deepcopy(model)
        layer = model_edit.transformer.h[lnum]
        original_dtype = layer.attn.q_proj.weight.dtype

        if attention_matrix == "Q":
            print("Extracting weight Q")
            tensor = layer.attn.q_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            )
        elif attention_matrix == "K":
            print("Extracting weight K")
            tensor = layer.attn.k_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            )
        elif attention_matrix == "V":
            print("Extracting weight V")
            tensor = layer.attn.v_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            )
        elif attention_matrix == "O":
            print("Extracting weight O")
            tensor = layer.attn.out_proj.weight.detach().T.contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            )
        else:
            raise ValueError(f"Unknown attention_matrix={attention_matrix}")

        tl.set_backend("pytorch")
        _clear_cuda_cache(device)

        # TensorLy decomposition must run in float32. Half can trigger:
        # "geqrf_cuda not implemented for 'Half'".
        tensorly_tensor = tl.tensor(
            tensor.to(device=device, dtype=torch.float32).contiguous(),
            device=device,
        )

        print("=" * 50)
        print("Partial Tucker decomposition")

        assert qkvo_rank <= GPTJLaser.head_dim, (
            f"rank exceeds head_dim. head_dim={GPTJLaser.head_dim}, rank={qkvo_rank}"
        )

        result = partial_tucker(
            tensorly_tensor,
            modes=[0, 2],
            rank=[qkvo_rank * GPTJLaser.n_heads, qkvo_rank],
            init="svd",
            svd="randomized_svd",
            random_state=0,
            tol=1e-5,
            verbose=True,
        )
        (core, factors), _ = _parse_partial_tucker_result(result)

        core, factors = _maybe_quantize_tucker_core_factors(
            core,
            factors,
            bits=tucker_quant_bits,
            method=tucker_quant_method,
        )

        reconstructed_tensor = tl.tenalg.multi_mode_dot(
            core,
            [factors[0], factors[1]],
            modes=[0, 2],
        )

        reconstruction_error = torch.norm(reconstructed_tensor - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f"Reconstruction error: {reconstruction_error}")

        reconstructed_tensor = reconstructed_tensor.to(dtype=original_dtype)

        if attention_matrix == "Q":
            print("Updating weight Q")
            layer.attn.q_proj.weight = torch.nn.Parameter(
                reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).contiguous()
            )
        elif attention_matrix == "K":
            print("Updating weight K")
            layer.attn.k_proj.weight = torch.nn.Parameter(
                reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).contiguous()
            )
        elif attention_matrix == "V":
            print("Updating weight V")
            layer.attn.v_proj.weight = torch.nn.Parameter(
                reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).contiguous()
            )
        elif attention_matrix == "O":
            print("Updating weight O")
            layer.attn.out_proj.weight = torch.nn.Parameter(
                reconstructed_tensor.reshape(GPTJLaser.hidden_size, GPTJLaser.hidden_size).T.contiguous()
            )

        storage_bits = _tucker_storage_bits(
            core,
            factors,
            quant_bits=tucker_quant_bits,
            value_bits=tucker_storage_bits,
            scale_bits=tucker_scale_bits,
        )
        dense_bits = int(GPTJLaser.hidden_size * GPTJLaser.hidden_size * int(tucker_storage_bits))

        _attach_tensorllm_stats(
            model_edit,
            layer,
            {
                "method": "Tucker3D",
                "layer": int(lnum),
                "attention_matrix": attention_matrix,
                "qkvo_rank": int(qkvo_rank),
                "storage_bits": int(storage_bits),
                "dense_bits": int(dense_bits),
                "compression_ratio": float(dense_bits / max(storage_bits, 1)),
                "quantized_before_reconstruction": tucker_quant_bits is not None,
                "tucker_quant_bits": None if tucker_quant_bits is None else int(tucker_quant_bits),
                "tucker_quant_method": None if tucker_quant_bits is None else tucker_quant_method,
                "core_shape": list(core.shape),
                "factor_shapes": [list(f.shape) for f in factors],
            },
        )

        print("=" * 50)
        return model_edit

    @staticmethod
    def get_QKVO_tensor(model, lnum):
        layer = model.transformer.h[lnum]
        stacked_tensor = [
            layer.attn.k_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            ),
            layer.attn.q_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            ),
            layer.attn.v_proj.weight.detach().contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            ),
            layer.attn.out_proj.weight.detach().T.contiguous().view(
                GPTJLaser.hidden_size,
                GPTJLaser.n_heads,
                GPTJLaser.head_dim,
            ),
        ]
        return torch.stack(stacked_tensor, dim=3)

    @staticmethod
    def get_QKVO_edited_model(
        model,
        lnum,
        device,
        qkvo_rank,
        stack_rank,
        head_dim_rank=None,
        qkvo_intervention="partial_tucker",
        logger=None,
        in_place=True,
        tucker_quant_bits=None,
        tucker_quant_method="rtn_symmetric",
        tucker_storage_bits=16,
        tucker_scale_bits=16,
    ):
        device = _normalize_device(device)
        model_edit = model if in_place else deepcopy(model)
        layer = model_edit.transformer.h[lnum]
        original_dtype = layer.attn.q_proj.weight.dtype

        QKVO_tensor = GPTJLaser.get_QKVO_tensor(model_edit, lnum)

        tl.set_backend("pytorch")
        _clear_cuda_cache(device)

        # TensorLy decomposition must run in float32. Half can trigger:
        # "geqrf_cuda not implemented for 'Half'".
        tensorly_tensor = tl.tensor(
            QKVO_tensor.to(device=device, dtype=torch.float32).contiguous(),
            device=device,
        )

        if qkvo_intervention == "partial_tucker":
            print("=" * 50)
            print("Partial Tucker decomposition")
            result = partial_tucker(
                tensorly_tensor,
                modes=[0, 2, 3],
                rank=[qkvo_rank * GPTJLaser.n_heads, qkvo_rank, stack_rank],
                init="svd",
                svd="randomized_svd",
                random_state=0,
                tol=1e-5,
                verbose=True,
            )
            print("=" * 50)

        elif qkvo_intervention == "partial_tucker_v2":
            print("=" * 50)
            print("Partial Tucker decomposition v2")
            assert qkvo_rank * GPTJLaser.n_heads <= GPTJLaser.head_dim, (
                f"head dim rank: {GPTJLaser.hidden_size}; "
                f"qkvo_rank*{GPTJLaser.n_heads}: {qkvo_rank * GPTJLaser.n_heads}"
            )
            result = partial_tucker(
                tensorly_tensor,
                modes=[0, 2, 3],
                rank=[qkvo_rank, qkvo_rank * GPTJLaser.n_heads, stack_rank],
                init="svd",
                svd="randomized_svd",
                random_state=0,
                tol=1e-5,
                verbose=True,
            )
            print("=" * 50)

        elif qkvo_intervention == "partial_tucker_v3":
            print("=" * 50)
            print("Partial Tucker decomposition v3")
            assert qkvo_rank * (GPTJLaser.n_heads // 2) <= GPTJLaser.head_dim, (
                f"head dim rank: {GPTJLaser.hidden_size}; "
                f"rank={qkvo_rank * (GPTJLaser.n_heads // 2)}"
            )
            result = partial_tucker(
                tensorly_tensor,
                modes=[0, 2, 3],
                rank=[qkvo_rank, qkvo_rank * (GPTJLaser.n_heads // 2), stack_rank],
                init="svd",
                svd="randomized_svd",
                random_state=0,
                tol=1e-5,
                verbose=True,
            )
            print("=" * 50)

        elif qkvo_intervention == "partial_tucker_v4":
            print("=" * 50)
            print("Partial Tucker decomposition v4")
            assert qkvo_rank * (GPTJLaser.n_heads * 2) < GPTJLaser.hidden_size, (
                f"hidden dim rank: {GPTJLaser.hidden_size}; "
                f"rank={qkvo_rank * (GPTJLaser.n_heads * 2)}"
            )
            result = partial_tucker(
                tensorly_tensor,
                modes=[0, 2, 3],
                rank=[qkvo_rank * (GPTJLaser.n_heads * 2), qkvo_rank, stack_rank],
                init="svd",
                svd="randomized_svd",
                random_state=0,
                tol=1e-5,
                verbose=True,
            )
            print("=" * 50)

        elif qkvo_intervention == "partial_tucker_v5":
            print("=" * 50)
            print("Partial Tucker decomposition v5")
            assert qkvo_rank <= GPTJLaser.hidden_size, (
                f"qkvo_rank={qkvo_rank}, hidden dim = {GPTJLaser.hidden_size}"
            )
            assert head_dim_rank <= GPTJLaser.head_dim, (
                f"head_dim_rank={head_dim_rank}, head dim = {GPTJLaser.head_dim}"
            )
            result = partial_tucker(
                tensorly_tensor,
                modes=[0, 2, 3],
                rank=[qkvo_rank, head_dim_rank, stack_rank],
                init="svd",
                svd="randomized_svd",
                random_state=0,
                tol=1e-5,
                verbose=True,
            )
            print("=" * 50)

        else:
            raise ValueError(f"Unhandled qkvo_intervention={qkvo_intervention}")

        (core, factors), _ = _parse_partial_tucker_result(result)

        core, factors = _maybe_quantize_tucker_core_factors(
            core,
            factors,
            bits=tucker_quant_bits,
            method=tucker_quant_method,
        )

        reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(
            core,
            [factors[0], factors[1], factors[2]],
            modes=[0, 2, 3],
        )

        reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f"Reconstruction error: {reconstruction_error}")

        reconstructed_tensor_qkvo = reconstructed_tensor_qkvo.to(dtype=original_dtype)

        layer.attn.k_proj.weight = torch.nn.Parameter(
            reconstructed_tensor_qkvo[:, :, :, 0].reshape(
                GPTJLaser.hidden_size,
                GPTJLaser.hidden_size,
            ).contiguous()
        )
        layer.attn.q_proj.weight = torch.nn.Parameter(
            reconstructed_tensor_qkvo[:, :, :, 1].reshape(
                GPTJLaser.hidden_size,
                GPTJLaser.hidden_size,
            ).contiguous()
        )
        layer.attn.v_proj.weight = torch.nn.Parameter(
            reconstructed_tensor_qkvo[:, :, :, 2].reshape(
                GPTJLaser.hidden_size,
                GPTJLaser.hidden_size,
            ).contiguous()
        )
        layer.attn.out_proj.weight = torch.nn.Parameter(
            reconstructed_tensor_qkvo[:, :, :, 3].reshape(
                GPTJLaser.hidden_size,
                GPTJLaser.hidden_size,
            ).T.contiguous()
        )

        storage_bits = _tucker_storage_bits(
            core,
            factors,
            quant_bits=tucker_quant_bits,
            value_bits=tucker_storage_bits,
            scale_bits=tucker_scale_bits,
        )
        dense_bits = int(4 * GPTJLaser.hidden_size * GPTJLaser.hidden_size * int(tucker_storage_bits))

        _attach_tensorllm_stats(
            model_edit,
            layer,
            {
                "method": "Tucker4D",
                "layer": int(lnum),
                "qkvo_rank": int(qkvo_rank),
                "head_dim_rank": None if head_dim_rank is None else int(head_dim_rank),
                "stack_rank": None if stack_rank is None else int(stack_rank),
                "tucker_type": qkvo_intervention,
                "storage_bits": int(storage_bits),
                "dense_bits": int(dense_bits),
                "compression_ratio": float(dense_bits / max(storage_bits, 1)),
                "quantized_before_reconstruction": tucker_quant_bits is not None,
                "tucker_quant_bits": None if tucker_quant_bits is None else int(tucker_quant_bits),
                "tucker_quant_method": None if tucker_quant_bits is None else tucker_quant_method,
                "core_shape": list(core.shape),
                "factor_shapes": [list(f.shape) for f in factors],
            },
        )

        return model_edit

    @staticmethod
    def get_edited_model(
        model,
        lname,
        lnum,
        rate,
        intervention="rank-reduction",
        logger=None,
        in_place=True,
    ):
        model_edit = model if in_place else deepcopy(model)

        if lname == "dont":
            print("Not intervening at all")
            return model_edit

        converted_names = GPTJLaser.convert_name(lname)
        num_update = 0

        for name, param in model.named_parameters():
            modify_flag = GPTJLaser._modify_layer(
                name=name,
                lnum_to_modify=lnum,
                lname_to_modify=lname,
                converted_names=converted_names,
            )

            if not modify_flag:
                continue

            if logger is not None:
                logger.log(f"Updating Layer: {name}")
            print(f"Updating Layer: {name}")

            if intervention == 'dropout':
                param_device = param.device
                param_dtype = param.dtype

                mat_analysis = param.detach().cpu().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = prune(mat_analysis, mat_sort, rate)
                mat_analysis = torch.from_numpy(mat_analysis).to(device=param_device, dtype=param_dtype)

            elif intervention == 'rank-reduction':
                param_device = param.device
                param_dtype = param.dtype

                mat_analysis_tensor = deepcopy(param).to(device=param_device, dtype=torch.float32)
                mat_analysis = do_low_rank(mat_analysis_tensor, (10 - rate) * 0.1)
                mat_analysis = mat_analysis.to(device=param_device, dtype=param_dtype)

            elif intervention == 'zero':
                mat_analysis = torch.zeros_like(param)

            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            GPTJLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters GPTJ: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, (
                f"Was supposed to make 1 update to the model but instead made {num_update} updates."
            )

        return model_edit
