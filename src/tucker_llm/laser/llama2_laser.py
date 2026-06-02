from copy import deepcopy

import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker

from .abstract_laser import AbstractLaser
from .matrix_utils import do_low_rank, sorted_mat, prune
from .gptj_laser import (
    _normalize_device,
    _clear_cuda_cache,
    _parse_partial_tucker_result,
    _maybe_quantize_tucker_core_factors,
    _tucker_storage_bits,
    _attach_tensorllm_stats,
)


class LLAMA2Laser(AbstractLaser):
    n_embd = 4096
    n_heads = 32
    head_dim = n_embd // n_heads

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_name(name):
        if name == "k_proj":
            converted_names = "self_attn.k_proj.weight"
        elif name == "q_proj":
            converted_names = "self_attn.q_proj.weight"
        elif name == "v_proj":
            converted_names = "self_attn.v_proj.weight"
        elif name == "out_proj":
            converted_names = "self_attn.o_proj.weight"
        elif name == "fc_in":
            converted_names = "mlp.gate_proj.weight"
        elif name == "fc_up":
            converted_names = "mlp.up_proj.weight"
        elif name == "fc_out":
            converted_names = "mlp.down_proj.weight"
        elif name == "None":
            converted_names = "None"
        elif name == "mlp":
            converted_names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        elif name == "attn":
            converted_names = [
                "self_attn.k_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            ]
        elif name == "all":
            converted_names = [
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            ]
        else:
            raise AssertionError(f"Unhandled name {name}")
        return converted_names

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):
        if lnum_to_modify != -1 and not name.startswith(f"model.layers.{lnum_to_modify}."):
            return False
        if isinstance(converted_names, list):
            return any(name.endswith(converted_name) for converted_name in converted_names)
        if isinstance(converted_names, str):
            return name.endswith(converted_names)
        raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):
        model_edit = model if in_place else deepcopy(model)
        if lname == "dont":
            print("Not intervening at all")
            return model_edit

        converted_names = LLAMA2Laser.convert_name(lname)
        num_update = 0
        for name, param in model.named_parameters():
            if not LLAMA2Laser._modify_layer(name, lnum, lname, converted_names):
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

            LLAMA2Laser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters Llama: {lnum}, {lname}"
        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")
        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, (
                f"Was supposed to make 1 update to the model but instead made {num_update} updates."
            )
        return model_edit

    @staticmethod
    def get_QKVO_tensor(model, lnum):
        layer = model.model.layers[lnum]
        stacked_tensor = [
            layer.self_attn.q_proj.weight.detach().contiguous().view(LLAMA2Laser.n_embd, LLAMA2Laser.n_heads, LLAMA2Laser.head_dim),
            layer.self_attn.k_proj.weight.detach().contiguous().view(LLAMA2Laser.n_embd, LLAMA2Laser.n_heads, LLAMA2Laser.head_dim),
            layer.self_attn.v_proj.weight.detach().contiguous().view(LLAMA2Laser.n_embd, LLAMA2Laser.n_heads, LLAMA2Laser.head_dim),
            layer.self_attn.o_proj.weight.detach().T.contiguous().view(LLAMA2Laser.n_embd, LLAMA2Laser.n_heads, LLAMA2Laser.head_dim),
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
        new_reshape=False,
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
        layer = model_edit.model.layers[lnum]
        original_dtype = layer.self_attn.q_proj.weight.dtype

        tl.set_backend("pytorch")
        _clear_cuda_cache(device)

        tensorly_tensor = (
            LLAMA2Laser.get_QKVO_tensor(model_edit, lnum)
            .detach()
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

        if qkvo_intervention == "partial_tucker":
            print("=" * 50)
            print("Partial Tucker decomposition")
            assert qkvo_rank <= LLAMA2Laser.head_dim, f"rank exceeds head_dim. head_dim={LLAMA2Laser.head_dim}, rank={qkvo_rank}"
            rank = [qkvo_rank * LLAMA2Laser.n_heads, qkvo_rank, stack_rank]
        elif qkvo_intervention == "partial_tucker_v2":
            print("=" * 50)
            print("Partial Tucker decomposition v2")
            assert qkvo_rank * LLAMA2Laser.n_heads < LLAMA2Laser.head_dim, f"rank exceeds head_dim. head_dim={LLAMA2Laser.head_dim}, rank={qkvo_rank * LLAMA2Laser.n_heads}"
            rank = [qkvo_rank, qkvo_rank * LLAMA2Laser.n_heads, stack_rank]
        elif qkvo_intervention == "partial_tucker_v3":
            print("=" * 50)
            print("Partial Tucker decomposition v3")
            assert qkvo_rank * (LLAMA2Laser.n_heads // 2) <= LLAMA2Laser.head_dim, f"rank exceeds head_dim. head_dim={LLAMA2Laser.head_dim}, rank={qkvo_rank * (LLAMA2Laser.n_heads // 2)}"
            rank = [qkvo_rank, qkvo_rank * (LLAMA2Laser.n_heads // 2), stack_rank]
        elif qkvo_intervention == "partial_tucker_v4":
            print("=" * 50)
            print("Partial Tucker decomposition v4")
            assert qkvo_rank * (LLAMA2Laser.n_heads * 2) <= LLAMA2Laser.n_embd, f"rank exceeds hidden_dim. hidden_dim={LLAMA2Laser.n_embd}, rank={qkvo_rank * (LLAMA2Laser.n_heads * 2)}"
            rank = [qkvo_rank * (LLAMA2Laser.n_heads * 2), qkvo_rank, stack_rank]
        elif qkvo_intervention == "partial_tucker_v5":
            print("=" * 50)
            print("Partial Tucker decomposition v5")
            assert qkvo_rank <= LLAMA2Laser.n_embd, f"qkvo_rank={qkvo_rank}, hidden dim = {LLAMA2Laser.n_embd}"
            assert head_dim_rank <= LLAMA2Laser.head_dim, f"head_dim_rank={head_dim_rank}, head dim = {LLAMA2Laser.head_dim}"
            rank = [qkvo_rank, head_dim_rank, stack_rank]
        else:
            raise ValueError(f"Unhandled qkvo_intervention={qkvo_intervention}")

        result = partial_tucker(
            tensorly_tensor,
            modes=[0, 2, 3],
            rank=rank,
            init="svd",
            svd="randomized_svd",
            random_state=0,
            tol=1e-5,
            verbose=True,
        )
        print("=" * 50)

        (core, factors), _ = _parse_partial_tucker_result(result)
        core, factors = _maybe_quantize_tucker_core_factors(core, factors, bits=tucker_quant_bits, method=tucker_quant_method)
        reconstructed_tensor_qkvo = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1], factors[2]], modes=[0, 2, 3])
        reconstruction_error = torch.norm(reconstructed_tensor_qkvo - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f"Reconstruction error: {reconstruction_error}")
        reconstructed_tensor_qkvo = reconstructed_tensor_qkvo.to(dtype=original_dtype)

        layer.self_attn.q_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 0].reshape(LLAMA2Laser.n_embd, LLAMA2Laser.n_embd).contiguous())
        layer.self_attn.k_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 1].reshape(LLAMA2Laser.n_embd, LLAMA2Laser.n_embd).contiguous())
        layer.self_attn.v_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 2].reshape(LLAMA2Laser.n_embd, LLAMA2Laser.n_embd).contiguous())
        layer.self_attn.o_proj.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 3].reshape(LLAMA2Laser.n_embd, LLAMA2Laser.n_embd).T.contiguous())

        storage_bits = _tucker_storage_bits(core, factors, quant_bits=tucker_quant_bits, value_bits=tucker_storage_bits, scale_bits=tucker_scale_bits)
        dense_bits = int(4 * LLAMA2Laser.n_embd * LLAMA2Laser.n_embd * int(tucker_storage_bits))
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
