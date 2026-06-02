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


class RobertaLaser(AbstractLaser):
    n_heads = 12
    hidden_size = 768
    head_dim = hidden_size // n_heads

    def __init__(self):
        super().__init__()

    @staticmethod
    def convert_name(name):
        if name == "k_proj":
            converted_name = "attention.self.key.weight"
        elif name == "q_proj":
            converted_name = "attention.self.query.weight"
        elif name == "v_proj":
            converted_name = "attention.self.value.weight"
        elif name == "out_proj":
            converted_name = "attention.output.dense.weight"
        elif name == "fc_in":
            converted_name = "intermediate.dense.weight"
        elif name == "fc_out":
            converted_name = "output.dense.weight"
        elif name == "mlp":
            converted_name = ["intermediate.dense.weight", "output.dense.weight"]
        elif name == "attn":
            converted_name = [
                "attention.self.key.weight",
                "attention.self.query.weight",
                "attention.self.value.weight",
                "attention.output.dense.weight",
            ]
        elif name == "all":
            converted_name = [
                "intermediate.dense.weight",
                "output.dense.weight",
                "attention.self.key.weight",
                "attention.self.query.weight",
                "attention.self.value.weight",
                "attention.output.dense.weight",
            ]
        elif name == "None":
            converted_name = "None"
        else:
            raise AssertionError(f"Unhandled name {name}")
        return converted_name

    @staticmethod
    def _modify_layer(name, converted_names, lnum_to_modify):
        # Original Roberta code used lnum=12 as all-layers sentinel.
        if lnum_to_modify != 12 and not name.startswith(f"roberta.encoder.layer.{lnum_to_modify}."):
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

        converted_names = RobertaLaser.convert_name(lname)
        num_update = 0
        for name, param in model.named_parameters():
            if not RobertaLaser._modify_layer(name=name, converted_names=converted_names, lnum_to_modify=lnum):
                continue
            if logger is not None:
                logger.log(f"Updating Layer: {name}")

            if intervention == "dropout":
                mat_analysis_np = param.detach().cpu().numpy().copy()
                mat_sort = sorted_mat(mat_analysis_np)
                mat_analysis_np = prune(mat_analysis_np, mat_sort, rate)
                mat_analysis = torch.from_numpy(mat_analysis_np)
            elif intervention == "rank-reduction":
                mat_analysis = do_low_rank(param.detach().to(torch.float32), (10 - rate) * 0.1, niter=20)
            elif intervention == "zero":
                mat_analysis = torch.zeros_like(param.detach())
            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            RobertaLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters Roberta: {lnum}, {lname}"
        return model_edit

    @staticmethod
    def get_QKVO_tensor(model, lnum):
        layer = model.roberta.encoder.layer[lnum]
        stacked_tensor = [
            layer.attention.self.key.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim),
            layer.attention.self.query.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim),
            layer.attention.self.value.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim),
            layer.attention.output.dense.weight.detach().T.contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim),
        ]
        return torch.stack(stacked_tensor, dim=3)

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
        layer = model_edit.roberta.encoder.layer[lnum]
        original_dtype = layer.attention.self.query.weight.dtype

        if attention_matrix == "Q":
            print("Extracting weight Q")
            tensor = layer.attention.self.query.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == "K":
            print("Extracting weight K")
            tensor = layer.attention.self.key.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == "V":
            print("Extracting weight V")
            tensor = layer.attention.self.value.weight.detach().contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        elif attention_matrix == "O":
            print("Extracting weight O")
            tensor = layer.attention.output.dense.weight.detach().T.contiguous().view(RobertaLaser.hidden_size, RobertaLaser.n_heads, RobertaLaser.head_dim)
        else:
            raise ValueError(f"Unknown attention_matrix={attention_matrix}")

        tl.set_backend("pytorch")
        _clear_cuda_cache(device)
        tensorly_tensor = tl.tensor(tensor.to(device=device, dtype=torch.float32).contiguous(), device=device)

        print("=" * 50)
        print("Partial Tucker decomposition")
        assert qkvo_rank <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank}"
        result = partial_tucker(
            tensorly_tensor,
            modes=[0, 2],
            rank=[qkvo_rank * RobertaLaser.n_heads, qkvo_rank],
            init="svd",
            svd="randomized_svd",
            random_state=0,
            tol=1e-5,
            verbose=True,
        )
        (core, factors), _ = _parse_partial_tucker_result(result)
        core, factors = _maybe_quantize_tucker_core_factors(core, factors, bits=tucker_quant_bits, method=tucker_quant_method)
        reconstructed_tensor = tl.tenalg.multi_mode_dot(core, [factors[0], factors[1]], modes=[0, 2])
        reconstruction_error = torch.norm(reconstructed_tensor - tensorly_tensor) / torch.norm(tensorly_tensor)
        print(f"Reconstruction error: {reconstruction_error}")
        reconstructed_tensor = reconstructed_tensor.to(dtype=original_dtype)

        if attention_matrix == "Q":
            print("Updating weight Q")
            layer.attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        elif attention_matrix == "K":
            print("Updating weight K")
            layer.attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        elif attention_matrix == "V":
            print("Updating weight V")
            layer.attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        elif attention_matrix == "O":
            print("Updating weight O")
            layer.attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor.reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).T.contiguous())

        storage_bits = _tucker_storage_bits(core, factors, quant_bits=tucker_quant_bits, value_bits=tucker_storage_bits, scale_bits=tucker_scale_bits)
        dense_bits = int(RobertaLaser.hidden_size * RobertaLaser.hidden_size * int(tucker_storage_bits))
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
        layer = model_edit.roberta.encoder.layer[lnum]
        original_dtype = layer.attention.self.query.weight.dtype

        tensorly_tensor = tl.tensor(
            RobertaLaser.get_QKVO_tensor(model_edit, lnum).to(device=device, dtype=torch.float32).contiguous(),
            device=device,
        )
        tl.set_backend("pytorch")
        _clear_cuda_cache(device)

        if qkvo_intervention == "partial_tucker":
            print("=" * 50)
            print("Partial Tucker decomposition")
            assert qkvo_rank <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank}"
            rank = [qkvo_rank * RobertaLaser.n_heads, qkvo_rank, stack_rank]
        elif qkvo_intervention == "partial_tucker_v2":
            print("=" * 50)
            print("Partial Tucker decomposition v2")
            assert qkvo_rank * RobertaLaser.n_heads < RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank * RobertaLaser.n_heads}"
            rank = [qkvo_rank, qkvo_rank * RobertaLaser.n_heads, stack_rank]
        elif qkvo_intervention == "partial_tucker_v3":
            print("=" * 50)
            print("Partial Tucker decomposition v3")
            assert qkvo_rank * (RobertaLaser.n_heads // 2) <= RobertaLaser.head_dim, f"rank exceeds head_dim. head_dim={RobertaLaser.head_dim}, rank={qkvo_rank * (RobertaLaser.n_heads // 2)}"
            rank = [qkvo_rank, qkvo_rank * (RobertaLaser.n_heads // 2), stack_rank]
        elif qkvo_intervention == "partial_tucker_v4":
            print("=" * 50)
            print("Partial Tucker decomposition v4")
            assert qkvo_rank * 24 <= RobertaLaser.hidden_size, f"rank exceeds hidden_dim. hidden_dim={RobertaLaser.hidden_size}, rank={qkvo_rank * 24}"
            rank = [qkvo_rank * 24, qkvo_rank, stack_rank]
        elif qkvo_intervention == "partial_tucker_v5":
            print("=" * 50)
            print("Partial Tucker decomposition v5")
            assert qkvo_rank <= RobertaLaser.hidden_size, f"qkvo_rank={qkvo_rank}, hidden dim = {RobertaLaser.hidden_size}"
            assert head_dim_rank <= RobertaLaser.head_dim, f"head_dim_rank={head_dim_rank}, head dim = {RobertaLaser.head_dim}"
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

        layer.attention.self.key.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 0].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        layer.attention.self.query.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 1].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        layer.attention.self.value.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 2].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).contiguous())
        layer.attention.output.dense.weight = torch.nn.Parameter(reconstructed_tensor_qkvo[:, :, :, 3].reshape(RobertaLaser.hidden_size, RobertaLaser.hidden_size).T.contiguous())

        storage_bits = _tucker_storage_bits(core, factors, quant_bits=tucker_quant_bits, value_bits=tucker_storage_bits, scale_bits=tucker_scale_bits)
        dense_bits = int(4 * RobertaLaser.hidden_size * RobertaLaser.hidden_size * int(tucker_storage_bits))
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
