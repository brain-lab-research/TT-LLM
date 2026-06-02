from transformers import LlamaForCausalLM
from transformers import RobertaForMaskedLM
from transformers import GPTJForCausalLM

from .gptj_laser import GPTJLaser
from .llama2_laser import LLAMA2Laser

try:
    from .roberta_laser import RobertaLaser
except Exception:
    RobertaLaser = None


def _log(logger, msg):
    if logger is not None:
        logger.log(msg)


class LaserWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_3D_Tucker_edited_model(
        model,
        lnum,
        device,
        qkvo_rank,
        attention_matrix,
        logger=None,
        in_place=True,
        tucker_quant_bits=None,
        tucker_quant_method="rtn_symmetric",
        tucker_storage_bits=16,
        tucker_scale_bits=16,
    ):
        if isinstance(model, GPTJForCausalLM):
            _log(logger, "Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_3D_Tucker_edited_model(
                model=model,
                lnum=lnum,
                device=device,
                qkvo_rank=qkvo_rank,
                attention_matrix=attention_matrix,
                in_place=in_place,
                tucker_quant_bits=tucker_quant_bits,
                tucker_quant_method=tucker_quant_method,
                tucker_storage_bits=tucker_storage_bits,
                tucker_scale_bits=tucker_scale_bits,
            )

        if isinstance(model, RobertaForMaskedLM):
            if RobertaLaser is None:
                raise ImportError("RobertaLaser is not available")
            _log(logger, "Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_3D_Tucker_edited_model(
                model=model,
                lnum=lnum,
                device=device,
                qkvo_rank=qkvo_rank,
                attention_matrix=attention_matrix,
                in_place=in_place,
                tucker_quant_bits=tucker_quant_bits,
                tucker_quant_method=tucker_quant_method,
                tucker_storage_bits=tucker_storage_bits,
                tucker_scale_bits=tucker_scale_bits,
            )

        raise AssertionError(f"Unhandled model of type {type(model)}.")

    @staticmethod
    def get_QKVO_edited_model(
        model,
        lnum,
        device,
        qkvo_rank,
        head_dim_rank=None,
        stack_rank=None,
        new_reshape=False,
        qkvo_intervention="partial_tucker",
        logger=None,
        in_place=True,
        tucker_quant_bits=None,
        tucker_quant_method="rtn_symmetric",
        tucker_storage_bits=16,
        tucker_scale_bits=16,
    ):
        if isinstance(model, GPTJForCausalLM):
            _log(logger, "Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_QKVO_edited_model(
                model=model,
                lnum=lnum,
                device=device,
                qkvo_rank=qkvo_rank,
                stack_rank=stack_rank,
                head_dim_rank=head_dim_rank,
                qkvo_intervention=qkvo_intervention,
                logger=logger,
                in_place=in_place,
                tucker_quant_bits=tucker_quant_bits,
                tucker_quant_method=tucker_quant_method,
                tucker_storage_bits=tucker_storage_bits,
                tucker_scale_bits=tucker_scale_bits,
            )

        if isinstance(model, LlamaForCausalLM):
            _log(logger, "Editing a LlamaForCausalLM Model")
            return LLAMA2Laser.get_QKVO_edited_model(
                model=model,
                lnum=lnum,
                device=device,
                qkvo_rank=qkvo_rank,
                head_dim_rank=head_dim_rank,
                stack_rank=stack_rank,
                qkvo_intervention=qkvo_intervention,
                logger=logger,
                in_place=in_place,
                tucker_quant_bits=tucker_quant_bits,
                tucker_quant_method=tucker_quant_method,
                tucker_storage_bits=tucker_storage_bits,
                tucker_scale_bits=tucker_scale_bits,
            )

        if isinstance(model, RobertaForMaskedLM):
            if RobertaLaser is None:
                raise ImportError("RobertaLaser is not available")
            _log(logger, "Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_QKVO_edited_model(
                model=model,
                lnum=lnum,
                device=device,
                qkvo_rank=qkvo_rank,
                stack_rank=stack_rank,
                head_dim_rank=head_dim_rank,
                qkvo_intervention=qkvo_intervention,
                logger=logger,
                in_place=in_place,
                tucker_quant_bits=tucker_quant_bits,
                tucker_quant_method=tucker_quant_method,
                tucker_storage_bits=tucker_storage_bits,
                tucker_scale_bits=tucker_scale_bits,
            )

        raise AssertionError(f"Unhandled model of type {type(model)}.")

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
        if isinstance(model, LlamaForCausalLM):
            _log(logger, "Editing a LlamaForCausalLM Model")
            return LLAMA2Laser.get_edited_model(
                model=model,
                lname=lname,
                lnum=lnum,
                rate=rate,
                intervention=intervention,
                logger=logger,
                in_place=in_place,
            )

        if isinstance(model, RobertaForMaskedLM):
            if RobertaLaser is None:
                raise ImportError("RobertaLaser is not available")
            _log(logger, "Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_edited_model(
                model=model,
                lname=lname,
                lnum=lnum,
                rate=rate,
                intervention=intervention,
                logger=logger,
                in_place=in_place,
            )

        if isinstance(model, GPTJForCausalLM):
            _log(logger, "Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_edited_model(
                model=model,
                lname=lname,
                lnum=lnum,
                rate=rate,
                intervention=intervention,
                logger=logger,
                in_place=in_place,
            )

        raise AssertionError(f"Unhandled model of type {type(model)}.")
