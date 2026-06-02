import torch


class AbstractLaser:
    """Base helpers for LASER/TensorLLM weight edits."""

    @staticmethod
    def get_parameter(model, name):
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    @staticmethod
    def update_model(model, name, params):
        """Update a named parameter while preserving target device/dtype.

        This is intentionally tolerant of params being a Tensor, Parameter,
        NumPy-derived tensor, CPU tensor, or different dtype.
        """
        target = AbstractLaser.get_parameter(model, name)

        if isinstance(params, torch.nn.Parameter):
            params = params.data

        if not torch.is_tensor(params):
            params = torch.as_tensor(params)

        params = params.to(device=target.device, dtype=target.dtype)

        with torch.no_grad():
            target.copy_(params)

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
        raise NotImplementedError()
