"""FSDP (Fully Sharded Data Parallel) utilities."""

import functools

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from core.models.moe.llama_moe_model import MoETransformer, MoEBlock


def get_trainable_params(model: MoETransformer) -> list[torch.nn.Parameter]:
    """
    Configure which parameters are trainable.

    Only router gate weights and expert weights are trainable.
    Everything else (embedding, attention, norms, lm_head) is frozen.

    Args:
        model: The MoE model to configure.

    Returns:
        List of trainable parameters.
    """
    trainable_params = []

    for name, param in model.named_parameters():
        # Train only:
        # - Router gate: layers.*.ffn.router.gate.weight
        # - Expert weights: layers.*.ffn.experts.{gate_proj, up_proj, down_proj}
        if "ffn.router.gate" in name or "ffn.experts" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    return trainable_params


def register_expert_freeze_hooks(model: MoETransformer):
    """
    Register gradient hooks to freeze sliced experts (indices 0-7).

    The Experts module stores all experts (sliced + learned) in the same tensor.
    This hook zeros out gradients for the first num_sliced experts during backward.
    """
    for layer in model.layers:
        layer.ffn.experts.register_freeze_hooks()


def wrap_model_with_fsdp(
    model: MoETransformer,
    local_rank: int,
    mixed_precision: bool = True,
) -> FSDP:
    """
    Wrap model with Fully Sharded Data Parallel (FSDP).

    Args:
        model: The model to wrap.
        local_rank: Local GPU rank for device placement.
        mixed_precision: Whether to use bfloat16 mixed precision.

    Returns:
        FSDP-wrapped model.
    """
    # Mixed precision policy for bfloat16 training
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Wrap each MoEBlock separately for better memory efficiency
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MoEBlock},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,  # Required for freezing individual parameters
    )

    return model


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
):
    """
    Save a full state dict checkpoint from FSDP model.

    Gathers the full state dict to rank 0 and saves it.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer to save.
        step: Current training step.
        path: Path to save checkpoint.
    """
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()

        checkpoint = {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(checkpoint, path)
