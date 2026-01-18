import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.moe.llama_moe_loader import load_moe_weights
from core.models.moe.llama_moe_model import Experts, MoEModelConfig, MoETransformer
from core.utils.constants import MODEL_NAME
from core.utils.device import DEVICE, DTYPE
from core.utils.generate import generate
from core.utils.sanity_check_test import get_sanity_check_input_ids


def recompile_dense_from_experts(experts: Experts, config: MoEModelConfig) -> dict[str, torch.Tensor]:
    """
    Reverse the slicing operation to reconstruct the original dense FFN weights
    from the first 8 sliced experts.
    """
    num_experts = config.num_sliced_experts
    dense_weights = {}

    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        moe_weight = getattr(experts, proj_name)[:num_experts]

        if proj_name == "down_proj":
            # [8, 1792, 4096] -> [4096, 14336]
            dense = moe_weight.permute(2, 0, 1).reshape(config.hidden_size, -1)
        else:
            # [8, 4096, 1792] -> [14336, 4096]
            dense = moe_weight.transpose(1, 2).reshape(-1, config.hidden_size)

        dense_weights[proj_name] = dense

    return dense_weights


def select_singular_values(
    total_singular_values: int,
    num_to_select: int,
    expert_idx: int,
    base_seed: int = 42,
) -> list[int]:
    """
    Select singular values: half biggest + half random from remaining.
    Each expert gets a different random selection.
    """
    half = num_to_select // 2

    # Top half (biggest) - same for all experts
    biggest_indices = list(range(half))

    # Random half from remaining - different per expert
    remaining_indices = list(range(half, total_singular_values))

    rng = random.Random(base_seed + expert_idx)
    random_indices = rng.sample(remaining_indices, half)

    selected = sorted(biggest_indices + random_indices)
    return selected


def create_svd_expert_weights(
    dense_weights: dict[str, torch.Tensor],
    num_svd_experts: int = 8,
    num_singular_values: int = 1792,
    base_seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """
    Create SVD-based expert weights from dense FFN weights.
    Each expert uses a different random selection of singular values.
    """
    expert_weights_list = []

    for expert_idx in range(num_svd_experts):
        expert_weights = {}

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            W = dense_weights[proj_name].float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            total_sv = len(S)

            selected = select_singular_values(
                total_singular_values=total_sv,
                num_to_select=num_singular_values,
                expert_idx=expert_idx,
                base_seed=base_seed,
            )
            selected_tensor = torch.tensor(selected, device=S.device)

            if proj_name == "down_proj":
                # Dense shape: [4096, 14336], Expert shape: [1792, 4096]
                # SVD: U [4096, 4096], S [4096], Vh [4096, 14336]
                # Expert weight = diag(S[selected]) @ U[:, selected].T
                # Shape: [1792, 4096]
                S_sel = S[selected_tensor]
                U_sel = U[:, selected_tensor]  # [4096, 1792]
                expert_weight = S_sel.unsqueeze(1) * U_sel.T  # [1792, 4096]
            else:
                # Dense shape: [14336, 4096], Expert shape: [4096, 1792]
                # SVD: U [14336, 4096], S [4096], Vh [4096, 4096]
                # Expert weight = Vh[selected, :].T @ diag(S[selected])
                # Shape: [4096, 1792]
                S_sel = S[selected_tensor]
                Vh_sel = Vh[selected_tensor, :]  # [1792, 4096]
                expert_weight = Vh_sel.T * S_sel  # [4096, 1792]

            expert_weights[proj_name] = expert_weight.to(DTYPE)

        expert_weights_list.append(expert_weights)

    return expert_weights_list


def assign_svd_experts_to_model(
    model: MoETransformer,
    layer_idx: int,
    svd_expert_weights: list[dict[str, torch.Tensor]],
    config: MoEModelConfig,
) -> None:
    """
    Assign SVD expert weights to expert slots 8-15 for a given layer.
    """
    experts = model.layers[layer_idx].ffn.experts  # type: ignore[union-attr]
    num_sliced = config.num_sliced_experts

    for expert_idx, expert_weights in enumerate(svd_expert_weights):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            target_idx = num_sliced + expert_idx
            getattr(experts, proj_name)[target_idx].copy_(expert_weights[proj_name])


# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

config = MoEModelConfig(routing="match_dense")
model = MoETransformer(config)

# Load dense weights
load_res = load_moe_weights(
    model,
    safetensors_path=str(Path(__file__).parent.joinpath("../../../artifacts/llama3_8b_moe.safetensors")),
    dtype=DTYPE,
    device=DEVICE,
    strict=True,
    verbose=True,
)
sd = load_res["state_dict"]

print("Creating SVD-based experts (indices 8-...)...")
with torch.no_grad():
    for layer_idx in range(config.num_hidden_layers):
        experts = model.layers[layer_idx].ffn.experts  # type: ignore[union-attr]

        dense_weights = recompile_dense_from_experts(experts, config)

        svd_expert_weights = create_svd_expert_weights(
            dense_weights=dense_weights,
            num_svd_experts=8,
            num_singular_values=config.expert_intermediate_size,  # 1792
            base_seed=42 + layer_idx,  # Different seed per layer
        )

        assign_svd_experts_to_model(model, layer_idx, svd_expert_weights, config)
        print(f"  Layer {layer_idx}: SVD experts created and assigned.")

print("All layers converted.")

# Test with text generation
print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("Starting text generation (using sliced experts only in match_dense mode)...")
generate(model, tokenizer, input_ids)

# Save the model with both sliced and SVD experts
print("Saving MoE version with SVD experts...")
save_file(
    model.state_dict(),
    Path(__file__).parent.joinpath("../../../../artifacts/llama3_8b_moe_svd.safetensors"),
)
print("Done.")
