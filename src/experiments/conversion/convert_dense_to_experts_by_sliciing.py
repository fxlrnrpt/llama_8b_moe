import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.dense.llama_dense_loader import load_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.utils.constants import MODEL_NAME
from core.utils.device import DEVICE, DTYPE
from core.utils.generate import generate
from core.utils.sanity_check_test import get_sanity_check_input_ids

# Set random seed for reproducibility
torch.manual_seed(42)

print(f"Downloading model weights for {MODEL_NAME}...")
model_dir = snapshot_download(repo_id=MODEL_NAME)

print("Initializing MoE model...")
config = MoEModelConfig(routing="match_dense")
model = MoETransformer(config)

load_res = load_weights(model, model_dir=model_dir, dtype=DTYPE, device=DEVICE, strict=False, verbose=True)
sd = load_res["state_dict"]

# Convert dense FFN weights to sliced MoE weights
num_sliced_experts = config.num_sliced_experts
num_learned_experts = config.num_learned_experts
expert_size = config.expert_intermediate_size

# Noise scale relative to weight standard deviation
NOISE_SCALE = 0.2

with torch.no_grad():
    dense_ffns_converted = 0
    for layer_idx in range(config.num_hidden_layers):
        sliced_ffn = model.layers[layer_idx].ffn.experts  # type: ignore[missing-attribute]

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            dense_key = f"layers.{layer_idx}.ffn.{proj_name}.weight"
            dense_weight = sd[dense_key]

            # Slice original dense weights into first 8 experts (exact reconstruction)
            if proj_name == "down_proj":
                # [hidden, intermediate] -> [num_experts, expert_size, hidden]
                moe_weight = dense_weight.reshape(config.hidden_size, num_sliced_experts, expert_size)
                moe_weight = moe_weight.permute(1, 2, 0)
            else:
                # [intermediate, hidden] -> [num_experts, hidden, expert_size]
                moe_weight = dense_weight.reshape(num_sliced_experts, expert_size, config.hidden_size)
                moe_weight = moe_weight.transpose(1, 2)

            getattr(sliced_ffn, proj_name)[:num_sliced_experts].copy_(moe_weight)

            # Create noisy experts (8-15) by adding noise to dense and slicing
            weight_std = dense_weight.float().std().item()
            noise = torch.randn_like(dense_weight) * weight_std * NOISE_SCALE
            noisy_dense = dense_weight + noise

            if proj_name == "down_proj":
                noisy_moe = noisy_dense.reshape(config.hidden_size, num_learned_experts, expert_size)
                noisy_moe = noisy_moe.permute(1, 2, 0)
            else:
                noisy_moe = noisy_dense.reshape(num_learned_experts, expert_size, config.hidden_size)
                noisy_moe = noisy_moe.transpose(1, 2)

            getattr(sliced_ffn, proj_name)[num_sliced_experts:].copy_(noisy_moe)

        dense_ffns_converted += 1

    print(f"Converted {dense_ffns_converted} dense FFNs to sliced MoE FFNs.")
    print(f"  - Experts 0-{num_sliced_experts - 1}: exact slices of dense FFN")
    print(
        f"  - Experts {num_sliced_experts}-{num_sliced_experts + num_learned_experts - 1}: noisy slices (noise scale={NOISE_SCALE})"
    )

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("\n\nStarting text generation in logit match mode...\n")
generate(model, tokenizer, input_ids)

print("\n\nStarting text generation in learned_only mode...\n")
model.switch_routing("learned_only")
generate(model, tokenizer, input_ids)

print("\n\nSaving MoE version...")
save_path = Path(__file__).parent.joinpath("../../../artifacts/llama3_8b_moe.safetensors").resolve()
os.makedirs(save_path.parent, exist_ok=True)
save_file(model.state_dict(), save_path)
print("Done.")
