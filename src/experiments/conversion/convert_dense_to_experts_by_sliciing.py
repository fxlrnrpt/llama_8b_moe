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

print(f"Downloading model weights for {MODEL_NAME}...")
model_dir = snapshot_download(repo_id=MODEL_NAME)

config = MoEModelConfig(routing="match_dense")
model = MoETransformer(config)

load_res = load_weights(model, model_dir=model_dir, dtype=DTYPE, device=DEVICE, strict=False, verbose=True)
sd = load_res["state_dict"]

# Convert dense FFN weights to sliced MoE weights
num_experts = config.num_sliced_experts
expert_size = config.expert_intermediate_size

with torch.no_grad():
    dense_ffns_converted = 0
    for layer_idx in range(config.num_hidden_layers):
        sliced_ffn = model.layers[layer_idx].ffn.experts  # type: ignore[missing-attribute]

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            dense_key = f"layers.{layer_idx}.ffn.{proj_name}.weight"
            dense_weight = sd[dense_key]

            if proj_name == "down_proj":
                # [hidden, intermediate] -> [num_experts, expert_size, hidden]
                moe_weight = dense_weight.reshape(config.hidden_size, num_experts, expert_size)
                moe_weight = moe_weight.permute(1, 2, 0)
            else:
                # [intermediate, hidden] -> [num_experts, hidden, expert_size]
                moe_weight = dense_weight.reshape(num_experts, expert_size, config.hidden_size)
                moe_weight = moe_weight.transpose(1, 2)

            getattr(sliced_ffn, proj_name)[:num_experts].copy_(moe_weight)
        dense_ffns_converted += 1

    print(f"Converted {dense_ffns_converted} dense FFNs to sliced MoE FFNs.")

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("Starting text generation...")
generate(model, tokenizer, input_ids)

print("Saving MoE version...")
save_file(model.state_dict(), Path(__file__).parent.joinpath("../../../artifacts/llama3_8b_moe.safetensors"))
print("Done.")
