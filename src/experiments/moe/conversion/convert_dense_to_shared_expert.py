from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.dense.llama_dense_loader import dense_mapping, load_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.utils.device import DEVICE, DTYPE
from core.utils.generate import generate
from core.utils.sanity_check_test import get_sanity_check_input_ids

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Downloading model weights for {model_name}...")
model_dir = snapshot_download(repo_id=model_name)

config = MoEModelConfig(toy_mode=True)
model = MoETransformer(config)

moe_mapping = {
    **dense_mapping,
    ".mlp": ".ffn.shared_expert",
}

load_weights(model, model_dir=model_dir, dtype=DTYPE, device=DEVICE, strict=False, verbose=True, mapping=moe_mapping)

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("Starting text generation...")
generate(model, tokenizer, input_ids)

print("Saving MoE version...")
save_file(model.state_dict(), Path(__file__).parent.joinpath("../../../../artifacts/llama3_8b_moe.safetensors"))
print("Done.")
