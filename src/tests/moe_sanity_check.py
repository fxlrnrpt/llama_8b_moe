from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.moe.llama_moe_loader import load_moe_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.utils.constants import MODEL_NAME
from core.utils.device import DEVICE, DTYPE
from core.utils.generate import generate
from core.utils.sanity_check_test import get_sanity_check_input_ids

print("Loading MoE model...")
config = MoEModelConfig(routing="match_dense")
model = MoETransformer(config)

load_moe_weights(
    model,
    safetensors_path=str(Path(__file__).parent.joinpath("../../artifacts/llama3_8b_moe.safetensors").resolve()),
    dtype=DTYPE,
    device=DEVICE,
    strict=True,
    verbose=True,
)

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("\n\nStarting text generation (match_dense)...\n\n")
generate(model, tokenizer, input_ids)

print("\n\nStarting text generation (learned_only)...\n\n")
model.switch_routing("learned_only")
generate(model, tokenizer, input_ids)

print("\n\nStarting text generation (auto)...\n\n")
model.switch_routing("auto")
generate(model, tokenizer, input_ids)
