from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.dense.llama_dense_loader import load_weights
from core.models.dense.llama_dense_model import ModelConfig, Transformer
from core.utils.constants import MODEL_NAME
from core.utils.device import DEVICE, DTYPE
from core.utils.generate import generate
from core.utils.sanity_check_test import get_sanity_check_input_ids

print(f"Downloading model weights for {MODEL_NAME}...")
model_dir = snapshot_download(repo_id=MODEL_NAME)

print("Loading dense model...")
config = ModelConfig()
model = Transformer(config)

load_weights(
    model,
    model_dir=model_dir,
    dtype=DTYPE,
    device=DEVICE,
    strict=True,
    verbose=True,
)

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = get_sanity_check_input_ids(tokenizer)
input_ids = input_ids.to(DEVICE)

print("Starting text generation...")
generate(model, tokenizer, input_ids)
