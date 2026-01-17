from audioop import add
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer

from core.models.dense.llama_dense_loader import load_weights
from core.models.dense.llama_dense_model import ModelConfig, Transformer
from core.utils.device import DEVICE, DTYPE

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print(f"Downloading model weights for {model_name}...")
model_dir = snapshot_download(repo_id=model_name)

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


system_prompt = "You are a helpful coding assistant. Answer the user's questions like a pirate."
user_prompt = "Explain the difference between a list and a tuple in Python."

conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

print("Model loaded. Downloading tokenizer...")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_ids = tokenizer.apply_chat_template(conversation, add_special_tokens=True, return_tensors="pt")
input_ids = input_ids.to(DEVICE)

print("Starting text generation...")
with torch.no_grad():
    while True:
        logits = model(input_ids)

        predicted_token_id = torch.argmax(logits[0, -1, :]).item()
        predicted_token = tokenizer.decode([predicted_token_id])

        print(predicted_token, end="", flush=True)
        if predicted_token_id == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]], device=input_ids.device)], dim=-1)
