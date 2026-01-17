import torch
from transformers import PreTrainedTokenizer

from core.models.dense.llama_dense_model import Transformer


def generate(model: Transformer, tokenizer: PreTrainedTokenizer, initial_input_ids: torch.Tensor):
    input_ids = initial_input_ids

    with torch.no_grad():
        while True:
            logits = model.forward(input_ids)

            predicted_token_id = torch.argmax(logits[0, -1, :]).item()
            predicted_token = tokenizer.decode([predicted_token_id])

            print(predicted_token, end="", flush=True)
            if predicted_token_id == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]], device=input_ids.device)], dim=-1)
