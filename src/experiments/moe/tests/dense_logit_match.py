from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from core.models.dense.llama_dense_loader import load_weights
from core.models.dense.llama_dense_model import ModelConfig, Transformer
from core.models.moe.llama_moe_loader import load_moe_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.utils.constants import BF16_TOLERANCE_ATOL, BF16_TOLERANCE_RTOL, MODEL_NAME
from core.utils.device import DEVICE, DTYPE


def test_logit_match(moe_safetensors_path: str, num_samples: int = 10):
    model_dir = snapshot_download(repo_id=MODEL_NAME)
    dense_model = Transformer(ModelConfig())

    load_weights(
        dense_model,
        model_dir=model_dir,
        dtype=DTYPE,
        device=DEVICE,
        strict=True,
        verbose=True,
    )

    config = MoEModelConfig(toy_mode=True)
    moe_model = MoETransformer(config)
    load_moe_weights(
        moe_model,
        safetensors_path=moe_safetensors_path,
        dtype=DTYPE,
        device=DEVICE,
        strict=True,
        verbose=True,
    )

    results = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len), device=DEVICE)

        with torch.no_grad():
            dense_logits = dense_model(input_ids)
            moe_logits = moe_model(input_ids)

        max_diff = (dense_logits - moe_logits).abs().max().item()
        assert torch.allclose(dense_logits, moe_logits, atol=BF16_TOLERANCE_ATOL, rtol=BF16_TOLERANCE_RTOL), (
            f"Logit mismatch: max diff = {max_diff}"
        )

        dense_tokens = dense_logits.argmax(dim=-1)
        moe_tokens = moe_logits.argmax(dim=-1)
        assert (dense_tokens == moe_tokens).all(), "Token mismatch in greedy decode"

        results.append(max_diff)

    return results


result = test_logit_match(
    moe_safetensors_path=str(Path(__file__).parent.joinpath("../../../../artifacts/llama3_8b_moe.safetensors")),
    num_samples=5,
)
print("Logit match test passed. Max differences per sample:", result)
