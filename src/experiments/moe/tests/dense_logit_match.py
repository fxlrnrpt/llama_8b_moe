from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from core.models.dense.llama_dense_loader import load_weights
from core.models.dense.llama_dense_model import ModelConfig, Transformer
from core.models.moe.llama_moe_loader import load_moe_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.utils.constants import BF16_TOLERANCE_ATOL, BF16_TOLERANCE_RTOL, MODEL_NAME
from core.utils.device import DEVICE, DTYPE


def test_logit_match(moe_safetensors_path: str, num_samples: int = 10):
    model_dir = snapshot_download(repo_id=MODEL_NAME)
    config = MoEModelConfig(toy_mode=True)

    # Generate all samples on CPU first
    all_input_ids = [torch.randint(0, config.vocab_size, (1, config.max_seq_len)) for _ in range(num_samples)]

    # Load dense model and get logits
    dense_model = Transformer(ModelConfig())
    load_weights(
        dense_model,
        model_dir=model_dir,
        dtype=DTYPE,
        device=DEVICE,
        strict=True,
        verbose=True,
    )

    dense_logits_list = []
    with torch.no_grad():
        for input_ids in tqdm(all_input_ids, desc="Dense model inference"):
            input_ids_device = input_ids.to(DEVICE)
            logits = dense_model(input_ids_device)
            dense_logits_list.append(logits.cpu())

    # Free dense model VRAM
    del dense_model
    torch.cuda.empty_cache()

    # Load MoE model and get logits
    moe_model = MoETransformer(config)
    load_moe_weights(
        moe_model,
        safetensors_path=moe_safetensors_path,
        dtype=DTYPE,
        device=DEVICE,
        strict=True,
        verbose=True,
    )

    moe_logits_list = []
    with torch.no_grad():
        for input_ids in tqdm(all_input_ids, desc="MoE model inference"):
            input_ids_device = input_ids.to(DEVICE)
            logits = moe_model(input_ids_device)
            moe_logits_list.append(logits.cpu())

    # Free MoE model VRAM
    del moe_model
    torch.cuda.empty_cache()

    # Compare logits on CPU
    results = []
    for dense_logits, moe_logits in tqdm(zip(dense_logits_list, moe_logits_list), desc="Comparing logits"):
        max_diff = (dense_logits - moe_logits).abs().max().item()
        assert torch.allclose(dense_logits, moe_logits, atol=BF16_TOLERANCE_ATOL, rtol=BF16_TOLERANCE_RTOL), (
            f"Logit mismatch: max diff = {max_diff}"
        )
        results.append(max_diff)

    return results


result = test_logit_match(
    moe_safetensors_path=str(Path(__file__).parent.joinpath("../../../../artifacts/llama3_8b_moe.safetensors")),
    num_samples=5,
)
print("Logit match test passed. Max differences per sample:", result)
