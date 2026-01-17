import json
import os

import torch
from safetensors.torch import load_file as safetensors_load_file

mapping = {
    ".mlp": ".ffn",
    ".post_attention_layernorm": ".ffn_norm",
    ".input_layernorm": ".attn_norm",
}


def load_weights(
    model: torch.nn.Module,
    model_dir: str | None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    strict: bool = False,
    verbose: bool = True,
):
    """
    Minimal loader for a model whose weights are sharded across multiple safetensors files.

    Maps between the llama weight names as given in the HF safetensors and the parameter names in the above implementation.
    """

    sd = {}
    if model_dir is not None:
        # Read model.safetensors.index.json
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing index file: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)

        if "weight_map" not in idx:
            raise ValueError(f"Index file missing 'weight_map': {index_path}")

        weight_map = idx["weight_map"]
        shard_files = sorted(set(weight_map.values()))

        # Load each sharded list in one merged state_dict
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard referenced by index not found: {shard_path}")
            sd.update(safetensors_load_file(shard_path))
    else:
        shard_files = []

    # iterate over keys and rename according to mapping
    # optionally cast dtype and/or move to device
    for param in list(sd.keys()):
        weights = sd.pop(param)
        if param.startswith("model."):
            param = param.replace("model.", "")
        for old, new in mapping.items():
            if old in param:
                param = param.replace(old, new)
        if dtype is not None:
            weights = weights.to(dtype)
        if device is not None:
            weights = weights.to(device)
        sd[param] = weights

    meta = {
        "model_dir": model_dir,
        "num_shards": len(shard_files),
        "shards": shard_files,
    }

    if not sd:
        for name, tensor in model.state_dict().items():
            if tensor.is_floating_point():
                rand = torch.randn_like(tensor)
            else:
                rand = torch.zeros_like(tensor)

            if dtype is not None and rand.is_floating_point():
                rand = rand.to(dtype)
            if device is not None:
                rand = rand.to(device)

            sd[name] = rand

    # Load state_dict into model
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    model.to(device=device)

    if verbose:
        print("=== load_weights_sharded_safetensors_only ===")
        print(f"model_dir: {model_dir}")
        print(f"num_shards: {meta['num_shards']}")
        print(f"num_tensors_loaded: {len(sd)}")
        print(f"missing_keys: {len(missing)}")
        print(f"unexpected_keys: {len(unexpected)}")

    return {
        "model_dir": model_dir,
        "checkpoint_meta": meta,
        "num_tensors_loaded": len(sd),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
