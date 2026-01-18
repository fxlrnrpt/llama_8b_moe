import torch
from safetensors.torch import load_file as safetensors_load_file


def load_moe_weights(
    model: torch.nn.Module,
    safetensors_path: str,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    strict: bool = True,
    verbose: bool = True,
):
    sd = safetensors_load_file(safetensors_path)

    # Convert dtype if specified
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}

    # Load state_dict into model
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    model.to(device=device)

    if verbose:
        print("=== load_moe_weights ===")
        print(f"safetensors_path: {safetensors_path}")
        print(f"missing_keys: {len(missing)}")
        print(f"unexpected_keys: {len(unexpected)}")

    return {
        "safetensors_path": safetensors_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "state_dict": sd,
    }
