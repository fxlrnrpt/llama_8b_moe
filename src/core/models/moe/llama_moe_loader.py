import torch
from safetensors.torch import load_file as safetensors_load_file


def load_moe_weights(
    model: torch.nn.Module,
    safetensors_path: str,
    device: torch.device,
    strict: bool = True,
    verbose: bool = True,
):
    sd = safetensors_load_file(safetensors_path, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=strict, assign=True)
    model.to(device)

    if verbose:
        print("=== load_moe_weights ===")
        print(f"safetensors_path: {safetensors_path}")
        print(f"missing_keys: {len(missing)}")
        print(f"unexpected_keys: {len(unexpected)}")

    return {
        "safetensors_path": safetensors_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
