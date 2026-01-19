"""Distributed training utilities."""

import os

import torch
import torch.distributed as dist


def setup_distributed() -> int:
    """
    Initialize distributed training with NCCL backend.

    Returns:
        local_rank: The local GPU rank for this process.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def get_world_size() -> int:
    """Get total number of processes in distributed training."""
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current process."""
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0
