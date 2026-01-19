"""Training loop and utilities."""

import os
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from core.training.distributed import is_main_process, get_rank
from core.training.fsdp import save_fsdp_checkpoint


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    # Optimization
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int

    # Logging & Checkpointing
    log_interval: int
    save_interval: int
    output_dir: str


def create_optimizer(
    model: FSDP,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer for trainable parameters.

    Args:
        model: FSDP-wrapped model.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured optimizer.
    """
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),  # Standard LLM training betas
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.

    Returns:
        Learning rate scheduler.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: FSDP,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    config: TrainerConfig,
    local_rank: int,
):
    """
    Main training loop.

    Args:
        model: FSDP-wrapped model.
        dataloader: Training data loader.
        tokenizer: Tokenizer (for pad token ID).
        config: Trainer configuration.
        local_rank: Local GPU rank.
    """
    rank = get_rank()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = create_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)

    # Training state
    model.train()
    step = 0
    accumulated_loss = 0.0

    # Progress bar on main process
    pbar = None
    if is_main_process():
        pbar = tqdm(total=config.max_steps, desc="Training")

    optimizer.zero_grad()

    for batch in dataloader:
        if step >= config.max_steps:
            break

        input_ids = batch["input_ids"].to(local_rank)
        labels = batch["labels"].to(local_rank)

        # Forward pass
        logits = model(input_ids)

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="mean",
        )

        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        accumulated_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient accumulation step
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            model.clip_grad_norm_(1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if is_main_process() and (step + 1) % config.log_interval == 0:
                avg_loss = accumulated_loss / config.log_interval
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                accumulated_loss = 0.0

            # Checkpointing
            if (step + 1) % config.save_interval == 0:
                _save_checkpoint(model, optimizer, step + 1, config, rank)

        step += 1

        if pbar is not None:
            pbar.update(1)

    # Final checkpoint
    _save_checkpoint(model, optimizer, step, config, rank)

    if pbar is not None:
        pbar.close()

    if is_main_process():
        print("Training complete!")


def _save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainerConfig,
    rank: int,
):
    """Save checkpoint (only on rank 0)."""
    if rank != 0:
        return

    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")

    save_fsdp_checkpoint(model, optimizer, step, path)
    print(f"Saved checkpoint to {path}")
