"""
MoE Expert Continuous Pretraining Script

Trains only the learned experts (indices 8-15) and router weights using FSDP on 2 H100 GPUs.
All other parameters (embedding, attention, norms, lm_head, sliced experts 0-7) are frozen.

Usage:
    torchrun --nproc_per_node=2 src/experiments/training/moe_expert_continuous_pretraining.py
"""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from core.models.moe.llama_moe_loader import load_moe_weights
from core.models.moe.llama_moe_model import MoEModelConfig, MoETransformer
from core.training import (
    TrainerConfig,
    cleanup_distributed,
    create_dataloader,
    get_rank,
    get_trainable_params,
    get_world_size,
    is_main_process,
    register_expert_freeze_hooks,
    setup_distributed,
    train,
    wrap_model_with_fsdp,
)
from core.utils.constants import MODEL_NAME


@dataclass
class ExperimentConfig:
    """
    Configuration for MoE expert continuous pretraining.

    This experiment trains only the learned experts and router on FineWeb-Edu,
    keeping all other model parameters frozen.
    """

    # ==================== Model ====================
    # Path to the converted MoE model weights (safetensors format)
    # This should be the output of convert_dense_to_experts_by_slicing.py
    model_path: str = "artifacts/llama3_8b_moe.safetensors"

    # ==================== Dataset ====================
    # FineWeb-Edu: High-quality educational web content filtered from CommonCrawl
    # - "sample-10BT" subset: 10 billion tokens, good for initial experiments
    # - Full dataset: ~1.3T tokens for production runs
    # Alternative datasets: "cerebras/SlimPajama-627B", "togethercomputer/RedPajama-Data-1T"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # Maximum sequence length for training
    # - 2048: Standard for initial training, fits well in memory
    # - 4096/8192: For longer context training (requires more memory)
    max_seq_len: int = 2048

    # ==================== Batch Size ====================
    # Per-GPU batch size
    # - Effective batch size = batch_size * gradient_accumulation_steps * num_gpus
    # - With defaults: 2 * 32 * 2 = 128 samples per optimization step
    batch_size: int = 2

    # Gradient accumulation steps to simulate larger batch sizes
    # - Higher values = more stable training, better gradient estimates
    # - 32 gives effective batch of 128, which is more typical for LLM training
    # - Trade-off: fewer optimizer steps per epoch
    gradient_accumulation_steps: int = 32

    # ==================== Optimization ====================
    # Peak learning rate after warmup
    # - 1e-4: Conservative, stable training for fine-tuning
    # - 3e-4: More aggressive, faster learning but may be unstable
    # - For frozen base model + new experts, 1e-4 is a safe starting point
    learning_rate: float = 1e-4

    # AdamW weight decay coefficient
    # - 0.1: Standard for LLM training (Chinchilla, LLaMA)
    # - Applied only to non-bias, non-LayerNorm parameters
    weight_decay: float = 0.1

    # Linear warmup steps before reaching peak learning rate
    # - 100-500 steps typical for fine-tuning
    # - Helps stabilize early training
    warmup_steps: int = 100

    # Total number of training steps
    # - 10000 steps * 128 effective batch * 2048 tokens ≈ 2B tokens
    # - Adjust based on dataset size and compute budget
    max_steps: int = 10000

    # ==================== Logging & Checkpointing ====================
    # How often to log loss and learning rate (in steps)
    log_interval: int = 10

    # How often to save checkpoints (in steps)
    save_interval: int = 1000

    # Directory for saving checkpoints
    output_dir: str = "artifacts/checkpoints"

    # ==================== FSDP Configuration ====================
    # Use bfloat16 mixed precision for faster training and lower memory
    # - True: Recommended for H100 (native bf16 support)
    # - False: Full precision, for debugging numerical issues
    mixed_precision: bool = True


def main():
    """Main entry point for training."""

    # Initialize distributed training
    local_rank = setup_distributed()
    world_size = get_world_size()
    rank = get_rank()

    # Load experiment configuration
    config = ExperimentConfig()

    if is_main_process():
        print(f"Training on {world_size} GPUs")
        print(f"Experiment config: {config}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Initialize model with auto routing (uses router for expert selection)
    model_config = MoEModelConfig(routing="auto")
    model = MoETransformer(model_config)

    # Load pretrained MoE weights
    if is_main_process():
        print(f"Loading weights from {config.model_path}")
    load_moe_weights(
        model,
        config.model_path,
        dtype=torch.bfloat16,
        device="cpu",
        verbose=is_main_process(),
    )

    # Enable gradient checkpointing to reduce memory usage
    model.enable_gradient_checkpointing()

    # Configure trainable parameters (experts + router only)
    trainable_params = get_trainable_params(model)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Frozen parameters: {total_params - trainable_count:,}")
        print(f"Trainable %: {100 * trainable_count / total_params:.2f}%")

    # Register hooks to freeze sliced experts (indices 0-7) during backward
    register_expert_freeze_hooks(model)

    # Wrap model with FSDP
    model = wrap_model_with_fsdp(model, local_rank, config.mixed_precision)

    # Create dataloader
    dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        world_size=world_size,
        rank=rank,
    )

    # Create trainer config
    trainer_config = TrainerConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        output_dir=config.output_dir,
    )

    # Run training
    train(model, dataloader, tokenizer, trainer_config, local_rank)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
