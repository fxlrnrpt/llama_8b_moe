# Llama 3 dense to MoE conversion

This repo showcases a potential route to convert Llama 3 8B from a dense model to mixture-of-experts. Take a look at full list of [requirements](./REQUIREMENTS.md) to have a better understanding of why certain design choices were made.

Below is a high-level final report. Find a detailed engineering diary [here](./DIARY.md).

- [Llama 3 dense to MoE conversion](#llama-3-dense-to-moe-conversion)
  - [Architecture](#architecture)
  - [Memory vs inference speed trade-off](#memory-vs-inference-speed-trade-off)
  - [Completed Tasks](#completed-tasks)
  - [Potential next steps](#potential-next-steps)
  - [How to run](#how-to-run)

## Architecture

1. 16 fine-grained experts following best practive set by DeepSeek - better perf and potentially smaller memory footprint.
2. Experts are created by slicing original dense FFN along its hidden dimension
3. 8 experts are frozen and never trained to match the output of the dense model if all activated simultaniously. 
4. 8 other experts are created by slicing original dense FFN with added noise. It should be a good starting point for tuning.
6. Learned router with bias instead of auxiliary loss. Simpler. Does not create a mixed training objective.
7. Token-choice routing as teh current industry standard.
8. 8 active routed experts to match the intial FFN size.
9. Experts are kept in a single tensor for batched matmuls.

## Memory vs inference speed trade-off

For best inference speed, we load all experts in VRAM at once. Potentially, adding pipeline, context and expert parallelism to process extremely long sequences. 
For smaller memory footprint, we can decrease top_k (doubtful performance as we keep first 8 experts untrained for logit parity with dense), and offload parts of the expert tensor to CPU RAM or to disk (not implemented). We could also make teh fine-grained experts even smaller.

## Completed Tasks

> Note: you can look at what I have been given as astarting point [here (first commit)](https://github.com/fxlrnrpt/llama_8b_moe/commit/37fd8622bc94ad63b2072b8c198f22b457d6e12e)


- [x] Toy mode that matches the output of the dense model
- [x] Test to check the logit match 
- [x] Try a better dense FFN replica
- [x] Init learned experts
- [x] MoE router
- [x] Start training learned experts and routers

## Potential next steps

- Offload unused experts to CPU RAM or to disk
- Distill experts first from the base model and do first continuous pre-training there (see [diary](./DIARY.md) for details)
- Add LoRA training for first 8 sliced experts, but do not merge the matrices. This way we could still match the original dense FFN and also allow training of these experts!
- Add post-training pipeline
- Ablations to find the best expert size
- Ablations o compare bias-based and auxiliary-loss-based routing
- Add support for training longer sequences (go beyond FSDP)
- Shard experts safetensors to load only necessary chunks from disk in memory-constrained envs

## How to run

1. `uv sync` - installs deps
2. Rent a single H100 machine
3. `uv run src/tests/dense_sanity_check.py` - check dense model produced expected output
4. `uv run src/experiments/conversion/convert_dense_to_experts_by_sliciing.py` - create a MoE version of the model
5. `uv run src/tests/moe_logit_match.py` - test dense logit match
6. Rent 3 x H100 machine
7. `uv run torchrun --nproc-per-node=3 src/experiments/training/moe_expert_continuous_pretraining.py` - start continuous pre-training of the learned experts
