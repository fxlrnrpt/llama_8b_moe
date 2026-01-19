# Llama 3 dense to MoE conversion

This repo showcases a potential route to convert Llama 3 8B from a dense model to mixture-of-experts. Take a look at full list of [requirements](./REQUIREMENTS.md) to have a better understanding of why certain design choices were made.

Below is a high-level final report. Find a detailed engineering diary [here](./DIARY.md).

- [Llama 3 dense to MoE conversion](#llama-3-dense-to-moe-conversion)
  - [Architecture](#architecture)
  - [Memory vs inference speed trade-off](#memory-vs-inference-speed-trade-off)
  - [Completed Tasks](#completed-tasks)
  - [Potential next steps](#potential-next-steps)

## Architecture

1. 16 fine-grained experts following best practive set by DeepSeek - better perf and potentially smaller memory footprint.
2. Experts are created by slicing original dense FFN along its hidden dimension
3. 8 experts are frozen and never trained to match the output of the dense model if all activated simultaniously. 
4. 8 other experts are created by slicing original dense FFN with added noise. It should be a good starting point for tuning.
5. Learned router with bias instead of auxiliary loss. Simpler. Does not create a mixed training objective.
6. Token-choice routing as teh current industry standard.
7. Experts are kept in a single tensor for batched matmuls.

## Memory vs inference speed trade-off

For best inference speed, we load all experts in VRAM at once. Potentially, adding pipeline, context and expert parallelism to process extremely long sequences. 
For smaller memory footprint, we can decrease top_k (doubtful performance as we keep first 8 experts untrained for logit parity with dense), and offload parts of the expert tensor to CPU RAM or to disk (not implemented). We could also make teh fine-grained experts even smaller.

## Completed Tasks

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