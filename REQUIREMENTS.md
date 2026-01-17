## Task Description

Your task is to take Llama 3.1 8B and rearchitect the dense layers to MoE layers. You are supplied with a Llama 3 architecture below.

Architect the MoE-ified Llama 8B such that it performs identically to the base model. That is, when inferencing upon a single prompt at temperature 0 the output of the base dense model and your MoE Llama should be the same. Write a test that demonstrates that the MoE-ified Llama produces the same logits as the base model to within a reasonable tolerance, such that greedy decoding produces the same token sequence in either setting. Describe and/or demonstrate the approach you took to ensure this and whether or not this differs from how you would expect to inference with the model in a production setting.

When writing the MoE layer you should consider issues regarding performance and memory when inferencing in real-world scenarios. Discuss the implications on performance when optimising for memory when inferencing your MoE-ified Llama, and vice-versa. The task is open ended in the sense that you may wish to produce a more performant MoE implementation, and another implementation better suited to situations where memory is the bottleneck. However, this is not required and a detailed discussion can also prove sufficient.

## Core Requirements

1. Replace the dense feed-forward layers (MLP blocks) in Llama with an MoE equivalent. Your MoE design should support:

- an “identity / toy mode” that guarantees logit fidelity with the base dense model
- a “production-like mode” that resembles a realistic MoE setup (e.g., learned router + top-k routing), even if it is not trained. You may decide what “production-like” means, but it should meaningfully reflect real MoE usage (routing, multiple experts, dispatch pattern, etc.) rather than only being a no-op wrapper.

2. Write a runnable test that demonstrates logit fidelity between the original dense Llama 3.1 8B model, and your MoE-ified model in identity/toy mode. The test must:

- load pretrained dense weights
- compare logits between models and assert they match within a reasonable tolerance (you decide what’s reasonable and justify it)
- ideally include a greedy decode check to confirm identical token outputs

3. A discussion regarding your MoE implementation and the performance/memory trade-offs you have considered. Consider the implications of optimizing for: 

- maximum throughput (fast inference)
- minimum memory footprint 

As well as runnability, you will be judged on code quality and readability. With regards to the asset you produce, there are no strict requirements - you can return a notebook, or split it out into a package along with a script.