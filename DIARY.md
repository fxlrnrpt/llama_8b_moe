# Engineering Diary

1. Converting the initial jupyter notebook into a package to make experiments more reproducible and modular.
2. Make sure sanity check (simple inference) for the dense model of LLaMA 3.1 8B works correctly.
3. Time to implment MoE. Reqs: 
    - Logit match with dense in toy mode
    - Ability to balance between inference speed and memory footprint
    - Not an explicit requirement, but we want this thing to perform as well as we can make it, right?
4. Considerations:
    - I'll keep the original FFN layer as one of the experts. This way we can easily get the logit parity with the dense model. Just route tokens to this single expert.
    - I'll use the original FFN as teh awlays-on shared expert. No time to run ablations, but shared expert seems to work for Deepseek. Their setup is different. Olmo also did not find any use for shared expert. But then matching the dense model is elegant - I just disable routed experts.
    - I'd like to use fine-grained experts for better perf (DeepSeek proved it). 
      Challenges:
      - Match the granularity level with the shared "fat" expert for batched matmuls
      - Training fine-grained experts is going to be complicated. We will not be able to re-use teh existing FFN as a starting point (following Qwen 1.5). However... Pruning?
      Pros: easy to adjust memory footprint (number of active small experts) based on the memory cap. Also, better expressivity compared to 2 active "fat" experts.
5. Keeping the fine-grained expert idea in the back of my mind. Will research details later. Implmenting the toy mode with logit match.
