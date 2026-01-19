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
5. Keeping the fine-grained expert idea in the back of my mind. Will research details later. Implementing the toy mode with logit match.
6. Toy mode works. Logit match is perfect (0 diff). I assumed there might be some numerical non-determenism, so added small tolerance (2^-7 ~ 1e-2). It still might happen in the future once I add a real router, so keeping it in place.
7. Examined some papers - better idea! Let's slice FFN into multiple experts along hidden dim. Then using all of the experts at once should match the output of the original FFN. We can't train them though, but we will have a separate batch of experts for that.
8. Yay! Still works. This time with acceptable numerical fluctations: `Logit match test passed. Max differences per sample: [0.000247955322265625, 0.0002758502960205078, 0.000232696533203125, 0.0003854334354400635, 0.00037860870361328125]`. With this architecture we can have a much smaller memory footprint if wanted.
9. Had a brilliant idea - what if we use SVD to decompose up, down and gate projections in FFN? Then we could randomly select a portion of singular values and only one of the matrices from the decomposition to get the resulting expert slice. We could go even further - select half of the biggest singular values and the other half randomly - capture the most important part of the transformation with enough randomness. There is a problem tough - if we pick only one of the 2 matrices, we are effectively changing the basis. Still wanted to check if it worked by some miracle. Well, it does not. Moving on.
10. Using the standard approach by adding noise to FFN and doing the same slicing.
11. Started with noise scaling = 0.01. Too little. Almost no difference compared to the original output. Increasing to 0.1.
12. Still too little. Let's try 1.
13. Long story short, 0.2 is good enough.
14. Added bias-based router. Looks cleaner than auxiliary loss if you ask me. Accroding to Deepseek should perform well. Luckily, I am not going to have enough resources and time to run proper ablations. Or unlickily. Anyhow, rolling with it.
Side note: I wish CC wrote cleaner code. Had to re-write most of it this time :cry:
15. Testing the final implmentation with untrained router. Replaced router initialization for small std normal ~ nearly uniform routing after softmax. Also increased top_k to 4 to match the initial numer of weigths in dense model. Otherwise, in the routed mode (whne the router and the experts are not trained) the model produces garbage. With the new parameters the output is still far from ideal, but sill better. Also, intuitevely we at least have a chance of something reasonable without a ton of continuous pre-training becasue we match the number of dense FFN params.
16. 