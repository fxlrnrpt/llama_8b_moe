from dataclasses import dataclass

import torch
from torch import nn

from core.models.dense.llama_dense_model import DenseBlock, ModelConfig
from core.models.dense.llama_dense_model import FeedForward as DenseFeedForward
from core.models.dense.llama_dense_model import Transformer as DenseTransformer


@dataclass
class MoEModelConfig(ModelConfig):
    # Logit match dense model
    toy_mode: bool = False


class ExpertBlock(nn.Module):
    def __init__(self, config: MoEModelConfig):
        super().__init__()

        self.num__routed_experts = 42  # TODO: define number of experts
        if config.toy_mode:
            self.num__routed_experts = 0

        self.shared_expert = DenseFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement routed experts logic
        return self.shared_expert.forward(x)


class MoEBlock(DenseBlock):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.ffn = ExpertBlock(config)


class MoETransformer(DenseTransformer):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MoEBlock(config) for _ in range(config.num_hidden_layers)])
