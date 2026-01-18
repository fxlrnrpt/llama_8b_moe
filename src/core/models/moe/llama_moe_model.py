from dataclasses import dataclass

import torch
from torch import nn

from core.models.dense.llama_dense_model import DenseBlock, ModelConfig
from core.models.dense.llama_dense_model import Transformer as DenseTransformer


@dataclass
class MoEModelConfig(ModelConfig):
    # Logit match dense model
    toy_mode: bool = False
    sliced_expert_intermediate_size: int = 1792  # 14336 / 8
    num_sliced_experts: int = 8
    num_learned_experts: int = 8  # TODO: define number of learned experts
    learned_expert_intermediate_size: int = 1792  # 14336 / 8


class SlicedDenseFeedForward(nn.Module):
    def __init__(self, config: MoEModelConfig):
        super().__init__()

        self.num_experts = config.num_sliced_experts
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, config.sliced_expert_intermediate_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, config.sliced_expert_intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, config.sliced_expert_intermediate_size, config.hidden_size)
        )

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, hidden]
        expert_mask: boolean mask of shape [num_experts], True = use expert (default: all)
        returns: [batch, seq, hidden]
        """
        gate = self.gate_proj[expert_mask]
        up = self.up_proj[expert_mask]
        down = self.down_proj[expert_mask]

        # Batched matmul over experts
        # x: [batch, seq, hidden], gate/up: [k, hidden, expert_size]
        g = torch.einsum("bsh,khe->bske", x, gate)
        u = torch.einsum("bsh,khe->bske", x, up)

        act = nn.functional.silu(g) * u  # [batch, seq, k, expert_size]

        # [batch, seq, k, expert_size] @ [k, expert_size, hidden] -> [batch, seq, k, hidden]
        out = torch.einsum("bske,keh->bskh", act, down)

        return out.sum(dim=2)  # [batch, seq, hidden]


class ExpertBlock(nn.Module):
    def __init__(self, config: MoEModelConfig):
        super().__init__()

        self.num_sliced_experts = config.num_sliced_experts
        self.num_learned_experts = config.num_learned_experts

        self.active_experts_mask = torch.tensor([False] * (self.num_sliced_experts + self.num_learned_experts))

        if config.toy_mode:
            self.active_experts_mask[: self.num_sliced_experts] = True

        self.sliced_dense_ffn = SlicedDenseFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement learned experts logic
        return self.sliced_dense_ffn.forward(x, self.active_experts_mask[: self.num_sliced_experts])


class MoEBlock(DenseBlock):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.ffn = ExpertBlock(config)


class MoETransformer(DenseTransformer):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MoEBlock(config) for _ in range(config.num_hidden_layers)])
