from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from core.models.dense.llama_dense_model import DenseBlock, ModelConfig
from core.models.dense.llama_dense_model import Transformer as DenseTransformer

TRouting = Literal["match_dense", "learned_only", "auto"]


@dataclass
class MoEModelConfig(ModelConfig):
    routing: TRouting = "auto"
    expert_intermediate_size: int = 1792  # 14336 / 8
    num_sliced_experts: int = 8
    num_learned_experts: int = 8
    router_top_k: int = 4
    router_bias_update_rate: float = 0.001


class BiasRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, bias_update_rate: float = 0.001):
        super().__init__()
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

        # Bias-based load balancing (not learned, updated dynamically)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq, hidden]
        returns: (weights, indices) both [batch, seq, top_k]
        """
        logits = self.gate(x) + self.expert_bias  # [batch, seq, num_experts]
        scores = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices

    def update_bias(self, expert_counts: torch.Tensor, target_count: float):
        """Call after each batch during training to balance expert load"""
        if not self.training:
            return
        # Increase bias for underused experts, decrease for overused
        deviation = target_count - expert_counts.float()
        self.expert_bias += self.bias_update_rate * deviation


class Experts(nn.Module):
    def __init__(self, config: MoEModelConfig):
        super().__init__()

        self.num_sliced = config.num_sliced_experts
        self.num_learned = config.num_learned_experts
        self.total_num_experts = self.num_sliced + self.num_learned

        self.gate_proj = nn.Parameter(
            torch.empty(self.total_num_experts, config.hidden_size, config.expert_intermediate_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.total_num_experts, config.hidden_size, config.expert_intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.total_num_experts, config.expert_intermediate_size, config.hidden_size)
        )

        self._freeze_hooks_registered = False

    def register_freeze_hooks(self):
        """Register hooks to freeze sliced experts (indices 0:num_sliced)"""
        if self._freeze_hooks_registered:
            return

        def make_hook(n_sliced):
            def hook(grad):
                grad[:n_sliced] = 0
                return grad

            return hook

        self.gate_proj.register_hook(make_hook(self.num_sliced))
        self.up_proj.register_hook(make_hook(self.num_sliced))
        self.down_proj.register_hook(make_hook(self.num_sliced))
        self._freeze_hooks_registered = True

    def forward(self, x: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, hidden]
        expert_mask: boolean mask of shape [num_experts], True = use expert
        returns: [batch, seq, hidden]
        """
        gate = self.gate_proj[expert_mask]
        up = self.up_proj[expert_mask]
        down = self.down_proj[expert_mask]

        # Batched matmul over experts
        # x: [batch, seq, hidden], gate/up: [k, hidden, expert_size]
        g = torch.einsum("bsh,khe->bske", x, gate)
        u = torch.einsum("bsh,khe->bske", x, up)

        act = F.silu(g) * u  # [batch, seq, k, expert_size]

        # [batch, seq, k, expert_size] @ [k, expert_size, hidden] -> [batch, seq, k, hidden]
        out = torch.einsum("bske,keh->bskh", act, down)

        return out


class ExpertBlock(nn.Module):
    def __init__(self, config: MoEModelConfig):
        super().__init__()

        self.num_sliced_experts = config.num_sliced_experts
        self.num_learned_experts = config.num_learned_experts
        self.total_num_experts = self.num_sliced_experts + self.num_learned_experts

        self.routing = config.routing
        self.experts = Experts(config)

        self.router = BiasRouter(
            hidden_size=config.hidden_size,
            num_experts=self.total_num_experts,
            top_k=config.router_top_k,
            bias_update_rate=config.router_bias_update_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, hidden = x.shape

        weights, indices = self._router(x)  # [batch, seq, top_k]
        unique_experts = indices.unique()

        active_experts_mask = torch.zeros(self.total_num_experts, dtype=torch.bool, device=x.device)
        active_experts_mask[unique_experts] = True

        out = self.experts.forward(x, active_experts_mask)

        if self.routing == "auto":
            # Map expert indices to mask indices for gathering
            expert_to_mask = torch.full((self.total_num_experts,), -1, device=x.device, dtype=torch.long)
            expert_to_mask[unique_experts] = torch.arange(len(unique_experts), device=x.device)
            mask_indices = expert_to_mask[indices]  # [batch, seq, top_k]

            # Gather: [batch, seq, top_k, hidden]
            mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, -1, hidden)
            selected_outputs = torch.gather(out, dim=2, index=mask_indices_expanded)

            # Weighted sum: [batch, seq, hidden]
            out = selected_outputs * weights.unsqueeze(-1)

            if self.training:
                expert_counts = torch.bincount(indices.flatten(), minlength=self.total_num_experts)
                target_count = indices.numel() / self.total_num_experts
                self.router.update_bias(expert_counts, target_count)

        return out.sum(dim=2)

    def switch_routing(self, routing: TRouting):
        self.routing = routing

    def _router(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.routing == "match_dense":
            return torch.empty(0), torch.arange(self.num_sliced_experts)
        elif self.routing == "learned_only":
            return torch.empty(0), torch.arange(self.num_sliced_experts, self.total_num_experts)
        return self.router(x)


class MoEBlock(DenseBlock):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.ffn = ExpertBlock(config)


class MoETransformer(DenseTransformer):
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MoEBlock(config) for _ in range(config.num_hidden_layers)])

    def switch_routing(self, routing: TRouting):
        for layer in self.layers:
            layer.ffn.switch_routing(routing)
