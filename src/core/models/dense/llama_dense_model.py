import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    max_seq_len: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_position_emb(q, k, cos, sin):
    # q, k: [bs, nheads, seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len_cache = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be multiple of num_kv_heads"

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        bsz, kv_heads, seq, hd = x.shape
        x = x[:, :, None, :, :].expand(bsz, kv_heads, n_rep, seq, hd)
        return x.reshape(bsz, kv_heads * n_rep, seq, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [bsz, seq, hidden]
        past_key_value:
            k: [bsz, kv_heads, past_seq, head_dim]
            v: [bsz, kv_heads, past_seq, head_dim]
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, heads, seq, hd]
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [bsz, kv, seq, hd]
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_position_emb(q, k, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        k = self.repeat_kv(k, n_rep)  # [bsz, heads, total_seq, hd]
        v = self.repeat_kv(v, n_rep)

        total_seq = k.size(2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [bsz, heads, seq, total_seq]
        causal = torch.tril(torch.ones(seq_len, total_seq, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, v)  # [bsz, heads, seq, hd]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.ffn = self._create_ffn(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _create_ffn(self, config: ModelConfig) -> nn.Module:
        return FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out = self.self_attn(self.attn_norm(x))
        x = residual + attn_out

        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DenseBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(x)  # [bsz, seq, hidden]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
