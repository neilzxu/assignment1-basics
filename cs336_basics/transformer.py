from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce


@dataclass(eq=False)
class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self._weight = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device)
        )
        sigma = np.sqrt(1 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self._weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self._weight.T)


@dataclass(eq=False)
class Embedding(torch.nn.Module):
    num_embeddings: int
    embedding_dim: int
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self._embeddings = torch.nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device)
        )
        sigma = 1
        torch.nn.init.trunc_normal_(self._embeddings, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embeddings[token_ids]


@dataclass(eq=False)
class RMSNorm(torch.nn.Module):
    d_model: int
    eps: float = 1e-5
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self._gain = torch.nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = reduce(x.square(), "... d -> ... 1", "mean")
        result = x * torch.rsqrt(rms_x + self.eps) * self._gain.to(torch.float32)
        return result.to(in_dtype)


@dataclass(eq=False)
class FFN(torch.nn.Module):
    d_model: int
    d_ff: int
    activation: str = "SwiGLU"
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    @staticmethod
    def calc_d_ff(d_model):
        return int(8 / 3 * d_model / 64) * 64

    def __post_init__(self):
        super().__init__()
        self.w1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        if self.activation == "SwiGLU":
            self.w3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner_prod = self.w1(x)
        activation_x = inner_prod * F.sigmoid(inner_prod)
        if self.activation == "SwiGLU":
            activation_x = activation_x * self.w3(x)
        return self.w2(activation_x)


@dataclass(eq=False)
class RoPE(torch.nn.Module):
    theta: float
    d_k: int
    max_seq_len: int
    device: torch.device | None = None

    def __post_init__(self):
        super().__init__()

        seq_indices = torch.arange(self.max_seq_len)
        theta_exps = self.theta ** (-1 * (2 * torch.arange(1, self.d_k // 2 + 1) - 2) / self.d_k)
        theta_vals = torch.outer(seq_indices, theta_exps)
        cos_vals = torch.cos(theta_vals)
        sin_vals = torch.sin(theta_vals)
        rot_matrices = rearrange(
            torch.stack([cos_vals, -1 * sin_vals, sin_vals, cos_vals], dim=-1),
            "seq d (rot_row rot_col)-> seq d rot_row rot_col",
            rot_row=2,
        )
        self.register_buffer("rot_matrices", rot_matrices, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rot_matrices = self.rot_matrices.to(token_positions.device)[token_positions]
        rotatable_x = rearrange(x, "... (d rot) -> ... d rot ()", rot=2)
        rotated_x = rot_matrices.matmul(rotatable_x)
        return rearrange(rotated_x, "... d rot a -> ... (d rot a)")


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_elem = torch.amax(x, dim=i, keepdim=True)
    deltas = x - max_elem
    return torch.exp(deltas) / torch.exp(deltas).sum(dim=i, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-1, -2) * (d_k ** (-0.5))
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(masked_scores, i=-1)
    return weights @ V


class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_layer: torch.nn.Module | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_layer = rope_layer
        self._Wq = Linear(self.d_model, self.d_model)
        self._Wk = Linear(self.d_model, self.d_model)
        self._Wv = Linear(self.d_model, self.d_model)
        self._Wo = Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor):

        seq_len = x.size(-2)
        Q, K, V = (self._Wq(x), self._Wk(x), self._Wv(x))

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        Q_head = rearrange(Q, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)
        K_head = rearrange(K, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)
        V_head = rearrange(V, "... seq_len (head d) -> ... head seq_len d", head=self.num_heads)
        if self.rope_layer is not None:
            positions = torch.arange(seq_len, device=x.device)
            Q_head = self.rope_layer(Q_head, positions)
            K_head = self.rope_layer(K_head, positions)
        head_attn = scaled_dot_product_attention(Q_head, K_head, V_head, causal_mask)
        concat_attn = head_attn.transpose(-2, -3).flatten(-2, -1)
        return self._Wo(concat_attn)
