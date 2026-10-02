import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_elem = torch.amax(x, dim=i, keepdim=True)
    deltas = x - max_elem
    return torch.exp(deltas) / torch.exp(deltas).sum(dim=i, keepdim=True)


def _cross_entropy_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    max_elem = torch.amax(logits, dim=-1, keepdim=True)
    deltas = logits - max_elem
    return -(
        deltas.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.log(torch.exp(deltas).sum(dim=-1, keepdim=True))
    )


def cross_entropy_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _cross_entropy_softmax(logits, targets).mean()


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy_softmax(logits, targets)).mean(dim=-1)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta_1, beta_2 = group["betas"]  # Get the learning rate.
            lam = group["weight_decay"]  # Get the learning rate.
            eps = group["eps"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", 0)  # Get iteration number from the state, or 0.
                v = state.get("v", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                alpha_t = lr * ((1 - beta_2**t) ** 0.5) / (1 - beta_1**t)
                state["m"] = beta_1 * m + (1 - beta_1) * grad
                state["v"] = beta_2 * v + (1 - beta_2) * (grad**2)
                state["t"] = t + 1  # Increment iteration number.

                p.data -= lr * lam * p.data  # weight decay
                p.data -= alpha_t * state["m"] / (state["v"] ** 0.5 + eps)  # Update weight tensor in-place.
        return loss


@dataclass(eq=False)
class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device)
        )
        sigma = np.sqrt(1 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)


@dataclass(eq=False)
class Embedding(torch.nn.Module):
    num_embeddings: int
    embedding_dim: int
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device)
        )
        sigma = 1
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


@dataclass(eq=False)
class RMSNorm(torch.nn.Module):
    d_model: int
    eps: float = 1e-5
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = reduce(x.square(), "... d -> ... 1", "mean")
        result = x * torch.rsqrt(rms_x + self.eps) * self.weight.to(torch.float32)
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

        seq_indices = torch.arange(self.max_seq_len, device=self.device)
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


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    d_k = K.size(-1)
    scores = Q @ K.transpose(-1, -2) * (d_k ** (-0.5))
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(masked_scores, i=-1)
    return weights @ V


class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_layer: torch.nn.Module | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_layer = rope_layer
        self.device = device
        self.dtype = dtype
        self.q_proj = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(-2)
        Q, K, V = (self.q_proj(x), self.k_proj(x), self.v_proj(x))

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
        return self.output_proj(concat_attn)


@dataclass(eq=False)
class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_layer: torch.nn.Module | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.rope_layer = rope_layer

        self.attn = CausalMultiheadSelfAttention(self.d_model, self.num_heads, self.rope_layer)
        self.ln1 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ln2 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ffn = FFN(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_out_1 = self.ln1(x)
        attn_out = x + self.attn(norm_out_1)
        norm_out_2 = self.ln2(attn_out)
        return attn_out + self.ffn(norm_out_2)


@dataclass(eq=False)
class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype

        self.d_k = self.d_model // self.num_heads

        self.rope_layer = RoPE(theta=self.rope_theta, d_k=self.d_k, max_seq_len=self.context_length, device=self.device)

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, self.device, self.dtype)
        self.layers = torch.nn.Sequential(
            *[
                TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.rope_layer, self.dtype, self.device)
                for i in range(self.num_layers)
            ]
        )
        self.ln_final = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_embs = self.token_embeddings(x)
        layer_out = self.layers(tok_embs)
        norm_out = self.ln_final(layer_out)
        return self.lm_head(norm_out)
