from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce


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
