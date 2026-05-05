import math
from typing import Tuple

import torch
from torch import nn

from .patching import patchify, unpatchify


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None] * max_period
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class LatentBridgeDiT(nn.Module):
    """A compact DiT-like model with bridge and high-resolution pixel heads."""

    def __init__(
        self,
        image_size: int = 256,
        bridge_size: int = 64,
        patch_size: int = 16,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if bridge_size % patch_size != 0:
            raise ValueError("bridge_size must be divisible by patch_size")

        self.image_size = image_size
        self.bridge_size = bridge_size
        self.patch_size = patch_size
        self.channels = channels
        self.patch_dim = channels * patch_size * patch_size
        self.n_hr = (image_size // patch_size) ** 2
        self.n_bridge = (bridge_size // patch_size) ** 2

        self.hr_in = nn.Linear(self.patch_dim, dim)
        self.bridge_in = nn.Linear(self.patch_dim, dim)
        self.hr_pos = nn.Parameter(torch.zeros(1, self.n_hr, dim))
        self.bridge_pos = nn.Parameter(torch.zeros(1, self.n_bridge, dim))
        self.modality = nn.Parameter(torch.zeros(2, dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.hr_out = nn.Linear(dim, self.patch_dim)
        self.bridge_out = nn.Linear(dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.hr_pos, std=0.02)
        nn.init.trunc_normal_(self.bridge_pos, std=0.02)
        nn.init.trunc_normal_(self.modality, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_hr_t: torch.Tensor,
        x_bridge_t: torch.Tensor,
        t_hr: torch.Tensor,
        t_bridge: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x_hr_t.shape[0]

        hr = patchify(x_hr_t, self.patch_size)
        bridge = patchify(x_bridge_t, self.patch_size)
        hr = self.hr_in(hr) + self.hr_pos + self.modality[0]
        bridge = self.bridge_in(bridge) + self.bridge_pos + self.modality[1]

        hr_time = self.time_mlp(timestep_embedding(t_hr, hr.shape[-1])).view(b, 1, -1)
        bridge_time = self.time_mlp(timestep_embedding(t_bridge, bridge.shape[-1])).view(b, 1, -1)
        hr = hr + hr_time
        bridge = bridge + bridge_time

        x = torch.cat([bridge, hr], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        bridge_tokens, hr_tokens = x[:, : self.n_bridge], x[:, self.n_bridge :]
        pred_bridge = self.bridge_out(bridge_tokens)
        pred_hr = self.hr_out(hr_tokens)

        pred_bridge = unpatchify(pred_bridge, self.patch_size, self.bridge_size, self.bridge_size, self.channels)
        pred_hr = unpatchify(pred_hr, self.patch_size, self.image_size, self.image_size, self.channels)
        return pred_hr, pred_bridge
