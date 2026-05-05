from dataclasses import dataclass
from typing import Tuple

import torch


def add_noise(x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow interpolation.

    t=0 is pure noise and t=1 is clean data.
    """

    while t.ndim < x.ndim:
        t = t[..., None]
    if noise is None:
        noise = torch.randn_like(x)
    z_t = t * x + (1.0 - t) * noise
    return z_t, noise


def loss_weight(t: torch.Tensor, mode: str = "velocity", min_denom: float = 0.05, max_weight: float = 100.0) -> torch.Tensor:
    """Weights x-prediction loss to mimic velocity loss.

    For rectified flow, if a model predicts x0, the implied velocity is
    (x0 - z_t) / (1 - t). A velocity MSE is therefore an x0 MSE weighted by
    1 / (1 - t)^2. The clamp avoids exploding loss very near clean data.
    """

    if mode == "none":
        return torch.ones_like(t)
    if mode != "velocity":
        raise ValueError(f"Unknown loss weight mode: {mode}")
    denom = (1.0 - t).clamp(min=min_denom)
    return (denom.reciprocal() ** 2).clamp(max=max_weight)


@dataclass
class FlowSchedule:
    """Maps a global progress variable to bridge and pixel times."""

    cascade_ratio: float = 0.5
    eps: float = 1e-4

    def __post_init__(self) -> None:
        if not 0.0 < self.cascade_ratio < 1.0:
            raise ValueError("cascade_ratio must be in (0, 1)")

    def times(self, global_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (t_bridge, t_pixel)."""

        c = self.cascade_ratio
        t_bridge = (global_t / c).clamp(0.0, 1.0)
        t_pixel = ((global_t - c) / (1.0 - c)).clamp(0.0, 1.0)
        t_bridge = t_bridge.clamp(self.eps, 1.0 - self.eps)
        t_pixel = t_pixel.clamp(self.eps, 1.0 - self.eps)
        return t_bridge, t_pixel

    def step_times(self, step: int, total_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scalar global, bridge, and pixel times for sampling."""

        g = torch.tensor(step / total_steps, device=device)
        tb, tp = self.times(g)
        return g, tb, tp

    def step_delta(self, step: int, total_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dt for bridge and pixel from step to step+1."""

        _, tb0, tp0 = self.step_times(step, total_steps, device)
        _, tb1, tp1 = self.step_times(step + 1, total_steps, device)
        return tb1 - tb0, tp1 - tp0
