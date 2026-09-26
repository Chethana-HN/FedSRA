from __future__ import annotations

import math

import torch


class ResidualDiffusionSchedule:
    def __init__(
        self,
        steps: int = 20,
        device: str | torch.device = "cpu",
        min_alpha_bar: float = 1e-4,
    ) -> None:
        self.steps = int(steps)
        self.device = torch.device(device)

        if self.steps < 2:
            raise ValueError("steps must be >= 2")

        x = torch.linspace( 0.0, 1.0, self.steps, device=self.device, )

        alpha_bar = torch.cos(x * math.pi / 2.0).pow(2)
        alpha_bar = alpha_bar.clamp( min=float(min_alpha_bar), max=0.9999, )

        alpha_bar[0] = 0.9999
        alpha_bar[-1] = float(min_alpha_bar)

        self.alpha_bars = alpha_bar

    def q_sample(
        self,
        clean_residual: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean_residual)

        alpha = self.alpha_bars[t].view(-1, 1, 1, 1)

        noisy = ( torch.sqrt(alpha) * clean_residual + torch.sqrt(1.0 - alpha) * noise )

        return noisy, noise

    def sample_low_mid_timesteps(
        self,
        batch_size: int,
        max_fraction: float = 0.5,
    ) -> torch.Tensor:
        max_fraction = min(max(float(max_fraction), 0.05), 1.0)

        high = max( 1, int(round(self.steps * max_fraction)), )
        high = min(high, self.steps)

        return torch.randint( 0, high, (int(batch_size),), device=self.device, dtype=torch.long, )

    def zero_start(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.zeros_like(reference)

        t = torch.zeros( reference.shape[0], device=reference.device, dtype=torch.long,  )

        return residual, t