"""Lightweight frequency-preserving pitch-only model (Step 10)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FREQ_AFTER_POOL = 180


class FreqPreservingPitchModel(nn.Module):
    """CQT [B,1,360,T] → keep frequency until a position-sensitive pitch head."""

    def __init__(
        self,
        *,
        n_pitch_bins: int | None = None,
        time_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.n_pitch_bins = n_pitch_bins
        pad_t = time_kernel // 2
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, time_kernel), padding=(0, pad_t)),
            nn.ReLU(inplace=True),
        )
        self.to_freq = nn.Conv2d(32, 1, kernel_size=1)
        if n_pitch_bins is None:
            self.freq_linear = nn.Linear(FREQ_AFTER_POOL, 1)
            self.bin_linear = None
        else:
            self.freq_linear = None
            self.bin_linear = nn.Linear(FREQ_AFTER_POOL, n_pitch_bins)

    def encode(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.frontend(spec)
        return self.to_freq(h).squeeze(1)  # [B, F', T]

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        freq = self.encode(spec).transpose(1, 2)  # [B, T, F']
        if self.n_pitch_bins is None:
            assert self.freq_linear is not None
            return self.freq_linear(freq).squeeze(-1)  # [B, T] standardized cents
        assert self.bin_linear is not None
        return self.bin_linear(freq)  # [B, T, n_bins]


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def bins_to_cents(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (argmax_cents, expected_cents) from [B,T,N] logits."""
    idx = logits.argmax(dim=-1)
    argmax_cents = bin_centers[idx]
    probs = F.softmax(logits, dim=-1)
    expected = (probs * bin_centers.view(1, 1, -1)).sum(dim=-1)
    return argmax_cents, expected
