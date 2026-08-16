"""Step 11 §8-9: shared harmonic-salience scorer.

The scorer is built entirely from kernel_size=(1,1) Conv2d layers applied
over the (channel, candidate-frequency, time) tensor. A 1x1 conv applies the
*same* weights at every spatial (candidate-frequency, time) position by
construction — this is what makes the scorer phi(...) structurally shared
across candidate frequencies (§9's frequency-equivariance requirement),
without any extra machinery. The same class serves both the harmonic-aware
variant (`harmonic_ks=(1,2,3,4)`) and the local-only control (`harmonic_ks=(1,)`,
default `hidden` widened by the caller to roughly match parameter count).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.pitch_diagnostics.salience_features import (
    build_harmonic_features,
    n_feature_channels,
)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class HarmonicSalienceModel(nn.Module):
    def __init__(
        self,
        *,
        candidate_lo_bin: int,
        candidate_hi_bin: int,
        harmonic_ks: tuple[int, ...] = (1, 2, 3, 4),
        delta: int = 2,
        bg_delta: int = 8,
        hidden: int = 16,
        temporal_smoothing: bool = False,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.candidate_lo_bin = candidate_lo_bin
        self.candidate_hi_bin = candidate_hi_bin
        self.harmonic_ks = tuple(harmonic_ks)
        self.delta = delta
        self.bg_delta = bg_delta
        self.n_candidates = candidate_hi_bin - candidate_lo_bin
        c_in = n_feature_channels(self.harmonic_ks)
        # Shared scorer phi(.) — identical weights applied at every candidate f.
        self.scorer = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=(1, 1)),
        )
        self.temporal_smoothing = temporal_smoothing
        self.temporal_kernel = temporal_kernel
        if temporal_smoothing:
            # One shared 1-in/1-out temporal kernel applied independently to
            # every candidate frequency (weight-shared across f, per §9).
            self.temporal_conv = nn.Conv1d(1, 1, kernel_size=temporal_kernel, padding=temporal_kernel // 2)
        else:
            self.temporal_conv = None

    def features(self, spec: torch.Tensor, *, ablate_ks: frozenset[int] = frozenset()) -> torch.Tensor:
        return build_harmonic_features(
            spec,
            candidate_lo_bin=self.candidate_lo_bin,
            candidate_hi_bin=self.candidate_hi_bin,
            harmonic_ks=self.harmonic_ks,
            delta=self.delta,
            bg_delta=self.bg_delta,
            ablate_ks=ablate_ks,
        )

    def forward(self, spec: torch.Tensor, *, ablate_ks: frozenset[int] = frozenset()) -> torch.Tensor:
        """spec: [B,1,N_BINS,T] normalized log-CQT. Returns logits [B,F_cand,T]."""
        feats = self.features(spec, ablate_ks=ablate_ks)
        logits = self.scorer(feats).squeeze(1)  # [B,F_cand,T]
        if self.temporal_conv is not None:
            b, f_cand, t = logits.shape
            flat = logits.reshape(b * f_cand, 1, t)
            flat = self.temporal_conv(flat)
            logits = flat.reshape(b, f_cand, t)
        return logits


__all__ = ["HarmonicSalienceModel", "count_params"]
