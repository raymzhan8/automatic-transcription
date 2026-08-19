"""Step 26 sections 4, 6, 7: the acoustic encoder and the two models that use
it (A1 audio-only, A2 CREPE+audio fusion). Deliberately small -- no
architecture search, no CNN-vs-GRU-vs-Transformer comparison, no pretrained
audio model, no attention.

`AcousticCNN` mirrors `training/framewise_models.py`'s `FrequencyCNN`
(already validated for CQT input throughout Steps 6-20), shrunk to
ContourCNN's own scale (hidden=16, matching `ContourCNN`'s pooled dim) and
with an added final pool over TIME as well as frequency, since this is a
one-embedding-per-primitive segment encoder rather than a framewise one.

`FusionModel`'s pitch branch duplicates `ContourCNN.net`'s exact
architecture (same Conv1d channels/kernels/dilations as Step 22/23's frozen
CNN) rather than importing the class directly, so that the pooled pitch
representation is available separately from the pooled audio representation
for section 20's modality-zeroing sanity check -- `ContourCNN.forward`
does not expose that seam. Fusion is a single `nn.Linear` over the
concatenated [h_pitch ; h_audio], nothing else: no fusion MLP, no
cross-modal attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PITCH_HIDDEN = 16   # == ContourCNN's own `hidden` default
AUDIO_HIDDEN = 16   # matched to PITCH_HIDDEN so fusion is a plain concat, not a projection


class AcousticCNN(nn.Module):
    """CQT log-magnitude patch [B, 1, N_BINS, 64] -> pooled embedding [B, hidden].
    Same conv pattern as `framewise_models.FrequencyCNN` (pools frequency),
    plus a final global pool over time (this encoder sees one fixed-size
    segment, not a framewise sequence)."""

    def __init__(self, in_channels: int = 1, hidden: int = AUDIO_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(8, hidden, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(), nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 1, N_BINS, 64]
        h = self.net(x)                # [B, hidden, F', 64]
        return h.mean(dim=(-2, -1))    # [B, hidden]


class AudioOnlyModel(nn.Module):
    """A1: acoustic embedding -> single linear head. No CREPE features."""

    def __init__(self, hidden: int = AUDIO_HIDDEN, n_classes: int = 4) -> None:
        super().__init__()
        self.encoder = AcousticCNN(hidden=hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x_audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x_audio))


class FusionModel(nn.Module):
    """A2 (and optional A4): [h_pitch ; h_audio] -> ONE linear head.
    `pitch_hidden`-architecture is byte-identical to `ContourCNN.net`
    (in_channels=2, kernel=5, dilations 1/2/4) -- duplicated, not
    subclassed, only so `encode()` can return h_pitch and h_audio
    separately for section 20's zeroing ablation."""

    def __init__(self, pitch_hidden: int = PITCH_HIDDEN, audio_hidden: int = AUDIO_HIDDEN, n_classes: int = 4) -> None:
        super().__init__()
        self.pitch_net = nn.Sequential(
            nn.Conv1d(2, pitch_hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.audio_encoder = AcousticCNN(hidden=audio_hidden)
        self.head = nn.Linear(pitch_hidden + audio_hidden, n_classes)

    def encode(self, x_contour: torch.Tensor, x_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pitch = self.pitch_net(x_contour).mean(dim=-1)  # [B, pitch_hidden]
        h_audio = self.audio_encoder(x_audio)               # [B, audio_hidden]
        return h_pitch, h_audio

    def forward(
        self, x_contour: torch.Tensor, x_audio: torch.Tensor, *,
        zero_pitch: bool = False, zero_audio: bool = False,
    ) -> torch.Tensor:
        h_pitch, h_audio = self.encode(x_contour, x_audio)
        if zero_pitch:
            h_pitch = torch.zeros_like(h_pitch)
        if zero_audio:
            h_audio = torch.zeros_like(h_audio)
        return self.head(torch.cat([h_pitch, h_audio], dim=-1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
