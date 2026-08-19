"""Step 27: L1, the one-hidden-layer nonlinear fusion condition. Encoders
(pitch_net, AcousticCNN) are architecturally identical to Step 26's
FusionModel -- literally the same class definitions, duplicated only so this
module has no import-time dependency on step26_model.py's L0-specific head.
The only new thing is the fusion head itself:

    [h_pitch ; h_audio] -> Linear(32,16) -> ReLU -> Linear(16,4)

hidden=16 fixed before looking at any result (spec section 5) -- no hidden-
size sweep, no second layer, no normalization/dropout/residual/attention/
gating anywhere in the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.shape_classification.step26_model import AUDIO_HIDDEN, PITCH_HIDDEN, AcousticCNN

FUSION_HIDDEN = 16


class NonlinearFusionModel(nn.Module):
    """L1: [h_pitch ; h_audio] -> Linear -> ReLU -> Linear. `encode()` exposes
    h_pitch/h_audio separately (section 15's zeroing check); `forward()`
    also returns the hidden activation `z` (section 16's interaction
    diagnostic) alongside the logits."""

    def __init__(
        self, pitch_hidden: int = PITCH_HIDDEN, audio_hidden: int = AUDIO_HIDDEN,
        fusion_hidden: int = FUSION_HIDDEN, n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.pitch_net = nn.Sequential(
            nn.Conv1d(2, pitch_hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.audio_encoder = AcousticCNN(hidden=audio_hidden)
        self.fusion1 = nn.Linear(pitch_hidden + audio_hidden, fusion_hidden)
        self.act = nn.ReLU()
        self.head = nn.Linear(fusion_hidden, n_classes)

    def encode(self, x_contour: torch.Tensor, x_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pitch = self.pitch_net(x_contour).mean(dim=-1)
        h_audio = self.audio_encoder(x_audio)
        return h_pitch, h_audio

    def forward(
        self, x_contour: torch.Tensor, x_audio: torch.Tensor, *,
        zero_pitch: bool = False, zero_audio: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_pitch, h_audio = self.encode(x_contour, x_audio)
        if zero_pitch:
            h_pitch = torch.zeros_like(h_pitch)
        if zero_audio:
            h_audio = torch.zeros_like(h_audio)
        z = self.act(self.fusion1(torch.cat([h_pitch, h_audio], dim=-1)))
        return self.head(z), z


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
