"""Step 28: C1/C2 context models. One shared encoder pair (pitch_net,
AcousticCNN -- architecturally identical to Step 26/27's, duplicated here
only for import isolation) is applied independently to the center, previous,
and next primitive's own (contour, audio) inputs -- not three separate
encoders. Missing neighbors are encoded from a zero placeholder and then
multiplied by their presence mask, so a missing neighbor's contribution is
exactly zero and contributes exactly zero gradient to the encoder (spec
section 3's "explicit mask or zero vector").

No hidden layer anywhere (spec section 5/6) -- both C1 and C2 are a single
Linear over the concatenated embeddings (+ 2 presence-mask scalars)."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.shape_classification.step26_model import AUDIO_HIDDEN, PITCH_HIDDEN, AcousticCNN

FUSION_DIM = PITCH_HIDDEN + AUDIO_HIDDEN  # 32, e_i's dimension


class SharedEncoder(nn.Module):
    """pitch_net + AcousticCNN, applied per-primitive. `encode` returns the
    full 32-dim [h_pitch;h_audio]; `encode_pitch_only` returns just h_pitch
    (16-dim) for C2's neighbor branches."""

    def __init__(self, pitch_hidden: int = PITCH_HIDDEN, audio_hidden: int = AUDIO_HIDDEN) -> None:
        super().__init__()
        self.pitch_net = nn.Sequential(
            nn.Conv1d(2, pitch_hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(pitch_hidden, pitch_hidden, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.audio_encoder = AcousticCNN(hidden=audio_hidden)

    def encode_pitch(self, x_contour: torch.Tensor) -> torch.Tensor:
        return self.pitch_net(x_contour).mean(dim=-1)

    def encode(self, x_contour: torch.Tensor, x_audio: torch.Tensor) -> torch.Tensor:
        h_pitch = self.encode_pitch(x_contour)
        h_audio = self.audio_encoder(x_audio)
        return torch.cat([h_pitch, h_audio], dim=-1)


class C1ContextModel(nn.Module):
    """[e_prev ; e_center ; e_next ; prev_mask ; next_mask] -> Linear(98,4)."""

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.head = nn.Linear(3 * FUSION_DIM + 2, n_classes)

    def forward(
        self, center_c, center_a, prev_c, prev_a, prev_mask, next_c, next_a, next_mask,
    ) -> torch.Tensor:
        e_center = self.encoder.encode(center_c, center_a)
        e_prev = self.encoder.encode(prev_c, prev_a) * prev_mask.unsqueeze(-1)
        e_next = self.encoder.encode(next_c, next_a) * next_mask.unsqueeze(-1)
        feat = torch.cat([e_prev, e_center, e_next, prev_mask.unsqueeze(-1), next_mask.unsqueeze(-1)], dim=-1)
        return self.head(feat)


class C2ContextModel(nn.Module):
    """center gets full [h_pitch;h_audio] (32-dim); neighbors get pitch-only
    (16-dim each). [h_pitch_prev ; e_center ; h_pitch_next ; masks] -> Linear(66,4)."""

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.head = nn.Linear(2 * PITCH_HIDDEN + FUSION_DIM + 2, n_classes)

    def forward(
        self, center_c, center_a, prev_c, prev_mask, next_c, next_mask,
    ) -> torch.Tensor:
        e_center = self.encoder.encode(center_c, center_a)
        h_prev = self.encoder.encode_pitch(prev_c) * prev_mask.unsqueeze(-1)
        h_next = self.encoder.encode_pitch(next_c) * next_mask.unsqueeze(-1)
        feat = torch.cat([h_prev, e_center, h_next, prev_mask.unsqueeze(-1), next_mask.unsqueeze(-1)], dim=-1)
        return self.head(feat)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
