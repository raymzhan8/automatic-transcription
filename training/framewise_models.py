"""Framewise model architectures (TCN Model B, BiGRU Model C)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyCNN(nn.Module):
    """Shared 2D frontend: pool frequency only, preserve time."""

    def __init__(self, in_channels: int = 1, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, None)),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.squeeze(2)


class TemporalConvNet(nn.Module):
    """Dilated Conv1d stack, same length in/out."""

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dilation in dilations:
            padding = (kernel_size - 1) // 2 * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.dilations = dilations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FramewiseTCNModel(nn.Module):
    """Model B: frequency CNN + TCN + type (and optional pitch) head."""

    def __init__(
        self,
        *,
        embed_channels: int = 128,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        num_types: int = 4,
        predict_pitch: bool = True,
    ) -> None:
        super().__init__()
        self.predict_pitch = predict_pitch
        self.freq_cnn = FrequencyCNN(out_channels=embed_channels)
        self.tcn = TemporalConvNet(
            channels=embed_channels,
            dilations=tcn_dilations,
        )
        self.type_head = nn.Linear(embed_channels, num_types)
        self.pitch_head = nn.Linear(embed_channels, 1) if predict_pitch else None

    def forward(self, spec: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.freq_cnn(spec)
        h = self.tcn(emb)
        h_t = h.transpose(1, 2)
        type_logits = self.type_head(h_t)
        if not self.predict_pitch:
            return type_logits
        assert self.pitch_head is not None
        pitch = self.pitch_head(h_t).squeeze(-1)
        return type_logits, pitch


class FramewiseBiGRUModel(nn.Module):
    """Model C: same frontend, BiGRU temporal encoder + dual heads."""

    def __init__(
        self,
        *,
        embed_channels: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        num_types: int = 4,
        dropout: float = 0.3,
        predict_pitch: bool = True,
    ) -> None:
        super().__init__()
        self.predict_pitch = predict_pitch
        self.freq_cnn = FrequencyCNN(out_channels=embed_channels)
        self.gru = nn.GRU(
            input_size=embed_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = gru_hidden * 2
        self.type_head = nn.Linear(feat_dim, num_types)
        self.pitch_head = nn.Linear(feat_dim, 1) if predict_pitch else None

    def forward(
        self,
        spec: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.freq_cnn(spec)
        h_seq = emb.transpose(1, 2)
        if lengths is not None:
            t_full = h_seq.size(1)
            packed = nn.utils.rnn.pack_padded_sequence(
                h_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            h_seq, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=t_full
            )
        else:
            h_seq, _ = self.gru(h_seq)
        type_logits = self.type_head(h_seq)
        if not self.predict_pitch:
            return type_logits
        assert self.pitch_head is not None
        pitch = self.pitch_head(h_seq).squeeze(-1)
        return type_logits, pitch


class FramewiseConditionalTCNModel(nn.Module):
    """Step 14: A/B/C/D feature-ablation model. Reuses FrequencyCNN and
    TemporalConvNet UNCHANGED across every condition -- only the input
    projection into the shared 128-channel TCN differs:

      A (audio only):        FrequencyCNN(spec) -> TCN
      B (pitch only):         Linear(pitch_dim -> embed) -> TCN
      C/D (audio + pitch):   concat(FrequencyCNN(spec), Linear(pitch_dim -> pitch_embed)) -> Linear(fuse) -> TCN

    so the temporal classifier itself (TCN + type_head) is byte-identical
    across all four conditions, per spec section 2.
    """

    def __init__(
        self,
        *,
        use_audio: bool = True,
        use_pitch: bool = False,
        pitch_dim: int = 4,
        embed_channels: int = 128,
        pitch_embed_dim: int = 16,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        num_types: int = 4,
    ) -> None:
        super().__init__()
        if not (use_audio or use_pitch):
            raise ValueError("at least one of use_audio/use_pitch must be True")
        self.use_audio = use_audio
        self.use_pitch = use_pitch
        self.freq_cnn = FrequencyCNN(out_channels=embed_channels) if use_audio else None
        if use_pitch:
            self.pitch_proj = nn.Linear(pitch_dim, embed_channels if not use_audio else pitch_embed_dim)
        else:
            self.pitch_proj = None
        self.fuse = nn.Linear(embed_channels + pitch_embed_dim, embed_channels) if (use_audio and use_pitch) else None
        self.tcn = TemporalConvNet(channels=embed_channels, dilations=tcn_dilations)
        self.type_head = nn.Linear(embed_channels, num_types)

    def forward(self, spec: torch.Tensor | None, pitch_feat: torch.Tensor | None) -> torch.Tensor:
        if self.use_audio and self.use_pitch:
            h_audio = self.freq_cnn(spec)  # [B, embed, T]
            r = self.pitch_proj(pitch_feat).transpose(1, 2)  # [B, pitch_embed, T]
            combined = self.fuse(torch.cat([h_audio, r], dim=1).transpose(1, 2)).transpose(1, 2)
        elif self.use_audio:
            combined = self.freq_cnn(spec)
        else:
            combined = self.pitch_proj(pitch_feat).transpose(1, 2)  # [B, embed, T]
        h = self.tcn(combined)
        type_logits = self.type_head(h.transpose(1, 2))
        return type_logits


class PitchMotionEncoder(nn.Module):
    """Step 15 P1/P3: small dedicated pitch-motion encoder. Input is the
    DENSE, per-frame, octave-unwrapped frame-to-frame delta (1 channel --
    the offset=1 slice of Step 14's phi, spec section 5's "frame-to-frame
    octave-unwrapped increments, let the temporal encoder integrate them"
    alternative), not the 4 hand-picked Step-14 offsets. A 1x1 Conv1d lifts
    it to `hidden` channels, then the SAME TemporalConvNet class used
    elsewhere in this project (kernel=5, dilations=(1,2,4,8) -> 61-frame /
    610ms receptive field, spec section 6's ~0.5-1.0s target) lets the
    model learn its own multi-scale combination instead of 4 fixed taps.
    """

    def __init__(self, *, in_channels: int = 1, hidden: int = 32,
                 tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.tcn = TemporalConvNet(channels=hidden, dilations=tcn_dilations)
        self.out_channels = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, T] -> [B, hidden, T]."""
        return self.tcn(self.proj(x))


class SalienceFrequencyCNN(nn.Module):
    """Step 15 P2: small 2D frontend for the windowed relative-salience
    input (73 bins, register-normalized -- see dense_relative_salience.py),
    structurally identical to FrequencyCNN (same kernel sizes / two (4,1)
    frequency-only pools / final AdaptiveMaxPool2d) but with much smaller
    channel widths (16/32 vs 32/64/128), since the input axis itself is
    already ~5x narrower and register-invariant."""

    def __init__(self, in_channels: int = 1, out_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(16, out_channels, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.AdaptiveMaxPool2d((1, None)),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(2)


class SalienceMotionEncoder(nn.Module):
    """Step 15 P2: SalienceFrequencyCNN + the same small TemporalConvNet
    used by PitchMotionEncoder, so P1 and P2 differ only in what enters the
    temporal encoder (a single decoded scalar vs. the full local salience
    shape), not in temporal-encoder capacity."""

    def __init__(self, *, hidden: int = 32, tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.freq_cnn = SalienceFrequencyCNN(out_channels=hidden)
        self.tcn = TemporalConvNet(channels=hidden, dilations=tcn_dilations)
        self.out_channels = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, W_BINS, T] -> [B, hidden, T]."""
        return self.tcn(self.freq_cnn(x))


class PitchOnlyClassifier(nn.Module):
    """Step 15: encoder (PitchMotionEncoder or SalienceMotionEncoder) + a
    linear type head -- the shared "temporal classifier" shape for P1/P2/P3
    (P0 reuses Step 14's FramewiseConditionalTCNModel(use_pitch-only)
    unchanged as the reproduction anchor)."""

    def __init__(self, encoder: nn.Module, num_types: int = 4) -> None:
        super().__init__()
        self.encoder = encoder
        self.type_head = nn.Linear(encoder.out_channels, num_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)  # [B, C, T]
        return self.type_head(h.transpose(1, 2))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_model(config: dict) -> nn.Module:
    arch = config.get("architecture", "tcn")
    predict_pitch = config.get("task", "type_and_pitch") != "type_only"
    if arch == "tcn":
        return FramewiseTCNModel(
            embed_channels=config.get("embed_channels", 128),
            tcn_dilations=tuple(config.get("tcn_dilations", [1, 2, 4, 8])),
            predict_pitch=predict_pitch,
        )
    if arch == "bigru":
        return FramewiseBiGRUModel(
            embed_channels=config.get("embed_channels", 128),
            gru_layers=config.get("gru_layers", 1),
            predict_pitch=predict_pitch,
        )
    raise ValueError(f"unknown architecture {arch!r}")


def component_param_counts() -> dict[str, int]:
    embed = 128
    freq = FrequencyCNN(out_channels=embed)
    tcn = TemporalConvNet(channels=embed, dilations=(1, 2, 4, 8))
    gru1 = nn.GRU(embed, 128, num_layers=1, batch_first=True, bidirectional=True)
    gru2 = nn.GRU(
        embed, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
    )
    tcn_type_head = nn.Linear(embed, 4)
    tcn_pitch_head = nn.Linear(embed, 1)
    gru_type_head = nn.Linear(256, 4)
    gru_pitch_head = nn.Linear(256, 1)

    model_tcn_full = FramewiseTCNModel(embed_channels=embed, predict_pitch=True)
    model_tcn_type = FramewiseTCNModel(embed_channels=embed, predict_pitch=False)
    model_gru1 = FramewiseBiGRUModel(embed_channels=embed, gru_layers=1)
    model_gru2 = FramewiseBiGRUModel(embed_channels=embed, gru_layers=2)

    return {
        "cnn": count_params(freq),
        "tcn": count_params(tcn),
        "gru_1layer": count_params(gru1),
        "gru_2layer": count_params(gru2),
        "tcn_type_head": count_params(tcn_type_head),
        "tcn_pitch_head": count_params(tcn_pitch_head),
        "gru_type_head": count_params(gru_type_head),
        "gru_pitch_head": count_params(gru_pitch_head),
        "total_tcn_type_only": count_params(model_tcn_type),
        "total_tcn_model": count_params(model_tcn_full),
        "total_gru_1layer_model": count_params(model_gru1),
        "total_gru_2layer_model": count_params(model_gru2),
    }


def tcn_receptive_field(
    kernel_size: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
) -> int:
    rf = 1
    for d in dilations:
        rf += (kernel_size - 1) * d
    return rf


if __name__ == "__main__":
    import json

    counts = component_param_counts()
    rf_frames = tcn_receptive_field()
    print(
        json.dumps(
            {
                **counts,
                "tcn_receptive_field_frames": rf_frames,
                "tcn_receptive_field_ms_at_10ms": rf_frames * 0.01 * 1000,
            },
            indent=2,
        )
    )
