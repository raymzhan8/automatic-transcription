"""Step 29: S2, the order-aware trajectory-sequence model. Operates on
complete trajectory embeddings [e_{i-1}, e_i, e_{i+1}] (sequence length 3),
NOT on 10ms acoustic frames -- unrelated to the earlier framewise BiGRU
(Step 9's Experiment C), which failed at a completely different timescale.

Reuses Step 28's `SharedEncoder` (imported directly, not re-duplicated,
so the architecture is guaranteed identical rather than just claimed so).
Each token is `[e_j ; present_j]` (33-dim, section 5's explicit
presence-bit mechanism), fed through ONE bidirectional GRU
(hidden=16/direction, 1 layer), reading the output at the CENTER position
only (h_center in R^32) into a single Linear(32,4) head. No second layer,
no dropout, no attention.

`center_only=True` (the S2-center-only capacity control, section 8) forces
BOTH neighbor slots to the zero/absent token regardless of what real
neighbor data is available, so the GRU sees a constant-shape sequence with
no neighbor content -- same parameter count, same architecture, only the
center trajectory's own information available.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.shape_classification.step28_model import FUSION_DIM, SharedEncoder

GRU_HIDDEN = 16


class BiGRUContextModel(nn.Module):
    def __init__(self, embed_dim: int = FUSION_DIM, hidden: int = GRU_HIDDEN, n_classes: int = 4) -> None:
        super().__init__()
        self.encoder = SharedEncoder()
        self.gru = nn.GRU(input_size=embed_dim + 1, hidden_size=hidden, num_layers=1,
                           bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(
        self, center_c, center_a, prev_c, prev_a, prev_mask, next_c, next_a, next_mask, *,
        center_only: bool = False,
    ) -> torch.Tensor:
        e_center = self.encoder.encode(center_c, center_a)
        ones = torch.ones_like(prev_mask)
        if center_only:
            e_prev = torch.zeros_like(e_center)
            e_next = torch.zeros_like(e_center)
            pm = torch.zeros_like(prev_mask)
            nm = torch.zeros_like(next_mask)
        else:
            e_prev = self.encoder.encode(prev_c, prev_a) * prev_mask.unsqueeze(-1)
            e_next = self.encoder.encode(next_c, next_a) * next_mask.unsqueeze(-1)
            pm, nm = prev_mask, next_mask

        tok_prev = torch.cat([e_prev, pm.unsqueeze(-1)], dim=-1)
        tok_center = torch.cat([e_center, ones.unsqueeze(-1)], dim=-1)
        tok_next = torch.cat([e_next, nm.unsqueeze(-1)], dim=-1)
        seq = torch.stack([tok_prev, tok_center, tok_next], dim=1)  # [B, 3, embed_dim+1]

        out, _ = self.gru(seq)  # [B, 3, 2*hidden]
        h_center = out[:, 1, :]
        return self.head(h_center)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
