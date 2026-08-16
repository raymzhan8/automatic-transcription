"""Step 11 §6: harmonic candidate-feature construction.

All features are computed on the SAME normalized log-CQT the rest of the
Step 10 pitch-diagnostics pipeline uses as model input (``spec`` from
``normalize_cqt``/fold CQT stats) — not raw linear magnitude. This keeps the
learned model consistent with ``FreqPreservingPitchModel``'s input contract.
Because the input is log-magnitude, a *sum* of per-harmonic branches is the
log-domain analog of the deterministic HPS *product* (log(ab) = log a + log
b) — so the cross-harmonic "harmonic_log_sum" feature below is a natural,
differentiable relative of the frozen HPS baseline, not an arbitrary feature.

For every candidate fundamental bin f and harmonic k in `harmonic_ks`, the
harmonic frequency k*f maps to a FIXED bin offset ``harmonic_bin_offset(k)``
on this log-frequency CQT grid (see salience_common.py's docstring for why
that offset is derived, not assumed). Feature channels per harmonic branch:
  - raw:  value at harmonic bin (shifted CQT slice)
  - local_max: max over a small +/-delta window around that bin (captures
    hits slightly off-partial due to imprecise harmonics/vibrato)
  - raw_minus_background: raw minus a wider local average (a coarse
    "is this a real peak or just ambient energy" signal)
Plus 2 cross-harmonic channels (see above + a normalized-vs-frame-max term).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.features import N_BINS
from training.pitch_diagnostics.salience_common import harmonic_bin_offset

DEFAULT_DELTA = 2  # local-max window half-width, bins (~33 cents)
DEFAULT_BG_DELTA = 8  # background window half-width, bins (~133 cents)


def n_feature_channels(harmonic_ks: tuple[int, ...]) -> int:
    return 3 * len(harmonic_ks) + 2


def build_harmonic_features(
    spec: torch.Tensor,
    *,
    candidate_lo_bin: int,
    candidate_hi_bin: int,
    harmonic_ks: tuple[int, ...],
    delta: int = DEFAULT_DELTA,
    bg_delta: int = DEFAULT_BG_DELTA,
    ablate_ks: frozenset[int] = frozenset(),
) -> torch.Tensor:
    """spec: [B,1,N_BINS,T] normalized log-CQT. Returns [B,C,F_cand,T].

    `ablate_ks`: harmonics to zero out at inference time (§27 ablation) while
    keeping the channel layout/count the trained scorer expects unchanged —
    zeroed harmonics also drop out of the cross-harmonic log_sum feature.
    """
    b, c1, f_bins, t = spec.shape
    assert c1 == 1 and f_bins == N_BINS
    offsets = {k: harmonic_bin_offset(k) for k in harmonic_ks}
    max_shift = max(offsets.values())

    # Pad frequency axis (dim=2) at the high-frequency end so every harmonic
    # slice [lo+shift : hi+shift] is in-bounds; out-of-range harmonics read as 0,
    # matching how baselines_b.harmonic_product_cents zero-fills.
    padded = F.pad(spec, (0, 0, 0, max_shift))  # pad (W_l,W_r,H_top,H_bot) -> H_bot=freq end

    frame_max = spec.amax(dim=2, keepdim=True)  # [B,1,1,T]

    branch_feats = []
    fundamental_raw = None
    log_sum = None
    for k in harmonic_ks:
        shift = offsets[k]
        lo = candidate_lo_bin + shift
        hi = candidate_hi_bin + shift
        raw = padded[:, :, lo:hi, :]  # [B,1,F_cand,T]
        if k in ablate_ks:
            raw = torch.zeros_like(raw)
        local_max = F.max_pool2d(raw, kernel_size=(2 * delta + 1, 1), stride=1, padding=(delta, 0))
        background = F.avg_pool2d(raw, kernel_size=(2 * bg_delta + 1, 1), stride=1, padding=(bg_delta, 0))
        raw_minus_bg = raw - background
        branch_feats.extend([raw, local_max, raw_minus_bg])
        if k not in ablate_ks:
            log_sum = raw if log_sum is None else (log_sum + raw)
        if k == 1:
            fundamental_raw = raw

    if log_sum is None:
        log_sum = torch.zeros_like(branch_feats[0])
    if fundamental_raw is None:
        # local-only control models may exclude k=1 explicitly at ablation time;
        # fall back to the first harmonic's raw channel for the frame-max-normalized feature.
        fundamental_raw = branch_feats[0]

    fund_over_max = fundamental_raw - frame_max  # log domain: E(f)/frame_max -> difference
    branch_feats.append(log_sum)
    branch_feats.append(fund_over_max)

    return torch.cat(branch_feats, dim=1)  # [B, 3*len(ks)+2, F_cand, T]


__all__ = ["build_harmonic_features", "n_feature_channels", "DEFAULT_DELTA", "DEFAULT_BG_DELTA"]
