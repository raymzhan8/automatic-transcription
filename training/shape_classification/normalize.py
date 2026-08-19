"""Step 22 section 3, 6, 11: turn a raw log2-Hz phase-grid contour into the
direction/span-normalized shape q(x), its normalized velocity dq/dx, and a
small set of interpretable geometric summary features.

T0 handling (section 11): the branch between "span-normalize" and "keep
relative-cents" is chosen purely from the contour's own start/end pitch
values (never from the GT canonical_type label), so a genuinely near-flat
T1/T2/T3 primitive is treated identically to a T0 one.
"""

from __future__ import annotations

import numpy as np

# Minimum |p(1)-p(0)| (cents) to treat a primitive as having a "real" span
# worth direction/span-normalizing. Half a semitone: comfortably above the
# natural-vibrato range Step 22's label definitions call out for Fixed
# ("minor natural vibrato" -- doc/label_definitions.md), comfortably below
# a real bend between adjacent swaras (typically >=100c). Contour-derived
# only, never label-derived.
MIN_SPAN_CENTS = 50.0


def relative_cents(log2_contour: np.ndarray) -> np.ndarray:
    """r(x) = p(x) - p(0), in cents. Defined for every primitive, including
    T0. Tonic-independent (an additive constant cancels in the difference)."""
    return 1200.0 * (log2_contour - log2_contour[0])


def shape_normalize(r: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """q(x) per section 3/11. Returns (q, signed_span_cents, was_span_normalized).

    If |r[-1]| >= MIN_SPAN_CENTS: q = r / r[-1] (q(0)=0, q(1)=1 for both
    rising and falling trajectories -- the direction-removing step).
    Otherwise (near-flat contour, T0 or a degenerate bend): q = r unchanged
    (still in cents, deliberately NOT forced onto a [0,1] scale -- its near-
    zero magnitude relative to the spanning class is itself the Fixed
    signal, per section 11)."""
    span = float(r[-1])
    if abs(span) >= MIN_SPAN_CENTS:
        return r / span, span, True
    return r.copy(), span, False


def velocity(q: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """dq/dx via a numerically stable (central-difference, one-sided at the
    endpoints) derivative -- numpy's standard, not a hand-rolled finite
    difference."""
    return np.gradient(q, x_grid)


def slope_features(q: np.ndarray, v: np.ndarray, x_grid: np.ndarray) -> dict[str, float]:
    """Section 6's small, fixed feature set. `q`/`v` may come from either
    normalization branch (span-normalized or raw relative-cents) -- the
    feature names are generic ("displacement"), not assumed to be in [0,1]."""
    q25 = float(np.interp(0.25, x_grid, q))
    q50 = float(np.interp(0.50, x_grid, q))
    q75 = float(np.interp(0.75, x_grid, q))

    early = x_grid < (1.0 / 3.0)
    mid = (x_grid >= (1.0 / 3.0)) & (x_grid < (2.0 / 3.0))
    late = x_grid >= (2.0 / 3.0)
    early_vel = float(np.mean(np.abs(v[early]))) if early.any() else 0.0
    mid_vel = float(np.mean(np.abs(v[mid]))) if mid.any() else 0.0
    late_vel = float(np.mean(np.abs(v[late]))) if late.any() else 0.0

    phase_of_max_vel = float(x_grid[int(np.argmax(np.abs(v)))])

    early_disp = q50  # displacement completed by the midpoint
    late_disp = float(q[-1]) - q50  # displacement completed after the midpoint
    total_excursion = float(np.max(np.abs(q)))  # peak |displacement| anywhere on the contour

    return {
        "q25": q25, "q50": q50, "q75": q75,
        "early_velocity": early_vel, "mid_velocity": mid_vel, "late_velocity": late_vel,
        "phase_of_max_velocity": phase_of_max_vel,
        "early_displacement": early_disp, "late_displacement": late_disp,
        "early_minus_late_displacement": early_disp - late_disp,
        "total_excursion": total_excursion,
    }


ANALYTIC_FEATURE_NAMES = (
    "total_excursion", "q25", "q50", "q75",
    "early_displacement", "late_displacement", "phase_of_max_velocity",
)


def analytic_feature_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in ANALYTIC_FEATURE_NAMES], dtype=np.float64)
