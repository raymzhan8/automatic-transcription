"""Step 24: the four canonical trajectory shapes as fixed, zero-free-
parameter phase templates, recovered directly from idtap's own
`Trajectory.id0/id1/id2/id3` formulas (`idtap/classes/trajectory.py`) -- the
exact functions `dataset/canonical/contour.py` already uses to regenerate
GT targets -- not hand-approximated.

    id0 (Fixed):        log2_freq(x) = log_freqs[0]                          -> constant
    id1 (Cosine):        pi_x(x) = (cos(pi*(x+1))/2)+0.5 = 0.5 - 0.5*cos(pi*x) -> NO free parameter
    id2 (Sloped-start):  log_freq_out = (a-b)*(1-x)^slope + b                 -> q(x) = 1-(1-x)^slope
    id3 (Sloped-end):    log_freq_out = (b-a)*x^slope + a                    -> q(x) = x^slope

`slope` is a per-annotation attribute of idtap's Trajectory (default 2.0,
`DEFAULT_SLOPE` in `dataset/canonical/schema.py`) -- real corpus slopes for
raw types 2/3 range ~1.1-8.0, not uniformly 2.0 (see
docs/step_24_template_fitting.md section 2), so a single fixed template is
necessarily an approximation of a curve *family*, not a per-instance-exact
fit. slope=2.0 is used here because it is idtap's own documented default and
is the value that reproduces Step 22's own reference statistics exactly
(Cosine q(0.5)=0.500, Sloped-start q(0.5)=0.750, Sloped-end q(0.5)=0.250) --
not tuned, not chosen after seeing any Step 24 result.

All four templates are expressed in the SAME units: predicted RELATIVE
CENTS r_hat(x) = span_cents * f_k(x), where span_cents is the primitive's
OWN observed r(1) (from whichever source -- oracle or CREPE -- is being
scored). This is deliberate and is exactly how section 3's "do not divide
by a near-zero endpoint difference" is satisfied: templates are projected
INTO cents space by multiplying by the observed span, never by dividing
anything by it, so Fixed (span~0) is just the k=0 case of the identical
formula (f_fixed=0 identically) rather than a separately branched rule.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from training.shape_classification.contours import X_GRID

TEMPLATE_NAMES = ("Fixed", "Cosine", "Sloped-start", "Sloped-end")
CANONICAL_SLOPE = 2.0  # idtap's own DEFAULT_SLOPE; fixed before any Step 24 result was examined

# Reuse Step 22's already-established, already-justified "real vs. noise"
# cents scale (MIN_SPAN_CENTS) as the Huber delta -- not a new tuned
# constant, and fixed before looking at any Step 24 classification result.
HUBER_DELTA_CENTS = 50.0


def _f_fixed(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def _f_cosine(x: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _f_sloped_start(x: np.ndarray, slope: float = CANONICAL_SLOPE) -> np.ndarray:
    return 1.0 - (1.0 - x) ** slope


def _f_sloped_end(x: np.ndarray, slope: float = CANONICAL_SLOPE) -> np.ndarray:
    return x ** slope


TEMPLATE_FNS: tuple[Callable[[np.ndarray], np.ndarray], ...] = (_f_fixed, _f_cosine, _f_sloped_start, _f_sloped_end)


def template_curves(x_grid: np.ndarray = X_GRID) -> np.ndarray:
    """[4, N] array of f_k(x) -- the raw, unscaled [0,1]-phase templates."""
    return np.stack([f(x_grid) for f in TEMPLATE_FNS])


def huber(residual: np.ndarray, delta: float = HUBER_DELTA_CENTS) -> np.ndarray:
    a = np.abs(residual)
    return np.where(a <= delta, 0.5 * residual ** 2, delta * (a - 0.5 * delta))


def template_errors(r: np.ndarray, span_cents: float, *, robust: bool = False, x_grid: np.ndarray = X_GRID) -> list[float]:
    """[E_fixed, E_cosine, E_sloped_start, E_sloped_end] for one primitive's
    OWN observed relative-cents contour `r` (source-specific: pass CREPE's
    r/span to score against CREPE, oracle's to score against oracle -- never
    mixed). `span_cents` is that same source's own r(1) (signed)."""
    curves = template_curves(x_grid)
    errs = []
    for k in range(4):
        pred = span_cents * curves[k]
        resid = r - pred
        e = huber(resid) if robust else resid ** 2
        errs.append(float(np.mean(e)))
    return errs


def predict(errs: list[float]) -> int:
    return int(np.argmin(errs))
