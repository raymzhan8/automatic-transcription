"""Step 25 section 4: the ONE fixed scale-normalized template-evidence
representation, reusing Step 24's `template_errors` API unchanged.

    m = min_k E_k
    d = mean_k E_k + eps
    z_k = (E_k - m) / d

Chosen before evaluating any Step 25 result: best-fitting template gets
z=0, differences between templates are retained, and the huge span/duration
-driven scale variation Step 24 §16 documented in raw E_k is substantially
divided out. No alternative normalization was tried.
"""

from __future__ import annotations

import numpy as np

from training.shape_classification.templates import template_errors

Z_EPS = 1e-6  # numerical stability only, not a tuned constant


def compute_z(r: np.ndarray, span_cents: float, *, robust: bool = False) -> list[float]:
    errs = np.array(template_errors(r, span_cents, robust=robust))
    m = errs.min()
    d = errs.mean() + Z_EPS
    return ((errs - m) / d).tolist()


def build_z_lookup(records: list[dict], source_key: str) -> dict[str, list[float]]:
    """primitive_id -> [z_fixed, z_cosine, z_sloped_start, z_sloped_end],
    computed without mutating the shared Step 22 corpus records."""
    out = {}
    for r in records:
        d = r[source_key]
        if d is None:
            continue
        out[r["primitive_id"]] = compute_z(d["r"], d["span_cents"])
    return out
