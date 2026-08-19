"""Step 22 section 7: measure, don't assume, whether the data matches the
declared Fixed/Cosine/Sloped-start/Sloped-end semantics:

  Fixed:         total motion ~ 0
  Cosine:        displacement evolution ~symmetric in phase
  Sloped-start:  more displacement early than Cosine
  Sloped-end:    more displacement late than Cosine

Restricted to the span-normalized subset for T1-T3 (section 3/5's q(x) is
only meaningful there); T0 is checked on its own cents-magnitude scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)),
        "std": float(np.std(a)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
    }


def check_source(records, source_key: str) -> dict:
    out: dict = {}
    # Fixed: total motion should be ~0 (measured in cents, all T0 rows are
    # unnormalized r(x) by construction -- section 11).
    t0_excursion = [abs(r[source_key]["features"]["total_excursion"]) for r in records
                    if r["canonical_type"] == 0 and r[source_key] is not None]
    out["Fixed_total_excursion_cents"] = _stats(t0_excursion)

    for t in (1, 2, 3):
        pool = [r for r in records if r["canonical_type"] == t and r[source_key] is not None
                and r[source_key]["span_normalized"]]
        q25 = [r[source_key]["features"]["q25"] for r in pool]
        q50 = [r[source_key]["features"]["q50"] for r in pool]
        q75 = [r[source_key]["features"]["q75"] for r in pool]
        sym = [r[source_key]["features"]["early_minus_late_displacement"] for r in pool]
        out[CLASS_NAMES[t]] = {
            "n_span_normalized": len(pool),
            "q25": _stats(q25), "q50": _stats(q50), "q75": _stats(q75),
            "early_minus_late_displacement": _stats(sym),
        }
    return out


def main() -> None:
    records = build()
    result = {"oracle": check_source(records, "oracle"), "crepe": check_source(records, "crepe")}
    (OUT_DIR / "semantic_check.json").write_text(json.dumps(result, indent=2) + "\n")

    print("=== Step 22 §7 semantic hypothesis check ===")
    for src in ("oracle", "crepe"):
        print(f"\n-- {src} --")
        fx = result[src]["Fixed_total_excursion_cents"]
        print(f"Fixed total excursion (cents): median={fx['median']:.2f} mean={fx['mean']:.2f}")
        for cls in ("Cosine", "Sloped-start", "Sloped-end"):
            d = result[src][cls]
            print(f"{cls:14s} n={d['n_span_normalized']:4d}  "
                  f"q25={d['q25']['median']:+.3f}  q50={d['q50']['median']:+.3f}  q75={d['q75']['median']:+.3f}  "
                  f"early-late={d['early_minus_late_displacement']['median']:+.3f}")
    print(f"\nsaved to {OUT_DIR / 'semantic_check.json'}")


if __name__ == "__main__":
    main()
