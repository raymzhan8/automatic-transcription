"""Step 22 section 18: direct Sloped-start (T2) vs. Sloped-end (T3)
comparison -- does CREPE preserve the front-loaded vs. back-loaded slope
distinction, the actual semantic difference between these two classes
(more important here than generic turning-point recall)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.shape_classification.contours import X_GRID  # noqa: E402
from training.shape_classification.dataset import OUT_DIR, build  # noqa: E402

FIG_DIR = OUT_DIR / "figures"


def _pool(records, source_key: str, t: int):
    return [r for r in records if r["canonical_type"] == t and r[source_key] is not None
            and r[source_key]["span_normalized"]]


def analyze_source(records, source_key: str) -> dict:
    t2 = _pool(records, source_key, 2)
    t3 = _pool(records, source_key, 3)
    sym2 = np.array([r[source_key]["features"]["early_minus_late_displacement"] for r in t2])
    sym3 = np.array([r[source_key]["features"]["early_minus_late_displacement"] for r in t3])
    phase_max2 = np.array([r[source_key]["features"]["phase_of_max_velocity"] for r in t2])
    phase_max3 = np.array([r[source_key]["features"]["phase_of_max_velocity"] for r in t3])

    return {
        "n_T2": len(t2), "n_T3": len(t3),
        "early_minus_late_T2": {"mean": float(sym2.mean()), "median": float(np.median(sym2))},
        "early_minus_late_T3": {"mean": float(sym3.mean()), "median": float(np.median(sym3))},
        "frac_T2_front_loaded": float(np.mean(sym2 > 0)),   # expect T2 (sloped-start) mostly positive
        "frac_T3_back_loaded": float(np.mean(sym3 < 0)),    # expect T3 (sloped-end) mostly negative
        "phase_of_max_velocity_T2": {"mean": float(phase_max2.mean()), "median": float(np.median(phase_max2))},
        "phase_of_max_velocity_T3": {"mean": float(phase_max3.mean()), "median": float(np.median(phase_max3))},
        # simplest possible "does the sign alone separate T2 from T3" classifier:
        "sign_separation_accuracy": float(
            (np.sum(sym2 > 0) + np.sum(sym3 < 0)) / (len(sym2) + len(sym3))
        ) if (len(sym2) + len(sym3)) else None,
    }


def plot(records) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, src in zip(axes, ("oracle", "crepe")):
        t2 = _pool(records, src, 2)
        t3 = _pool(records, src, 3)
        m2 = np.median(np.stack([r[src]["q"] for r in t2]), axis=0)
        m3 = np.median(np.stack([r[src]["q"] for r in t3]), axis=0)
        ax.plot(X_GRID, m2, label=f"Sloped-start median (n={len(t2)})", color="tab:green")
        ax.plot(X_GRID, m3, label=f"Sloped-end median (n={len(t3)})", color="tab:red")
        ax.axline((0, 0), (1, 1), color="gray", linestyle="--", linewidth=0.8, label="symmetric reference")
        ax.set_title(src)
        ax.set_xlabel("normalized phase x")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("q(x)")
    fig.suptitle("Step 22 §18: Sloped-start vs. Sloped-end median shape")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "t2_vs_t3.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    records = build()
    result = {src: analyze_source(records, src) for src in ("oracle", "crepe")}
    (OUT_DIR / "t2_t3_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(records)
    print("=== Step 22 §18 T2 (Sloped-start) vs T3 (Sloped-end) ===")
    for src in ("oracle", "crepe"):
        d = result[src]
        print(f"{src}: frac_T2_front_loaded={d['frac_T2_front_loaded']:.3f}  "
              f"frac_T3_back_loaded={d['frac_T3_back_loaded']:.3f}  "
              f"sign_separation_accuracy={d['sign_separation_accuracy']:.3f}  "
              f"phase_of_max_vel T2={d['phase_of_max_velocity_T2']['median']:.3f} "
              f"T3={d['phase_of_max_velocity_T3']['median']:.3f}")
    print(f"\nsaved to {OUT_DIR / 't2_t3_analysis.json'}")


if __name__ == "__main__":
    main()
