"""Step 22 section 5: plot normalized q(x) shapes by class, before any
training -- a sanity check that must pass (oracle shapes visibly separate)
before the rest of the step proceeds."""

from __future__ import annotations

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
from training.shape_classification.dataset import CLASS_NAMES, build  # noqa: E402

FIG_DIR = REPO_ROOT / "output" / "shape_classification" / "figures"
RNG_SEED = 0
N_EXAMPLES = 20


def _plot_source(records, source_key: str, title: str, out_path: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
    for t in range(4):
        ax = axes[t]
        # Section 5's q(x) "shape" comparison is only meaningful for the
        # span-normalized subset (section 3/11: near-zero-span primitives,
        # including every T0 by construction, keep the raw-cents r(x)
        # instead and are excluded here so one outlier's cents scale
        # doesn't swamp the [0,1] shape axis for the rest of the class).
        pool = [r for r in records if r["canonical_type"] == t and r[source_key] is not None]
        if t != 0:
            pool = [r for r in pool if r[source_key]["span_normalized"]]
        qs = [r[source_key]["q"] for r in pool]
        n_total = sum(1 for r in records if r["canonical_type"] == t and r[source_key] is not None)
        if not qs:
            ax.set_title(f"{CLASS_NAMES[t]} (n=0/{n_total})")
            continue
        qs = np.stack(qs)
        idxs = rng.choice(len(qs), size=min(N_EXAMPLES, len(qs)), replace=False)
        for i in idxs:
            ax.plot(X_GRID, qs[i], color="tab:blue", alpha=0.15, linewidth=0.8)
        median = np.median(qs, axis=0)
        q25 = np.percentile(qs, 25, axis=0)
        q75 = np.percentile(qs, 75, axis=0)
        ax.fill_between(X_GRID, q25, q75, color="tab:orange", alpha=0.3, label="IQR")
        ax.plot(X_GRID, median, color="tab:orange", linewidth=2.5, label="median")
        title_n = f"n={len(qs)}" if t == 0 else f"n={len(qs)}/{n_total} span-normalized"
        ax.set_title(f"{CLASS_NAMES[t]} ({title_n})")
        ax.set_xlabel("normalized phase x")
        if t == 0:
            ax.set_ylabel("q(x)  [span-normalized, or cents if near-flat]")
            ax.legend(fontsize=8)
        ax.axhline(0.0, color="gray", linewidth=0.5)
    fig.suptitle(title)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    records = build()
    _plot_source(records, "oracle", "Step 22 §5: Oracle normalized shape q(x) by class",
                 FIG_DIR / "oracle_normalized_shapes.png")
    _plot_source(records, "crepe", "Step 22 §5: CREPE normalized shape q(x) by class",
                 FIG_DIR / "crepe_normalized_shapes.png")


if __name__ == "__main__":
    main()
