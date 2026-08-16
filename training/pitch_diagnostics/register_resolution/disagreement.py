"""Step 12 spec §13: HPS vs learned-salience disagreement categorization.
Correct/wrong defined via octave_k==0 (i.e. within ~600c of true pitch, the
same "correct octave" definition used everywhere else in this step)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.metrics import NUM_TYPES, TYPE_NAMES  # noqa: E402
from training.pitch_diagnostics.common import write_json  # noqa: E402
from training.pitch_diagnostics.register_resolution.collect import build  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import REG_DIR  # noqa: E402


def main() -> None:
    records = build()
    cats = []  # per-frame category strings, plus metadata arrays
    entropy_l, entropy_h, types, ranks_spec, recs, durs, kdirs, margin_h, margin_l = ([] for _ in range(9))

    for rec in records:
        hps_k = np.asarray(rec["hps"]["octave_k"]) == 0
        learned_k = np.asarray(rec["learned"]["octave_k"]) == 0
        both = hps_k & learned_k
        hps_only = hps_k & ~learned_k
        learned_only = ~hps_k & learned_k
        neither = ~hps_k & ~learned_k
        n = len(hps_k)
        cat = np.full(n, "D_both_wrong", dtype=object)
        cat[both] = "A_both_correct"
        cat[hps_only] = "B_hps_correct_learned_wrong"
        cat[learned_only] = "C_learned_correct_hps_wrong"
        cats.append(cat)
        entropy_l.append(rec["learned"]["entropy"]); entropy_h.append(rec["hps"]["entropy"])
        types.append(rec["trajectory_type"])
        recs.append(np.full(n, rec["recording_id"], dtype=object))
        margin_h.append(rec["hps"]["margin12"]); margin_l.append(rec["learned"]["margin12"])

    cat_all = np.concatenate(cats)
    n_total = len(cat_all)
    fractions = {c: float((cat_all == c).mean()) for c in
                 ("A_both_correct", "B_hps_correct_learned_wrong", "C_learned_correct_hps_wrong", "D_both_wrong")}

    entropy_l_all = np.concatenate(entropy_l); entropy_h_all = np.concatenate(entropy_h)
    types_all = np.concatenate(types)
    recs_all = np.concatenate(recs)
    margin_h_all = np.concatenate(margin_h); margin_l_all = np.concatenate(margin_l)

    def _breakdown(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        by_type = {TYPE_NAMES[t]: int(((types_all == t) & mask).sum()) for t in range(NUM_TYPES)}
        by_rec = {}
        for rid in np.unique(recs_all[mask]):
            by_rec[str(rid)] = int(((recs_all == rid) & mask).sum())
        return {
            "n": int(mask.sum()),
            "mean_learned_entropy": float(entropy_l_all[mask].mean()),
            "mean_hps_entropy": float(entropy_h_all[mask].mean()),
            "mean_hps_margin12": float(margin_h_all[mask].mean()),
            "mean_learned_margin12": float(margin_l_all[mask].mean()),
            "by_trajectory_type": by_type,
            "by_recording": dict(sorted(by_rec.items(), key=lambda kv: -kv[1])[:10]),
        }

    out = {
        "n_total": n_total,
        "fractions": fractions,
        "B_hps_correct_learned_wrong": _breakdown(cat_all == "B_hps_correct_learned_wrong"),
        "C_learned_correct_hps_wrong": _breakdown(cat_all == "C_learned_correct_hps_wrong"),
        "complementarity_note": (
            "Complementary iff both B and C fractions are non-trivial (not one-sided); "
            "see fractions above for the actual verdict."
        ),
    }
    write_json(REG_DIR / "disagreement_analysis.json", out)
    print("disagreement fractions:", fractions)


if __name__ == "__main__":
    main()
