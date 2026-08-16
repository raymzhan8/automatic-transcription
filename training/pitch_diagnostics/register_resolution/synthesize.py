"""Step 12 Phase-A checkpoint (plan step 8): synthesize sections 2-7,9,11,13
against spec §31's success criteria and state explicitly which of D1-D4 are
justified for Phase B. Reads the JSON outputs already written by the other
Phase-A scripts; writes no new measurements of its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pitch_diagnostics.common import write_json  # noqa: E402
from training.pitch_diagnostics.register_resolution.common import REG_DIR  # noqa: E402


def _load(name: str) -> dict:
    return json.loads((REG_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    reconciliation = _load("step11_reconciliation.json")
    run_durs = _load("octave_run_durations.json")
    trans = _load("octave_transition_matrix.json")
    gt_topk = _load("gt_topk_given_wrong_octave.json")
    oracle_topk = _load("oracle_topk.json")
    oct_oracle = _load("octave_oracle.json")
    static_prior = _load("static_register_prior.json")
    disagree = _load("disagreement_analysis.json")

    findings = {}

    findings["range_fix"] = {
        "conclusions_survive_full_range": reconciliation["conclusions_survive"],
        "resolution": "learned model evaluated on native trained range for all Phase-A diagnostics; HPS unaffected",
    }

    findings["run_structure"] = {
        m: {
            "n_runs": run_durs[m].get("n_runs"),
            "median_s": run_durs[m].get("median_s"),
            "p90_s": run_durs[m].get("p90_s"),
            "verdict": "sustained (median run well above one frame)" if (run_durs[m].get("median_s") or 0) > 0.05 else "mostly isolated glitches",
        } for m in ("hps", "learned")
    }

    findings["transition_stickiness"] = {
        m: {
            "P_k0_given_k0": trans[m]["P_k0_given_k0"],
            "P_kplus1_given_kplus1": trans[m]["P_kplus1_given_kplus1"],
            "verdict": "wrong-octave state is sticky once entered" if trans[m]["P_kplus1_given_kplus1"] > 0.5 else "wrong-octave state is not particularly sticky",
        } for m in ("hps", "learned")
    }

    findings["gt_availability_given_wrong_octave"] = {
        m: {
            "p_top3": gt_topk[m]["p_gt_in_top3_given_wrong_octave"],
            "p_top5": gt_topk[m]["p_gt_in_top5_given_wrong_octave"],
            "verdict": "GT commonly present in top-k -> decoding problem" if (gt_topk[m]["p_gt_in_top3_given_wrong_octave"] or 0) > 0.3 else "GT often absent from top-k -> representation problem",
        } for m in ("hps", "learned")
    }

    findings["oracle_headroom"] = {
        m: {
            "argmax_mae": oracle_topk[m]["argmax"]["mae_cents"],
            "oracle_top5_mae": oracle_topk[m]["oracle_top5"]["mae_cents"],
            "headroom_cents": oracle_topk[m]["argmax"]["mae_cents"] - oracle_topk[m]["oracle_top5"]["mae_cents"],
            "octave_oracle_mae": oct_oracle[m]["mae_cents"],
        } for m in ("hps", "learned")
    }

    findings["static_prior_effect"] = {
        m: {
            "raw_mae": static_prior["test_summary"][m]["raw_argmax"]["mae_cents"],
            "prior_mae": static_prior["test_summary"][m]["static_prior_decode"]["mae_cents"],
            "improvement_cents": static_prior["test_summary"][m]["raw_argmax"]["mae_cents"] - static_prior["test_summary"][m]["static_prior_decode"]["mae_cents"],
        } for m in ("hps", "learned")
    }

    findings["disagreement"] = {
        "fractions": disagree["fractions"],
        "complementary": disagree["fractions"]["B_hps_correct_learned_wrong"] > 0.03 and disagree["fractions"]["C_learned_correct_hps_wrong"] > 0.03,
    }

    # ---- decoder justification verdicts ----
    d1_justified = any(findings["static_prior_effect"][m]["improvement_cents"] > 10 for m in ("hps", "learned"))
    d2_justified = findings["disagreement"]["complementary"]
    oracle_headroom_substantial = any(findings["oracle_headroom"][m]["headroom_cents"] > 30 for m in ("hps", "learned"))
    runs_sustained = any(findings["run_structure"][m]["verdict"].startswith("sustained") for m in ("hps", "learned"))
    gt_available = any(findings["gt_availability_given_wrong_octave"][m]["verdict"].startswith("GT commonly") for m in ("hps", "learned"))
    d3_d4_justified = oracle_headroom_substantial and gt_available and runs_sustained

    verdict = {
        "D1_static_prior_justified": d1_justified,
        "D2_fusion_justified": d2_justified,
        "D3_D4_viterbi_justified": d3_d4_justified,
        "reasoning": {
            "D1": f"static prior improvement > 10c for at least one method: {d1_justified}",
            "D2": f"HPS/learned disagreement is complementary (both B and C fractions > 3%): {d2_justified}",
            "D3_D4": f"oracle top-5 headroom > 30c AND GT commonly in top-k given wrong-octave AND runs are sustained (not just glitches): {d3_d4_justified}",
        },
    }
    findings["phase_b_scope_verdict"] = verdict

    write_json(REG_DIR / "phase_a_synthesis.json", findings)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
