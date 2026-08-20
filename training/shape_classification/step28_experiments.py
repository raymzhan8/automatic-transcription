"""Step 28: does immediately-adjacent observable trajectory context (C1) or
neighbor-pitch-only context (C2) improve on Step 26 L0's local-only baseline
(C0, reused unchanged)? Full comparison battery per the step spec.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import build_fold_split, load_kfold_manifest  # noqa: E402
from training.shape_classification.cnn_model import SEED, get_device  # noqa: E402
from training.shape_classification.dataset import CLASS_NAMES, OUT_DIR, build  # noqa: E402
from training.shape_classification.duration_span_analysis import DURATION_BUCKET_NAMES, DURATION_BUCKETS_S  # noqa: E402
from training.shape_classification.metrics_utils import eval_metrics, prediction_frequency  # noqa: E402
from training.shape_classification.step25_experiments import fold_consistency, recording_consistency  # noqa: E402
from training.shape_classification.step26_features import build as build_audio  # noqa: E402
from training.shape_classification.step28_model import count_params  # noqa: E402
from training.shape_classification.step28_neighbors import (  # noqa: E402
    build_neighbor_map, neighbor_coverage_report, train_only_transition_matrix,
)
from training.shape_classification.step28_train import FOUR_CLASS_NAMES, run_condition  # noqa: E402

STEP26_DIR = OUT_DIR / "step26"
STEP28_DIR = OUT_DIR / "step28"
N_FOLDS = 5


def load_cached(path: Path) -> dict:
    d = json.loads(path.read_text())
    d["per_fold_macro_f1"] = {int(k): v for k, v in d["per_fold_macro_f1"].items()}
    for fold in d["folds"]:
        fold["fold"] = int(fold["fold"])
    return d


def pooled_pred_true(result: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.concatenate([np.array(f["test_pred"]) for f in result["folds"]])
    true = np.concatenate([np.array(f["test_true"]) for f in result["folds"]])
    return pred, true


def _duration_lookup(records: list[dict]) -> dict[str, float]:
    return {r["primitive_id"]: r["duration_s"] for r in records}


def bucket_by_duration(result: dict, lookup: dict[str, float]) -> dict[str, dict]:
    by_pid_pred, by_pid_true = {}, {}
    for fold in result["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            by_pid_pred[pid] = p; by_pid_true[pid] = t
    out = {}
    for name, lo, hi in zip(DURATION_BUCKET_NAMES, DURATION_BUCKETS_S[:-1], DURATION_BUCKETS_S[1:]):
        pids = [pid for pid, v in lookup.items() if pid in by_pid_pred and lo <= v < hi]
        if not pids:
            out[name] = {"n": 0}; continue
        pred = np.array([by_pid_pred[pid] for pid in pids]); true = np.array([by_pid_true[pid] for pid in pids])
        m = eval_metrics(pred, true, FOUR_CLASS_NAMES)
        out[name] = {"n": len(pids), "macro_f1": m["macro_f1"], "sloped_start_f1": m["per_class"]["Sloped-start"]["f1"]}
    return out


def context_position_ablation(result: dict) -> dict:
    """Section 15: same trained weights, prev-only / next-only, no retraining."""
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in result["folds"]])
    normal = np.concatenate([np.array(f["test_pred"]) for f in result["folds"]])
    prev_only = np.concatenate([np.array(f["prev_only_test_pred"]) for f in result["folds"]])
    next_only = np.concatenate([np.array(f["next_only_test_pred"]) for f in result["folds"]])
    return {
        "both_macro_f1": eval_metrics(normal, pooled_true, FOUR_CLASS_NAMES)["macro_f1"],
        "prev_only_macro_f1": eval_metrics(prev_only, pooled_true, FOUR_CLASS_NAMES)["macro_f1"],
        "next_only_macro_f1": eval_metrics(next_only, pooled_true, FOUR_CLASS_NAMES)["macro_f1"],
        "prev_only_sloped_start_f1": eval_metrics(prev_only, pooled_true, FOUR_CLASS_NAMES)["per_class"]["Sloped-start"]["f1"],
        "next_only_sloped_start_f1": eval_metrics(next_only, pooled_true, FOUR_CLASS_NAMES)["per_class"]["Sloped-start"]["f1"],
        "both_sloped_start_f1": eval_metrics(normal, pooled_true, FOUR_CLASS_NAMES)["per_class"]["Sloped-start"]["f1"],
    }


def t2_recovery_breakage(records: list[dict], c0: dict, c1: dict, n: int = 4) -> dict[str, Any]:
    rec_by_pid = {r["primitive_id"]: r for r in records}
    c0_by_pid, c1_by_pid = {}, {}
    for fold in c0["folds"]:
        for pid, p, t in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"]):
            c0_by_pid[pid] = (p, t)
    for fold in c1["folds"]:
        for pid, p, t, probs in zip(fold["test_primitive_id"], fold["test_pred"], fold["test_true"], fold["test_probs"]):
            c1_by_pid[pid] = (p, t, probs)
    common = sorted(set(c0_by_pid) & set(c1_by_pid))
    set_a = [pid for pid in common if c0_by_pid[pid][0] == 1 and c0_by_pid[pid][1] == 2]
    recovered = [pid for pid in set_a if c1_by_pid[pid][0] == 2]
    set_b = [pid for pid in common if c0_by_pid[pid][0] == 1 and c0_by_pid[pid][1] == 1]
    broken = [pid for pid in set_b if c1_by_pid[pid][0] != 1]

    def describe(pid: str) -> dict:
        r = rec_by_pid[pid]
        return {"primitive_id": pid, "recording_id": r["recording_id"], "duration_s": round(r["duration_s"], 3),
                "true": CLASS_NAMES[r["canonical_type"]], "C0_pred": CLASS_NAMES[c0_by_pid[pid][0]],
                "C1_pred": CLASS_NAMES[c1_by_pid[pid][0]], "C1_confidence": round(max(c1_by_pid[pid][2]), 3)}

    recovered_sorted = sorted(recovered, key=lambda pid: max(c1_by_pid[pid][2]), reverse=True)
    broken_sorted = sorted(broken, key=lambda pid: max(c1_by_pid[pid][2]), reverse=True)
    return {"n_set_A": len(set_a), "n_recovered": len(recovered), "n_set_B": len(set_b), "n_broken": len(broken),
            "top_recovered": [describe(pid) for pid in recovered_sorted[:n]],
            "top_broken": [describe(pid) for pid in broken_sorted[:n]]}


def leakage_check(records: list[dict], neighbor_map: dict) -> dict:
    """Section 7: neighbor construction never pairs primitives across
    different recordings -- verified directly, not just asserted."""
    by_pid = {r["primitive_id"]: r for r in records}
    violations = 0
    checked = 0
    for pid, nb in neighbor_map.items():
        rid = by_pid[pid]["recording_id"]
        for key in ("prev", "next"):
            nid = nb[key]
            if nid is not None:
                checked += 1
                if by_pid[nid]["recording_id"] != rid:
                    violations += 1
    return {"n_neighbor_pairs_checked": checked, "n_cross_recording_violations": violations}


def oracle_neighbor_ceiling(records: list[dict], audio_lookup: dict, neighbor_map: dict) -> dict:
    """Section 17: O-context. center local embedding (frozen-style, but
    trained jointly here since we have no other frozen checkpoint available)
    + one-hot TRUE previous/next type -> Linear. Explicitly NOT a deployable
    condition; reported separately."""
    from training.shape_classification.step26_model import FusionModel
    from training.shape_classification.step26_train import (
        _channel_stats_contour, _standardize_contour, audio_channel_stats, select_audio_and_contour, standardize_audio,
    )
    from training.shape_classification.step23_train import class_weights_inverse_freq
    from training.shape_classification.cnn_model import BATCH_SIZE, MAX_EPOCHS, PATIENCE

    by_pid = {r["primitive_id"]: r for r in records}

    class OContextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = FusionModel(n_classes=4)
            self.head = nn.Linear(32 + 4 + 4, 4)

        def forward(self, xc, xa, prev_onehot, next_onehot):
            h_pitch, h_audio = self.fusion.encode(xc, xa)
            feat = torch.cat([h_pitch, h_audio, prev_onehot, next_onehot], dim=-1)
            return self.head(feat)

    manifest = load_kfold_manifest(REPO_ROOT)
    folds_out = []
    for fold in range(N_FOLDS):
        torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
        device = get_device()
        split = build_fold_split(manifest, fold, seed=SEED)

        def build_split(recording_ids):
            Xa, Xc, y, prevoh, nextoh = [], [], [], [], []
            for r in records:
                if r["recording_id"] not in recording_ids:
                    continue
                d = r["crepe"]; a = audio_lookup.get(r["primitive_id"])
                if d is None or a is None:
                    continue
                nb = neighbor_map[r["primitive_id"]]
                p_oh = np.zeros(4, dtype=np.float32); n_oh = np.zeros(4, dtype=np.float32)
                if nb["prev"] is not None and nb["prev"] in by_pid:
                    p_oh[by_pid[nb["prev"]]["canonical_type"]] = 1.0
                if nb["next"] is not None and nb["next"] in by_pid:
                    n_oh[by_pid[nb["next"]]["canonical_type"]] = 1.0
                Xc.append(np.stack([d["q"], d["v"]])); Xa.append(a); y.append(r["canonical_type"])
                prevoh.append(p_oh); nextoh.append(n_oh)
            return (np.stack(Xc).astype(np.float32), np.stack(Xa).astype(np.float32), np.array(y, dtype=np.int64),
                    np.stack(prevoh), np.stack(nextoh))

        Xc_tr, Xa_tr, y_tr, p_tr, n_tr = build_split(set(split.train_recording_ids))
        Xc_va, Xa_va, y_va, p_va, n_va = build_split(set(split.val_recording_ids))
        Xc_te, Xa_te, y_te, p_te, n_te = build_split(set(split.test_recording_ids))

        c_mu, c_sigma = _channel_stats_contour(Xc_tr)
        a_mu, a_sigma = audio_channel_stats(Xa_tr)
        Xc_tr, Xc_va, Xc_te = (_standardize_contour(x, c_mu, c_sigma) for x in (Xc_tr, Xc_va, Xc_te))
        Xa_tr, Xa_va, Xa_te = (standardize_audio(x, a_mu, a_sigma) for x in (Xa_tr, Xa_va, Xa_te))

        model = OContextModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        Xctr_t = torch.from_numpy(Xc_tr).to(device); Xatr_t = torch.from_numpy(Xa_tr).unsqueeze(1).to(device)
        ptr_t = torch.from_numpy(p_tr).to(device); ntr_t = torch.from_numpy(n_tr).to(device)
        ytr_t = torch.from_numpy(y_tr).to(device)
        Xcva_t = torch.from_numpy(Xc_va).to(device); Xava_t = torch.from_numpy(Xa_va).unsqueeze(1).to(device)
        pva_t = torch.from_numpy(p_va).to(device); nva_t = torch.from_numpy(n_va).to(device)
        Xcte_t = torch.from_numpy(Xc_te).to(device); Xate_t = torch.from_numpy(Xa_te).unsqueeze(1).to(device)
        pte_t = torch.from_numpy(p_te).to(device); nte_t = torch.from_numpy(n_te).to(device)

        n = len(y_tr)
        torch_gen = torch.Generator().manual_seed(SEED + fold)
        best_f1, best_state, stale = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            per_class_w = class_weights_inverse_freq(y_tr, 4)
            probs = per_class_w[y_tr]; probs = probs / probs.sum()
            perm = torch.multinomial(torch.from_numpy(probs), num_samples=n, replacement=True, generator=torch_gen).numpy()
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                opt.zero_grad(set_to_none=True)
                logits = model(Xctr_t[idx], Xatr_t[idx], ptr_t[idx], ntr_t[idx])
                loss = crit(logits, ytr_t[idx]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(Xcva_t, Xava_t, pva_t, nva_t).argmax(dim=-1).cpu().numpy()
            val_f1 = eval_metrics(val_pred, y_va, FOUR_CLASS_NAMES)["macro_f1"]
            if val_f1 > best_f1:
                best_f1, stale = val_f1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            test_pred = model(Xcte_t, Xate_t, pte_t, nte_t).argmax(dim=-1).cpu().numpy()
        folds_out.append({"fold": fold, "test_pred": test_pred.tolist(), "test_true": y_te.tolist()})
        print(f"  O-context fold {fold}: macro_f1={eval_metrics(test_pred, y_te, FOUR_CLASS_NAMES)['macro_f1']:.4f}")

    pooled_pred = np.concatenate([np.array(f["test_pred"]) for f in folds_out])
    pooled_true = np.concatenate([np.array(f["test_true"]) for f in folds_out])
    pooled = eval_metrics(pooled_pred, pooled_true, FOUR_CLASS_NAMES)
    per_fold_f1 = {f["fold"]: eval_metrics(np.array(f["test_pred"]), np.array(f["test_true"]), FOUR_CLASS_NAMES)["macro_f1"] for f in folds_out}
    return {"pooled": pooled, "per_fold_macro_f1": per_fold_f1,
            "grouped_mean_macro_f1": float(np.mean(list(per_fold_f1.values())))}


def main(*, run_oracle_ceiling: bool = True) -> None:
    STEP28_DIR.mkdir(parents=True, exist_ok=True)
    records = build()
    audio_lookup = build_audio()
    neighbor_map = build_neighbor_map(records)

    print("=== §3 neighbor coverage ===")
    coverage = neighbor_coverage_report(neighbor_map)
    print(coverage)

    print("\n=== §7 leakage check ===")
    leakage = leakage_check(records, neighbor_map)
    print(leakage)
    assert leakage["n_cross_recording_violations"] == 0

    c0 = load_cached(STEP26_DIR / "a2_full.json")
    c1 = load_cached(STEP28_DIR / "c1_full.json")
    c2 = load_cached(STEP28_DIR / "c2_full.json")

    print("\n=== §5 C0 reproduction check ===")
    print(f"C0 pooled macro_f1={c0['pooled']['macro_f1']:.4f} (Step 26 A2/Step 27 L0 reference: 0.3668)")

    print("\n=== §8 primary result table ===")
    for tag, res in (("C0", c0), ("C1", c1), ("C2", c2)):
        p = res["pooled"]
        row = " ".join(f"{p['per_class'][n]['f1']:.3f}" for n in FOUR_CLASS_NAMES)
        print(f"{tag:4s} macro_f1={p['macro_f1']:.4f} acc={p['accuracy']:.4f} [{','.join(FOUR_CLASS_NAMES)}]={row} "
              f"grouped_mean={res['grouped_mean_macro_f1']:.4f}+/-{res['grouped_std_macro_f1']:.4f} n_params={res.get('n_params')}")

    # §12 fold consistency
    fc = {"C1_vs_C0": fold_consistency(c0, c1), "C2_vs_C0": fold_consistency(c0, c2)}
    print("\n=== §12 fold consistency (C1 vs C0) ===", fc["C1_vs_C0"])
    print("=== §12 fold consistency (C2 vs C0) ===", fc["C2_vs_C0"])

    # §13 recording consistency
    rc = {"C1_vs_C0": recording_consistency(c0, c1), "C2_vs_C0": recording_consistency(c0, c2)}
    print("\n=== §13 recording consistency (C1 vs C0) ===", {k: v for k, v in rc["C1_vs_C0"].items() if k != "per_recording_delta"})

    # §10-11 per-class + confusion
    per_class = {tag: res["pooled"]["per_class"] for tag, res in (("C0", c0), ("C1", c1), ("C2", c2))}
    confusion = {tag: res["pooled"]["confusion_matrix"] for tag, res in (("C0", c0), ("C1", c1), ("C2", c2))}
    print("\n=== §11 confusion matrices ===")
    for tag in ("C0", "C1", "C2"):
        print(tag)
        for row in confusion[tag]:
            print(" ", row)

    # prediction frequency (for completeness)
    _, true_pooled = pooled_pred_true(c0)
    pred_freq = {"true": prediction_frequency(true_pooled, FOUR_CLASS_NAMES)}
    for tag, res in (("C0", c0), ("C1", c1), ("C2", c2)):
        pred, _ = pooled_pred_true(res)
        pred_freq[tag] = prediction_frequency(pred, FOUR_CLASS_NAMES)
    print("\n=== prediction frequency ===")
    for k, v in pred_freq.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})

    # §14 T2 recovery/breakage
    recovery = t2_recovery_breakage(records, c0, c1)
    print(f"\n=== §14 T2 recovery/breakage (C1 vs C0) === set_A(C0 Cosine/true SlS)={recovery['n_set_A']} "
          f"recovered={recovery['n_recovered']}  set_B(C0 correct Cosine)={recovery['n_set_B']} broken={recovery['n_broken']}")

    # §15 context-position ablation
    pos_ablation = {"C1": context_position_ablation(c1), "C2": context_position_ablation(c2)}
    print("\n=== §15 context-position ablation ===")
    for tag, d in pos_ablation.items():
        print(f"  {tag}: both={d['both_macro_f1']:.4f} prev_only={d['prev_only_macro_f1']:.4f} next_only={d['next_only_macro_f1']:.4f}  "
              f"SlS_F1: both={d['both_sloped_start_f1']:.3f} prev_only={d['prev_only_sloped_start_f1']:.3f} next_only={d['next_only_sloped_start_f1']:.3f}")

    # §16 transition diagnostic (TRAIN-only, descriptive)
    manifest = load_kfold_manifest(REPO_ROOT)
    transition_by_fold = {}
    for fold in range(N_FOLDS):
        split = build_fold_split(manifest, fold, seed=SEED)
        transition_by_fold[fold] = train_only_transition_matrix(records, neighbor_map, set(split.train_recording_ids))
    fold0_trans = transition_by_fold[0]["transition_probs"]
    print("\n=== §16 transition diagnostic (fold 0 TRAIN, descriptive only) ===")
    for prev_cls in FOUR_CLASS_NAMES:
        print(f"  P(*|{prev_cls}):", {k: round(v, 3) for k, v in fold0_trans[prev_cls].items()})

    # duration buckets
    dur_lookup = _duration_lookup(records)
    duration_report = {tag: bucket_by_duration(res, dur_lookup) for tag, res in (("C0", c0), ("C1", c1))}

    oracle_ceiling = None
    if run_oracle_ceiling:
        print("\n=== §17 O-context (oracle-neighbor-label ceiling, NOT deployable) ===")
        oracle_ceiling = oracle_neighbor_ceiling(records, audio_lookup, neighbor_map)
        print(f"O-context pooled macro_f1={oracle_ceiling['pooled']['macro_f1']:.4f} "
              f"grouped_mean={oracle_ceiling['grouped_mean_macro_f1']:.4f}")

    out = {
        "neighbor_coverage": coverage, "leakage_check": leakage,
        "C0": {k: v for k, v in c0.items() if k != "folds"}, "C1": {k: v for k, v in c1.items() if k != "folds"},
        "C2": {k: v for k, v in c2.items() if k != "folds"},
        "fold_consistency": fc, "recording_consistency": rc,
        "per_class": per_class, "confusion_matrices": confusion, "prediction_frequency": pred_freq,
        "t2_recovery_breakage": recovery, "context_position_ablation": pos_ablation,
        "transition_matrix_fold0": fold0_trans, "duration_buckets": duration_report,
        "oracle_neighbor_ceiling": (oracle_ceiling and {k: v for k, v in oracle_ceiling.items()}),
    }
    (STEP28_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nsaved to {STEP28_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
