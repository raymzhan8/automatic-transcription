"""Paired B1 vs C comparison, with B0 as context only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.framewise_dataset import RecordingLaneIndex  # noqa: E402
from training.metrics import TYPE_NAMES  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return float(new) - float(old)


def _get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _train_gap(run_dir: Path, eval_summary: dict[str, Any]) -> dict[str, Any]:
    result_path = run_dir / "train_result.json"
    if not result_path.exists():
        return {}
    result = _load(result_path)
    best_epoch = result.get("best_epoch")
    train_f1 = train_pitch = val_f1 = val_pitch = None
    for rec in result.get("history") or []:
        if rec.get("epoch") == best_epoch:
            train_f1 = _get(rec, "train", "macro_f1")
            train_pitch = _get(rec, "train", "pitch_mae_cents")
            val_f1 = _get(rec, "val", "macro_f1")
            val_pitch = _get(rec, "val", "pitch_mae_cents")
            break
    test_f1 = _get(eval_summary, "frame_metrics", "macro_f1")
    test_pitch = _get(eval_summary, "pitch", "overall", "mae_cents")
    return {
        "best_epoch": best_epoch,
        "train_macro_f1": train_f1,
        "val_macro_f1": val_f1 if val_f1 is not None else result.get("best_val_macro_f1"),
        "test_macro_f1": test_f1,
        "train_pitch_mae_cents": train_pitch,
        "val_pitch_mae_cents": val_pitch,
        "test_pitch_mae_cents": test_pitch,
        "train_minus_val_f1": _delta(train_f1, val_f1),
        "val_minus_test_f1": _delta(val_f1, test_f1),
        "test_minus_train_pitch_mae": _delta(test_pitch, train_pitch),
    }


def fold_delta(b1: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    fm1, fmc = b1["frame_metrics"], c["frame_metrics"]
    tm1 = b1.get("trajectory_metrics") or {}
    tmc = c.get("trajectory_metrics") or {}
    per_class = {}
    for cls in ("T0", "T1", "T2", "T3"):
        a = fm1.get("per_class", {}).get(cls, {})
        b = fmc.get("per_class", {}).get(cls, {})
        per_class[cls] = {
            "b1_f1": a.get("f1"),
            "c_f1": b.get("f1"),
            "delta_f1": _delta(b.get("f1"), a.get("f1")),
            "b1_recall": a.get("recall"),
            "c_recall": b.get("recall"),
            "delta_recall": _delta(b.get("recall"), a.get("recall")),
        }
    pairs = {}
    p1 = b1.get("confusion_pairs") or {}
    pc = c.get("confusion_pairs") or {}
    for k in sorted(set(p1) | set(pc)):
        pairs[k] = {
            "b1": p1.get(k, 0),
            "c": pc.get(k, 0),
            "delta": pc.get(k, 0) - p1.get(k, 0),
        }
    pitch1 = _get(b1, "pitch", "overall") or {}
    pitchc = _get(c, "pitch", "overall") or {}
    base1 = _get(b1, "pitch", "mean_pitch_baseline") or {}
    basec = _get(c, "pitch", "mean_pitch_baseline") or {}

    recs = {}
    ids = sorted(set(b1.get("per_recording", {})) | set(c.get("per_recording", {})))
    for rid in ids:
        r1 = b1.get("per_recording", {}).get(rid, {})
        rc = c.get("per_recording", {}).get(rid, {})
        recs[rid] = {
            "frame_macro_f1": {
                "b1": r1.get("frame_macro_f1"),
                "c": rc.get("frame_macro_f1"),
                "delta": _delta(rc.get("frame_macro_f1"), r1.get("frame_macro_f1")),
            },
            "pitch_mae_cents": {
                "b1": r1.get("pitch_mae_cents"),
                "c": rc.get("pitch_mae_cents"),
                "delta": _delta(rc.get("pitch_mae_cents"), r1.get("pitch_mae_cents")),
            },
        }

    return {
        "fold_index": c.get("fold_index", b1.get("fold_index")),
        "frame_accuracy": {
            "b1": fm1.get("accuracy"),
            "c": fmc.get("accuracy"),
            "delta": _delta(fmc.get("accuracy"), fm1.get("accuracy")),
        },
        "frame_macro_f1": {
            "b1": fm1.get("macro_f1"),
            "c": fmc.get("macro_f1"),
            "delta": _delta(fmc.get("macro_f1"), fm1.get("macro_f1")),
        },
        "trajectory_accuracy": {
            "b1": tm1.get("accuracy"),
            "c": tmc.get("accuracy"),
            "delta": _delta(tmc.get("accuracy"), tm1.get("accuracy")),
        },
        "trajectory_macro_f1": {
            "b1": tm1.get("macro_f1"),
            "c": tmc.get("macro_f1"),
            "delta": _delta(tmc.get("macro_f1"), tm1.get("macro_f1")),
        },
        "per_class": per_class,
        "confusion_pairs": pairs,
        "class_distribution": {
            "b1": b1.get("class_distribution"),
            "c": c.get("class_distribution"),
        },
        "boundary_distance": {
            "b1": b1.get("boundary_distance"),
            "c": c.get("boundary_distance"),
        },
        "duration": {"b1": b1.get("duration"), "c": c.get("duration")},
        "trajectory_by_duration": {
            "b1": b1.get("trajectory_by_duration"),
            "c": c.get("trajectory_by_duration"),
        },
        "pitch": {
            "mae_cents": {
                "b1": pitch1.get("mae_cents"),
                "c": pitchc.get("mae_cents"),
                "delta": _delta(pitchc.get("mae_cents"), pitch1.get("mae_cents")),
            },
            "median_ae_cents": {
                "b1": pitch1.get("median_ae_cents"),
                "c": pitchc.get("median_ae_cents"),
                "delta": _delta(pitchc.get("median_ae_cents"), pitch1.get("median_ae_cents")),
            },
            "pct_within_25": {
                "b1": pitch1.get("pct_within_25"),
                "c": pitchc.get("pct_within_25"),
                "delta": _delta(pitchc.get("pct_within_25"), pitch1.get("pct_within_25")),
            },
            "pct_within_50": {
                "b1": pitch1.get("pct_within_50"),
                "c": pitchc.get("pct_within_50"),
                "delta": _delta(pitchc.get("pct_within_50"), pitch1.get("pct_within_50")),
            },
            "vs_mean_baseline_mae": {
                "b1": _delta(pitch1.get("mae_cents"), base1.get("mae_cents")),
                "c": _delta(pitchc.get("mae_cents"), basec.get("mae_cents")),
            },
            "b1_overall": pitch1,
            "c_overall": pitchc,
            "b1_by_type": _get(b1, "pitch", "by_type"),
            "c_by_type": _get(c, "pitch", "by_type"),
            "b1_by_duration": _get(b1, "pitch", "by_duration"),
            "c_by_duration": _get(c, "pitch", "by_duration"),
            "type_correctness": {
                "b1": _get(b1, "pitch", "pitch_error_by_type_correctness"),
                "c": _get(c, "pitch", "pitch_error_by_type_correctness"),
            },
            "fine_pitch_error": {
                "b1": _get(b1, "pitch", "type_by_fine_pitch_error"),
                "c": _get(c, "pitch", "type_by_fine_pitch_error"),
            },
        },
        "per_recording": recs,
    }


def summarize(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...], *, lower_better: bool = False) -> list[float]:
        vals = []
        for d in deltas:
            cur: Any = d
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    def pack(vals: list[float], *, lower_better: bool = False) -> dict[str, Any]:
        improved = sum(v < 0 for v in vals) if lower_better else sum(v > 0 for v in vals)
        return {
            "values": vals,
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "n_folds_improved": int(improved),
            "n_folds": len(vals),
        }

    keys = {
        "delta_frame_accuracy": (("frame_accuracy", "delta"), False),
        "delta_frame_macro_f1": (("frame_macro_f1", "delta"), False),
        "delta_traj_accuracy": (("trajectory_accuracy", "delta"), False),
        "delta_traj_macro_f1": (("trajectory_macro_f1", "delta"), False),
        "delta_T0_recall": (("per_class", "T0", "delta_recall"), False),
        "delta_T1_recall": (("per_class", "T1", "delta_recall"), False),
        "delta_T2_recall": (("per_class", "T2", "delta_recall"), False),
        "delta_T3_recall": (("per_class", "T3", "delta_recall"), False),
        "delta_T0_f1": (("per_class", "T0", "delta_f1"), False),
        "delta_T1_f1": (("per_class", "T1", "delta_f1"), False),
        "delta_T2_f1": (("per_class", "T2", "delta_f1"), False),
        "delta_T3_f1": (("per_class", "T3", "delta_f1"), False),
        "delta_pitch_mae_cents": (("pitch", "mae_cents", "delta"), True),
        "delta_pitch_median_ae_cents": (("pitch", "median_ae_cents", "delta"), True),
        "delta_pct_within_25": (("pitch", "pct_within_25", "delta"), False),
        "delta_pct_within_50": (("pitch", "pct_within_50", "delta"), False),
    }
    out: dict[str, Any] = {}
    for name, (path, lower_better) in keys.items():
        out[name] = pack(collect(path), lower_better=lower_better)

    try:
        from scipy.stats import wilcoxon

        f1 = collect(("frame_macro_f1", "delta"))
        if len(f1) >= 5 and any(v != 0 for v in f1):
            stat = wilcoxon(f1)
            out["wilcoxon_frame_macro_f1_delta"] = {
                "statistic": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "note": "exploratory only; n=5",
            }
        mae = collect(("pitch", "mae_cents", "delta"))
        if len(mae) >= 5 and any(v != 0 for v in mae):
            stat = wilcoxon(mae)
            out["wilcoxon_pitch_mae_delta"] = {
                "statistic": float(stat.statistic),
                "pvalue": float(stat.pvalue),
                "note": "exploratory only; n=5; negative delta means C better",
            }
    except Exception:
        pass
    return out


def three_way_summary(
    b0s: list[dict[str, Any]],
    b1s: list[dict[str, Any]],
    cs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    n = min(len(b0s), len(b1s), len(cs))
    for i in range(n):
        rows.append(
            {
                "fold_index": cs[i].get("fold_index", i),
                "b0_frame_macro_f1": _get(b0s[i], "frame_metrics", "macro_f1"),
                "b1_frame_macro_f1": _get(b1s[i], "frame_metrics", "macro_f1"),
                "c_frame_macro_f1": _get(cs[i], "frame_metrics", "macro_f1"),
                "b0_traj_acc": _get(b0s[i], "trajectory_metrics", "accuracy"),
                "b1_traj_acc": _get(b1s[i], "trajectory_metrics", "accuracy"),
                "c_traj_acc": _get(cs[i], "trajectory_metrics", "accuracy"),
                "b1_pitch_mae": _get(b1s[i], "pitch", "overall", "mae_cents"),
                "c_pitch_mae": _get(cs[i], "pitch", "overall", "mae_cents"),
                "b1_mean_baseline_mae": _get(
                    b1s[i], "pitch", "mean_pitch_baseline", "mae_cents"
                ),
                "c_mean_baseline_mae": _get(
                    cs[i], "pitch", "mean_pitch_baseline", "mae_cents"
                ),
            }
        )
    return rows


def plot_overlay(
    b1_npz: np.lib.npyio.NpzFile,
    c_npz: np.lib.npyio.NpzFile,
    primitives: list[dict[str, Any]],
    out_path: Path,
    *,
    t0: float = 0.0,
    window_s: float = 8.0,
    recording_id: str,
    fold_index: int,
) -> None:
    times = np.asarray(c_npz["frame_time_s"])
    t1 = t0 + window_s
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        return
    t = times[mask]
    gt = np.asarray(c_npz["trajectory_type"])[mask]
    valid = np.asarray(c_npz["valid_target"])[mask]
    c_logits = np.asarray(c_npz["logits"])[mask]
    c_pred = c_logits.argmax(axis=-1)
    b1_pred = None
    if "logits" in b1_npz.files:
        b1_pred = np.asarray(b1_npz["logits"])[mask].argmax(axis=-1)

    n_rows = 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"B1 vs C  {recording_id}  fold {fold_index}  [{t0:.1f}, {t1:.1f}] s")

    axes[0].step(t, gt, where="mid", label="GT", linewidth=1.8, color="black")
    if b1_pred is not None:
        axes[0].step(t, b1_pred, where="mid", label="B1", alpha=0.85)
    axes[0].step(t, c_pred, where="mid", label="C", alpha=0.85)
    axes[0].set_ylabel("type")
    axes[0].set_yticks(list(range(4)))
    axes[0].set_yticklabels([TYPE_NAMES[i] for i in range(4)])
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)

    has_pitch = "pred_cents" in c_npz.files and "true_cents" in c_npz.files
    if has_pitch:
        axes[1].plot(t, np.asarray(c_npz["true_cents"])[mask], label="GT", color="black", linewidth=1.2)
        if "pred_cents" in b1_npz.files:
            axes[1].plot(t, np.asarray(b1_npz["pred_cents"])[mask], label="B1", alpha=0.85)
        axes[1].plot(t, np.asarray(c_npz["pred_cents"])[mask], label="C", alpha=0.85)
        axes[1].set_ylabel("cents")
        axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    else:
        axes[1].set_ylabel("pitch (missing)")

    axes[2].fill_between(t, 0, 1, where=valid, alpha=0.35, step="mid", label="valid")
    if "padding_mask" in c_npz.files:
        axes[2].fill_between(
            t, 0, 1, where=np.asarray(c_npz["padding_mask"])[mask],
            alpha=0.3, step="mid", color="red", label="padding",
        )
    axes[2].set_ylabel("mask")
    axes[2].legend(loc="upper right", fontsize=8)

    for prim in primitives:
        start, end = prim["start_s"], prim["end_s"]
        if end < t0 or start > t1:
            continue
        axes[3].axvline(start, color="gray", linewidth=0.8, alpha=0.8)
        axes[3].axvline(end, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        mid = 0.5 * (max(start, t0) + min(end, t1))
        axes[3].text(
            mid, 0.5, f"T{int(prim['canonical_type'])}",
            ha="center", va="center", fontsize=7, alpha=0.8,
        )
    axes[3].set_ylabel("GT bounds")
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel("time (s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_overlays(
    b1_dir: Path,
    c_dir: Path,
    fig_dir: Path,
    index: RecordingLaneIndex,
) -> list[str]:
    written: list[str] = []
    for fold in range(5):
        c_eval = c_dir / f"fold_{fold}" / "eval"
        b1_eval = b1_dir / f"fold_{fold}" / "eval"
        summary_path = c_eval / "eval_summary.json"
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        rec_ids = list(summary.get("test_recordings") or [])
        per = summary.get("per_recording") or {}
        if per:
            rec_ids = sorted(per, key=lambda r: per[r].get("frame_macro_f1") or 0.0)
        chosen = []
        if rec_ids:
            chosen.append(rec_ids[0])
        if len(rec_ids) > 1:
            chosen.append(rec_ids[-1])
        if len(rec_ids) > 2:
            chosen.append(rec_ids[len(rec_ids) // 2])
        for rid in dict.fromkeys(chosen):
            c_npz_path = c_eval / "pitch_contours" / f"{rid}.npz"
            b1_npz_path = b1_eval / "pitch_contours" / f"{rid}.npz"
            if not c_npz_path.exists() or not b1_npz_path.exists():
                continue
            c_npz = np.load(c_npz_path, allow_pickle=True)
            b1_npz = np.load(b1_npz_path, allow_pickle=True)
            prims = index.primitives_for_recording(rid)
            for t0, tag in ((0.0, "0s"), (30.0, "mid")):
                out = fig_dir / f"fold{fold}_{rid}_{tag}.png"
                plot_overlay(
                    b1_npz, c_npz, prims, out,
                    t0=t0, recording_id=rid, fold_index=fold,
                )
                written.append(str(out.relative_to(REPO_ROOT)) if out.exists() else str(out))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b0-dir",
        type=Path,
        default=REPO_ROOT / "output" / "framewise_runs" / "b0_tcn_type_only",
    )
    parser.add_argument(
        "--b1-dir",
        type=Path,
        default=REPO_ROOT / "output" / "framewise_runs" / "b1_tcn_type_pitch",
    )
    parser.add_argument(
        "--c-dir",
        type=Path,
        default=REPO_ROOT / "output" / "framewise_runs" / "c_bigru_type_pitch",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    b0s, b1s, cs = [], [], []
    deltas = []
    gaps = []
    for i in range(5):
        p0 = args.b0_dir / f"fold_{i}" / "eval" / "eval_summary.json"
        p1 = args.b1_dir / f"fold_{i}" / "eval" / "eval_summary.json"
        pc = args.c_dir / f"fold_{i}" / "eval" / "eval_summary.json"
        missing = [p for p in (p1, pc) if not p.exists()]
        if missing:
            print(f"skip fold {i}: missing {missing[0]}", file=sys.stderr)
            continue
        b1 = _load(p1)
        c = _load(pc)
        b1s.append(b1)
        cs.append(c)
        if p0.exists():
            b0s.append(_load(p0))
        deltas.append(fold_delta(b1, c))
        gaps.append(
            {
                "fold_index": i,
                "b1": _train_gap(args.b1_dir / f"fold_{i}", b1),
                "c": _train_gap(args.c_dir / f"fold_{i}", c),
            }
        )

    fig_dir = args.c_dir / "b1_vs_c_figures"
    overlays: list[str] = []
    if not args.skip_plots and cs:
        index = RecordingLaneIndex.build(REPO_ROOT)
        overlays = make_overlays(args.b1_dir, args.c_dir, fig_dir, index)

    report = {
        "note": (
            "Primary contrast is B1 vs C under matched data/losses/folds. "
            "B0 is context only; B0 vs C is not a single-variable contrast. "
            "C is offline/bidirectional. A C win is not evidence that "
            "'longer context helps' in isolation (architecture, parameter "
            "count, and inductive bias also change)."
        ),
        "folds": deltas,
        "summary": summarize(deltas),
        "generalization_gaps": gaps,
        "b0_b1_c": three_way_summary(b0s, b1s, cs) if len(b0s) == len(cs) else [],
        "overlay_figures": overlays,
    }
    out = args.output or (args.c_dir / "b1_vs_c.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
