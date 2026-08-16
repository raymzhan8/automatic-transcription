"""Evaluate framewise models on held-out recordings (B0 / B1 / C)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.folds import prepare_fold  # noqa: E402
from training.framewise_dataset import (  # noqa: E402
    FullRecordingDataset,
    RecordingLaneIndex,
    collate_variable_length,
)
from training.framewise_models import build_model  # noqa: E402
from training.metrics import (  # noqa: E402
    BOUNDARY_MS_BUCKETS,
    DURATION_S_BUCKETS,
    TYPE_NAMES,
    aggregate_trajectory_predictions,
    baseline_frame_metrics,
    class_distribution,
    collect_frame_predictions,
    confusion_pair_counts,
    duration_s_per_frame,
    frame_metrics,
    majority_baseline,
    mean_softmax_entropy,
    metrics_by_buckets,
    nearest_boundary_distance_ms,
    pitch_error_by_type_correctness,
    pitch_error_metrics,
    pitch_metrics_by_provenance,
    pitch_metrics_by_type,
    t1_provenance_metrics,
    trajectory_metrics,
    type_metrics_by_fine_pitch_error,
    type_metrics_by_pitch_error,
    trajectory_metrics_by_duration,
)
from training.normalization import (  # noqa: E402
    denormalize_pitch,
    load_fold_cqt_stats,
    load_fold_pitch_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-recordings", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_recording(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    index: RecordingLaneIndex,
    *,
    predict_pitch: bool,
    mu_pitch: float | None,
    sigma_pitch: float | None,
) -> dict[str, Any]:
    spec = batch["spec"].to(device)
    pad = batch["padding_mask"][0]
    lengths = (~batch["padding_mask"]).sum(dim=1).clamp(min=1)
    if hasattr(model, "gru"):
        outputs = model(spec, lengths.to(device))
    else:
        outputs = model(spec)
    pitch_std = None
    if isinstance(outputs, tuple):
        logits, pitch_std = outputs
    else:
        logits = outputs
    logits_np = logits[0].cpu().numpy()
    pad = batch["padding_mask"][0].numpy()
    valid = batch["valid_target"][0].numpy()
    times = np.asarray(batch["frame_time_s"][0])
    traj_type = batch["trajectory_type"][0].numpy()

    pred, label = collect_frame_predictions(
        logits.cpu(),
        batch["trajectory_type"],
        batch["valid_target"],
        batch["padding_mask"],
    )
    frame_m = frame_metrics(pred, label)

    eval_mask = (~pad) & valid
    prim_ids = batch["primitive_id"][0]
    provenance_full = np.array(
        [
            (index.get_primitive(batch["recording_id"][0], str(pid)) or {}).get(
                "t1_provenance", ""
            )
            for pid in prim_ids
        ],
        dtype=object,
    )
    t1_m = t1_provenance_metrics(pred, label, provenance_full[eval_mask])

    primitives = index.primitives_for_recording(batch["recording_id"][0])
    traj_results, frame_hist = aggregate_trajectory_predictions(
        times, logits_np, pad, primitives
    )
    traj_m = trajectory_metrics(traj_results)

    pred_cents = None
    true_cents = batch["pitch_cents"][0].numpy() if "pitch_cents" in batch else None
    if predict_pitch and pitch_std is not None and mu_pitch is not None and sigma_pitch is not None:
        pred_cents = denormalize_pitch(pitch_std[0].cpu().numpy(), mu_pitch, sigma_pitch)

    return {
        "recording_id": batch["recording_id"][0],
        "frame_metrics": frame_m,
        "t1_provenance": t1_m,
        "trajectory_metrics": traj_m,
        "trajectory_accuracy": traj_m["accuracy"],
        "trajectory_n": traj_m.get("n", 0),
        "trajectory_frame_histogram": frame_hist,
        "logits": logits_np,
        "padding_mask": pad,
        "valid_target": valid,
        "trajectory_type": traj_type,
        "frame_time_s": times,
        "pred": pred,
        "label": label,
        "eval_mask": eval_mask,
        "provenance": provenance_full,
        "pred_cents": pred_cents,
        "true_cents": true_cents,
        "spec": batch["spec"][0, 0].numpy(),
    }


def plot_recording_eval(
    result: dict[str, Any],
    out_path: Path,
    *,
    t0: float = 0.0,
    window_s: float = 8.0,
) -> None:
    times = result["frame_time_s"]
    t1 = t0 + window_s
    mask = (times >= t0) & (times <= t1)
    if not np.any(mask):
        return

    has_pitch = result.get("pred_cents") is not None
    n_rows = 6 if has_pitch else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows), sharex=True)
    fig.suptitle(f"{result['recording_id']} [{t0:.1f}, {t1:.1f}] s")

    gt = result["trajectory_type"][mask]
    pr = result["logits"][mask].argmax(axis=-1)
    valid = result["valid_target"][mask]
    probs = torch.softmax(torch.from_numpy(result["logits"][mask]), dim=-1).numpy()

    spec = result["spec"][:, mask] if result["spec"].shape[1] == len(times) else None
    if spec is not None:
        axes[0].imshow(spec, aspect="auto", origin="lower", interpolation="nearest")
        axes[0].set_ylabel("CQT")
    else:
        axes[0].set_ylabel("CQT (missing)")

    axes[1].step(times[mask], gt, where="mid", label="GT type")
    axes[1].step(times[mask], pr, where="mid", label="Pred type", alpha=0.7)
    axes[1].set_ylabel("type")
    axes[1].legend(loc="upper right", fontsize=8)

    if has_pitch:
        axes[2].plot(times[mask], result["true_cents"][mask], label="GT pitch", linewidth=1.2)
        axes[2].plot(times[mask], result["pred_cents"][mask], label="Pred pitch", alpha=0.8, linewidth=1.0)
        axes[2].set_ylabel("cents")
        axes[2].legend(loc="upper right", fontsize=8)
        mask_ax, prob_ax = axes[3], axes[4]
        last = axes[5]
    else:
        mask_ax, prob_ax, last = axes[2], axes[3], None

    mask_ax.fill_between(times[mask], 0, 1, where=valid, alpha=0.3, step="mid", label="valid")
    mask_ax.fill_between(
        times[mask], 0, 1, where=result["padding_mask"][mask], alpha=0.3,
        step="mid", color="red", label="padding",
    )
    mask_ax.set_ylabel("mask")
    mask_ax.legend(loc="upper right", fontsize=8)

    for i in range(4):
        prob_ax.plot(times[mask], probs[:, i], label=TYPE_NAMES[i], alpha=0.8)
    prob_ax.set_ylabel("softmax")
    prob_ax.legend(loc="upper right", ncol=4, fontsize=7)
    if last is None:
        prob_ax.set_xlabel("time (s)")
    else:
        last.plot(times[mask], np.abs(result["pred_cents"][mask] - result["true_cents"][mask]))
        last.set_ylabel("|pitch err|")
        last.set_xlabel("time (s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _jsonable_frame_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return m


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    fold_index = ckpt["fold_index"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_pitch = config.get("task") == "type_and_pitch"

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    split, fold_summary = prepare_fold(args.repo_root, fold_index, seed=config.get("seed", 42))
    mu, sigma = load_fold_cqt_stats(fold_index, args.repo_root)
    mu_pitch, sigma_pitch = load_fold_pitch_stats(fold_index, args.repo_root)
    index = RecordingLaneIndex.build(args.repo_root)

    test_ids = split.test_recording_ids
    if args.max_recordings:
        test_ids = test_ids[: args.max_recordings]

    ds = FullRecordingDataset(
        index, test_ids, mu, sigma,
        mu_pitch=mu_pitch if predict_pitch else None,
        sigma_pitch=sigma_pitch if predict_pitch else None,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_variable_length)

    run_dir = Path(args.checkpoint).parent
    out_dir = args.output_dir or (run_dir / "eval")
    fig_dir = out_dir / "figures"
    contour_dir = out_dir / "pitch_contours"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred, all_label = [], []
    all_pred_c, all_true_c, all_prov = [], [], []
    all_logits, all_boundary_ms, all_duration_s = [], [], []
    per_rec_results: list[dict[str, Any]] = []

    for batch in loader:
        result = evaluate_recording(
            model, batch, device, index,
            predict_pitch=predict_pitch, mu_pitch=mu_pitch, sigma_pitch=sigma_pitch,
        )
        per_rec_results.append(result)
        all_pred.append(result["pred"])
        all_label.append(result["label"])
        em = result["eval_mask"]
        all_prov.append(result["provenance"][em])
        all_logits.append(result["logits"][em])
        prims = index.primitives_for_recording(result["recording_id"])
        all_boundary_ms.append(
            nearest_boundary_distance_ms(result["frame_time_s"][em], prims)
        )
        all_duration_s.append(duration_s_per_frame(result["frame_time_s"], prims)[em])
        plot_recording_eval(result, fig_dir / f"{result['recording_id']}_0s.png", t0=0.0)
        plot_recording_eval(result, fig_dir / f"{result['recording_id']}_mid.png", t0=30.0)

        contour_dir.mkdir(parents=True, exist_ok=True)
        save_arrays: dict[str, Any] = {
            "frame_time_s": result["frame_time_s"],
            "logits": result["logits"],
            "valid_target": result["valid_target"],
            "padding_mask": result["padding_mask"],
            "trajectory_type": result["trajectory_type"],
        }
        if result["pred_cents"] is not None:
            save_arrays["pred_cents"] = result["pred_cents"]
            save_arrays["true_cents"] = result["true_cents"]
            all_pred_c.append(result["pred_cents"][em])
            all_true_c.append(result["true_cents"][em])
        np.savez_compressed(contour_dir / f"{result['recording_id']}.npz", **save_arrays)

    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    y_true = np.concatenate(all_label) if all_label else np.array([])
    overall = frame_metrics(y_pred, y_true)
    pairs = confusion_pair_counts(y_true, y_pred)

    train_hist = {int(k): int(v) for k, v in fold_summary["train_class_histogram"].items()}
    maj = majority_baseline(train_hist)
    baseline = baseline_frame_metrics(y_true, maj)

    traj_all: list[dict[str, Any]] = []
    for r in per_rec_results:
        prims = index.primitives_for_recording(r["recording_id"])
        results, _ = aggregate_trajectory_predictions(
            r["frame_time_s"], r["logits"], r["padding_mask"], prims
        )
        traj_all.extend(results)
    traj_m = trajectory_metrics(traj_all)

    pitch_block: dict[str, Any] = {}
    if all_true_c:
        pc = np.concatenate(all_pred_c)
        tc = np.concatenate(all_true_c)
        pv = np.concatenate(all_prov)
        mean_baseline = np.full_like(tc, mu_pitch)
        pitch_block = {
            "overall": pitch_error_metrics(pc, tc),
            "mean_pitch_baseline": pitch_error_metrics(mean_baseline, tc),
            "by_type": pitch_metrics_by_type(pc, tc, y_true),
            "by_t1_provenance": pitch_metrics_by_provenance(pc, tc, y_true, pv),
            "type_by_pitch_error": type_metrics_by_pitch_error(y_pred, y_true, pc, tc),
            "type_by_fine_pitch_error": type_metrics_by_fine_pitch_error(
                y_pred, y_true, pc, tc
            ),
            "pitch_error_by_type_correctness": pitch_error_by_type_correctness(
                y_pred, y_true, pc, tc
            ),
            "fold_mu_cents": mu_pitch,
            "fold_sigma_cents": sigma_pitch,
        }

    t1_merged: dict[str, Any] = {}
    if all_pred and all_prov:
        pv = np.concatenate(all_prov)
        if len(pv) == len(y_true):
            t1_merged = t1_provenance_metrics(y_pred, y_true, pv)

    pc = np.concatenate(all_pred_c) if all_true_c else None
    tc = np.concatenate(all_true_c) if all_true_c else None
    bd = np.concatenate(all_boundary_ms) if all_boundary_ms else np.array([])
    dur = np.concatenate(all_duration_s) if all_duration_s else np.array([])
    logits_eval = np.concatenate(all_logits) if all_logits else np.zeros((0, 4))
    boundary_block = metrics_by_buckets(
        y_pred, y_true, bd, BOUNDARY_MS_BUCKETS, pred_cents=pc, true_cents=tc
    )
    duration_block = metrics_by_buckets(
        y_pred, y_true, dur, DURATION_S_BUCKETS, pred_cents=pc, true_cents=tc
    )
    if pitch_block:
        pitch_block["by_duration"] = {
            name: duration_block[name].get("pitch") for name in duration_block
        }
        pitch_block["by_recording"] = {}
        for r in per_rec_results:
            if r["pred_cents"] is None:
                continue
            em = r["eval_mask"]
            pitch_block["by_recording"][r["recording_id"]] = pitch_error_metrics(
                r["pred_cents"][em], r["true_cents"][em]
            )

    def _per_rec_pitch(r: dict[str, Any]) -> float | None:
        if r["pred_cents"] is None:
            return None
        em = r["eval_mask"]
        return pitch_error_metrics(r["pred_cents"][em], r["true_cents"][em])["mae_cents"]

    summary = {
        "checkpoint": str(args.checkpoint),
        "fold_index": fold_index,
        "experiment": config.get("experiment"),
        "test_recordings": test_ids,
        "frame_metrics": overall,
        "confusion_pairs": pairs,
        "majority_baseline": baseline,
        "majority_class": maj,
        "trajectory_metrics": traj_m,
        "trajectory_by_duration": trajectory_metrics_by_duration(traj_all),
        "t1_provenance": t1_merged,
        "pitch": pitch_block,
        "class_distribution": {
            "predicted": class_distribution(y_pred),
            "ground_truth": class_distribution(y_true),
            "mean_softmax_entropy": mean_softmax_entropy(logits_eval),
        },
        "boundary_distance": boundary_block,
        "duration": duration_block,
        "per_recording": {
            r["recording_id"]: {
                "frame_accuracy": r["frame_metrics"]["accuracy"],
                "frame_macro_f1": r["frame_metrics"]["macro_f1"],
                "trajectory_accuracy": r["trajectory_accuracy"],
                "trajectory_macro_f1": r["trajectory_metrics"].get("macro_f1"),
                "trajectory_n": r["trajectory_n"],
                "pitch_mae_cents": _per_rec_pitch(r),
            }
            for r in per_rec_results
        },
    }
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_recording"}, indent=2))


if __name__ == "__main__":
    main()
