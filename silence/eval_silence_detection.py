"""Evaluate silent / non-silent detection against IDTAP trajectory labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from idtap import Piece, SwaraClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from silence.silence_audio_prep import AudioVariant, load_variant_audio  # noqa: E402
from silence.silence_detection import (  # noqa: E402
    DEFAULT_HOP_LENGTH,
    DEFAULT_MIN_DURATION,
    DEFAULT_SR,
    DetectionMethod,
    label_interval,
    segment_audio,
)

RMS_THRESHOLD_GRID = [-50, -45, -40, -35, -30, -25]
LIBROSA_TOP_DB_GRID = [20, 30, 40, 50, 60]
DEFAULT_VARIANT_OUTPUT_DIRS = {
    "raw": PROJECT_ROOT / "output" / "silence_detection" / "raw_baseline",
    "denoised": PROJECT_ROOT / "output" / "silence_detection" / "denoised_baseline",
    "vocals": PROJECT_ROOT / "output" / "silence_detection" / "vocals_baseline",
}


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    precision_silent: float
    recall_silent: float
    f1_silent: float
    n_samples: int
    n_silent_gt: int
    n_nonsilent_gt: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def find_raw_audio(audio_dir: Path, piece_id: str, piece_title: str) -> Path | None:
    candidates = list(audio_dir.glob(f"*_{piece_id}.wav"))
    if candidates:
        return candidates[0]
    title_path = audio_dir / f"{piece_title}_{piece_id}.wav"
    if title_path.exists():
        return title_path
    return None


def split_pieces(piece_ids: list[str], *, seed: int) -> tuple[set[str], set[str], set[str]]:
    if len(piece_ids) < 3:
        raise ValueError(f"Need at least 3 pieces for dev/test split, got {len(piece_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = sorted(piece_ids)
    rng.shuffle(shuffled)

    n_pieces = len(shuffled)
    n_train = max(1, int(round(n_pieces * 0.6)))
    n_val = max(1, int(round(n_pieces * 0.2)))
    if n_train + n_val >= n_pieces:
        n_val = max(1, n_pieces - n_train - 1)
    n_test = n_pieces - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_pieces - n_val - n_test)

    train_pieces = set(shuffled[:n_train])
    val_pieces = set(shuffled[n_train : n_train + n_val])
    test_pieces = set(shuffled[n_train + n_val :])
    return train_pieces, val_pieces, test_pieces


def load_piece_objects(piece_ids: list[str]) -> dict[str, Piece]:
    client = SwaraClient()
    pieces: dict[str, Piece] = {}
    for piece_id in piece_ids:
        try:
            pieces[piece_id] = Piece.from_json(client.get_piece(piece_id))
        except Exception as exc:
            print(f"Warning: could not load IDTAP piece {piece_id}: {exc}")
    return pieces


def load_piece_audio_cache(
    frame: pd.DataFrame,
    audio_dir: Path,
    denoise_root: Path,
    *,
    variant: AudioVariant,
    sr: int,
    piece_objects: dict[str, Piece] | None = None,
    skip_denoise: bool = False,
    force_vocal_separation: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Path], list[str], list[str]]:
    audio_by_piece: dict[str, np.ndarray] = {}
    paths_by_piece: dict[str, Path] = {}
    missing_pieces: list[str] = []
    skipped_pieces: list[str] = []

    piece_groups = frame.groupby("piece_id", sort=False)
    for piece_id, group in piece_groups:
        piece_id_str = str(piece_id)
        title = str(group.iloc[0]["piece_title"])
        piece_obj = (piece_objects or {}).get(piece_id_str)
        try:
            y, stem_path, _meta = load_variant_audio(
                piece_id=piece_id_str,
                piece_title=title,
                audio_dir=audio_dir,
                denoise_root=denoise_root,
                variant=variant,
                sr=sr,
                piece_obj=piece_obj,
                skip_denoise=skip_denoise,
                force_vocal_separation=force_vocal_separation,
            )
        except FileNotFoundError as exc:
            if variant == "raw" and find_raw_audio(audio_dir, piece_id_str, title) is None:
                missing_pieces.append(piece_id_str)
            else:
                skipped_pieces.append(piece_id_str)
                print(f"Skipping {piece_id_str}: {exc}")
            continue

        audio_by_piece[piece_id_str] = y
        paths_by_piece[piece_id_str] = stem_path

    return audio_by_piece, paths_by_piece, missing_pieces, skipped_pieces


def predict_for_rows(
    rows: pd.DataFrame,
    audio_by_piece: dict[str, np.ndarray],
    *,
    method: DetectionMethod,
    sr: int,
    params: dict[str, float | int],
) -> tuple[list[bool], list[bool]]:
    segment_cache: dict[str, list] = {}
    y_true: list[bool] = []
    y_pred: list[bool] = []

    for _, row in rows.iterrows():
        piece_id = str(row["piece_id"])
        y = audio_by_piece[piece_id]
        if piece_id not in segment_cache:
            segment_cache[piece_id] = segment_audio(y, sr, method, **params)

        start = float(row["abs_start"])
        duration = float(row["duration"])
        gt_silent = str(row["label"]) == "silent"
        pred_silent = label_interval(segment_cache[piece_id], start, start + duration)
        y_true.append(gt_silent)
        y_pred.append(pred_silent)

    return y_true, y_pred


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> EvalMetrics:
    if not y_true:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    return EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_silent=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_silent=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_silent=float(f1_score(y_true, y_pred, zero_division=0)),
        n_samples=len(y_true),
        n_silent_gt=int(sum(y_true)),
        n_nonsilent_gt=int(len(y_true) - sum(y_true)),
    )


def per_piece_metrics(
    rows: pd.DataFrame,
    audio_by_piece: dict[str, np.ndarray],
    *,
    method: DetectionMethod,
    sr: int,
    params: dict[str, float | int],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for piece_id, group in rows.groupby("piece_id", sort=False):
        piece_id_str = str(piece_id)
        if piece_id_str not in audio_by_piece:
            continue
        y_true, y_pred = predict_for_rows(
            group,
            audio_by_piece,
            method=method,
            sr=sr,
            params=params,
        )
        metrics = compute_metrics(y_true, y_pred)
        records.append(
            {
                "piece_id": piece_id_str,
                "piece_title": str(group.iloc[0]["piece_title"]),
                "method": method,
                **metrics.to_dict(),
            }
        )
    return pd.DataFrame.from_records(records)


def sweep_method(
    rows: pd.DataFrame,
    audio_by_piece: dict[str, np.ndarray],
    *,
    method: DetectionMethod,
    sr: int,
    param_name: str,
    param_grid: list[float],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for value in param_grid:
        params = {param_name: value}
        y_true, y_pred = predict_for_rows(
            rows,
            audio_by_piece,
            method=method,
            sr=sr,
            params=params,
        )
        metrics = compute_metrics(y_true, y_pred)
        records.append(
            {
                "method": method,
                param_name: value,
                **metrics.to_dict(),
            }
        )
    return pd.DataFrame.from_records(records)


def choose_best_params(sweep_df: pd.DataFrame, param_name: str) -> dict[str, float | int]:
    best = sweep_df.sort_values(
        ["f1_silent", "accuracy", param_name],
        ascending=[False, False, True],
    ).iloc[0]
    return {param_name: float(best[param_name])}


def save_confusion_matrix(
    y_true: list[bool],
    y_pred: list[bool],
    output_path: Path,
    *,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-silent", "silent"])
    ax.set_yticks([0, 1], labels=["non-silent", "silent"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_variant_eval(
    *,
    frame: pd.DataFrame,
    audio_dir: Path,
    denoise_root: Path,
    output_dir: Path,
    variant: AudioVariant,
    seed: int,
    sr: int,
    min_duration: float,
    hop_length: int,
    piece_objects: dict[str, Piece] | None,
    skip_denoise: bool,
    force_vocal_separation: bool,
) -> dict[str, object]:
    audio_by_piece, paths_by_piece, missing_pieces, skipped_pieces = load_piece_audio_cache(
        frame,
        audio_dir,
        denoise_root,
        variant=variant,
        sr=sr,
        piece_objects=piece_objects,
        skip_denoise=skip_denoise,
        force_vocal_separation=force_vocal_separation,
    )
    eval_frame = frame[frame["piece_id"].isin(audio_by_piece)].copy()
    if eval_frame.empty:
        raise RuntimeError(f"No evaluable pieces for audio variant {variant!r}")

    piece_ids = sorted(eval_frame["piece_id"].unique())
    train_pieces, val_pieces, test_pieces = split_pieces(piece_ids, seed=seed)
    val_rows = eval_frame[eval_frame["piece_id"].isin(val_pieces)]
    test_rows = eval_frame[eval_frame["piece_id"].isin(test_pieces)]

    shared_params = {
        "hop_length": hop_length,
        "min_duration": min_duration,
    }

    rms_sweep = sweep_method(
        val_rows,
        audio_by_piece,
        method="rms",
        sr=sr,
        param_name="threshold_db",
        param_grid=RMS_THRESHOLD_GRID,
    )
    librosa_sweep = sweep_method(
        val_rows,
        audio_by_piece,
        method="librosa_split",
        sr=sr,
        param_name="top_db",
        param_grid=LIBROSA_TOP_DB_GRID,
    )

    best_rms = choose_best_params(rms_sweep, "threshold_db")
    best_librosa = choose_best_params(librosa_sweep, "top_db")
    best_rms.update(shared_params)
    best_librosa.update(shared_params)

    method_configs: list[tuple[str, DetectionMethod, dict[str, float | int]]] = [
        ("rms", "rms", best_rms),
        ("librosa_split", "librosa_split", best_librosa),
    ]

    comparison_rows: list[dict[str, object]] = []
    test_predictions: dict[str, tuple[list[bool], list[bool]]] = {}

    for method_name, method, params in method_configs:
        y_true, y_pred = predict_for_rows(
            test_rows,
            audio_by_piece,
            method=method,
            sr=sr,
            params=params,
        )
        metrics = compute_metrics(y_true, y_pred)
        test_predictions[method_name] = (y_true, y_pred)
        comparison_rows.append(
            {
                "audio_variant": variant,
                "method": method_name,
                "split": "test",
                **params,
                **metrics.to_dict(),
            }
        )

    per_piece = pd.concat(
        [
            per_piece_metrics(
                test_rows,
                audio_by_piece,
                method=method,
                sr=sr,
                params=params,
            ).assign(audio_variant=variant, detection_method=method_name)
            for method_name, method, params in method_configs
        ],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rms_sweep.to_csv(output_dir / "param_sweep_rms.csv", index=False)
    librosa_sweep.to_csv(output_dir / "param_sweep_librosa.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison.csv", index=False)
    per_piece.to_csv(output_dir / "per_piece_metrics.csv", index=False)

    for method_name, (y_true, y_pred) in test_predictions.items():
        save_confusion_matrix(
            y_true,
            y_pred,
            output_dir / f"confusion_matrix_{method_name}.png",
            title=f"{variant} / {method_name} (test split)",
        )

    summary = {
        "audio_variant": variant,
        "metadata": str(frame.attrs.get("metadata_path", "")),
        "audio_dir": str(audio_dir),
        "denoise_root": str(denoise_root),
        "skip_denoise": skip_denoise,
        "force_vocal_separation": force_vocal_separation,
        "n_rows_total": int(len(frame)),
        "n_rows_evaluated": int(len(eval_frame)),
        "n_pieces_total": int(frame["piece_id"].nunique()),
        "n_pieces_evaluated": len(piece_ids),
        "missing_pieces": missing_pieces,
        "skipped_pieces": skipped_pieces,
        "split_seed": seed,
        "train_pieces": sorted(train_pieces),
        "val_pieces": sorted(val_pieces),
        "test_pieces": sorted(test_pieces),
        "best_rms_params": best_rms,
        "best_librosa_params": best_librosa,
        "test_metrics": {
            row["method"]: {
                key: row[key]
                for key in (
                    "accuracy",
                    "precision_silent",
                    "recall_silent",
                    "f1_silent",
                    "n_samples",
                )
            }
            for row in comparison_rows
        },
        "stem_paths": {piece_id: str(path) for piece_id, path in sorted(paths_by_piece.items())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {variant} (test split) ===")
    print(f"Evaluated {len(eval_frame)} clips from {len(piece_ids)} pieces")
    if missing_pieces:
        print(f"Missing raw audio for {len(missing_pieces)} piece(s): {missing_pieces}")
    if skipped_pieces:
        print(f"Skipped {len(skipped_pieces)} piece(s) for variant {variant}: {skipped_pieces}")
    print(f"Best RMS params (val): {best_rms}")
    print(f"Best librosa params (val): {best_librosa}")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print(f"Saved results to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "output" / "cnn_dataset" / "all" / "metadata.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
    )
    parser.add_argument(
        "--denoise-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "denoised",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default depends on --audio-variant)",
    )
    parser.add_argument(
        "--audio-variant",
        choices=("raw", "denoised", "vocals", "all"),
        default="raw",
        help="Audio stem to segment: raw, denoised, vocals, or all",
    )
    parser.add_argument(
        "--skip-denoise",
        action="store_true",
        help="Use cached stems under --denoise-root instead of running vocalprep",
    )
    parser.add_argument(
        "--force-vocal-separation",
        action="store_true",
        help="Run karaoke separation even for instrumental pieces",
    )
    parser.add_argument(
        "--skip-idtap",
        action="store_true",
        help="Do not query IDTAP for instrument type (karaoke only with --force-vocal-separation)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=DEFAULT_SR)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    args = parser.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Missing metadata: {args.metadata}")

    frame = pd.read_csv(args.metadata)
    frame["piece_id"] = frame["piece_id"].astype(str)
    frame.attrs["metadata_path"] = str(args.metadata)

    variants: list[AudioVariant]
    if args.audio_variant == "all":
        variants = ["raw", "denoised", "vocals"]
    else:
        variants = [args.audio_variant]

    piece_objects: dict[str, Piece] | None = None
    if not args.skip_idtap and any(v != "raw" for v in variants):
        piece_ids = sorted(frame["piece_id"].unique())
        print("Loading IDTAP piece metadata for vocal/instrumental routing ...")
        piece_objects = load_piece_objects(piece_ids)

    summaries: list[dict[str, object]] = []
    for variant in variants:
        output_dir = args.output_dir
        if output_dir is None:
            if args.audio_variant == "all":
                output_dir = PROJECT_ROOT / "output" / "silence_detection" / f"{variant}_baseline"
            else:
                output_dir = DEFAULT_VARIANT_OUTPUT_DIRS.get(
                    variant,
                    PROJECT_ROOT / "output" / "silence_detection" / f"{variant}_baseline",
                )

        try:
            summary = run_variant_eval(
                frame=frame,
                audio_dir=args.audio_dir,
                denoise_root=args.denoise_root,
                output_dir=output_dir,
                variant=variant,
                seed=args.seed,
                sr=args.sr,
                min_duration=args.min_duration,
                hop_length=args.hop_length,
                piece_objects=piece_objects,
                skip_denoise=args.skip_denoise,
                force_vocal_separation=args.force_vocal_separation,
            )
            summaries.append(summary)
        except RuntimeError as exc:
            print(f"Skipping variant {variant}: {exc}")

    if len(summaries) > 1:
        rows: list[dict[str, object]] = []
        for summary in summaries:
            variant = summary["audio_variant"]
            for method_name, metrics in summary["test_metrics"].items():
                rows.append(
                    {
                        "audio_variant": variant,
                        "method": method_name,
                        **metrics,
                    }
                )
        cross_dir = PROJECT_ROOT / "output" / "silence_detection"
        cross_dir.mkdir(parents=True, exist_ok=True)
        cross_path = cross_dir / "cross_variant_comparison.csv"
        pd.DataFrame(rows).to_csv(cross_path, index=False)
        print(f"\nSaved cross-variant comparison to {cross_path}")


if __name__ == "__main__":
    main()
