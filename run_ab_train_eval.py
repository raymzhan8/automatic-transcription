"""Train CNN on raw / denoised / vocals datasets and compare test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TC_ROOT = PROJECT_ROOT / "trajectory-classification"
TRAIN_SCRIPT = TC_ROOT / "scripts" / "train_cnn.py"
OUTPUT_ROOT = TC_ROOT / "outputs" / "ab_denoise_runs"
ALIGNED_DIR = PROJECT_ROOT / "output" / "ab_aligned_metadata"

METADATA_COLUMNS = [
    "piece_id",
    "piece_title",
    "traj_index",
    "segment_index",
    "unique_id",
    "idtap_id",
    "idtap_name",
    "label",
    "abs_start",
    "abs_end",
    "duration",
    "image_path",
    "clip_path",
]


def align_metadata_variants(
    variants: dict[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Inner-join metadata on (piece_id, traj_index) so A/B splits match."""
    frames: dict[str, pd.DataFrame] = {}
    for name, path in variants.items():
        frame = pd.read_csv(path)
        frame["piece_id"] = frame["piece_id"].astype(str)
        frame["traj_index"] = frame["traj_index"].astype(int)
        frames[name] = frame

    keys = ["piece_id", "traj_index"]
    common = frames["raw"][keys].drop_duplicates()
    for name, frame in frames.items():
        if name == "raw":
            continue
        common = common.merge(frame[keys].drop_duplicates(), on=keys, how="inner")

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        aligned = frame.merge(common, on=keys, how="inner").sort_values(keys)
        out_path = output_dir / f"metadata_{name}.csv"
        aligned.to_csv(out_path, index=False)
        aligned_paths[name] = out_path
        print(f"Aligned {name}: {len(aligned)} rows -> {out_path}")

    return aligned_paths


def run_training(
    *,
    metadata: Path,
    run_name: str,
    seed: int,
    epochs: int,
    patience: int,
) -> Path:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--metadata",
        str(metadata),
        "--output-dir",
        str(OUTPUT_ROOT),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--split-by",
        "piece_id",
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
    ]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    slug = run_name.lower().replace(" ", "-")
    run_dir = OUTPUT_ROOT / slug
    if not run_dir.exists():
        candidates = sorted(OUTPUT_ROOT.glob(f"{slug}*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No run directory found for {run_name}")
        run_dir = candidates[-1]
    return run_dir


def load_run_metrics(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    summary: dict[str, object] = {"run_dir": str(run_dir)}
    history_path = run_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        best = history.loc[history["val_macro_f1"].idxmax()]
        summary["best_val_macro_f1"] = float(best["val_macro_f1"])
        summary["best_val_acc"] = float(best["val_acc"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    variants = {
        "raw": PROJECT_ROOT / "output" / "cnn_dataset" / "all" / "metadata.csv",
        "denoised": PROJECT_ROOT / "output" / "cnn_dataset_denoised" / "all" / "metadata.csv",
        "vocals": PROJECT_ROOT / "output" / "cnn_dataset_vocals" / "all" / "metadata.csv",
    }

    missing = [name for name, path in variants.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing metadata for variants: {missing}. Run export_denoised_cnn_dataset.py first."
        )

    aligned_variants = align_metadata_variants(variants, ALIGNED_DIR)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []

    for name, metadata in aligned_variants.items():
        run_dir = run_training(
            metadata=metadata,
            run_name=f"ab-{name}",
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
        )
        metrics = load_run_metrics(run_dir)
        comparison_rows.append(
            {
                "variant": name,
                "run_dir": str(run_dir),
                "test_acc": metrics.get("test_accuracy"),
                "test_macro_f1": metrics.get("test_macro_f1"),
                "best_val_macro_f1": metrics.get("best_val_macro_f1"),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = OUTPUT_ROOT / "ab_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n=== A/B comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {comparison_path}")


if __name__ == "__main__":
    main()
