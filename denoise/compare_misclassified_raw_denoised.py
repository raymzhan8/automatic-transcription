"""Side-by-side raw vs denoised PNGs for misclassified denoised-model test clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models import build_model  # noqa: E402
from training.spec_dataset import (  # noqa: E402
    SpecImageDataset,
    build_label_config,
    label_to_idx,
    normalize_label,
    resolve_image_path,
)
from training.train_cnn import (  # noqa: E402
    collect_predictions,
    split_indices_by_piece,
)

LABEL_NAMES = {
    "0": "Fixed",
    "1": "Bend: Simple",
    "2": "Bend: Sloped Start",
    "3": "Bend: Sloped End",
    "silent": "Silent",
}


def parse_error_filter(token: str) -> tuple[int, int]:
    true_s, pred_s = token.split("->", maxsplit=1)
    label_map, _, _, _ = build_label_config([])
    return label_to_idx(true_s.strip(), label_map), label_to_idx(pred_s.strip(), label_map)


def load_image_rgb(path: Path) -> np.ndarray:
    return plt.imread(path)


def save_comparison_grid(
    examples: list[dict[str, object]],
    output_path: Path,
    *,
    title: str,
    max_examples: int = 12,
) -> None:
    sample = examples[:max_examples]
    if not sample:
        print(f"No examples for {title}; skipping.")
        return

    cols = 2
    rows = len(sample)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_idx, example in enumerate(sample):
        for col_idx, variant in enumerate(("raw", "denoised")):
            ax = axes[row_idx, col_idx]
            image = load_image_rgb(Path(example[f"{variant}_image"]))
            ax.imshow(image)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(variant.capitalize())
            if col_idx == 0:
                ax.set_ylabel(
                    f"#{example['traj_index']} {example['idtap_name']}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(sample)} examples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "ab_denoise_runs" / "ab-denoised",
        help="Denoised CNN run directory with best_model.pt",
    )
    parser.add_argument(
        "--raw-metadata",
        type=Path,
        default=PROJECT_ROOT / "output" / "ab_aligned_metadata" / "metadata_raw.csv",
    )
    parser.add_argument(
        "--denoised-metadata",
        type=Path,
        default=PROJECT_ROOT / "output" / "ab_aligned_metadata" / "metadata_denoised.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "misclassified_comparisons",
    )
    parser.add_argument(
        "--error-patterns",
        nargs="+",
        default=["0->1", "3->1"],
        help="True->pred label patterns to visualize (e.g. 0->1)",
    )
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_map, idx_to_label, class_names, num_classes = build_label_config([])
    error_patterns = [parse_error_filter(token) for token in args.error_patterns]

    raw_frame = pd.read_csv(args.raw_metadata)
    denoised_frame = pd.read_csv(args.denoised_metadata)
    for frame in (raw_frame, denoised_frame):
        frame["piece_id"] = frame["piece_id"].astype(str)
        frame["traj_index"] = frame["traj_index"].astype(int)
        frame["label"] = frame["label"].map(normalize_label)

    keys = ["piece_id", "traj_index"]
    merged = denoised_frame.merge(
        raw_frame[keys + ["image_path"]].rename(columns={"image_path": "raw_image_path"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    ).rename(columns={"image_path": "denoised_image_path"})

    train_idx, val_idx, test_idx = split_indices_by_piece(
        denoised_frame,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        label_map=label_map,
    )

    split_info_path = args.run_dir / "split_info.json"
    if split_info_path.exists():
        split_info = json.loads(split_info_path.read_text())
        print(
            f"Run {args.run_dir.name}: test_size={split_info.get('test_size')} "
            f"pieces={split_info.get('unique_pieces')}"
        )

    test_dataset = SpecImageDataset(
        args.denoised_metadata,
        project_root=PROJECT_ROOT,
        indices=test_idx,
        augment=False,
        label_to_idx_map=label_map,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location=device)
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_preds, test_labels, _ = collect_predictions(model, test_loader, device)

    test_keys = denoised_frame.loc[test_idx, keys].reset_index(drop=True)
    test_keys["true_idx"] = test_labels
    test_keys["pred_idx"] = test_preds
    test_keys["true_label"] = test_keys["true_idx"].map(idx_to_label)
    test_keys["pred_label"] = test_keys["pred_idx"].map(idx_to_label)

    lookup = merged.set_index(keys)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for true_idx, pred_idx in error_patterns:
        pattern = f"{idx_to_label[true_idx]}->{idx_to_label[pred_idx]}"
        pattern_name = (
            f"{LABEL_NAMES.get(idx_to_label[true_idx], idx_to_label[true_idx])} "
            f"predicted as {LABEL_NAMES.get(idx_to_label[pred_idx], idx_to_label[pred_idx])}"
        )
        matches = test_keys[
            (test_keys["true_idx"] == true_idx) & (test_keys["pred_idx"] == pred_idx)
        ]
        print(f"\n{pattern}: {len(matches)} test misclassifications")

        examples: list[dict[str, object]] = []
        for _, row in matches.iterrows():
            key = (row["piece_id"], int(row["traj_index"]))
            meta = lookup.loc[key]
            raw_image = resolve_image_path(meta["raw_image_path"], PROJECT_ROOT)
            denoised_image = resolve_image_path(meta["denoised_image_path"], PROJECT_ROOT)
            examples.append(
                {
                    "piece_id": key[0],
                    "traj_index": key[1],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "idtap_name": meta["idtap_name"],
                    "unique_id": meta["unique_id"],
                    "raw_image": raw_image,
                    "denoised_image": denoised_image,
                }
            )
            summary_rows.append(
                {
                    "error_pattern": pattern,
                    "piece_id": key[0],
                    "traj_index": key[1],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "idtap_name": meta["idtap_name"],
                    "unique_id": meta["unique_id"],
                    "raw_image": str(raw_image),
                    "denoised_image": str(denoised_image),
                }
            )

        save_comparison_grid(
            examples,
            args.output_dir / f"raw_vs_denoised_{pattern.replace('->', '_to_')}.png",
            title=f"Denoised-model test errors: {pattern_name} ({len(matches)} total)",
            max_examples=args.max_examples,
        )

    summary_path = args.output_dir / "misclassified_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
