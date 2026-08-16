"""Train a CNN to classify CQT spectrogram trajectory images."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "cnn_runs",
        help="Directory for checkpoints and evaluation artifacts",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-by",
        choices=("label", "piece_id"),
        default="label",
        help="Split strategy: stratified by label (single song) or grouped by piece_id",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-readable run title (e.g. 'phase-2-no-silent')",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        default=[],
        help="Labels to drop from training/eval (e.g. silent)",
    )
    return parser.parse_args()


def slugify_run_name(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug or "run"


def resolve_output_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if run_name:
        slug = slugify_run_name(run_name)
        candidate = output_root / slug
        if candidate.exists():
            candidate = output_root / f"{slug}_{timestamp}"
        return candidate
    return output_root / timestamp


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def split_indices_stratified(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    label_map: dict[str, int],
) -> tuple[list[int], list[int], list[int]]:
    labels = [label_to_idx(v, label_map) for v in frame["label"]]
    indices = frame.index.to_list()

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=seed,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def split_indices_by_piece(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    label_map: dict[str, int],
) -> tuple[list[int], list[int], list[int]]:
    piece_ids = sorted(frame["piece_id"].astype(str).unique())
    if len(piece_ids) < 3:
        warnings.warn(
            f"Only {len(piece_ids)} piece(s) found; falling back to stratified label split.",
            stacklevel=2,
        )
        return split_indices_stratified(
            frame,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            label_map=label_map,
        )

    rng = np.random.default_rng(seed)
    shuffled = piece_ids.copy()
    rng.shuffle(shuffled)

    n_pieces = len(shuffled)
    n_train = max(1, int(round(n_pieces * train_ratio)))
    n_val = max(1, int(round(n_pieces * val_ratio)))
    if n_train + n_val >= n_pieces:
        n_val = max(1, n_pieces - n_train - 1)
    n_test = n_pieces - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_pieces - n_val - n_test)

    train_pieces = set(shuffled[:n_train])
    val_pieces = set(shuffled[n_train : n_train + n_val])
    test_pieces = set(shuffled[n_train + n_val :])

    train_idx = frame.index[frame["piece_id"].astype(str).isin(train_pieces)].tolist()
    val_idx = frame.index[frame["piece_id"].astype(str).isin(val_pieces)].tolist()
    test_idx = frame.index[frame["piece_id"].astype(str).isin(test_pieces)].tolist()
    return train_idx, val_idx, test_idx


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs)
    return all_preds, all_labels, all_probs


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    path: Path,
    *,
    class_names: list[str],
    num_classes: int,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified_grid(
    dataset: SpecImageDataset,
    indices: list[int],
    y_true: list[int],
    y_pred: list[int],
    path: Path,
    *,
    idx_to_label: dict[int, str],
    max_images: int = 12,
) -> None:
    mistakes = [
        (idx, true, pred)
        for idx, true, pred in zip(indices, y_true, y_pred)
        if true != pred
    ]
    if not mistakes:
        return

    sample = mistakes[:max_images]
    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (dataset_idx, true, pred) in zip(axes_array, sample):
        row = dataset.row(dataset_idx)
        image_path = resolve_image_path(row["image_path"], dataset.project_root)
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(f"true={idx_to_label[true]} pred={idx_to_label[pred]}")
        ax.axis("off")

    for ax in axes_array[len(sample) :]:
        ax.axis("off")

    fig.suptitle("Misclassified examples")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    excluded_labels = [normalize_label(label) for label in args.exclude_labels]
    label_map, idx_to_label, class_names, num_classes = build_label_config(
        excluded_labels
    )

    metadata_path = args.metadata.resolve()
    frame = pd.read_csv(metadata_path)
    frame["label"] = frame["label"].map(normalize_label)
    if excluded_labels:
        before = len(frame)
        frame = frame[~frame["label"].isin(excluded_labels)]
        print(
            f"Excluded labels {excluded_labels}: {before} -> {len(frame)} samples "
            f"({num_classes} classes: {', '.join(class_names)})"
        )

    unique_pieces = frame["piece_id"].astype(str).nunique()
    if unique_pieces == 1:
        warnings.warn(
            "All samples come from a single recording; metrics may overstate generalization.",
            stacklevel=2,
        )

    if args.split_by == "piece_id":
        train_idx, val_idx, test_idx = split_indices_by_piece(
            frame,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            label_map=label_map,
        )
    else:
        train_idx, val_idx, test_idx = split_indices_stratified(
            frame,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            label_map=label_map,
        )

    train_dataset = SpecImageDataset(
        metadata_path,
        project_root=PROJECT_ROOT,
        indices=train_idx,
        augment=True,
        label_to_idx_map=label_map,
    )
    val_dataset = SpecImageDataset(
        metadata_path,
        project_root=PROJECT_ROOT,
        indices=val_idx,
        augment=False,
        label_to_idx_map=label_map,
    )
    test_dataset = SpecImageDataset(
        metadata_path,
        project_root=PROJECT_ROOT,
        indices=test_idx,
        augment=False,
        label_to_idx_map=label_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_labels = [
        label_to_idx(frame.loc[i]["label"], label_map) for i in train_idx
    ]
    class_weights = compute_class_weights(train_labels, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = resolve_output_dir(args.output_dir, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "run_name": args.run_name,
        "metadata": str(metadata_path),
        "split_by": args.split_by,
        "exclude_labels": excluded_labels,
        "class_names": class_names,
        "num_classes": num_classes,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "unique_pieces": int(unique_pieces),
        "seed": args.seed,
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        val_preds, val_labels, _ = collect_predictions(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} macro_f1={val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "label_to_idx": label_map,
                "exclude_labels": excluded_labels,
                "run_name": args.run_name,
                "epoch": epoch,
                "val_macro_f1": val_f1,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_preds, test_labels, _ = collect_predictions(model, test_loader, device)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    print("\nTest results")
    print(f"  accuracy: {test_acc:.3f}")
    print(f"  macro F1: {test_f1:.3f}")
    print(report)

    (output_dir / "classification_report.txt").write_text(report)
    save_confusion_matrix(
        test_labels,
        test_preds,
        output_dir / "confusion_matrix.png",
        class_names=class_names,
        num_classes=num_classes,
    )
    save_misclassified_grid(
        test_dataset,
        list(range(len(test_labels))),
        test_labels,
        test_preds,
        output_dir / "misclassified.png",
        idx_to_label=idx_to_label,
    )

    summary = {
        "run_name": args.run_name,
        "exclude_labels": excluded_labels,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
