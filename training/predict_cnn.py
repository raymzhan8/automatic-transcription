"""Run inference with a trained trajectory CNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more PNG paths, or directories containing PNGs",
    )
    return parser.parse_args()


def collect_image_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.png")))
        elif item.suffix.lower() == ".png":
            paths.append(item)
        else:
            raise ValueError(f"Unsupported input: {item}")
    if not paths:
        raise ValueError("No PNG images found in inputs")
    return paths


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, dict[str, float]]:
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_label = class_names[pred_idx]
    prob_map = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return pred_label, prob_map


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    run_name = checkpoint.get("run_name")
    if run_name:
        print(f"Model run: {run_name}")
    print(f"Classes: {', '.join(class_names)}")

    image_paths = collect_image_paths(args.inputs)
    for image_path in image_paths:
        pred_label, prob_map = predict_image(
            model, image_path, device, class_names
        )
        ranked = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
        print(f"\n{image_path.name}")
        print(f"  predicted: {pred_label}")
        print("  probabilities:")
        for label, prob in ranked:
            print(f"    {label}: {prob:.3f}")


if __name__ == "__main__":
    main()
