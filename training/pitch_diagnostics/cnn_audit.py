"""Audit FrequencyCNN frequency-axis collapse (no weight changes)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.framewise_models import FrequencyCNN  # noqa: E402
from training.pitch_diagnostics.common import OUT_DIR, write_json  # noqa: E402


def _layer_shapes(model: FrequencyCNN, spec: torch.Tensor) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    x = spec
    b, c, f, t = x.shape
    rows.append({"layer": "input", "channels": c, "freq": f, "time": t})
    for name, module in model.net.named_children():
        x = module(x)
        _, c, f, t = x.shape
        rows.append(
            {
                "layer": f"{name}:{type(module).__name__}",
                "channels": c,
                "freq": f,
                "time": t,
            }
        )
    return rows


def shift_invariance_check(model: FrequencyCNN, spec: torch.Tensor, shifts: tuple[int, ...] = (4, 8, 16)) -> dict:
    model.eval()
    with torch.no_grad():
        base = model(spec)
        out: dict[str, float] = {}
        for s in shifts:
            rolled = torch.zeros_like(spec)
            if s > 0:
                rolled[:, :, s:, :] = spec[:, :, :-s, :]
            else:
                rolled[:, :, :s, :] = spec[:, :, -s:, :]
            shifted = model(rolled)
            num = (base - shifted).pow(2).mean().sqrt().item()
            den = base.pow(2).mean().sqrt().item() + 1e-8
            cosine = torch.nn.functional.cosine_similarity(
                base.flatten(1), shifted.flatten(1)
            ).mean().item()
            out[f"shift_{s}"] = {
                "rel_l2": num / den,
                "cosine": cosine,
            }
        return out


def main() -> None:
    torch.manual_seed(0)
    model = FrequencyCNN()
    spec = torch.randn(2, 1, 360, 400)
    shapes = _layer_shapes(model, spec)
    inv = shift_invariance_check(model, spec)
    collapsed = [r for r in shapes if r["freq"] == 1]
    report = {
        "input": "[B,1,360,T]",
        "cents_per_cqt_bin": 1200 / 72,
        "layer_table": shapes,
        "explicit_pitch_lost_after": collapsed[0]["layer"] if collapsed else None,
        "note": (
            "AdaptiveMaxPool2d((1, None)) removes the frequency axis. "
            "A translation along CQT frequency that remains inside the 22-bin map "
            "can produce a similar max-pooled embedding."
        ),
        "shift_invariance_random_input": inv,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    out = OUT_DIR / "cnn_audit.json"
    write_json(out, report)
    print(json_dumps_brief(report))
    print(f"wrote {out}")


def json_dumps_brief(report: dict) -> str:
    import json

    return json.dumps(
        {
            "layer_table": report["layer_table"],
            "explicit_pitch_lost_after": report["explicit_pitch_lost_after"],
            "shift_invariance_random_input": report["shift_invariance_random_input"],
        },
        indent=2,
    )


if __name__ == "__main__":
    main()
