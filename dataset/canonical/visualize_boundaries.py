"""Boundary-centered diagnostic plots for Step 5.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.boundary_geometry import (  # noqa: E402
    _central_acceleration,
    _central_velocity,
    sample_dense_cents,
)
from dataset.canonical.schema import (  # noqa: E402
    canonical_root,
    figures_step5_5_dir,
    frames_npz_name,
    frames_dir,
)
from dataset.canonical.visualize_targets import TYPE_COLORS  # noqa: E402

WINDOW_S = 0.25


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_boundary_times(
    recording_doc: dict,
    lane_id: str,
    t0: float,
    t1: float,
) -> list[float]:
    times = []
    for traj in recording_doc["trajectories"]:
        if traj["lane_id"] != lane_id:
            continue
        end_s = float(traj["derived"]["end_s"])
        if t0 <= end_s <= t1:
            times.append(end_s)
    return sorted(set(times))


def plot_boundary(
    recording_doc: dict,
    primitives_doc: dict,
    example: dict,
    *,
    category: str,
    repo_root: Path,
    out_path: Path,
) -> None:
    recording_id = example["recording_id"]
    lane_id = example["lane_id"]
    boundary_s = float(example["boundary_s"])
    t0 = boundary_s - WINDOW_S
    t1 = boundary_s + WINDOW_S

    lane_trajs = [
        t for t in recording_doc["trajectories"] if t["lane_id"] == lane_id
    ]
    times, cents = sample_dense_cents(recording_doc, lane_trajs, t0, t1)
    velocities = _central_velocity(times, cents)
    accelerations = _central_acceleration(times, cents)

    npz_path = frames_dir(repo_root) / frames_npz_name(recording_id, lane_id)
    frame_times = np.array([])
    phase = np.array([])
    traj_type = np.array([])
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        mask = (data["frame_time_s"] >= t0) & (data["frame_time_s"] <= t1)
        frame_times = data["frame_time_s"][mask]
        phase = data["phase"][mask]
        traj_type = data["trajectory_type"][mask]
        valid = data["valid_target"][mask]
        frame_times = frame_times[valid]
        phase = phase[valid]
        traj_type = traj_type[valid]

    raw_bounds = raw_boundary_times(recording_doc, lane_id, t0, t1)

    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"{category}\n{recording_id} lane {lane_id} @ {boundary_s:.3f}s",
        fontsize=10,
    )

    axes[0].plot(times, cents, color="#333", linewidth=1.0)
    axes[0].axvline(boundary_s, color="#e45756", linewidth=1.5, label="canonical")
    for rb in raw_bounds:
        if abs(rb - boundary_s) > 1e-4:
            axes[0].axvline(rb, color="#888", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("pitch (cents)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(times, velocities, color="#54a24b", linewidth=0.9)
    axes[1].axvline(boundary_s, color="#e45756", linewidth=1.5)
    axes[1].set_ylabel("dp/dt\n(cents/s)")

    axes[2].plot(times, accelerations, color="#4c78a8", linewidth=0.9)
    axes[2].axvline(boundary_s, color="#e45756", linewidth=1.5)
    axes[2].set_ylabel("d²p/dt²\n(cents/s²)")

    for prim in primitives_doc["primitives"]:
        if prim["lane_id"] != lane_id:
            continue
        if prim["end_s"] < t0 or prim["start_s"] > t1:
            continue
        ct = int(prim["canonical_type"])
        axes[3].broken_barh(
            [
                (
                    max(prim["start_s"], t0),
                    min(prim["end_s"], t1) - max(prim["start_s"], t0),
                )
            ],
            (ct - 0.4, 0.8),
            facecolors=TYPE_COLORS[ct],
        )
    axes[3].axvline(boundary_s, color="#e45756", linewidth=1.5)
    axes[3].set_ylabel("type")
    axes[3].set_yticks([0, 1, 2, 3])

    if len(frame_times):
        for ct in np.unique(traj_type):
            m = traj_type == ct
            axes[4].scatter(
                frame_times[m],
                phase[m],
                s=12,
                c=TYPE_COLORS.get(int(ct), "#999"),
                label=f"T{int(ct)}",
            )
    axes[4].axvline(boundary_s, color="#e45756", linewidth=1.5)
    axes[4].set_ylabel("phase")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylim(-0.05, 1.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=None,
        help="Path to step_5_5_analysis.json (default: canonical/v1/)",
    )
    args = parser.parse_args()

    root = canonical_root(args.repo_root)
    analysis_path = args.analysis_json or (root / "step_5_5_analysis.json")
    if not analysis_path.exists():
        raise SystemExit(f"Run analyze_step_5_5.py first; missing {analysis_path}")

    analysis = load_json(analysis_path)
    examples = analysis.get("visualization_examples", {})
    out_dir = figures_step5_5_dir(args.repo_root)

    for category, items in examples.items():
        for item in items:
            recording_id = item["recording_id"]
            rec = load_json(root / "recordings" / f"{recording_id}.json")
            prim = load_json(root / "primitives" / f"{recording_id}.json")
            lane_token = item["lane_id"].replace(":", "_")
            boundary_tag = f"{item['boundary_s']:.3f}".replace(".", "p")
            out = out_dir / f"{category}_{recording_id}_{lane_token}_{boundary_tag}.png"
            plot_boundary(
                rec,
                prim,
                item,
                category=category,
                repo_root=args.repo_root,
                out_path=out,
            )
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
