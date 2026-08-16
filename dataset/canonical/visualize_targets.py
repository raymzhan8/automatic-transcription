"""Visualization for Step 5 framewise targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.schema import figures_step5_dir, frames_npz_name  # noqa: E402

TYPE_COLORS = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b", 3: "#e45756", -1: "#bbbbbb"}
RAW_CMAP = plt.cm.tab20


def source_audio_path(recording_doc: dict, repo_root: Path) -> Path | None:
    for item in recording_doc.get("audio", {}).get("files") or []:
        if item.get("role") == "source" and item.get("relpath"):
            path = repo_root / item["relpath"]
            if path.exists():
                return path
    return None


def plot_recording_lane(
    recording_doc: dict,
    primitives_doc: dict,
    npz_path: Path,
    *,
    repo_root: Path,
    out_path: Path,
    t0: float = 0.0,
    t1: float | None = None,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    times = data["frame_time_s"]
    valid = data["valid_target"]
    pitch = data["pitch_log2_hz"]
    phase = data["phase"]
    traj_type = data["trajectory_type"]

    if t1 is None:
        t1 = float(times[-1]) if len(times) else 0.0
    mask = (times >= t0) & (times <= t1)

    audio_path = source_audio_path(recording_doc, repo_root)
    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{recording_doc['recording_id']} lane {data['lane_id']}")

    if audio_path:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, offset=t0, duration=t1 - t0)
        ax_t = np.linspace(t0, t0 + len(y) / sr, num=len(y), endpoint=False)
        axes[0].plot(ax_t, y, color="0.2", linewidth=0.5)
    axes[0].set_ylabel("waveform")

    if audio_path:
        cqt = np.abs(
            librosa.cqt(y, sr=sr, hop_length=512, fmin=75, n_bins=72, bins_per_octave=72)
        )
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        librosa.display.specshow(
            cqt_db,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="cqt_hz",
            ax=axes[1],
            x_coords=np.linspace(t0, t1, cqt_db.shape[1]),
        )
    axes[1].set_ylabel("CQT")

    lane = str(data["lane_id"])
    raw_lane = [t for t in recording_doc["trajectories"] if t["lane_id"] == lane]
    for traj in raw_lane:
        d = traj["derived"]
        if d["end_s"] < t0 or d["start_s"] > t1:
            continue
        tid = int(d["type_id"])
        axes[2].broken_barh(
            [(max(d["start_s"], t0), min(d["end_s"], t1) - max(d["start_s"], t0))],
            (tid - 0.4, 0.8),
            facecolors=RAW_CMAP(tid % 20),
        )
    axes[2].set_ylabel("raw type")
    axes[2].set_yticks(range(14))

    for prim in primitives_doc["primitives"]:
        if prim["lane_id"] != lane:
            continue
        if prim["end_s"] < t0 or prim["start_s"] > t1:
            continue
        ct = prim["canonical_type"]
        axes[3].broken_barh(
            [(max(prim["start_s"], t0), min(prim["end_s"], t1) - max(prim["start_s"], t0))],
            (ct - 0.4, 0.8),
            facecolors=TYPE_COLORS[ct],
        )
    axes[3].set_ylabel("prim")
    axes[3].set_yticks([0, 1, 2, 3])

    axes[4].plot(times[mask], pitch[mask], color="#333", linewidth=0.8)
    axes[4].fill_between(
        times[mask],
        np.nanmin(pitch[mask]) if np.any(valid[mask]) else 0,
        np.nanmax(pitch[mask]) if np.any(valid[mask]) else 1,
        where=~valid[mask],
        color="#dddddd",
        alpha=0.4,
    )
    axes[4].set_ylabel("log2 pitch")

    axes[5].plot(times[mask], phase[mask], color="#666", linewidth=0.8)
    for ct in np.unique(traj_type[mask & valid]):
        axes[5].scatter(
            times[mask & valid & (traj_type == ct)],
            phase[mask & valid & (traj_type == ct)],
            s=4,
            c=TYPE_COLORS.get(int(ct), "#999"),
        )
    axes[5].set_ylabel("phase")
    axes[5].set_xlabel("time (s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-ids",
        nargs="*",
        default=[
            "645ff354deeaf2d1e33b3c44",
            "6491d48d608d1718e0311003",
            "6417585554a0bfbd8de2d3ff",
            "65b2ab707f607fb14920201a",
            "6824de49abc4705438ce918b",
        ],
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--window-s", type=float, default=30.0)
    args = parser.parse_args()

    root = args.repo_root / "output" / "canonical" / "v1"
    out_dir = figures_step5_dir(args.repo_root)

    for recording_id in args.recording_ids:
        rec = json.loads((root / "recordings" / f"{recording_id}.json").read_text())
        prim = json.loads((root / "primitives" / f"{recording_id}.json").read_text())
        for lane in rec.get("lanes", []):
            lane_id = lane["lane_id"]
            npz = root / "frames" / frames_npz_name(recording_id, lane_id)
            if not npz.exists():
                continue
            # pick a window with valid frames
            data = np.load(npz, allow_pickle=True)
            valid_times = data["frame_time_s"][data["valid_target"]]
            t0 = float(valid_times[0]) if len(valid_times) else 0.0
            t1 = t0 + args.window_s
            out = out_dir / f"{recording_id}_{lane_id.replace(':', '_')}.png"
            plot_recording_lane(
                rec, prim, npz, repo_root=args.repo_root, out_path=out, t0=t0, t1=t1
            )
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
