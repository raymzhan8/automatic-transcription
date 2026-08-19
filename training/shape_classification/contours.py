"""Step 22 section 1-2: load canonical primitives and resample their pitch
contour onto a fixed normalized-phase grid, x=(t-start_s)/(end_s-start_s) in
[0,1], from two sources:

  - oracle (O): the analytic IDTAP parametric curve itself, evaluated
    directly at the desired phase points via the same reconstruction
    (`Trajectory.compute`) `dataset/canonical/contour.py` already uses to
    regenerate framewise GT targets -- no intermediate 10ms-grid
    interpolation, the "cleanest possible representation of the annotation".
  - CREPE (C): Step 21's frozen dense CREPE path, linearly interpolated from
    its native 10ms grid onto the same phase grid. Supports a boundary
    perturbation (start_delta_s/end_delta_s) for Step 22 section 17 -- the
    window fed to CREPE shifts, the trajectory's true canonical_type label
    does not.

Audio is never time-stretched; only the extracted pitch contour is
resampled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.canonical.contour import compute_log2_at_time, primitive_trajectory_at_time  # noqa: E402
from dataset.canonical.schema import primitives_dir, recordings_dir  # noqa: E402

N_PHASE_POINTS = 64
X_GRID = np.linspace(0.0, 1.0, N_PHASE_POINTS)
EPS_S = 1e-9


def load_recording_and_primitives(rid: str, repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rec_doc = json.loads((recordings_dir(repo_root) / f"{rid}.json").read_text(encoding="utf-8"))
    prim_doc = json.loads((primitives_dir(repo_root) / f"{rid}.json").read_text(encoding="utf-8"))
    return rec_doc, prim_doc["primitives"]


def oracle_contour(
    primitive: dict[str, Any],
    rec_doc: dict[str, Any],
    by_idx: dict[int, dict[str, Any]],
    *,
    cache: dict[int, Any],
    x_grid: np.ndarray = X_GRID,
) -> np.ndarray | None:
    """log2-Hz array [N] sampled directly from the analytic parametric curve
    at the primitive's own phase points -- section 1's "dense pitch
    generated directly from the canonical IDTAP trajectory"."""
    start_s, end_s = primitive["start_s"], primitive["end_s"]
    dur = end_s - start_s
    out = np.empty(len(x_grid))
    for i, x in enumerate(x_grid):
        t = start_s + x * dur
        t = min(t, end_s - EPS_S)
        hit = primitive_trajectory_at_time(primitive, by_idx, t)
        if hit is None:
            return None
        entry, x_local = hit
        v = compute_log2_at_time(rec_doc, entry, x_local, cache=cache)
        if v is None:
            return None
        out[i] = v
    return out


def crepe_contour(
    frame_time_s: np.ndarray,
    crepe_log2_hz: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    x_grid: np.ndarray = X_GRID,
    start_delta_s: float = 0.0,
    end_delta_s: float = 0.0,
    lane_duration_s: float | None = None,
) -> np.ndarray | None:
    """log2-Hz array [N], linearly interpolated from CREPE's native 10ms
    grid onto the primitive's phase grid. `start_delta_s`/`end_delta_s`
    perturb the extraction WINDOW only (section 17) -- the audio and CREPE
    path are untouched; only which slice of it we read changes."""
    s = start_s + start_delta_s
    e = end_s + end_delta_s
    if lane_duration_s is not None:
        s = min(max(s, 0.0), lane_duration_s)
        e = min(max(e, 0.0), lane_duration_s)
    if e - s < 2 * EPS_S:
        return None
    t_grid = s + x_grid * (e - s)
    return np.interp(t_grid, frame_time_s, crepe_log2_hz).astype(np.float64)


def all_recording_ids(repo_root: Path = REPO_ROOT) -> list[str]:
    return sorted(p.stem for p in primitives_dir(repo_root).glob("*.json"))
