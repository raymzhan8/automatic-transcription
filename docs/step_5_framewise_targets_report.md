# Step 5 — Canonical Framewise Targets Report

Implementation and validation of the four-primitive canonical representation and 10 ms framewise targets. **No model was trained.**

Reproduce the pipeline from the repository root:

```bash
python dataset/canonical/build_primitives.py
python dataset/canonical/verify_contours.py
python dataset/canonical/build_frames.py
python dataset/canonical/validate_targets.py
python dataset/canonical/visualize_targets.py
python -m pytest dataset/canonical/test_targets.py -q
```

---

## 1. Implemented canonicalization code

| Module | Purpose |
|--------|---------|
| [`dataset/canonical/decompose.py`](../dataset/canonical/decompose.py) | Approved 4/5/6 decomposition rules (mirrors export) |
| [`dataset/canonical/primitives.py`](../dataset/canonical/primitives.py) | Build primitive trajectories with provenance |
| [`dataset/canonical/build_primitives.py`](../dataset/canonical/build_primitives.py) | CLI → `primitives/*.json` |
| [`dataset/canonical/contour.py`](../dataset/canonical/contour.py) | Shared pitch-at-time / contour comparison |
| [`dataset/canonical/verify_contours.py`](../dataset/canonical/verify_contours.py) | Raw vs canonical contour verification |
| [`dataset/canonical/frames.py`](../dataset/canonical/frames.py) | 10 ms framewise target generation |
| [`dataset/canonical/build_frames.py`](../dataset/canonical/build_frames.py) | CLI → `frames/*.npz` |
| [`dataset/canonical/validate_targets.py`](../dataset/canonical/validate_targets.py) | Corpus validation + statistics |
| [`dataset/canonical/test_targets.py`](../dataset/canonical/test_targets.py) | Pytest sanity checks |
| [`dataset/canonical/visualize_targets.py`](../dataset/canonical/visualize_targets.py) | Debug figures |

Raw recording documents under `recordings/` are **never modified**.

---

## 2. Canonical primitive schema

Each primitive in `primitives/<recording_id>.json`:

| Field | Description |
|-------|-------------|
| `primitive_id` | `{recording_id}:{lane_token}:p{seq:05d}` |
| `lane_id` | e.g. `0:0` |
| `canonical_type` | 0–3 only |
| `source_raw_trajectory_ids` | Provenance (list of traj_id strings) |
| `source_raw_indices` | Raw trajectory indices |
| `source_raw_type` | Wire IDTAP type before mapping |
| `source_subsegment_index` | 0-based within composite; null otherwise |
| `source_x_start`, `source_x_end` | Normalized span in source raw traj for `compute()` |
| `start_s`, `end_s`, `duration_s` | Absolute times |
| `start_pitch`, `end_pitch` | Derived pitch blocks |
| `rule_applied` | `keep`, `merge_t0`, `decompose_4`, `decompose_5`, `decompose_6` |

Mapping rules (no invented decomposition):

| Raw | Canonical primitive(s) |
|-----|------------------------|
| 0 | T0 (merged when same pitch_key) |
| 1,2,3 | T1, T2, T3 |
| 4 | T2 segment → T1 segment |
| 5 | T1 segment → T3 segment |
| 6 | T1 × N (via `dur_array`) |
| 7,8,9,10,11,12,13 | **masked** (no primitive emitted) |

**Step 5 choice:** Type 6 is fully decomposed into T1 primitives (overrides Step 4.5 KEEP_COMPOSITE recommendation).

---

## 3. Framewise target schema

Per lane: `frames/<recording_id>_<lane>.npz`

| Array | dtype | ML target? |
|-------|-------|------------|
| `frame_time_s` | float64 | metadata |
| `valid_target` | bool | gates loss |
| `trajectory_type` | int8 | **yes** (0–3; -1 masked) |
| `pitch_log2_hz` | float32 | **yes** (parametric annotation) |
| `phase` | float32 | **yes** ([0,1] within primitive) |
| `dp_dt_log2_hz_per_s` | float32 | diagnostic only |
| `primitive_id` | object | metadata |
| `source_raw_trajectory_ids` | object | metadata |
| `boundary_start_frame_indices` | int32 | eval metadata |
| `boundary_end_frame_indices` | int32 | eval metadata |

**Parameters:**

- `hop_s = 0.01` (10 ms)
- Frame centers: `t_k = (k + 0.5) * hop_s`
- Membership: half-open `[start_s, end_s)` on frame centers
- Pitch: `Trajectory.compute(x, log_scale=True)` — **not measured F0**
- Phase: `(t - primitive_start) / primitive_duration`, clamped to [0, 1]
- Phase resets automatically at each new `primitive_id` (including T1|T1)

**Not included in v1:** boundary head, activity head, dp/dt as ML target.

---

## 4. Statistics before and after canonicalization

| Metric | Value |
|--------|------:|
| Recordings | 17 |
| Raw trajectories | 5,209 |
| Canonical primitives | 7,177 |
| T0 merge runs | 57 |
| Type 4 decompositions | 192 |
| Type 5 decompositions | 28 |
| Type 6 T1 segments emitted | 4,654 |

---

## 5. Type 0 merge statistics

- **Pitch tolerance:** exact `pitch_key` (`swara|oct|raised|log_offset`)
- **Rule:** consecutive T0 with identical endpoint pitch_key and `meets` interval → one primitive
- **Merge runs:** 57 (1,306 T0 primitives after merge vs 1,363 raw Fixed trajectories)
- Different-pitch T0 pairs remain separate (206 in transition analysis)

---

> **Correction (Step 5.5):** The duration tables in §6–7 below misread JSON values (e.g. `0.028` reported as 2.8%, and a 30 ms bucket labeled as 50 ms). Corrected statistics are in [`docs/step_5_5_boundary_learnability_report.md`](step_5_5_boundary_learnability_report.md) §1.

## 6. Type 4/5/6 decomposition statistics

| Type | Raw count (approx) | Primitive segments added |
|------|-------------------:|-------------------------:|
| 4 | 96 | 192 (2 per raw) |
| 5 | 14 | 28 (2 per raw) |
| 6 | 1,237 | 4,654 T1 segments |

Type 6 primitive duration (from decomposed T1 segments):

| Stat | Value |
|------|------:|
| min | 9.2 ms |
| p01 | 32.2 ms |
| median | 109.1 ms |
| mean | 168.3 ms |
| max | 3.91 s |
| % shorter than 10 ms | 4.3% |
| % shorter than 20 ms | 40.8% |
| % shorter than 50 ms | 4.1% |

---

## 7. Trajectory duration statistics (10 ms suitability)

**Overall primitives:**

| Stat | Value |
|------|------:|
| min | 9.2 ms |
| p01 | 35.1 ms |
| p05 | 51.9 ms |
| median | 132.1 ms |
| mean | 236.5 ms |
| p95 | 726.3 ms |
| max | 7.86 s |
| % < 10 ms | 2.8% |
| % < 20 ms | 27.9% |
| % < 50 ms | 47.4% |
| % < 100 ms | 31.8% |

**By canonical type (% shorter than 10 ms):**

| Type | % < 10 ms | median duration |
|------|----------:|----------------:|
| T0 | 0.0% | 235 ms |
| T1 | 4.0% | 109 ms |
| T2 | 0.0% | 226 ms |
| T3 | 0.0% | 235 ms |

**Assessment:** 10 ms is adequate for most primitives (97% ≥ 10 ms overall), but **T1 from Type 6 decomposition** has 4% of segments shorter than one frame and 41% shorter than two frames. Phase and type supervision on sub-10 ms T1 segments may be sparse; consider this when weighting loss or evaluating segmentation.

---

## 8. Contour preservation verification

Sampled at **1 ms** over annotated spans only (with Trajectory object caching).

| Metric | Value |
|--------|------:|
| Tolerance | 1e-9 log2 Hz |
| Recordings failed | 0 |
| Max error over corpus | 8.34e-13 log2 Hz |

**Conclusion:** canonicalization (T0 merge + 4/5/6 decomposition) preserves the dense parametric pitch contour to numerical precision.

Full report: [`output/canonical/v1/step_5_contour_verification.json`](../output/canonical/v1/step_5_contour_verification.json)

---

## 9. Same-type boundary analysis

| Metric | Value |
|--------|------:|
| Same-type canonical boundaries | 4,054 |
| T1 → T1 boundaries | 3,686 |
| Classified pitch_step (>1 cent) | 1,848 |
| Classified phase_only | 2,206 |

Many T1|T1 boundaries follow Type 6 decomposition. A substantial fraction (46%) show endpoint pitch differences > 1 cent — often at transitions between different raw Type 6 objects or at internal segment boundaries where control-point metadata differs. Phase resets still occur at every primitive boundary regardless.

---

## 10. Visualizations

Generated under [`output/canonical/v1/figures/step5/`](../output/canonical/v1/figures/step5/):

- `645ff354deeaf2d1e33b3c44_0_0.png`
- `6491d48d608d1718e0311003_0_0.png`
- `6417585554a0bfbd8de2d3ff_0_0.png`
- `65b2ab707f607fb14920201a_0_0.png`
- `6824de49abc4705438ce918b_0_0.png` (+ lane `0:1`)

Each figure: waveform, CQT, raw type bands, canonical primitive bands, pitch target, phase (30 s window with valid frames).

---

## 11. Sanity checks and tests

All checks in [`output/canonical/v1/step_5_validation.json`](../output/canonical/v1/step_5_validation.json):

| Check | Result |
|-------|--------|
| Primitives exist | pass |
| canonical_type ∈ {0,1,2,3} | pass |
| Contour preservation | pass |
| Split leakage | pass |
| Frame npz validation | pass |

```bash
python -m pytest dataset/canonical/test_targets.py -q
# 6 passed
```

---

## 12. Assumptions and ambiguous cases

1. **Type 6 decomposed to T1** — explicit Step 5 decision; creates many short T1 primitives and T1|T1 phase resets.
2. **Silent (type 12) masked** — not treated as acoustic inactivity.
3. **Types 7, 13 masked** — not forced into four-class vocabulary.
4. **T0 merge** may cross vowel annotation changes (12/66 in Step 4.5); low risk for pitch/type.
5. **Pitch target** is parametric IDTAP annotation, not measured F0.
6. **No boundary head** — segmentation recoverable from type + pitch + phase in principle; not yet tested with a model.

---

## 13. Tentative model target bundle

```text
audio
   ↓
shared temporal encoder
   ↓
┌─────────────────┬───────────────┬───────────────┐
│ trajectory type │ dense pitch   │ phase         │
│ 0 / 1 / 2 / 3   │ log2_hz       │ [0, 1]        │
└─────────────────┴───────────────┴───────────────┘
         masked where valid_target == False
```

Transformation pipeline:

```text
RAW IDTAP (5,209 trajectories, unchanged)
    ↓
primitives_v1: MERGE T0 same pitch, MAP 4/5/6, MASK 7–13 & silent
    ↓
CANONICAL PRIMITIVES (7,177)
    ↓
10 ms frames: type, pitch_log2_hz, phase, valid_target
```
