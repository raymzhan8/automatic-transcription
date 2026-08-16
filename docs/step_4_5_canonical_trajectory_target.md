# Step 4.5 — Canonical Trajectory Target Report

Analysis-only deliverable. No dataset or training code was modified.

**Corpus:** 17 recordings, 5,209 raw trajectories, 5,189 consecutive same-lane transitions under `output/canonical/v1/`.

**Reproduce statistics:**

```bash
python dataset/canonical/analyze_step_4_5.py --output output/canonical/v1/step_4_5_analysis.json
```

---

## 1. Exact current four-class trajectory definition

The project does **not** have one consistent "four-class" definition. Four coexisting schemes must be distinguished.

### 1.1 Legacy four-class clip dataset

[`inventory/dataset_utils.py`](../inventory/dataset_utils.py) defines exactly four shape classes, mapping 1:1 to raw IDTAP types 0–3:

```python
TARGET_IDTAP_IDS = {0, 1, 2, 3}

LABEL_MAP = {
    "trajectory_1": {"idtap_id": 0, "name": "Fixed"},
    "trajectory_2": {"idtap_id": 1, "name": "Bend: Simple"},
    "trajectory_3": {"idtap_id": 2, "name": "Bend: Sloped Start"},
    "trajectory_4": {"idtap_id": 3, "name": "Bend: Sloped End"},
}
```

- Composites (4, 5, 6), silent (12), and ornaments (7, 13, …) are **excluded** at candidate iteration time.
- This is the strictest interpretation of "four-class."

### 1.2 Live CNN five-class experiment

[`training/spec_dataset.py`](../training/spec_dataset.py) and the [README label scheme](../README.md) use **five** classes:

```python
FULL_LABEL_TO_IDX = {
    "0": 0, "1": 1, "2": 2, "3": 3, "silent": 4,
}
```

| Label | IDTAP ID | IDTAP name |
|-------|----------|------------|
| `0` | 0 | Fixed |
| `1` | 1 | Bend: Simple |
| `2` | 2 | Bend: Sloped Start |
| `3` | 3 | Bend: Sloped End |
| `silent` | 12 | Silent |

Training can drop `silent` via `--exclude-labels silent`, recovering a four-class shape experiment.

### 1.3 CNN export pipeline mapping

[`dataset/export_denoised_cnn_dataset.py`](../dataset/export_denoised_cnn_dataset.py) applies additional rules:

```python
SKIP_IDTAP_IDS = {7, 8, 9, 10, 11, 13}
COMPOSITE_LABELS = {4: [2, 1], 5: [1, 3]}
```

| Raw ID | Handling |
|--------|----------|
| 0, 1, 2, 3 | Direct label |
| 4 (Ladle) | Decompose via `dur_array` → segments labeled `[2, 1]` |
| 5 (Reverse Ladle) | Decompose → `[1, 3]` |
| 6 (Simple Multiple) | Decompose into N segments, all labeled `1` |
| 12 | `silent` |
| 7, 8, 9, 10, 11, 13 | **Skipped** (logged to `skipped.csv`) |

**Known bug:** `trajectory_export_label()` returns only the **first** segment's label for composites, so clip metadata loses internal segmentation.

### 1.4 Notebook evaluation mapping

[`notebooks/f0s.ipynb`](../notebooks/f0s.ipynb) `trajectory_label()` collapses composites to a single class:

```python
if traj.id in {0, 1, 2, 3}: return traj.id
if traj.id == 6: return 1
if traj.id == 4: return 2
if traj.id == 5: return 3
return "other"  # includes skipped types
```

### 1.5 Answers to the mapping questions

| Question | Answer |
|----------|--------|
| What are the four trajectory types? | IDTAP types 0–3: Fixed, Bend Simple, Bend Sloped Start, Bend Sloped End ([`docs/label_definitions.md`](label_definitions.md)) |
| Map to raw IDs 0–13? | See table above; **not** 1:1 for the full corpus |
| Literally four raw types? | Only in the legacy clip builder; the live CNN adds silent and partial composite mapping |
| Multiple raw types → one ML class? | **Yes:** 6→1, 4→2 (first segment only in export label), 5→3 (first segment only) |
| Some raw types excluded? | **Yes:** 7, 8, 9, 10, 11, 13 skipped in export |
| Composites decomposed? | **Partially:** export knows `[2,1]` / `[1,3]` / N×`1` splits, but clip label uses first segment only |
| Rationale | CNN targets the four fundamental pitch-shape primitives; composites are treated as sequences of primitives; krintin/vibrato/slide families are out of scope for the shape classifier |

**Flag:** The phrase "four-class experiment" is ambiguous — it may mean four shapes (0–3), four shapes plus silent (five heads), or four legacy clip labels excluding all composites.

---

## 2. Raw boundary statistics

All statistics from [`output/canonical/v1/index/transitions.csv`](../output/canonical/v1/index/transitions.csv), computed by [`dataset/canonical/analyze_step_4_5.py`](../dataset/canonical/analyze_step_4_5.py).

**Pitch tolerance:** exact `pitch_key` equality (`swara|oct|raised|log_offset`), per [`dataset/canonical/schema.py`](../dataset/canonical/schema.py) `derivation_params.pitch_match_rule`. No cent tolerance is applied; adjacent pairs with identical swara/oct/raised/log_offset register as `same_pitch=True` with 0.0 cents delta.

### 2.1 Transition categories (n = 5,189)

| Category | Count |
|----------|------:|
| different type + different pitch | 2,529 |
| different type + same pitch | 1,736 |
| same type + same pitch | 553 |
| same type + different pitch | 371 |

Every transition has `interval_relation == "meets"` and `gap_s == 0.0`.

### 2.2 Type 0 (Fixed) adjacent pairs (n = 272)

| Pattern | Count | Meaning |
|---------|------:|---------|
| T0(P) → T0(P) | **66** | Same pitch; merge candidates |
| T0(P1) → T0(P2) | **206** | Different pitch; must stay distinct |

T0 diff-pitch step sizes: min ≈ 100 cents, p50 = 200 cents, max = 1100 cents (swara steps in the stratified ratio grid).

### 2.3 Non-T0 same-type + same-pitch (n = 487)

| Type | Count |
|------|------:|
| 6 (Simple Multiple) | 362 |
| 3 (Sloped End) | 64 |
| 2 (Sloped Start) | 29 |
| 1 (Simple) | 12 |
| 7 (Krintin) | 9 |
| 4 (Ladle) | 6 |
| 13 (Vibrato) | 5 |

These are raw annotation boundaries **without** a geometric pitch discontinuity.

### 2.4 Top different-type + same-pitch pairs

| From → To | Count |
|-----------|------:|
| 6 → 0 | 232 |
| 3 → 0 | 149 |
| 0 → 6 | 140 |
| 2 → 0 | 123 |
| 0 → 3 | 91 |

These are common musically (bend ending on a fixed swara at the same pitch) but are still **type** boundaries.

---

## 3. Type 0 canonicalization analysis

### 3.1 Rule under test

```
type == 0
AND pitch_key(prev_end) == pitch_key(next_start)
AND interval_relation == "meets"
→ MERGE into one canonical unit
```

Already implemented as an additive overlay in [`dataset/canonical/canonicalize.py`](../dataset/canonical/canonicalize.py) (`type0_v1`).

### 3.2 Corpus counts

| Metric | Value |
|--------|------:|
| Merge candidates (adjacent T0 same-pitch) | 66 |
| Boundaries removed | 66 |
| Fixed raw trajectories | 1,363 |
| Fixed canonical units | 1,297 |
| Merged units (runs of 2+) | 54 |
| Run-length histogram | 46×2, 5×3, 2×4, 1×5 |

### 3.3 Safety checks

| Check | Result |
|-------|--------|
| Pitch contour change after merge | **0** — concatenation of identical `Trajectory.compute` pieces |
| Combined duration p50 / max | 0.54 s / 7.86 s |
| Vowel differs across boundary | 12 / 66 |
| Consonant differs | 0 (field mostly null in corpus) |
| Articulation differs | 0 (field mostly null) |
| Slope differs | 0 |
| Crosses phrase boundary | 0 (`merge_across_phrase_boundary=True`) |

### 3.4 Verdict

**Defensible for pitch and shape targets.** The rule is deterministic and contour-preserving.

**Caveat:** 12/66 merges cross a vowel annotation change (e.g. `'a'` → `'ai'` on pitch S at `645ff354deeaf2d1e33b3c44:129|130`). That is a **symbolic** lyric boundary, not an acoustic one. Low risk for pitch/type supervision; ambiguous for lyric-aligned downstream tasks.

### 3.5 Representative merge candidates

| Recording | Indices | Pitch (sargam) | Durations (s) | Vowel L → R |
|-----------|---------|----------------|---------------|-------------|
| `645ff354deeaf2d1e33b3c44` | 129–130 | S | 0.55 + 1.47 | a → ai |
| `6491d48d608d1718e0311003` | 33–36 | d | 0.62 + 0.61 + 0.52 + 0.34 | a → a → ū → ō |
| `6417585554a0bfbd8de2d3ff` | 304–305 | N | 0.39 + 0.26 | a → a |
| `645ff354deeaf2d1e33b3c44` | 135–139 | P | five-part run | mixed vowels |
| `6503e348d9ff49d3988d0b3f` | 314–315 | S | 0.28 + 0.26 | null → null |

Full list of 20 examples: `step_4_5_analysis.json` → `t0_merge_analysis.examples`.

---

## 4. Simple vs composite trajectory equivalence

### 4.1 Composite types in corpus

| Type | Name | Count | `dur_array` structure |
|------|------|------:|----------------------|
| 4 | Bend: Ladle | 96 | always 2 segments |
| 5 | Bend: Reverse Ladle | 14 | always 2 segments |
| 6 | Bend: Simple Multiple | 1,237 | 1–33 segments (median 3–4) |
| 7 | Krintin | 47 | always 2 segments |

Types 8–11 do not occur in this corpus.

### 4.2 Internal boundary smoothness

Measured as |Δ log2 Hz| between `compute(x−ε)` and `compute(x+ε)` at each internal `dur_array` fraction:

| Type | Internal kink p50 (log2 Hz) | Internal kink max | Interpretation |
|------|----------------------------:|------------------:|----------------|
| 4 | 2.2×10⁻¹⁰ | 2.5×10⁻⁸ | C0 smooth — **not** a geometric boundary |
| 5 | 1.4×10⁻¹⁰ | 2.7×10⁻⁸ | C0 smooth |
| 6 | 1.0×10⁻⁹ | 1.9×10⁻⁶ | C0 smooth |
| 7 | **0.167** | **0.417** | **Meaningful pitch kink** at internal split |

### 4.3 Can composites be rewritten as simpler primitives?

**Types 4, 5, 6:** The single-object `Trajectory.compute` contour is identical to the export decomposition (same parametric object; labels change, geometry does not). An equivalent annotation could use separate type 1/2/3 trajectories **or** one composite object — the segmentation is **not unique**.

**Type 6 specifically:** 362 consecutive T6→T6 transitions at the same pitch show that annotators sometimes split one continuous simple-multiple run into multiple raw objects. Merging those would require a rule symmetric with T0 merging, but type 6 internal motion means "same pitch at boundary" does **not** imply "same trajectory."

**Type 7:** Internal splits are geometrically real but the type is excluded from the four-class vocabulary anyway.

### 4.4 `dur_array` as sub-boundaries

- For types 4, 5, 6: `dur_array` marks **parametric segment labels** used by IDTAP's composite types, not acoustic event boundaries.
- For type 7: internal splits coincide with geometric kinks but target class is undefined in the four-class scheme.

### 4.5 Do internal pieces map to the four target classes?

| Composite | Export mapping | Maps to {0,1,2,3}? |
|-----------|----------------|---------------------|
| 4 | `[2, 1]` | Yes |
| 5 | `[1, 3]` | Yes |
| 6 | N × `1` | Partially — all segments labeled Simple Bend |
| 7 | skipped | No |

---

## 5. Three notions of boundary

### 5.1 Definitions

| Notion | Definition |
|--------|------------|
| **Raw annotation boundary** | Two consecutive `Trajectory` objects in the same lane (`meets`, gap 0) |
| **Geometric boundary** | A point where pitch or shape meaningfully changes: type change with pitch step, T0(P1)→T0(P2), or internal kink (type 7) |
| **Canonical ML boundary** | A boundary the automatic transcription model should recover |

These three are **not equivalent**.

### 5.2 Agreement and disagreement

| Scenario | Raw | Geometric | Canonical ML |
|----------|:---:|:---------:|:--------------:|
| T0(S) → T0(S) same pitch | ✓ | ✗ | ✗ (merge) |
| T0(N) → T0(S) Δ100c | ✓ | ✓ | ✓ |
| T6(P) → T6(P) same pitch | ✓ | ✗ | **ambiguous** — keep raw |
| T3 → T0 same pitch | ✓ | ✗ | ✓ (type change) |
| Type 6 internal `dur_array` split | ✗ | ✗ | ✗ (unless decomposing) |
| Type 7 internal split | ✗ | ✓ | mask (out of vocabulary) |
| T0(S,a) → T0(S,ai) same pitch | ✓ | ✗ | ✗ (merge; vowel symbolic) |

---

## 6. Proposed canonicalization rules

Extend the existing overlay in [`dataset/canonical/canonicalize.py`](../dataset/canonical/canonicalize.py):

```python
def canonicalize(raw_trajectories) -> list[CanonicalUnit]:
    """
    Each unit contains:
      start_s, end_s, target_type, member_indices, member_traj_ids,
      rule_applied, merged (bool)
    Pitch contour: Trajectory.compute on each member, unchanged by all rules below.
    """
```

| Operation | Condition | target_type | Contour error |
|-----------|-----------|-------------|---------------|
| **KEEP** | types 1, 2, 3; T0 diff-pitch; ambiguous splits | raw type (or mapped) | 0 |
| **MERGE** | T0 same-pitch contiguous | 0 | 0 |
| **MAP_TYPE** | type 4 | `[2, 1]` sub-units via `dur_array` | 0 |
| **MAP_TYPE** | type 5 | `[1, 3]` sub-units | 0 |
| **KEEP_COMPOSITE** | type 6 | 1 (or retain `source_type_id=6`) | 0 |
| **MASK** | types 7, 13 (and 8–11 if present) | null (masked) | N/A |
| **MASK** | type 12 silent | null (masked for type/pitch loss) | N/A |

**Provenance:** every unit carries `member_indices` and `member_traj_ids`; raw trajectories are never deleted (already guaranteed by the additive overlay design).

### 6.1 Cases that cannot be canonicalized reliably

1. **T6→T6 same-pitch splits (362):** no merge rule without absorbing whole performance-span runs.
2. **Composite vs decomposed-simple equivalence:** multiple valid segmentations, identical contour.
3. **Vowel/consonant boundaries inside T0 same-pitch merges:** symbolic, not acoustic.
4. **Type 6 as one unit vs N simple bends:** both valid; choose one convention and document it.

### 6.2 Projected unit counts

| Stage | Trajectory units |
|-------|-----------------:|
| Raw | 5,209 |
| After T0 merge (current overlay) | 5,143 |
| After MAP_TYPE 4/5 split | ~5,143 + 96 + 14 = ~5,253 sub-units (if splitting composites into labeled segments) |
| If type 6 kept as single units | no change from T0-merge count for type 6 |

Recommendation: apply T0 merge + MAP_TYPE for 4/5 + KEEP_COMPOSITE for 6 + MASK for 7/12/13.

---

## 7. Boundary prediction head recommendation

**Revise the Step 4 proposal.** Compare three target strategies:

| Option | Definition | Verdict |
|--------|------------|---------|
| A. Raw onsets | Every raw trajectory start (5,209) | **Too noisy** — 66 spurious T0 + 362 T6 same-pitch splits |
| B. Canonical onsets | After T0 merge (~5,143) + type mapping | Better, still ambiguous for composites and T6 splits |
| C. **No explicit boundary head** | Derive from type and pitch changes | **Recommended initially** |

### Why option C is sufficient

- T0(C)→T0(D) produces a ≥100 cent pitch step detectable from `pitch_t` alone.
- Type changes (e.g. Fixed→Simple Bend) are directly supervised via `target_type_t`.
- An explicit boundary head largely duplicates information already in type + pitch.
- Avoids premature commitment to raw vs canonical boundary semantics.

**Add `boundary_onset` later** only if experiments show type + pitch cannot recover segmentation (e.g. boundaries we deliberately keep: T6→T6 same pitch).

---

## 8. Dense pitch target

**Accurate description:**

> Dense pitch reconstructed from the human IDTAP parametric annotation via `Trajectory.compute(x, log_scale=True)` — **not** independently measured acoustic F0 ground truth.

This target is unchanged.

### Canonicalization impact

| Operation | Effect on pitch contour |
|-----------|------------------------|
| T0 merge | **Zero change** |
| MAP_TYPE 4/5 | **Zero change** (same parametric object) |
| KEEP_COMPOSITE 6 | **Zero change** |
| MASK 7/12/13 | Frames excluded from pitch loss via `mask` |

Quantified: [`dataset/canonical/verify_roundtrip.py`](../dataset/canonical/verify_roundtrip.py) reports 0.0 log2 Hz max error across all 5,209 raw trajectories when rebuilding from the stored `raw` block.

---

## 9. Before/after examples (13 cases)

Each example is in `output/canonical/v1/step_4_5_analysis.json` → `examples[]`.

### Example 1 — T0 same-pitch merge (2 members)

**Recording:** `645ff354deeaf2d1e33b3c44`, indices 129–130

```
RAW:       T0(S) 0.55s | T0(S) 1.47s
CANONICAL: T0(S) ───────────── 2.02s
```

**Why:** identical pitch_key `0|0|R|+0.000000`; contour-preserving. **Ambiguous:** vowel a → ai.

### Example 2 — T0 same-pitch merge (4 members)

**Recording:** `6491d48d608d1718e0311003`, indices 33–36

```
RAW:       T0(d) | T0(d) | T0(d) | T0(d)
CANONICAL: T0(d) ─────────────────── 2.09s
```

**Why:** single sustained swara d; longest merged run in corpus.

### Example 3 — T0 different pitch kept

**Recording:** `6417585554a0bfbd8de2d3ff`, indices 396–397

```
RAW:       T0(N) 0.31s | T0(S) 0.28s
CANONICAL: T0(N) | T0(S)   (unchanged)
```

**Why:** Δpitch ≈ 100 cents (N→S); geometric boundary.

### Example 4 — T0 large pitch jump

**Recording:** see `t0_diff_pitch_large` in analysis JSON (Δpitch = 1100 cents)

```
RAW:       T0(P1) | T0(P2)
CANONICAL: T0(P1) | T0(P2)
```

**Why:** large swara step; must remain two trajectories.

### Example 5 — T3 → T0 same pitch

**Recording:** see `t3_to_t0_same_pitch` in analysis JSON

```
RAW:       T3(P) | T0(P)
CANONICAL: T3(P) | T0(P)
```

**Why:** type change is a canonical ML boundary even without pitch step.

### Example 6 — T6 → T6 same pitch (ambiguous)

**Recording:** see `t6_to_t6_same_pitch` in analysis JSON

```
RAW:       T6(P) | T6(P)
CANONICAL: T6(P) | T6(P)   [AMBIGUOUS — do not auto-merge]
```

**Why:** annotator representation choice; 362 such pairs in corpus.

### Example 7 — Type 4 Ladle

**Recording:** `6417585554a0bfbd8de2d3ff`, index 1402

```
RAW:       T4 (Ladle) 0.51s
CANONICAL: T2 segment | T1 segment  (via dur_array fractions)
```

**Why:** export mapping `[2, 1]`; internal boundary is smooth (not geometric).

### Example 8 — Type 5 Reverse Ladle

```
RAW:       T5
CANONICAL: T1 segment | T3 segment
```

**Why:** export mapping `[1, 3]`.

### Example 9 — Type 6 Simple Multiple

```
RAW:       T6 (N internal segments)
CANONICAL: T6 as one unit, target_type=1  [AMBIGUOUS decomposition]
```

**Why:** decomposition to N×T1 is possible but not unique; internal splits are not geometric.

### Example 10 — Type 7 Krintin

```
RAW:       T7
CANONICAL: [MASKED — excluded from four-class target]
```

**Why:** SKIP_IDTAP_IDS; internal kink is geometric but class undefined.

### Example 11 — Type 13 Vibrato

```
RAW:       T13
CANONICAL: [MASKED]
```

**Why:** outside shape vocabulary.

### Example 12 — Type 12 Silent

```
RAW:       T12 (Silent)
CANONICAL: [MASKED for type/pitch loss]
```

**Why:** silent is an explicit annotation, not acoustic inactivity; do not label as "inactive."

### Example 13 — T0 merge with vowel change

**Recording:** `645ff354deeaf2d1e33b3c44`, indices 129–130

```
RAW:       T0(S,a) | T0(S,ai)
CANONICAL: T0(S) ────────  [AMBIGUOUS for lyric tasks]
```

**Why:** merge defensible for pitch; vowel boundary is symbolic.

---

## 10. Proposed final framewise target bundle

After canonicalization, at `hop_s = 0.01`, per lane:

```python
mask: bool                 # True: silent (12), excluded (7,13), unannotated
target_type: int | null    # 0, 1, 2, 3 when unmasked
pitch_log2_hz: float       # Trajectory.compute at frame center
# NO boundary_onset initially
```

| Field | Supervised? | Notes |
|-------|:-----------:|-------|
| `mask` | no (metadata) | Gates all losses |
| `target_type` | yes | Four shape classes after MAP_TYPE |
| `pitch_log2_hz` | yes | Parametric annotation contour |
| `boundary_onset` | **skip initially** | Derive from type + pitch if needed |

**Vocabulary:** five effective classes `{0,1,2,3,masked}`. For legacy CNN parity: `--exclude-labels silent` → four shape classes.

**Membership:** half-open `[start_s, end_s)` per Step 4.

---

## 11. Transformation pipeline

```text
RAW IDTAP (5,209 trajectories)
    ↓
[Deterministic canonicalization rules]
    • MERGE: T0 same-pitch contiguous (66 boundaries removed → 5,143 units)
    • MAP_TYPE: 4→[2,1], 5→[1,3] via dur_array
    • KEEP_COMPOSITE: 6 as single unit, target_type=1
    • MASK: 7, 13, 12 (silent)
    • KEEP: types 1,2,3; T0 diff-pitch; ambiguous T6 splits
    ↓
CANONICAL TRAJECTORIES
    ↓
10 ms targets:
    target_type (0–3 + mask)
    pitch_log2_hz (parametric annotation)
    [boundary derived from type/pitch — no explicit head initially]
    mask
```

**Step 5 (`dataset/canonical/frames.py`) is blocked** until this representation is reviewed.

---

## Summary recommendations

1. **Resolve vocabulary:** treat the ML target as four shape classes `{0,1,2,3}` plus mask; silent is masked, not a sixth competing shape class.
2. **Apply T0 same-pitch merge** — already implemented; defensible and contour-preserving.
3. **MAP_TYPE for 4 and 5** at canonicalization time; **KEEP_COMPOSITE for 6** unless a corpus-wide T6 merge policy is adopted.
4. **MASK types 7 and 13** (and 8–11 if they appear in expanded corpus).
5. **Skip explicit boundary head** initially; supervise type + pitch; derive segmentation post hoc.
6. **Do not treat raw annotation boundaries as ground truth** for segmentation — 619 same-type/same-pitch raw boundaries (66 T0 + 362 T6 + others) are largely annotator artifacts.

---

## References

| Artifact | Path |
|----------|------|
| Analysis script | [`dataset/canonical/analyze_step_4_5.py`](../dataset/canonical/analyze_step_4_5.py) |
| Analysis output | [`output/canonical/v1/step_4_5_analysis.json`](../output/canonical/v1/step_4_5_analysis.json) |
| Canonical overlay | [`dataset/canonical/canonicalize.py`](../dataset/canonical/canonicalize.py) |
| Transition logic | [`dataset/canonical/transitions.py`](../dataset/canonical/transitions.py) |
| Export mapping | [`dataset/export_denoised_cnn_dataset.py`](../dataset/export_denoised_cnn_dataset.py) |
| Findings notebook | [`notebooks/canonical_dataset_findings.ipynb`](../notebooks/canonical_dataset_findings.ipynb) |
