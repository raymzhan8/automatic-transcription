# Step 5.5 — Boundary Learnability and Class-Balance Audit

Diagnostic-only analysis of whether canonical primitive boundaries carry observable pitch-geometric evidence. **No model was trained.** Canonicalization rules and targets were not changed.

Reproduce from the repository root:

```bash
python dataset/canonical/analyze_step_5_5.py
python dataset/canonical/visualize_boundaries.py
python -m pytest dataset/canonical/test_targets.py -q
```

Full machine-readable output: [`output/canonical/v1/step_5_5_analysis.json`](../output/canonical/v1/step_5_5_analysis.json)

---

## 1. Corrected duration statistics

Recomputed from `primitives/*.json` with thresholds [10, 20, 50, 100] ms and a monotonicity assertion.

### All primitives (n = 7,177)

| Stat | Value |
|------|------:|
| min | 9.2 ms |
| p01 | 35.1 ms |
| p05 | 51.9 ms |
| median | 132.1 ms |
| mean | 236.5 ms |
| p95 | 726.3 ms |
| max | 7.86 s |
| % &lt; 10 ms | **0.03%** (2 primitives) |
| % &lt; 20 ms | 0.28% |
| % &lt; 50 ms | 2.9% |
| % &lt; 100 ms | 31.8% |

### By canonical type (% shorter than threshold)

| Type | n | % &lt; 10 ms | % &lt; 20 ms | % &lt; 50 ms | % &lt; 100 ms | median |
|------|--:|---:|---:|---:|---:|---:|
| T0 | 1,306 | 0.0% | 0.0% | 0.15% | 7.7% | 235 ms |
| T1 | 4,950 | 0.04% | 0.40% | 4.0% | 42.0% | 109 ms |
| T2 | 468 | 0.0% | 0.0% | 0.85% | 12.6% | 226 ms |
| T3 | 453 | 0.0% | 0.0% | 0.88% | 10.4% | 235 ms |

### T1 from Type 6 decomposition (n = 4,654)

| Stat | Value |
|------|------:|
| min | 9.2 ms |
| median | 109.1 ms |
| mean | 168.3 ms |
| % &lt; 10 ms | 0.04% |
| % &lt; 20 ms | 0.41% |
| % &lt; 50 ms | 4.1% |
| % &lt; 100 ms | 42.4% |

---

## 2. Step 5 error diagnosis

**Reporting-only bug.** The Step 5 report misread JSON values:

- Fractions like `0.028` were written as “2.8%” instead of **0.03%**.
- The removed **30 ms** bucket (`0.473`) was labeled “&lt; 50 ms”.
- Cumulative percentages in the underlying data were always monotonic once thresholds are read correctly.

**No structural issue** in primitive generation or duration fields. The automated monotonicity assertion in `duration_stats()` now prevents future misreporting.

---

## 3. Primitive-level class balance

| Type | Count | % of primitives |
|------|------:|----------------:|
| T0 | 1,306 | 18.2% |
| T1 | 4,950 | **69.0%** |
| T2 | 468 | 6.5% |
| T3 | 453 | 6.3% |

Type 6 decomposition inflates T1 at the primitive level: 94.0% of T1 primitives originate from raw Type 6.

---

## 4. Frame-level class balance (valid_target == True)

| Type | Frames | % of valid frames |
|------|-------:|------------------:|
| T0 | 58,755 | 34.6% |
| T1 | 84,854 | **50.0%** |
| T2 | 13,746 | 8.1% |
| T3 | 12,372 | 7.3% |
| **Total** | **169,727** | |

Frame-level balance is less skewed than primitive counts because T6-derived T1 segments tend to be shorter (fewer frames each) while T0 segments are longer.

---

## 5. T1 breakdown by raw source

### Primitives

| Origin | Count | % of T1 |
|--------|------:|--------:|
| raw T1 | 186 | 3.8% |
| Type 4 decomposition | 96 | 1.9% |
| Type 5 decomposition | 14 | 0.3% |
| **Type 6 decomposition** | **4,654** | **94.0%** |

### Valid frames

| Origin | Frames | % of T1 frames |
|--------|-------:|-----------------:|
| raw T1 | 3,426 | 4.0% |
| Type 4 decomposition | 2,833 | 3.3% |
| Type 5 decomposition | 255 | 0.3% |
| **Type 6 decomposition** | **78,340** | **92.3%** |

---

## 6. Canonical boundary inventory

| Metric | Value |
|--------|------:|
| Total boundaries | 7,159 |
| Same-type | 4,765 |
| Different-type | 2,394 |
| Introduced by decomposition | 3,527 |
| Raw IDTAP preserved | 3,632 |
| T1 → T1 | 4,123 |

**Provenance distinction:**

- **Decomposition-introduced (B):** internal splits within a composite raw trajectory (Types 4/5/6), identified by shared `source_raw_indices` and decomposition rules.
- **Raw preserved (A):** boundaries between distinct raw trajectory objects, including T0 merges, raw T1|T1, and cross-type transitions.

Top type pairs: T1|T1 (4,123), T0|T1 (475), T1|T0 (466), T0|T0 (511).

> **Note:** Step 5 reported 4,054 same-type boundaries because `validate_targets.same_type_boundary_analysis` grouped primitives by `lane_id` across recordings (e.g. all `"0:0"` lanes merged). Per-recording analysis gives the correct count of 4,765.

---

## 7. Same-type boundary analysis

Of 4,765 same-type boundaries, 4,123 are T1 → T1.

### Pitch step (endpoint metadata)

| Threshold | % with \|Δp\| &gt; threshold |
|-----------|-------------------------------:|
| 1 cent | 11.7% |
| 5 cents | 11.3% |
| 10 cents | 11.2% |
| 20 cents | 11.2% |

Median endpoint pitch step is **0 cents**; a long tail of T0|T0 and cross-raw boundaries drives the mean up.

### Dynamics change (1 ms parametric contour, ±250 ms window)

| Threshold | % with \|Δv\| &gt; threshold | % with \|Δa\| &gt; threshold |
|-----------|-------------------------------:|-------------------------------:|
| 50 / 500 | 93.9% | 91.1% |
| 100 / 1,000 | 93.6% | 89.8% |
| 200 / 2,000 | 92.8% | 87.8% |
| 500 / 5,000 | 89.2% | 83.3% |

High velocity/acceleration hit rates partly reflect **numerical differentiation sensitivity** on smooth curves, not necessarily perceptual discontinuities. Interpret alongside kink analysis (§8).

### Evidence tiers (same-type, multi-threshold)

| Tier | Description | Approx. coverage |
|------|-------------|-----------------:|
| B — pitch step | \|Δp\| &gt; 1 cent | 11.7% |
| C — dynamics | \|Δv\| &gt; 50 cents/s (same-type) | 84.6% |
| D — no obvious cue (strict) | same type, \|Δp\| ≤ 1c, \|Δv\| ≤ 50, \|Δa\| ≤ 500 | **4.4%** |

---

## 8. Type 6 internal boundary analysis (priority)

| Metric | Value |
|--------|------:|
| Internal T6 T1 → T1 boundaries | 3,417 |
| At durArray control-point transition | **100%** |
| Median internal kink (log2 Hz) | ~1×10⁻⁹ |

### Geometric cues at T6 internal boundaries

| Signal | % exceeding threshold |
|--------|----------------------:|
| Pitch step &gt; 1 cent | **0.0%** |
| \|Δv\| &gt; 50 cents/s | 99.4% |
| \|Δv\| &gt; 200 cents/s | 98.3% |
| \|Δa\| &gt; 500 cents/s² | 96.1% |
| No obvious cue (strict) | **0.4%** |

**Interpretation:** Type 6 decomposition boundaries align exactly with `durArray` control-point transitions. Parametric pitch is **C⁰-smooth** there (median kink ≈ 0). Reported velocity changes are slopes of a continuous contour evaluated on either side of a symbolic split — they reflect **local dynamics differences**, not pitch jumps.

**Representative examples** (see `figures/step5_5/`):

- **Obvious dynamics:** large \|Δv\| but zero pitch step — e.g. `6912841f213d07041b95a800` @ 1229.9 s.
- **No obvious cue:** `6417585554a0bfbd8de2d3ff` @ 145.8 s — smooth contour, near-zero kink, low \|Δv\| and \|Δa\|.

---

## 9. Raw T1 → T1 vs T6-introduced T1 → T1

| Population | n boundaries | Median \|Δp\| | Median \|Δv\| | Adjacent primitive median duration |
|------------|-------------:|--------------:|--------------:|-----------------------------------:|
| Raw T1 \| T1 | 17 | 0 cents | 2,077 cents/s | 109 ms |
| T6 internal T1 \| T1 | 3,417 | 0 cents | 2,151 cents/s | 106 ms |

KS tests do not reject equal distributions for pitch step, Δv, Δa, or duration (p &gt; 0.15). **Geometrically, the two populations look similar** — but T6 boundaries are guaranteed to be control-point splits within one parametric object, while raw T1|T1 are separate annotation objects (potentially different musical intent).

Only **17** raw T1 → T1 boundaries exist in the corpus — too few for strong statistical conclusions.

---

## 10. Pitch / velocity / curvature methodology

Documented in [`boundary_geometry.py`](../dataset/canonical/boundary_geometry.py):

| Parameter | Value |
|-----------|-------|
| Sample grid | 1 ms |
| Local window | ±250 ms |
| Pitch unit | cents = 1200 × log2_hz |
| Velocity | Central difference, ±10 ms → cents/s |
| Acceleration | Second central difference, ±10 ms → cents/s² |
| v/a aggregation | Mean over [−50, −20] ms and [+20, +50] ms around boundary |

Single-sample adjacent differences are **not** used.

---

## 11. Geometry vs acoustic learnability

This audit uses the **ground-truth parametric pitch contour** from IDTAP shape parameters. A boundary that is geometrically identifiable (e.g. large \|Δv\| on a smooth curve) does **not** imply it is easy to infer from audio.

Recordings contain drone, accompaniment, source interference, and timbral variation. **Acoustic learnability will be tested only when a model is trained.** This step answers whether the target representation itself encodes boundary evidence — a necessary but not sufficient condition.

---

## 12. Oracle: recovering boundaries from type + pitch alone

Cumulative recoverability on ground-truth boundaries (no phase):

### All boundaries (n = 7,159)

| Tier | Cumulative % recoverable |
|------|-------------------------:|
| Type change alone | 33.4% |
| + pitch step &gt; 1 cent | 41.2% |
| + pitch step &gt; 10 cents | 40.9% |
| + \|Δv\| &gt; 50 cents/s | 89.7% |
| + \|Δv\| &gt; 200 cents/s | 89.1% |
| No obvious cue (strict) | 2.9% |

### Same-type only (n = 4,765)

| Tier | Cumulative % |
|------|-------------:|
| Pitch step &gt; 1 cent | 11.7% |
| \|Δv\| &gt; 50 cents/s | 84.6% |
| No obvious cue (strict) | **4.4%** |

### T6 internal only (n = 3,417)

| Tier | Cumulative % |
|------|-------------:|
| Pitch step &gt; 1 cent | 0.0% |
| \|Δv\| &gt; 50 cents/s | 99.3% |
| No obvious cue (strict) | **0.4%** |

**Phase potentially adds information** for the ~4% of same-type boundaries without strict type/pitch/dynamics cues, and more broadly for enforcing consistent within-primitive progress where dynamics cues are ambiguous or numerically unstable.

---

## 13. Phase target analysis

Correlation of normalized phase with pitch displacement on valid frames:

| Type | n frames | Spearman(phase, disp from start) | R² |
|------|--------:|----------------------------------:|---:|
| T0 | 58,755 | 0.0002 | ~0 |
| **T1** | **84,854** | **0.0018** | **0.0006** |
| T2 | 13,746 | −0.21 | 0.007 |
| T3 | 12,372 | 0.22 | 0.061 |

For **T1** (dominated by Type 6 decomposition), phase is **essentially uncorrelated** with pitch displacement or progress along the contour. Phase is a **linear clock** over primitive duration, not a proxy for geometric progress on bends.

T2/T3 show modest correlation — expected for sloped trajectories where parametric time aligns with pitch motion.

---

## 14. 10 ms frame resolution

### Frames per primitive (half-open [start, end) on 10 ms centers)

| Type | 0 frames | 1 frame | 2 frames | 3+ frames |
|------|--------:|--------:|---------:|----------:|
| T0 | 0 | 0 | 0 | 100% |
| T1 | 0 | 0.18% | 0.30% | 99.5% |
| T2 | 0 | 0 | 0 | 100% |
| T3 | 0 | 0 | 0 | 100% |

### T6-origin T1 specifically

| Bucket | Count | % |
|--------|------:|--:|
| 1 frame | 8 | 0.17% |
| 2 frames | 15 | 0.32% |
| 3+ frames | 4,631 | 99.5% |

**0.49%** of T6-origin T1 primitives have only 1–2 frames (phase ramp poorly resolved). Median duration for 1–2 frame buckets ≈ 17 ms.

The corrected duration stats show sub-10 ms primitives are **extremely rare** (2 total), so frame starvation is not a corpus-wide problem — but the few 1–2 frame T1 segments are almost all from Type 6.

---

## 15. Diagnostic visualizations

Generated under [`output/canonical/v1/figures/step5_5/`](../output/canonical/v1/figures/step5_5/):

| Category | Files |
|----------|-------|
| T0 \| T0 different pitch | 2 |
| Raw T1 \| T1 | 2 |
| T6 T1 \| T1 obvious | 2 |
| T6 T1 \| T1 subtle | 2 |
| T6 T1 \| T1 no cue | 2 |
| T1 \| T2 | 2 |
| T2 \| T1 (Type 4) | 2 |
| T1 \| T3 (Type 5) | 2 |

Each plot: dense pitch (cents), dp/dt, d²p/dt², canonical type band, phase, canonical boundary (solid red), raw boundary if different (dashed gray).

---

## 16. Step 6 recommendation

### Recommended: **Option B — type + pitch primary; phase auxiliary**

| Option | Verdict | Rationale |
|--------|---------|-----------|
| A — phase primary | **No** | T1 phase does not track geometric progress (R² ≈ 0.0006); 92% of T1 frames are T6 splits where phase resets are symbolic |
| **B — phase auxiliary** | **Yes** | Type change resolves 33% of all boundaries; pitch step helps another ~8%; dynamics cues are widespread but numerically noisy on smooth T6 contours; phase still provides dense segmentation supervision where type+pitch alone underdetermine boundaries (~4% strict same-type remainder) |
| C — remove phase | Partially attractive | Would simplify the head, but sacrifices explicit within-primitive progress signal; risky for T2/T3 where phase aligns somewhat with geometry |
| D — revise canonicalization | **Not yet** | T6 decomposition creates convention-dependent boundaries, but they coincide with control-point structure; evidence does not yet justify changing rules before acoustic experiments |

### Practical Step 6 guidance

1. **Primary heads:** trajectory type (4-class) + dense pitch (log2_hz), masked by `valid_target`.
2. **Auxiliary head:** phase with **lower loss weight**, especially on T1 / T6-origin frames.
3. **Do not add** boundary, dp/dt, or curvature heads in v1.
4. **Monitor** same-type boundary errors on T6 segments after training — if phase fails to learn, revisit Option C or Type 6 KEEP_COMPOSITE (Step 4.5 alternative).
5. **Class imbalance:** T1 dominates primitives (69%) but not frames (50%); consider per-type loss weighting in Step 6 training design (not applied in this audit).

### If canonicalization were reconsidered later

The rule most implicated is **Type 6 → T1 × N via durArray** (Step 5 override of Step 4.5 KEEP_COMPOSITE). It creates 3,417 same-type internal boundaries with zero pitch step and near-zero parametric kink — boundaries that exist primarily to match export decomposition semantics rather than contour discontinuities.

---

## Implementation

| Module | Purpose |
|--------|---------|
| [`boundary_geometry.py`](../dataset/canonical/boundary_geometry.py) | Boundary enumeration + dense pitch dynamics |
| [`analyze_step_5_5.py`](../dataset/canonical/analyze_step_5_5.py) | Full audit CLI |
| [`visualize_boundaries.py`](../dataset/canonical/visualize_boundaries.py) | Boundary-centered figures |
| [`validate_targets.py`](../dataset/canonical/validate_targets.py) | Fixed `duration_stats()` + monotonicity assertion |
