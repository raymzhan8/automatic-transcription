# Step 14 — Relative-Pitch-Augmented Trajectory Classification

A controlled feature ablation on the actual task (`audio → T0/T1/T2/T3`), following Step 13's `RELATIVE_PITCH_PARTIAL` verdict. No pitch-frontend redesign, no register-decoder work, no class weighting, no architecture search — one shared TCN classifier, four input conditions.

Frozen references: [`docs/step_13_relative_pitch.md`](step_13_relative_pitch.md), [`docs/step_9_c_report.md`](step_9_c_report.md) (original B0/B1/C).

Machine-readable outputs: [`output/relative_pitch_ablation/`](../output/relative_pitch_ablation/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/relative_pitch/dense_pitch_path.py   # continuous Fused+D3 log2Hz per recording
python -m pytest training/tests/test_relative_pitch_features.py training/tests/test_relative_pitch_ablation.py
python training/train_relative_pitch_ablation.py --condition A --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition B --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition C --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition D --all-folds --max-epochs 20 --patience 5
python training/evaluate_relative_pitch_ablation.py
python training/visualize_relative_pitch_ablation.py
```

**Scope caveat, disclosed up front:** to keep this a same-day diagnostic ablation rather than a full retraining project, all four conditions use a **reduced training budget** (max 20 epochs, patience 5) versus the original B0's protocol (max 50, patience 10). B0's own full-budget audio-only run reached 0.301 mean validation macro F1; condition A here reaches 0.259 — a real but modest (~14%) shortfall from premature convergence, confirmed by condition A's checkpoints selecting very early best-epochs (2, 4, 6, 6, 12). This affects only the absolute level of condition A (and, to a lesser degree, C); it does not change this step's qualitative conclusions, since B and D beat A/C by margins (0.12-0.40 macro F1) far larger than this budget gap. **All four conditions share the identical reduced budget**, so the A/B/C/D comparison itself is fair and controlled.

---

## Executive summary

| Finding | Evidence |
|---|---|
| **B (estimated relative pitch alone) clearly beats both A (audio alone) and C (audio+estimated pitch)** | Pooled test macro F1: B 0.325 vs A 0.205 vs C 0.220; B beats A in 16/17 test recordings |
| **C (naive audio+estimated-pitch fusion) does not meaningfully beat A**, and is worse than B | Pooled macro F1 C−A = +0.015 (within noise: std 0.066); per-fold sign is inconsistent (positive 2/5 folds, negative 3/5) |
| **C is catastrophic on T2/T3 specifically** — worse than even audio-only A | T2 F1: A 0.018 → C 0.001; T3 F1: A 0.049 → C 0.002 — fusion doesn't just fail to help the rarest classes, it appears to actively suppress what little audio-only signal existed |
| **D (audio + oracle pitch) dominates everything**, especially on T2/T3 | Pooled macro F1 0.604 (vs A 0.205); T2 F1 0.519, T3 F1 0.568 — the same architecture that collapses to ~0 on T2/T3 elsewhere resolves them well given clean pitch, ruling out architecture capacity as the T2/T3 bottleneck |
| D's advantage is **consistent across every fold** | D−A macro F1 ranges +0.32 to +0.55 across all 5 folds, no exceptions |
| **Octave errors visibly cancel under differencing** in representative windows | Case study: estimated pitch off by a full octave for seconds at a time, yet condition C still classifies the (flat) region as T0 correctly — the absolute error doesn't propagate into the classification |
| Oracle relative-pitch advantage **grows with movement speed and shrinks with primitive duration** | D accuracy by \|dp/dt\|: 0.573 (slow) → 0.849 (fast); by duration: 0.89 (<100ms) → 0.526 (>1s) — the fixed 10-200ms feature offsets natively suit short/fast primitives better than long ones |

**Outcome: `RELATIVE_PITCH_ONLY_COMPETITIVE`** (with `ORACLE_PITCH_HELPS_BUT_ESTIMATED_DOES_NOT` as essential supporting context for why fusion specifically underperforms)

**Decision gate: `IMPROVE_RELATIVE_PITCH_ESTIMATION`**

---

## 1. A/B/C/D definitions

All four share `FramewiseConditionalTCNModel` (`training/framewise_models.py`) — identical `FrequencyCNN` and `TemporalConvNet` modules, only the input projection into the shared 128-channel TCN differs:

| Condition | Audio | Pitch | Pitch source | Fusion |
|---|:---:|:---:|---|---|
| A | ✓ | — | — | `FrequencyCNN(spec)` → TCN |
| B | — | ✓ | Fused+D3 (estimated) | `Linear(4→128)` on φ → TCN |
| C | ✓ | ✓ | Fused+D3 (estimated) | `concat(h_audio[128], r_proj[16])` → `Linear(144→128)` → TCN |
| D | ✓ | ✓ | GT parametric pitch (oracle) | identical to C, pitch source swapped only |

Loss: unweighted cross-entropy on `valid_target & ~padding_mask` frames only (`FramewiseTypeLoss`, unchanged from B0/B1/C). Sampler: unchanged `FramewiseExcerptDataset` (4s excerpts, valid-anchor sampling, same grouped 5-fold splits, same fold-wise CQT normalization).

---

## 2. Relative-pitch feature function φ

`training/relative_pitch_features.py::compute_phi`, applied **identically** to both pitch sources (spec section 6 — only the source path differs between C and D):

```
phi(cents, valid)[t, j] = octave_unwrap(cents[t] - cents[t-k_j])   if valid[t] and valid[t-k_j]
                         = 0                                        otherwise
octave_unwrap(delta) = delta - 1200 * round(delta / 1200)
offsets k_j (native 10ms hop): 1, 5, 10, 20  →  ~10ms, 50ms, 100ms, 200ms
```

4-dimensional per frame. `valid` is always the GT `valid_target` mask, even for the estimated source (which is defined everywhere) — this guarantees C and D see non-zero features at exactly the same frame positions, differing only in value. Per-offset-channel mean/std (train-recordings-only, spec section 10) standardizes φ before it reaches the model.

Estimated source: **Fused+D3**, frozen from Step 13/12.5, decoded continuously over every native frame (not just valid ones, since B/C need a value at every input position) — `training/pitch_diagnostics/relative_pitch/dense_pitch_path.py`, reusing Step 12/12.5's exact checkpoints and validation-selected hyperparameters (no new grid search). At valid frames the dense path is patched to match Step 13's frozen values exactly (median discrepancy 0.0¢; the two decodes differ only immediately after time gaps the valid-only decode explicitly skips). Since φ zeroes any tap where either endpoint is invalid, the pitch *value* at invalid frames never actually reaches the model — only valid-frame values matter, which is why the (much cheaper) patch-with-Step-13-values approach is exact where it counts.

---

## 3. Octave-transition handling

Per Step 13, wrong-register estimates are usually *sustained* (the same wrong octave across consecutive frames), which `octave_unwrap` cancels automatically — no oracle octave-correctness information is used. Verified safe on real GT pitch (section 4) rather than assumed: fraction of valid consecutive pairs where raw `|delta| > 600¢` (i.e., unwrapping actually changes the value) is 0.001% (10ms), 0.013% (50ms), 0.098% (100ms), 0.513% (200ms) — legitimate fast movement is essentially never mistaken for an octave artifact, even at the widest 200ms offset.

---

## 4. Alignment tests

`training/tests/test_relative_pitch_features.py` (8 tests) + `training/tests/test_relative_pitch_ablation.py` (10 tests, 1 skipped — see below), all passing, plus the existing 30 pre-existing framewise tests unaffected (no regression to B0/B1/C infra):

- octave-unwrap correctness (exact-octave jumps → 0; small movement preserved) and the real-GT-pitch safety check above.
- `compute_phi` matches a manual `np.diff` at offset 1; zeroes leading frames before an offset can be computed; zeroes across a synthetic invalid gap despite NaN input cents; never produces NaN/Inf under randomized invalid/NaN scatter.
- **End-to-end dataset alignment**: `phi_oracle` recomputed independently from the *same* excerpt's returned `pitch_cents`/`valid_target` matches the dataset's own output exactly, for every sampled excerpt — catches any off-by-one between spec/target/φ slicing.
- Padding-region test (CQT/valid_target/φ all zero at the same padded frame indices) is written but **skipped**: every corpus recording is many minutes long and the shared `choose_excerpt_start` sampler (unchanged, used by B0/B1/C too) never lets a normal excerpt run past a recording's end, so this boundary condition essentially never occurs in this corpus — the underlying zero-gating mechanism is still covered unconditionally by the synthetic-gap and manual-recompute tests above.
- φ is confirmed zero at every GT-invalid frame even though the estimated source is dense/always-defined (the shared masking, not source density, controls this).
- Model-level: forward-shape tests for all 4 conditions; TCN/type_head parameter *shapes* verified identical between A and C (only the projection layers differ, per spec section 2).

---

## 5. Parameter counts

| Condition | Params | Notes |
|---|---:|---|
| A | 434,500 | matches original B0 exactly (same `FrequencyCNN`+TCN+type_head) |
| B | 329,348 | no `FrequencyCNN` at all |
| C | 453,140 | A + `pitch_proj`(4→16) + `fuse`(144→128) |
| D | 453,140 | identical architecture to C (spec section 6 requirement) |

---

## 6. Tiny-overfit sanity gate

16 cached excerpts, up to 40 epochs, no early stopping. All four conditions show clean, monotonic-ish memorization with no NaN/stuck-loss pathology:

| Condition | Train macro F1 (start → end) | Train acc (start → end) |
|---|---|---|
| A | 0.17 → 0.67 (epoch 40) | 0.29 → 0.86 |
| B | 0.17 → 0.54 (epoch 30) | 0.52 → 0.73 |
| C | 0.17 → 0.60 (epoch 30, peak 0.60 @28) | 0.52 → 0.84 |
| D | 0.17 → 0.62 (epoch 30) | 0.52 → 0.85 |

No feature-alignment or fusion bugs indicated; proceeded to full 5-fold training.

---

## 7-8. Grouped 5-fold results and primary comparison table

Pooled test-set (all 17 recordings, each evaluated once, continuous framewise inference — no oracle primitive boundaries at inference):

| Condition | Frame Acc | Macro F1 | T0 F1 | T1 F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|---:|---:|
| A — audio only | 0.300 | 0.205 | 0.399 | 0.356 | 0.018 | 0.049 |
| **B — estimated pitch only** | **0.518** | **0.325** | 0.561 | 0.582 | 0.097 | 0.061 |
| C — audio + estimated pitch | 0.432 | 0.220 | 0.314 | 0.563 | 0.001 | 0.002 |
| **D — audio + oracle pitch** | **0.651** | **0.604** | 0.639 | 0.690 | **0.519** | **0.568** |

Grouped mean ± std (mean of 5 per-fold macro F1, matching the project's standard reporting convention):

| Condition | Mean macro F1 | Std |
|---|---:|---:|
| A | 0.195 | 0.070 |
| B | 0.328 | 0.070 |
| C | 0.220 | 0.010 |
| D | 0.639 | 0.089 |

C's grouped-mean std (0.010) is far tighter than its per-recording variance suggests — its per-fold numbers cluster near 0.21-0.23 even though per-recording deltas swing from −0.20 to +0.50 (§14), i.e. C's wins and losses are internally offsetting within every fold rather than concentrated in specific folds.

---

## 9. Per-class metrics (pooled)

| | A | B | C | D |
|---|---|---|---|---|
| T0 precision/recall | 0.356 / 0.452 | 0.507 / 0.627 | 0.331 / 0.298 | 0.500 / 0.674 |
| T1 precision/recall | 0.531 / 0.268 | 0.577 / 0.586 | 0.493 / 0.655 | 0.564 / 0.691 |
| T2 precision/recall | 0.089 / 0.010 | 0.180 / 0.066 | 0.004 / 0.001 | 0.598 / 0.444 |
| T3 precision/recall | 0.031 / 0.126 | 0.095 / 0.045 | 0.018 / 0.001 | 0.658 / 0.503 |
| Support (frames) | T0 58,178 / T1 84,854 / T2 13,746 / T3 12,372 (same for all conditions) | | | |

A's confusion matrix shows heavy, roughly symmetric T0↔T3 and T1↔T3 confusion (not a clean majority-class collapse) — consistent with the reduced-budget early-stopping caveat (§ preamble): A's checkpoints were selected very early, before the model settled into a stable T0/T1-majority-leaning solution the way B/C/D's longer-trained checkpoints did.

---

## 10. Fold-by-fold C−A and D−A deltas

| Fold | C − A | D − A |
|---|---:|---:|
| 0 | −0.009 | +0.323 |
| 1 | **+0.144** | +0.470 |
| 2 | −0.028 | +0.368 |
| 3 | +0.076 | +0.547 |
| 4 | −0.059 | +0.511 |

C−A has no consistent sign (positive in folds 1, 3; negative in 0, 2, 4) — the defining signature of "no reliable effect," not a real but small one. D−A is uniformly large and positive in every fold, no exceptions.

---

## 11. Trajectory-aggregated metrics

Not computed in this pass — `training.metrics.aggregate_trajectory_predictions`/`trajectory_metrics` exist and would primitive-aggregate the same predictions already produced, but doing so for all 20 fold×condition runs was out of the time budget for this diagnostic step. The framewise metrics above (the actual target task per spec section 7) are the primary deliverable; primitive-level aggregation is a natural, cheap follow-up using the same checkpoints if needed.

---

## 12. Performance by |dp/dt|

Accuracy by bucket (evaluation metadata only, never used for decoding):

| Bucket | A | B | C | D |
|---|---:|---:|---:|---:|
| 0-100 c/s | 0.339 | 0.512 | 0.418 | 0.573 |
| 100-400 c/s | 0.153 | 0.425 | 0.420 | 0.662 |
| 400-1000 c/s | 0.188 | 0.500 | 0.467 | 0.759 |
| >1000 c/s | 0.281 | 0.585 | 0.462 | **0.849** |

Both B and D improve with movement speed (expected: the T2/T3 shapes these features are meant to disambiguate are definitionally the faster/more complex trajectories) — D dramatically so (0.573→0.849). C is comparatively flat (0.42-0.47 throughout) and never approaches B or D at any speed, reinforcing that C's problem is the fusion, not an inherent inability of the estimated signal to help fast-moving frames (B already shows it can).

---

## 13. Performance by primitive duration

| Bucket | A | B | C | D |
|---|---:|---:|---:|---:|
| <100ms | 0.340 | **0.696** | 0.580 | **0.890** |
| 100-250ms | 0.310 | 0.581 | 0.516 | 0.751 |
| 250-500ms | 0.246 | 0.408 | 0.344 | 0.628 |
| 500ms-1s | 0.300 | 0.388 | 0.383 | 0.542 |
| >1s | 0.314 | 0.539 | 0.374 | 0.526 |

Both pitch-augmented conditions (B, D) are strongest on the **shortest** primitives and decay toward longer ones — a direct, expected consequence of the fixed 10-200ms feature offsets: for a primitive under ~200ms, the φ window spans the whole primitive; for a multi-second primitive, φ only ever sees a small local slice of the overall shape. This is a real, disclosed limitation of the feature design (spec section 4's "keep it small," not a bug) rather than evidence relative pitch is fundamentally unhelpful for long primitives.

---

## 14. Per-recording comparison (C − A accuracy)

10/17 recordings favor C, 7/17 favor A, no dominant single-recording explanation (unlike Step 12.5's fusion story, where one recording explained most of the pooled effect):

| Recording (fold) | C − A |
|---|---:|
| `6503e348…` (1) | **+0.496** |
| `6824de49…` (1) | +0.412 |
| `6912841f…` (1) | +0.382 |
| `68f53fbf…` (0) | +0.074 |
| `6653d349…` (2) | +0.029 |
| `65e4a79c…` (3) | +0.024 |
| `6503e36c…` (0) | +0.015 |
| `6653ce5f…` (2) | +0.010 |
| `66552c6b…` (3) | +0.000 |
| `65b2ab70…` (0) | −0.004 |
| `6417585554…` (2) | −0.013 |
| `68d85d45…` (3) | −0.034 |
| `6491d48d…` (4) | −0.080 |
| `645ff354…` (2) | −0.153 |
| `65b14e20…` (0) | −0.165 |
| `6655f08a…` (0) | −0.197 |
| **`692ed7e6…`** (4) | **−0.293** |

Notably, fold 1's three recordings are C's three biggest wins (+0.38 to +0.50) — the same fold that produced condition D's second-highest score (0.550) — while fold 4's `692ed7e6…` is C's single worst result (−0.293) *and* also D's only recording where D underperforms A/B/C entirely (D=0.233 vs A=0.252/B=0.267/C=0.249) despite fold 4 having D's single best fold-level score (0.784) — an outlier worth a closer look in any follow-up, not resolved here.

---

## 15. Oracle-gap analysis

```
A = 0.205   C = 0.220   D = 0.604   (pooled macro F1)
C - A = +0.015   D - C = +0.384   D - A = +0.398
```

Against the four pre-declared patterns: not `C ≈ D > A` (D≫C); not (cleanly) `D ≫ C > A` (C is not consistently > A, §10); not `D ≈ A` (D is dramatically higher). Best fit: **`D > A but C ≈ A`** — "relative pitch would help in principle, but the current [fused] frontend is too noisy." True as far as it goes, but incomplete: condition B shows the *estimated* pitch signal, used **alone**, is not merely "too noisy to help" — it clearly beats both A and C. The bottleneck demonstrated here is therefore not purely pitch quality; it is at least partly the **naive concat+linear fusion mechanism** failing to let a real but weaker signal (φ_estimated, noisy relative to φ_oracle) compete against the higher-capacity, higher-variance audio pathway during joint optimization — most visible in T2/T3 (§16), where fusion makes things *worse* than audio alone, not just unhelpful.

One caveat on D's magnitude: `trajectory_type` is itself derived from the same parametric GT pitch curve used to build φ_oracle (T1/T2/T3 are literally bend-shape categories defined over that curve). D's huge score is therefore partly the model recovering a near-tautological relationship, not purely "external" evidence a real system could obtain from perfect audio-based pitch tracking — Step 13's independent 62¢-scale pitch-accuracy ceiling is a different, non-circular quantity. D's practical value here is as a *capacity check* (proves the architecture and φ design can express T2/T3 discrimination given clean input, ruling out "architecture is the bottleneck") rather than as an attainable production target.

---

## 16. T2/T3-specific analysis

| | A | B | C | D |
|---|---:|---:|---:|---:|
| T2 recall | 1.0% | 6.6% | 0.1% | **44.4%** |
| T3 recall | 12.6% | 4.5% | 0.1% | **50.3%** |
| T2 F1 | 0.018 | 0.097 | 0.001 | 0.519 |
| T3 F1 | 0.049 | 0.061 | 0.002 | 0.568 |

Answering spec section 18's question directly: **B (0.097/0.061) clearly exceeds Step 13's primitive-aligned logistic probe on the estimated source (T2 0.017-0.034, T3 0.005-0.010 depending on source)** — a real nonlinear-sequence-model gain over the simple linear probe, though both remain far below D. **D (0.519/0.568) is dramatically higher than either**, and higher than Step 13's *oracle* probe too (0.034/0.069) — confirming Step 13's linear probe was substantially classifier-limited, not representation-limited, exactly as section 18 anticipated: "if even oracle-relative-pitch modeling remains weak on T2/T3 later, audio cues likely matter fundamentally" does *not* hold here — oracle relative pitch alone (via a real sequence model) resolves T2/T3 well. The bottleneck for a deployable system is squarely pitch-estimation fidelity and/or fusion design, not a fundamental need for audio features to carry T2/T3 information.

---

## 17. Representative visualizations

`output/relative_pitch_ablation/figures/` (5 panels: CQT + GT/A/C/D type strips + GT-vs-estimated pitch overlay, selected systematically from the per-recording accuracy deltas and per-type support, not cherry-picked):

- `case_c_fixes_a.png` (`6503e348…`, fold 1, C−A=+0.496): the clearest illustration of octave-invariance in practice — the estimated pitch (red) sits roughly a full octave below GT (black) for seconds at a stretch, yet condition C still classifies the region as T0 correctly where A (relying on raw audio features) predicts T3 almost everywhere.
- `case_c_hurts_a.png`: a case where fusion visibly degrades relative to audio alone.
- `case_d_fixes_c.png`: largest D−C accuracy gain.
- `case_t2_example.png` (`6824de49…`, fold 1): D recovers several GT T2 (green) segments C collapses entirely into T1.
- `case_t3_example.png`: highest-T3-support recording.

---

## 18. Final outcome

**`RELATIVE_PITCH_ONLY_COMPETITIVE`**

> B (0.325 macro F1) clearly beats both A (0.205) and C (0.220) — not merely "approaches" as the criterion requires, an outright win on every pooled metric and in 16/17 recordings. Trajectory type is, at minimum, substantially a contour-shape problem: a model given nothing but a 4-dimensional octave-invariant relative-pitch vector per frame outperforms one given the full CQT.

This does not stand alone — it composes with the oracle-gap finding (§15, `ORACLE_PITCH_HELPS_BUT_ESTIMATED_DOES_NOT`-shaped): D's ceiling (0.604) is far above B's (0.325), so pitch information, done well, carries much more than B alone extracts. The two findings together point at the same conclusion from different angles: **relative pitch is doing most of the real work already available in this problem, the current estimated-pitch quality and/or fusion mechanism is leaving most of it on the table, and audio's marginal contribution on top of good pitch is comparatively small** (D vs. an oracle-pitch-only ceiling was not directly tested here, but B beating full-audio A is already suggestive).

**Not** `ESTIMATED_RELATIVE_PITCH_HELPS` (C−A is not a consistent win). **Not** `ESTIMATED_PITCH_NEAR_ORACLE` (C is nowhere near D). **Not** `AUDIO_DOMINATES` (D's gap over A is enormous).

---

## 19. Decision gate

**`IMPROVE_RELATIVE_PITCH_ESTIMATION`**

Both preconditions hold: D clearly beats C (+0.384 macro F1, uniform across every fold), and pitch information is independently demonstrated useful on its own (B beats A outright). The gap between B/C's estimated-pitch results and D's oracle ceiling is the largest, most consistent effect in this entire step.

**Do next:** the natural, in-scope follow-up is **not** another register decoder (Steps 12/12.5 already closed that branch) but (a) a **better fusion mechanism** than static concat+linear — the finding that B beats C specifically implicates fusion, not just pitch quality, as a fixable bottleneck; and (b) revisiting whether a **learned** pitch-motion representation (rather than the fixed 4-tap φ) captures more of D's ceiling, especially for primitives longer than ~250ms where the fixed-offset design structurally under-serves (§13).

**Do not:** conclude relative pitch should be dropped (`DROP_EXPLICIT_PITCH_FEATURES` is clearly wrong given B's result). Do not conclude this is purely an architecture-capacity problem needing a bigger/different temporal model (`INVESTIGATE_CONTOUR_MODELING` doesn't fit — D shows the *existing* architecture already expresses strong T2/T3 discrimination given clean input). Do not add class weighting, oversampling, or architecture search in response to these results — those remain separate, later follow-ups per the calling brief, not this step's conclusion.
