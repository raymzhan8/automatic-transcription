# Step 22 — Oracle-Boundary Normalized Contour Shape Classification

Step 21 froze CREPE as the pitch source and closed pitch-frontend research. This step asks a narrower question before attempting continuous segmentation: **given the correct trajectory boundaries, does normalized pitch-contour geometry distinguish the four canonical trajectory shapes at all** — from CREPE, not just from the oracle parametric curve? GT `start_s`/`end_s` are used only because this is a segment-normalized shape diagnostic in isolation; the eventual transcription system will not receive them.

Semantic definitions (unchanged `canonical_type` ids from `dataset/canonical/schema.py`'s `PRIMITIVE_TYPE_IDS`, renamed for this step):

| id | Name here | `TRAJECTORY_ID_TO_NAME` | Geometric definition |
|---|---|---|---|
| 0 | Fixed | Fixed | pitch ≈ constant |
| 1 | Cosine | Bend: Simple | smooth cosine-like start→end transition |
| 2 | Sloped-start | Bend: Sloped Start | same transition, displacement concentrated **early** in phase |
| 3 | Sloped-end | Bend: Sloped End | same transition, displacement concentrated **late** in phase |

Machine-readable outputs: `output/shape_classification/{corpus_summary,semantic_check,analytic_baseline,cnn_results,duration_span_analysis,t2_t3_analysis,boundary_perturbation}.json`, figures in `output/shape_classification/figures/`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.dataset               # §1-2: corpus + primitive/duration/class counts
python -m training.shape_classification.visualize              # §5: normalized-shape sanity-check plots
python -m training.shape_classification.semantic_check          # §7: semantic-hypothesis validation
python -m training.shape_classification.baseline                # §8: analytic (logistic regression) baseline
python -m training.shape_classification.cnn_model                # §9-10: 1D CNN, shape-only + shape+velocity
python -m training.shape_classification.boundary_perturbation    # §17: CP condition
python -m training.shape_classification.duration_span_analysis   # §15-16
python -m training.shape_classification.t2_t3_analysis           # §18
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| 7,177 canonical primitives extracted cleanly, 0 oracle/CREPE extraction failures | §2 |
| Oracle normalized shapes **visibly and cleanly separate** by class (sanity gate passes) | §5 |
| The Fixed/Cosine/Sloped-start/Sloped-end semantic hypothesis is **confirmed analytically** (oracle `q(0.5)`: Cosine=0.500, Sloped-start=0.750, Sloped-end=0.250, exactly symmetric/front/back-loaded) and **directionally preserved, though attenuated, in CREPE** (`q(0.5)`: 0.456 / 0.679 / 0.282) | §7 |
| Oracle geometry is highly separable, both from 7 hand-picked features (analytic macro F1 0.775) and from the full 64-point CNN (macro F1 0.772, rising to **0.801** with a velocity channel; Sloped-start/Sloped-end F1 0.90-1.00) | §8-10 |
| **CREPE collapses on the two bend-subtype classes specifically — not partially, totally**: Sloped-start and Sloped-end F1 = **0.000** for both the analytic baseline and the CNN (shape-only and shape+velocity), confirmed by an all-zero confusion-matrix column for both classes | §8-11, §13 |
| The collapse is a **majority-class collapse under unweighted CE + severe class imbalance (69% Cosine) interacting with CREPE noise** — not a duration or pitch-span artifact (T2/T3 F1 is exactly 0.000 in *every* duration bucket and *every* pitch-span bucket) and not a boundary-precision artifact (T2/T3 F1 is already exactly 0.000 at the GT boundary, before any perturbation) | §15-17 |
| CREPE genuinely **does** carry real, above-chance front-loaded-vs-back-loaded shape signal — a single-feature sign test separates Sloped-start from Sloped-end at 76.8% accuracy (vs. 50% chance, vs. oracle's 100%) — the CNN is failing to exploit signal that measurably exists, not working with signal that doesn't | §18 |
| **Directly contradicts the hoped-for Step 21 explanation**: Step 21 speculated CREPE's T3 collapse (F1 0.003) might be an artifact of the fixed-time framewise representation being mismatched to CREPE's continuous signal. Here, even with GT boundaries *and* full phase/span normalization — removing that exact mismatch — CREPE's Sloped-end (T3-equivalent) F1 is still 0.000. The representation was not the (sole) problem | §19 |

**Primary outcome: `ORACLE_SHAPE_WORKS_CREPE_DEGRADES`**

**Decision gate: `INVESTIGATE_CREPE_SHAPE_NOISE`**

---

## 1-2. Corpus: primitive counts, durations, class counts

All primitives from `output/canonical/v1/primitives/*.json` across all 17 recordings, GT `start_s`/`end_s` used only to slice the pitch contour (never combined with neighboring-primitive labels, never used to alter feature construction beyond defining the window).

| Class | Count | Median duration | Mean duration | Min | Max |
|---|---:|---:|---:|---:|---:|
| Fixed | 1,306 | 235ms | 450ms | 49ms | 7.86s |
| Cosine | 4,950 | 109ms | 171ms | 9ms | 3.91s |
| Sloped-start | 468 | 226ms | 294ms | 47ms | 2.59s |
| Sloped-end | 453 | 235ms | 273ms | 40ms | 1.50s |
| **Total** | **7,177** | | | | |

0 oracle extraction failures, 0 CREPE extraction failures (every primitive's phase grid is fully populated, finite, on both sources).

Severe class imbalance: Cosine is 69% of all primitives; Sloped-start/Sloped-end are 6.5%/6.3% each. This imbalance turns out to be central to §11-13's result.

## 2-3. Normalized-phase transform and pitch-span/direction normalization

`x = (t - start_s)/(end_s - start_s)`, resampled onto `N=64` fixed phase points (`training/shape_classification/contours.py`). The audio itself is never time-stretched; only the extracted pitch contour is resampled. Two sources, same transform:

- **Oracle (O)**: the analytic IDTAP parametric curve, evaluated directly at each of the 64 phase points via the exact reconstruction (`Trajectory.compute(x, log_scale=True)`) `dataset/canonical/contour.py` already uses to regenerate GT framewise targets — no intermediate 10ms-grid interpolation.
- **CREPE (C)**: Step 21's frozen dense CREPE path, linearly interpolated from its native 10ms grid onto the same 64-point phase grid.

Relative pitch `r(x) = p(x) - p(0)` (cents, tonic-independent) is always computed first. Span/direction normalization: `q(x) = r(x)/r(1)` when `|r(1)| >= MIN_SPAN_CENTS = 50¢` (both rising and falling trajectories map to `q(1)=1`); otherwise `q(x) = r(x)` unchanged, kept in cents (section 11's T0 rule — see below). The branch is chosen purely from the sampled contour's own endpoints, never from the GT label.

**Span-normalization rate by class (oracle)**: Fixed 0% (by construction — a Fixed trajectory's analytic `p(1)==p(0)` exactly), Cosine 68.2%, Sloped-start 93.4%, Sloped-end 99.6%. Note: 31.8% of "Cosine" (Bend: Simple) primitives have a near-zero net span — real ornamental micro-bends that leave and return without net pitch displacement, not annotation noise. This population resurfaces in §16 as a genuinely hard, low-separability slice for every source, oracle included.

## 5. Normalized-shape plots (sanity gate)

`output/shape_classification/figures/{oracle,crepe}_normalized_shapes.png` (span-normalized subset for Cosine/Sloped-start/Sloped-end; all of Fixed, which stays unnormalized by construction).

**Oracle**: textbook-clean separation — Fixed flat at 0, Cosine a symmetric S-curve, Sloped-start a front-loaded concave rise, Sloped-end a back-loaded convex rise, essentially zero within-class spread for Cosine (it is a literal analytic cosine function).

**CREPE**: noisier, wide IQR bands, but the **class medians still visibly separate in the same direction** — Sloped-start's median rises faster early and plateaus, Sloped-end's median stays low longer and rises late, Cosine's median sits between them, roughly symmetric. Fixed shows real ±20-80¢ jitter (CREPE's noise floor), much larger than oracle's near-exact zero.

**Gate result: PASS** — the oracle shapes visibly separate (proceeding is warranted per spec), and even CREPE's noisy medians retain the expected ordering, motivating the rest of the step.

## 7. Semantic-hypothesis validation

Measured, not assumed (span-normalized subset only, T1-T3; Fixed's near-zero total excursion checked separately):

| | Fixed total excursion (median, ¢) | Cosine `q(0.5)` | Sloped-start `q(0.5)` | Sloped-end `q(0.5)` | Cosine early−late | Sloped-start early−late | Sloped-end early−late |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 0.00 | **0.500** | **0.750** | **0.250** | **0.000** | **+0.500** | **−0.500** |
| CREPE | 13.79 | 0.456 | 0.679 | 0.282 | −0.087 | +0.359 | −0.436 |

Oracle matches the assumed semantics exactly (it is the analytic definition, so this is a reproduction check, not new evidence). **CREPE preserves the same ordering and sign** (Sloped-start > Cosine > Sloped-end on every statistic) with realistic attenuation toward the symmetric (Cosine-like) point — exactly the shrinkage expected from measurement noise on a real, not fabricated, geometric distinction. The data supports rather than contradicts the semantic hypothesis; proceeding to modeling was warranted.

## 8. Analytic baseline (logistic regression, 7 features, grouped 5-fold)

Features: `total_excursion, q25, q50, q75, early_displacement, late_displacement, phase_of_max_velocity` (`training/shape_classification/normalize.py::ANALYTIC_FEATURE_NAMES`). No hyperparameter search. Same grouped 5-fold recording-level splits as the rest of the project (`grouped_kfold_k5_seed42.json`).

| Source | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 0.775 | 0.416 | 0.799 | 0.887 | **0.999** | 0.678 ± 0.158 |
| **CREPE** | **0.224** | 0.083 | 0.814 | **0.000** | **0.000** | 0.214 ± 0.031 |

Answers section 8's question cleanly: oracle geometry is already nearly separable from 7 hand-picked numbers alone. CREPE, fed the *same* 7 numbers, cannot separate Sloped-start or Sloped-end from Cosine at all (F1 exactly 0 for both) — foreshadowing §11's CNN result.

## 9-10. Small 1D CNN (contour-only, + one velocity ablation)

`ContourCNN`: 3 dilated Conv1d blocks (kernel 5, dilations 1/2/4) → global average pool → linear(4). ~2.8k params. Same architecture, same protocol (Adam, lr 1e-3, wd 1e-4, batch 32, max 100 epochs / patience 15, seed 42+fold, unweighted CE, train-only channel standardization), same grouped 5-fold splits, for Oracle and CREPE alike. No architecture search, no audio, no absolute duration as input.

Section 10's ablation — shape-only `q(x)` (`[B,1,64]`) vs. shape+velocity `q(x),dq/dx` (`[B,2,64]`) — run for both sources.

## 11. Central experiment table

| Input | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|---:|
| Oracle analytic | 0.775 | 0.416 | 0.799 | 0.887 | 0.999 |
| Oracle 1D CNN (shape only) | 0.772 | 0.450 | 0.797 | 0.945 | 0.896 |
| Oracle 1D CNN (+ velocity) | **0.801** | 0.450 | 0.805 | 0.955 | **0.996** |
| CREPE analytic | 0.224 | 0.083 | 0.814 | 0.000 | 0.000 |
| CREPE 1D CNN (shape only) | 0.269 | 0.268 | 0.809 | **0.000** | **0.000** |
| CREPE 1D CNN (+ velocity) | 0.290 | 0.343 | 0.817 | **0.000** | **0.000** |

Grouped 5-fold mean ± std: Oracle shape-only 0.789±0.067, Oracle +velocity 0.820±0.077, CREPE shape-only 0.277±0.029, CREPE +velocity 0.291±0.012.

**The velocity ablation clearly helps oracle** (macro F1 0.772→0.801, driven mainly by Sloped-end 0.896→0.996) — retained as evidence slope-distribution information matters, per section 10. It does **not** rescue CREPE (0.269→0.290, Sloped-start/Sloped-end still exactly 0) — the failure is not "the model lacks a velocity channel," it is something upstream of that.

**The CNN barely moves CREPE's outcome versus the 7-feature analytic baseline** (0.224→0.269) — a full 64-point sequence model, which could in principle average out per-point CREPE noise far better than 3 point-samples, gains almost nothing. This argues against "the analytic feature set was just too crude" as the explanation.

Confusion matrix, CREPE shape-only, pooled test set (rows=true, cols=pred, order Fixed/Cosine/Sloped-start/Sloped-end):

```
Fixed        [258, 1048,   0,   0]
Cosine       [298, 4652,   0,   0]
Sloped-start [ 40,  428,   0,   0]
Sloped-end   [ 25,  428,   0,   0]
```

Columns 3 and 4 (Sloped-start, Sloped-end) are **entirely zero** — the model never predicts either class, for any input, in the entire pooled test set. This is not "weak discrimination," it is complete majority/near-majority collapse (predictions land almost entirely on Cosine, with some Fixed).

## 12. Non-oracle / grouped-evaluation preservation

Both O and C features are constructed purely from the sliced pitch contour (analytic function or CREPE path) plus the GT window itself — no neighboring-primitive label, no future annotation metadata. Grouped 5-fold splits are the same `performance_group_id`-grouped manifest the framewise pipeline uses (`training.folds.build_fold_split`); every primitive is assigned to train/val/test purely by which recording it belongs to, never split independently. Standardization (CNN channel mean/std; analytic `StandardScaler`) is fit on TRAIN recordings' primitives only, per fold.

## 13. Metrics

Reported throughout: pooled accuracy/macro F1/per-class precision-recall-F1-support and confusion matrices (`training.metrics.frame_metrics`, reused unmodified — its 0-3 label convention already matches `canonical_type`), plus grouped 5-fold mean±std. Macro F1 is primary, consistent with the rest of the project.

## 15. Duration analysis

CREPE shape-only CNN, pooled test predictions bucketed by primitive duration:

| Bucket | n | Oracle macro F1 | CREPE macro F1 | CREPE T2/T3 F1 |
|---|---:|---:|---:|---:|
| <100ms | 2,285 | 0.749 | 0.241 | 0.000 / 0.000 |
| 100-250ms | 3,273 | 0.755 | 0.226 | 0.000 / 0.000 |
| 250-500ms | 1,020 | 0.763 | 0.211 | 0.000 / 0.000 |
| 500ms-1s | 364 | 0.706 | 0.236 | 0.000 / 0.000 |
| >1s | 235 | 0.411 | 0.247 | 0.000 / 0.000 |

Oracle is fairly duration-independent through 1s (0.71-0.76) with a real drop past 1s (0.41 — the longest, rarest primitives are hardest even for the clean analytic curve, plausibly because very long primitives are almost all a distinct sustained-note population). **CREPE is flat and uniformly poor across every duration bucket** (0.21-0.25) — normalized-phase modeling does remove most of the gross duration-dependence a fixed-time representation would show, but that is not what is limiting CREPE here: the Sloped-start/Sloped-end F1 is exactly 0.000 in *every single bucket*, ruling out "CREPE only fails on very short/long primitives" as an explanation.

## 16. Pitch-span analysis (moving primitives only, Fixed excluded)

| Span bucket | n (oracle / CREPE) | Oracle macro F1 | Oracle acc | CREPE macro F1 | CREPE acc |
|---|---|---:|---:|---:|---:|
| <50¢ | 1,608 / 2,605 | 0.110 | 0.234 | 0.221 | 0.774 |
| 50-100¢ | 1,253 / 1,017 | 0.716 | 0.982 | 0.239 | 0.916 |
| 100-200¢ | 1,427 / 1,032 | 0.720 | 0.986 | 0.228 | 0.839 |
| 200-400¢ | 1,124 / 805 | 0.710 | 0.964 | 0.211 | 0.728 |
| >400¢ | 459 / 412 | 0.716 | 0.961 | 0.190 | 0.612 |

(The oracle and CREPE n differ slightly because the two sources' span-normalization branch is computed independently from each one's own contour.)

**Oracle**: the `<50¢` bucket (exactly the un-span-normalized, "near-flat bend" population from §3) is genuinely hard even analytically (macro F1 0.110, accuracy 0.234) — these primitives are intrinsically ambiguous relative-cents blips, not a representation defect. Once span reaches the 50¢ normalization threshold, oracle performance jumps immediately to ~0.71-0.72 and **stays flat across every larger span bucket** — confirms span normalization does what it is supposed to: shape recognition becomes span-independent once there is a real span to normalize.

**CREPE**: macro F1 stays low and roughly flat (0.19-0.24) at every span level — again, T2/T3 F1 is exactly 0.000 in every bucket (not shown in the table but confirmed in `duration_span_analysis.json`), so span is not the limiting variable either. Accuracy actually *declines* as span grows (0.916→0.612) — consistent with the majority-class-collapse story: in the small-span bucket, guessing Cosine is often simply correct (many real Cosine primitives live there too), but as span grows, more of the true labels are genuinely Sloped-start/Sloped-end and blindly guessing Cosine costs more accuracy, while macro F1 (which does not reward the correct-by-luck majority guesses on Cosine itself as much) stays uniformly poor throughout.

## 17. Boundary-sensitivity experiment (CP condition)

Frozen CREPE shape-only CNN (no retraining), symmetric window shift:

| Δ (start & end) | Macro F1 | Fixed F1 | Cosine F1 | Sloped-start F1 | Sloped-end F1 |
|---:|---:|---:|---:|---:|---:|
| −100ms | 0.249 | 0.194 | 0.802 | 0.000 | 0.000 |
| −50ms | 0.252 | 0.208 | 0.799 | 0.000 | 0.000 |
| −20ms | 0.263 | 0.249 | 0.804 | 0.000 | 0.000 |
| **0 (GT)** | **0.269** | 0.268 | 0.809 | 0.000 | 0.000 |
| +20ms | 0.267 | 0.258 | 0.808 | 0.000 | 0.000 |
| +50ms | 0.258 | 0.228 | 0.803 | 0.000 | 0.000 |
| +100ms | 0.256 | 0.222 | 0.803 | 0.000 | 0.000 |

Start-only (end fixed at GT) and end-only (start fixed at GT), ±50ms: macro F1 0.255-0.267 — no meaningful asymmetry between start- and end-boundary error at this magnitude.

**Sloped-start/Sloped-end F1 is exactly 0.000 at every single perturbation level, including 0 (the exact GT boundary)** — the collapse is not caused or worsened by boundary imprecision; it is already total before any perturbation is applied. Overall macro F1 degrades only mildly and gracefully with perturbation magnitude (0.269→0.249 at −100ms, a ~7% relative decline), driven almost entirely by the Fixed class becoming harder to distinguish as the window pulls in adjacent-primitive content (Fixed F1 0.268→0.194 at −100ms); Cosine is essentially unaffected (0.80-0.81 throughout, consistent with it being the default/majority prediction regardless of window). **This rules out `BOUNDARY_ACCURACY_DOMINATES`**: there is no sharp perturbation-driven collapse to explain, because the collapse is already complete at zero perturbation.

## 18. T2 (Sloped-start) vs. T3 (Sloped-end) direct comparison

Does CREPE preserve the front-loaded vs. back-loaded distinction — the actual semantic difference between these classes, more targeted than generic turning-point recall? `output/shape_classification/figures/t2_vs_t3.png` (median `q(x)`, span-normalized subset).

| | n (T2/T3) | frac. T2 front-loaded (early>late) | frac. T3 back-loaded (late>early) | Sign-test separation accuracy | Median phase-of-max-|v| (T2, T3) |
|---|---|---:|---:|---:|---|
| Oracle | 437 / 451 | 1.000 | 1.000 | **1.000** | 0.00, 1.00 |
| **CREPE** | 306 / 324 | 0.765 | 0.772 | **0.768** | 0.30, 0.69 |

CREPE's own median shapes are visibly separated in the correct direction (plot), and a trivial single-bit statistic (sign of early-minus-late displacement) separates Sloped-start from Sloped-end at **76.8%** accuracy — far above chance (50%), though far below oracle's exact 100%. **CREPE does preserve real, usable front-loaded-vs-back-loaded signal.** This makes §11's exact-zero CNN/analytic F1 for both classes especially notable: the representation is not empty of the relevant information, but neither the 7-feature analytic model nor the full-sequence CNN, trained with unweighted CE against a 69%-Cosine prior, ever chooses to use it.

## 19. Comparison with Step 21's framewise CREPE P0

Step 21 (`docs/step_21_crepe_baseline.md`): CREPE-trained, fixed-time framewise P0 achieved pooled macro F1 ≈ 0.320, with T3 (Sloped-end-equivalent) F1 ≈ **0.003** — near-total collapse — while T0 (Fixed) *improved* over D1. Step 21 speculated this might be an artifact of feeding a continuous, non-staircased CREPE signal through a representation (fixed 10/50/100/200ms deltas) implicitly tuned against D1's heavily-staircased signal texture, and recommended testing a segment-normalized representation as the next trajectory-modeling question.

This step is exactly that test, and the result does **not** confirm the hoped-for explanation: even with GT boundaries (removing the segmentation question entirely) *and* full phase+span normalization (removing the fixed-time-vs-variable-duration mismatch entirely), CREPE's Sloped-end analog still collapses — F1 = **0.000**, if anything worse in absolute terms than Step 21's 0.003, and the collapse is now confirmed uniform across every duration bucket, every pitch-span bucket, and every boundary-perturbation level (§15-17), and traced to a specific, demonstrable mechanism: majority-class collapse under class imbalance, not a lack of underlying signal (§18 shows 76.8% usable signal exists). **CREPE pitch-contour noise, interacting with class imbalance under unweighted training, is at least as important a contributor to the T2/T3 failure as the choice of time representation** — Step 21's framewise-mismatch hypothesis is not supported as the primary or sole explanation.

## 20. Interpretation — primary outcome

**`ORACLE_SHAPE_WORKS_CREPE_DEGRADES`**

Oracle performs strongly and consistently across every diagnostic (analytic F1 0.775, CNN F1 0.772-0.801, near-perfect Sloped-start/Sloped-end F1, clean duration/span independence, 100% T2-vs-T3 sign separation) — the normalized phase/span representation itself is validated and correct. CREPE performs much worse specifically on the two bend-subtype classes (F1 exactly 0.000, both analytic and CNN, every duration/span/boundary slice) while still doing reasonably on Fixed/Cosine, and independently retains real (76.8%) discriminative signal for the T2-vs-T3 direction that the trained classifiers simply never use. `SHAPE_REPRESENTATION_INSUFFICIENT` is ruled out (oracle separates cleanly). `BOUNDARY_ACCURACY_DOMINATES` is ruled out (§17: no perturbation-driven degradation of an otherwise-working result — T2/T3 is already zero at Δ=0). `NORMALIZED_SHAPE_SOLVES_CLASSIFICATION` is ruled out (CREPE, the realistic condition, does not give strong four-class performance). This is squarely CREPE noise (as it interacts with class imbalance under unweighted training) corrupting trajectory geometry, per spec's own description of this outcome — **do not reopen pitch-frontend research** (Step 21's scope boundary stands; this is a modeling/training question about how to use CREPE's contour, not a question about CREPE's estimation quality per se, which Step 21 §12 already showed is good by classic motion-fidelity metrics).

## 21. Decision gate

**`INVESTIGATE_CREPE_SHAPE_NOISE`**

## Recommendation for Step 23

Section 18 already shows CREPE's normalized contour retains real, above-chance shape information (76.8% single-feature T2-vs-T3 separation) that neither classifier tested here ever uses, because both default to the 69%-prevalence Cosine class whenever the input is ambiguous under unweighted training. Before concluding CREPE cannot support shape-based trajectory typing, Step 23 should test whether that buried signal is recoverable with the two most direct, minimal interventions implied by this step's own "do not yet" list — both explicitly deferred here, not new ideas: (1) class-weighted loss or class-balanced sampling (spec explicitly separated "no class weighting initially" from the primary experiment, anticipating this as the natural next lever), and (2) a light, fixed denoising/smoothing of the CREPE contour before shape-feature extraction (a trajectory-input preprocessing choice, not a frontend change, consistent with Step 21's own closing recommendation). Only after that should segmentation (`MOVE_TO_SEGMENTAL_TRANSCRIPTION`) be revisited — building a boundary detector for a shape classifier that cannot yet see its own two rarest classes would be premature.
