# Step 23 — Can Balanced Training Recover CREPE Shape Information?

Step 22 established that oracle-boundary normalized contour shape works (CNN macro F1 ≈0.801, Sloped-start/Sloped-end F1 ≈0.955/0.996), that CREPE's normalized contour retains real geometric information (a trivial early-vs-late sign statistic separates Sloped-start from Sloped-end at 76.8% accuracy, well above the 50% chance level), and yet that both the analytic and CNN four-class CREPE classifiers **completely** collapse on Sloped-start/Sloped-end (F1 = 0.000 for both, the CNN never predicting either class). The training distribution is 69% Cosine. This step asks whether that collapse is a class-prior/objective artifact or a genuine information bottleneck — **without touching the pitch representation**.

Frozen exactly from Step 22: CREPE extraction, `MIN_SPAN_CENTS`, interpolation, the 64-point phase grid, `q(x)`/`dq/dx` normalization, `ContourCNN` architecture, optimizer/LR/epoch budget/patience, and the grouped 5-fold `performance_group_id` manifest. The only experimental variable is how class imbalance is handled during training. GT boundaries remain frozen (no segmentation); CREPE remains frozen (no frontend work).

Machine-readable outputs: `output/shape_classification/step23/{results,results_full}.json`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step23_experiments
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| **B0 reproduces Step 22 essentially exactly** (macro F1 0.2899 vs. Step 22's 0.290 reference; T2/T3 F1 both 0.000; predictions 90.4% Cosine against a 69% true rate) | §4 |
| **Both balancing interventions break the collapse**: T2/T3 F1 goes from exactly 0.000 to 0.202/0.203 (B1, sampler) and 0.138/0.077 (B2, weighted CE) — real, above-base-rate precision (12.7% vs. 6.5% true prevalence for B1), not a trivial "predict everywhere" artifact, confirmed by confusion matrices | §6-8 |
| Four-class macro F1 improves only **modestly and inconsistently**: pooled 0.290→0.311 (B1) but 0.290→**0.276** (B2, a decline); grouped mean 0.290±0.012→0.323±0.089 (B1) / 0.318±0.095 (B2) — Cosine F1 pays a large, real cost (0.817→0.46-0.50) | §6, §12 |
| Fold consistency is real but not uniform: B1 improves 4/5 folds (median Δ+0.019), B2 improves 3/5 folds (median Δ+0.016); one fold (3) shows a sizeable B1 regression (−0.078) | §12 |
| Prediction frequency confirms balancing genuinely moves the decision boundary: B0 predicts Cosine 90.4% of the time (T2/T3 0%); B1/B2 predict T2/T3 a combined ~40-46% of the time — the classifiers now clearly enter the T2/T3 decision regions | §7 |
| Confidence analysis shows the same story at the probability level: B0 assigns true-T2/T3 examples a median probability of 0.05-0.06 for their own class vs. 0.74-0.78 for Cosine; B1/B2 raise median P(true class) to ~0.28-0.33, now roughly tied with P(Cosine) — a >5x shift from the SAME architecture and features, changing only the objective | §14 |
| **Binary T2-vs-T3 result is confounded by small-sample training instability, not a clean below-baseline result**: pooled accuracy (53%) looks well below the 76.8% sign-test baseline, but per-fold inspection shows 2 of 5 folds collapsed at `best_epoch` 1-2 (early-stopping degeneracy on a tiny validation set, predicting only one class), while the other 3 folds — where training actually proceeded — score 64-76% accuracy, roughly matching the sign baseline | §9, §15 |
| 3-way bend-only (Cosine/Sloped-start/Sloped-end) balancing (M1) improves **every one of the 5 folds** over natural (M0) (0.28→0.49, 0.30→0.31, 0.31→0.33, 0.27→0.41, 0.32→0.37), the most consistent balancing result in the step, and gives CREPE's best absolute T2/T3 F1 anywhere in this step (0.243/0.222) | §10-11 |

**Primary outcome: `CLASS_BALANCING_REVEALS_TRADEOFF`**

**Decision gate: `INVESTIGATE_ROBUST_CREPE_SHAPE_REPRESENTATION`**

---

## 1-2. Frozen representation and model

Source: frozen CREPE (Step 21/22, unchanged). Segmentation: GT primitive boundaries (unchanged). Phase grid: `x∈[0,1]`, `N=64` (unchanged). Feature: `q(x)` + `dq/dx` (Step 22's best CREPE condition — `crepe_shape_velocity`). Model: `ContourCNN` (3 dilated Conv1d blocks, global average pool, linear head, ~2.8k params), Adam (lr 1e-3, wd 1e-4), batch 32, max 100 epochs / patience 15, seed 42+fold, grouped 5-fold `grouped_kfold_k5_seed42.json` manifest — every one of these reused unmodified from `training/shape_classification/cnn_model.py`. `training/shape_classification/step23_train.py` adds exactly one new axis: `balancing ∈ {none, sampler, weighted_ce}`, applied via `torch.multinomial`-based per-epoch resampling (B1) or a `CrossEntropyLoss(weight=...)` term (B2) — nothing else in the training loop changes.

## 3. Train-set class frequencies per fold

Representative fold (fold 0) train counts: Fixed 1183, Cosine 4643, Sloped-start 408, Sloped-end 397 (full per-fold counts in `results_full.json`'s `train_class_counts`). B2's inverse-frequency weights for fold 0 (mean-normalized to 1): Fixed 0.561, Cosine 0.143, Sloped-start 1.626, Sloped-end 1.671 — computed fresh per fold from TRAIN primitives only, never from val/test.

## 4. B0 — unweighted baseline (reproduction check)

| | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| Step 22 reference | ≈0.290 | 0.343 | 0.817 | 0.000 | 0.000 | — |
| **B0 (this step)** | **0.2899** | 0.343 | 0.817 | 0.000 | 0.000 | 0.2905 ± 0.0122 |

Exact reproduction (same seeds, same code path through the new generic trainer). Confusion matrix confirms total collapse: `[[342,964,0,0],[278,4672,0,0],[36,432,0,0],[35,418,0,0]]` — columns 3/4 (Sloped-start, Sloped-end) are entirely zero. Proceeding to interpret B1/B2 is warranted.

## 6. Central four-class table

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| B0 unweighted | 0.290 | 0.343 | 0.817 | 0.000 | 0.000 | 0.290 ± 0.012 |
| B1 balanced sampler | **0.311** | 0.379 | 0.459 | 0.202 | 0.203 | 0.323 ± 0.089 |
| B2 weighted CE | 0.276 | 0.387 | 0.503 | 0.138 | 0.077 | 0.318 ± 0.095 |
| Oracle reference (Step 22) | 0.801 | 0.450 | 0.805 | 0.955 | 0.996 | 0.820 ± 0.077 |

Macro F1 is primary and the picture it gives is genuinely mixed: B1 improves pooled macro F1 modestly (+0.021) and grouped mean more (+0.033); B2 *lowers* pooled macro F1 (−0.014) despite a higher grouped mean (+0.028) — pooled and grouped-mean disagree in direction for B2 because of fold-level variance (§12). Neither balancing condition comes remotely close to oracle's ceiling.

## 7. Prediction-frequency diagnostic

| | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| True distribution (pooled test) | 18.2% | 69.0% | 6.5% | 6.3% |
| B0 | 9.6% | **90.4%** | **0.0%** | **0.0%** |
| B1 | 19.5% | 30.4% | 25.5% | 24.6% |
| B2 | 19.0% | 35.2% | 29.6% | 16.1% |

Answers section 7's question unambiguously: **yes**, balancing causes the classifier to genuinely enter the T2/T3 decision regions — from literally never predicting either class (B0) to predicting them a combined 40-46% of the time (B1/B2), well above their 12.8% combined true prevalence (both methods now *over*-predict the minority classes, the expected signature of a successful prior-correction, not evidence of a trivial failure mode). B2 shows a real asymmetry (Sloped-start 29.6% vs. Sloped-end 16.1%, roughly matching their similar true rates 6.5%/6.3%) — weighted CE recovers Sloped-start more than Sloped-end.

## 8. Precision-recall tradeoff for T2/T3

| Condition | Sloped-start P / R / F1 | Sloped-end P / R / F1 |
|---|---|---|
| B0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| B1 | 0.127 / 0.496 / 0.202 | 0.127 / 0.497 / 0.203 |
| B2 | 0.084 / 0.382 / 0.138 | **0.054** / 0.137 / 0.077 |

Base rate for comparison: Sloped-start 6.5%, Sloped-end 6.3% of test primitives. B1's precision (12.7% for both classes) is roughly **double** the base rate — real, if weak, separability, not "predict T2/T3 everywhere" (which would produce precision ≈ base rate). B2's Sloped-end precision (5.4%) is essentially **at** its base rate (6.3%) — for Sloped-end specifically, weighted CE's positive predictions are barely better than chance, the weakest result in the step and worth flagging directly: B2 recovers usable Sloped-start signal but not Sloped-end.

## 9. Binary T2-vs-T3 diagnostic

Pooled: accuracy 52.99%, macro F1 0.435, Sloped-start F1 0.667, Sloped-end F1 0.203 — well below the 76.8% sign-test baseline. Per-fold breakdown (grouped mean 0.506, very different from the pooled number) reveals why the pooled figure is misleading:

| Fold | n_test | best_epoch | Macro F1 | Accuracy | Sloped-start F1 | Sloped-end F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 73 | 37 | 0.643 | 64.4% | 0.658 | 0.629 |
| 1 | 299 | **1** | 0.431 | 75.6% | 0.861 | **0.000** |
| 2 | 454 | **2** | 0.262 | 35.5% | 0.524 | **0.000** |
| 3 | 66 | 6 | 0.485 | 48.5% | 0.485 | 0.485 |
| 4 | 29 | 16 | 0.709 | 75.9% | 0.588 | 0.829 |

Folds 1 and 2 — which together hold 82% of all pooled test examples (753/921) — both stopped at `best_epoch` 1 or 2 and collapsed to predicting Sloped-start exclusively (Sloped-end F1 = 0.000 in both, confusion matrices `[[226,0],[73,0]]` and `[[161,0],[293,0]]`). This is early-stopping degeneracy on a tiny, high-variance validation set (per-fold binary val sets are a small slice of an already-small ~921-primitive population), not evidence that the CNN cannot learn the binary distinction: folds 0, 3, and 4 — where training visibly proceeded past epoch 1 — score 48-76% accuracy, macro F1 0.49-0.71, roughly comparable to (fold 4 exceeds) the 76.8% sign-test baseline. **Per section 15's own instruction, this is a "training/normalization to investigate" flag, not a clean "the CNN loses information the sign statistic has" result** — the pooled number should not be read as such. The most defensible summary: on the folds where training worked, the CNN roughly matches a trivial 1-D statistic; it does not clearly exceed it, so it is not adding value over the sign test on this binary subtask, but the specific claim "the CNN underperforms" is an artifact of 2/5 degenerate folds rather than a robust finding.

## 10-11. Natural vs. balanced 3-way bend classification (Cosine / Sloped-start / Sloped-end, no Fixed)

| Task | Condition | Macro F1 (pooled) | Cosine | Sloped-start | Sloped-end | Grouped mean |
|---|---|---:|---:|---:|---:|---:|
| 3-way bend | M0 natural | 0.305 | 0.915 | 0.000 | 0.000 | 0.297 |
| 3-way bend | M1 balanced | **0.339** | 0.551 | 0.243 | 0.222 | **0.383** |

Per-fold macro F1, M0→M1: 0.283→**0.490**, 0.305→**0.309**, 0.307→**0.330**, 0.271→**0.409**, 0.319→**0.375** — **balancing improves every single fold**, the most fold-consistent result in this step, and gives CREPE its best absolute Sloped-start/Sloped-end F1 anywhere (0.243/0.222, better than either four-class B1 or B2).

**Interpreting binary vs. three-way together (section 11)**: the intended contrast — "binary T2/T3 = high, 3-way = poor" vs. "both poor" — does not resolve cleanly here because the binary result is confounded by training instability (§9), not because the answer is ambiguous. What the *reliable* evidence shows: (a) on the folds where the binary CNN actually trained, it performs at roughly the sign-test's level (comparable to "high" for this dataset), and (b) the 3-way task, freed only of Fixed, still tops out at 0.24/0.22 F1 for Sloped-start/Sloped-end even with balancing — clearly below the binary folds' performance and far below oracle. This is closest to the spec's first framing: **CREPE preserves usable front-loaded-vs-back-loaded information (consistent with the 76.8% sign test and the well-trained binary folds), but Sloped-start/Sloped-end remain difficult to cleanly separate from noisy Cosine contours specifically** — Cosine, not the Sloped-start/Sloped-end distinction itself, is the dominant source of remaining confusion, exactly as the B1/B2 confusion matrices already show (the largest single error mode in every condition is Cosine↔minority confusion, not Sloped-start↔Sloped-end confusion).

## 12. Fold consistency (four-class B1/B2 vs. B0)

| Fold | B0 | B1 | B2 |
|---|---:|---:|---:|
| 0 | 0.299 | 0.473 | 0.493 |
| 1 | 0.276 | 0.295 | 0.305 |
| 2 | 0.295 | 0.302 | 0.236 |
| 3 | 0.276 | 0.199 | 0.234 |
| 4 | 0.306 | 0.348 | 0.322 |

B1 vs. B0: 4/5 folds improve, 1 worsens (fold 3, Δ−0.078), median Δ **+0.019**.
B2 vs. B0: 3/5 folds improve, 2 worsen (folds 2 and 3, Δ−0.059/−0.042), median Δ **+0.016**.

Real, directionally positive, but modest and not unanimous — fold 0 alone shows a large gain for both (+0.17 to +0.19) that pulls the mean up substantially; the other folds move by only a few points either way. **Not a one-fold result** (the median is positive and computed correctly across all 5), but not the "reasonable fold consistency" bar `CLASS_BALANCING_RECOVERS_SHAPE_CLASSES` requires either.

## 13. Per-recording consistency

B1 vs. B0: 9 recordings improved, 8 worsened, median Δ +0.010.
B2 vs. B0: 9 recordings improved, 8 worsened, median Δ +0.008.

Essentially a coin flip at the recording level despite the positive fold-level medians — the pooled/grouped-mean gains are not a general, recording-independent improvement; they are concentrated unevenly (consistent with §12's fold-0-driven pattern). This argues against a strong, general "balancing solves the learning problem" story and toward the more measured trade-off interpretation.

## 14. Confidence analysis for true T2/T3

| Condition | True Sloped-start: mean P(true) / P(Cosine) | True Sloped-end: mean P(true) / P(Cosine) |
|---|---|---|
| B0 | 0.061 / 0.702 | 0.055 / 0.709 |
| B1 | 0.312 / 0.275 | 0.285 / 0.273 |
| B2 | 0.270 / 0.276 | 0.279 / 0.263 |

Under B0, true-T2 and true-T3 examples are assigned almost no probability mass on their own class (medians 0.05-0.06, an order of magnitude below the 0.25 uniform-chance level) while Cosine absorbs the great majority (median 0.74-0.78) — in isolation this superficially resembles spec section 14's "Case B" (representation failing fundamentally). But B1/B2, using the *identical* architecture and features, raise mean P(true class) roughly **5x** (to 0.27-0.31), now statistically tied with P(Cosine) rather than dominated by it — this is only possible if the representation already carried the relevant signal and the unweighted objective was suppressing its use, i.e. **Case A dynamics revealed by the balancing intervention itself**, even though B0 alone looks like Case B. The honest reading is that B0's raw confidence numbers are a symptom of the training objective, not proof of a representation ceiling.

## 15. Comparison with the semantic sign-baseline

Sign-test T2-vs-T3 accuracy (Step 22 §18): **76.8%**. Binary CNN pooled accuracy: 52.99% (misleading, per §9's fold breakdown — driven by 2/5 degenerate folds). Binary CNN accuracy on the 3 folds where training proceeded normally: 64.4%, 48.5%, 75.9% (mean 62.9%) — below 76.8% but not by the dramatic margin the pooled number suggests, and fold 4 alone slightly exceeds it. **Conclusion: the learned model does not clearly beat the trivial 1-D statistic on this task, and on the evidence available cannot be said to reliably lose information relative to it either** — the honest result is "roughly comparable, with a training-stability problem serious enough that a confident comparison isn't yet possible." Per section 15, this is flagged rather than used to motivate a more complex model.

## 16-18. Scope discipline

No CREPE smoothing/denoising/pitch-correction was applied (frozen exactly, per section 16). No pitch-frontend, CREPE-model, or vocals/source-audio variant was touched (section 17; CREPE remains frozen infrastructure per Step 21). GT boundaries remain frozen; no boundary detector, segmental decoding, or DP segmentation was implemented or proposed as part of this step's own work (section 18).

## 19. Primary scientific question

> Was Step 22's zero-F1 Sloped-start/Sloped-end failure primarily caused by class imbalance / the unweighted objective?

**Partially, and demonstrably so, but not solely.** Every balancing-sensitive diagnostic moved in the same direction and by a large margin: prediction frequency (0%→24-30% each for T2/T3), precision above base rate (not a trivial recall-only flip), and mean confidence (5x increase, tying Cosine). This conclusively shows the exact-zero F1 in Step 22 was an objective/class-prior artifact, not a hard information ceiling — `CLASS_IMBALANCE_NOT_PRIMARY` is ruled out. But the recovered performance is modest and inconsistent at the fold/recording level (§12-13), Cosine pays a real cost (§6, §8), and even the most favorable balanced condition (3-way M1) tops out at F1 0.24/0.22 for Sloped-start/Sloped-end — far below oracle's 0.96/1.00 and below even the raw sign-test's 76.8% ceiling. Class imbalance was a real, load-bearing part of the problem; it is not the whole problem. What remains — per §11's Cosine-dominated confusion pattern — looks like genuine class overlap between noisy CREPE Cosine contours and noisy CREPE Sloped-start/Sloped-end contours, not a lack of front-loaded-vs-back-loaded signal per se (§10-11, §15).

## 20. Interpretation — primary outcome

**`CLASS_BALANCING_REVEALS_TRADEOFF`**

T2/T3 move substantially and measurably (F1 0.000→0.08-0.24 across every condition tested, with real above-base-rate precision and a 5x confidence shift) while Cosine degrades substantially (F1 0.82→0.46-0.55 depending on condition), and four-class macro F1 improves only modestly for B1 and *declines* pooled for B2 — squarely `CLASS_BALANCING_REVEALS_TRADEOFF`'s definition: "minority-class information exists, but current features/model do not separate all four classes cleanly enough." `CLASS_BALANCING_RECOVERS_SHAPE_CLASSES` is ruled out by the modest/inconsistent net macro-F1 gain and the merely-coin-flip recording-level consistency (§13). `CLASS_IMBALANCE_NOT_PRIMARY` is ruled out by the uniformly strong, mechanistically-consistent response to both balancing interventions (§7-8, §14). `T2_T3_SIGNAL_EXISTS_BUT_COSINE_OVERLAPS` is a close secondary reading — the confusion-matrix pattern (§11) and the sign-test comparison (§15) both point at Cosine overlap specifically as the dominant remaining error mode — but is not selected as primary because its own precondition ("T2-vs-T3 binary classification is strong") cannot be confirmed cleanly given §9's training instability.

## 21. Decision gate

**`INVESTIGATE_ROBUST_CREPE_SHAPE_REPRESENTATION`**

Usable Sloped-start/Sloped-end information demonstrably exists in CREPE's normalized contour (§7-8, §14-15) and the dominant remaining error mode in every balanced condition is confusion with Cosine specifically (§6, §8, §11), not a lack of front-loaded-vs-back-loaded signal — exactly the precondition this gate names. Two standard objective-level fixes (balanced sampling, inverse-frequency weighted CE) were tried per spec and both hit the same Cosine-overlap ceiling rather than diverging in a way that would motivate a third, more elaborate loss (which would point at `INVESTIGATE_CLASS_OBJECTIVE` instead) — the bottleneck has shifted from "the model won't look at T2/T3" (Step 22's finding) to "T2/T3 and Cosine are hard to tell apart in CREPE's own signal" (this step's finding), which is a representation-robustness question, not a further training-recipe question.

## Recommendation for Step 24

Step 22 §16 explicitly deferred CREPE-contour smoothing/denoising specifically so this class-prior question could be answered cleanly first; that condition is now met. Step 24 should test a light, fixed, non-learned smoothing or denoising of the CREPE contour (e.g. a small moving-average or median filter applied before `q(x)`/`dq/dx` extraction — a trajectory-input preprocessing choice, not a pitch-frontend change, consistent with Step 21's own scope boundary) under the SAME balanced training protocol established here (B1, balanced sampling — the more evenly-recovering of the two interventions per §7-8's Sloped-start/Sloped-end symmetry) so that only the representation changes next, mirroring this step's single-variable discipline. Question to answer: does reducing CREPE's frame-to-frame jitter shrink the Cosine/Sloped-start/Sloped-end confusion identified in §11, or is the overlap intrinsic to what CREPE's pitch estimate captures regardless of smoothing? Only after that should `MOVE_TO_SEGMENTAL_TRANSCRIPTION` be revisited.
