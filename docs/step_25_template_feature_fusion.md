# Step 25 — Do Canonical Template Residuals Add Complementary Information?

Step 24 showed the four canonical trajectory shapes are known parametric curves whose hard `argmin`-template classification is strong on oracle contours (macro F1 ≈0.849) but worse than Step 23's balanced CNN on CREPE (≈0.241 vs. ≈0.311) — and that the two methods fail in *opposite* directions along the same Cosine↔Sloped-start/Sloped-end axis (the CNN collapses toward Cosine; template argmin over-predicts the curved templates). Step 24 also produced a reusable per-primitive four-template error vector. This step asks the natural next, and per spec the *final*, small-feature question: **do the template residuals carry information the balanced ContourCNN doesn't already extract from `q(x)+dq/dx`, even though the hard argmin decision rule is a poor classifier on its own?**

Frozen exactly from Steps 22-24: CREPE extraction, `MIN_SPAN_CENTS`, interpolation, GT boundaries, phase normalization, `N=64`, grouped 5-fold manifest, canonical labels, and Step 24's `template_errors` API (unmodified). Training protocol frozen from Step 23's B1 (the better, more symmetric balancing intervention): class-balanced sampling, unweighted cross entropy, `ContourCNN` backbone, optimizer/LR/WD/batch/epochs/patience/seeds all unchanged.

Machine-readable outputs: `output/shape_classification/step25/results.json`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step25_experiments
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| F0 (contour-only, reused Step 23 B1 protocol) reproduces Step 23 exactly: macro F1 0.3107, grouped mean 0.3234±0.0893 | §5 |
| Normalized template evidence `z_k=(E_k-min E)/(mean E+ε)` is well-behaved: no NaN/Inf, range [0, 3.35], and by-class medians match Step 24's argmin recall pattern exactly (e.g. median `z_sloped_start=0.000` for true Sloped-start, consistent with its 52.4% argmin recall) | §4 |
| **F2 (contour+template fusion) is statistically indistinguishable from F0**: macro F1 0.3115 vs. 0.3107 (Δ+0.0008), grouped mean 0.3224±0.0683 vs. 0.3234±0.0893 — for +16 parameters (2852 vs. 2836) | §9 |
| Fold consistency is a wash: 3/5 folds improve, 2/5 worsen, with two large opposing swings (fold 3: +0.118, fold 4: −0.109) that cancel almost exactly; median Δ+0.004 | §10 |
| Recording consistency is a near coin-flip: 9 improved / 8 worsened, median Δ+0.004, **mean Δ−0.004** (slightly negative) | §11 |
| **Decisive negative sanity check**: zeroing `z` at test time (same trained F2 weights, no retraining) gives macro F1 0.3177 — *higher* than F2 with real z (0.3115) and higher than F0 itself (0.3107) — despite the trained head assigning `z` larger average weight magnitude than the contour features (ratio 1.14-1.68x across folds) | §15 |
| F1 (template evidence alone, `Linear(4→4)`, 20 params) scores macro F1 **0.2107 — worse than Step 24's own hard argmin (0.241)**, not comparable to or better than it | §6, §17 |
| Prediction frequency: F2 does not move meaningfully closer to the true class distribution than F0; if anything Sloped-start over-prediction increases slightly (25.5%→28.5%, true rate 6.5%) | §13 |
| **Oracle control tells a different story**: F1-oracle (0.862) > F2-oracle (0.844) > F0-oracle (0.819) — on clean contours, template evidence is genuinely, substantially informative, even *more* so alone than the raw-contour CNN. Template information is not fundamentally redundant; it is CREPE's specific noise that destroys its value | §18 |

**Primary outcome: `TEMPLATE_FEATURES_REDUNDANT`** (on CREPE, the realistic condition this step must decide on)

**Decision gate: `STOP_INCREMENTAL_CONTOUR_FEATURE_ENGINEERING`**

---

## 1-2. Frozen upstream and training protocol

Data: Step 22's corpus (`dataset.build()`), untouched. Training: Step 23's B1 protocol exactly — `ContourCNN`, Adam (lr 1e-3, wd 1e-4), batch 32, max 100 epochs / patience 15, seed 42+fold, class-balanced `torch.multinomial` sampling from TRAIN labels only, unweighted `CrossEntropyLoss`, grouped 5-fold `grouped_kfold_k5_seed42.json`. F0 is literally `training.shape_classification.step23_train.run_condition(..., balancing="sampler")`, not a reimplementation — guaranteeing exact reproduction.

## 3. Template scorer

`training.shape_classification.templates.template_errors` reused with zero modification (same `slope=2.0` templates, same MSE scoring, same CREPE contour preprocessing as Step 24). The hard-argmin prediction itself is never used as a feature — only the four raw errors `[E_fixed, E_cosine, E_sloped_start, E_sloped_end]`.

## 4. Normalized template-evidence formula and sanity check

```
m = min_k E_k
d = mean_k E_k + eps        (eps = 1e-6, numerical stability only)
z_k = (E_k - m) / d
```

Fixed before evaluating any Step 25 result; no alternative formula tried (`training/shape_classification/step25_features.py`).

**Sanity check** (7,177 primitives, CREPE): 0 NaN, 0 Inf, global range `[0.0, 3.35]`. By-class medians:

| True class | median `z_fixed` | median `z_cosine` | median `z_sloped_start` | median `z_sloped_end` |
|---|---:|---:|---:|---:|
| Fixed | 0.404 | 0.193 | 0.048 | 0.169 |
| Cosine | 1.567 | 0.180 | 0.231 | 0.198 |
| Sloped-start | 2.430 | 0.129 | **0.000** | 0.399 |
| Sloped-end | 1.681 | 0.221 | 0.643 | **0.021** |

The bolded near-zero medians for each class's own template exactly reproduce Step 24's argmin recall pattern (Sloped-start template is literally the best fit for the median true-Sloped-start primitive; Cosine's own median `z_cosine=0.180`, not 0, matches its poor 21.6% argmin recall). Pairwise margins (`z_cosine - z_sloped_start` etc.) were inspected as a diagnostic cross-check only, per spec, and are algebraically redundant with `z` — not added as separate features.

## 5-6. F0 reproduction and F1 (template evidence only)

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 (contour, Step 23 B1) | 0.3107 | 0.379 | 0.459 | 0.202 | 0.203 | 0.3234 ± 0.0893 | 2,836 |
| F1 (`z` only, `Linear(4→4)`) | 0.2107 | 0.272 | 0.269 | 0.151 | 0.151 | 0.2089 ± 0.0278 | 20 |

F0 reproduces Step 23 B1 exactly (same code path). **F1 is worse than F0 on every class**, and — the more informative comparison — **worse than Step 24's own deterministic hard-argmin template classifier** (0.211 vs. 0.241). This does not match either of section 17's anticipated "F1 > argmin" or "F1 ≈ argmin" cases; a learned linear read-out over balanced-sampled `z`, under this exact protocol, underperforms the simple deterministic rule it was meant to improve on. Reported as a genuine negative finding, not smoothed into the closest anticipated bucket.

## 7-9. F2 (fusion) and the central table

F2: same `ContourCNN` backbone as F0, `z` concatenated to the pooled contour representation immediately before the final linear layer (`Linear(hidden+4, 4)` replacing `Linear(hidden, 4)` — no other architectural change, per section 6).

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| F0 contour | 0.3107 | 0.379 | 0.459 | 0.202 | 0.203 | 0.3234 ± 0.0893 |
| F1 template evidence | 0.2107 | 0.272 | 0.269 | 0.151 | 0.151 | 0.2089 ± 0.0278 |
| **F2 contour + template evidence** | **0.3115** | 0.403 | 0.440 | 0.198 | 0.205 | 0.3224 ± 0.0683 |
| Step 24 hard template argmin | 0.2408 | 0.291 | 0.339 | 0.152 | 0.181 | 0.2315 ± 0.0233 |
| Oracle reference (Step 22 CNN) | 0.8014 | 0.450 | 0.805 | 0.955 | 0.996 | 0.8204 ± 0.0772 |

**Parameter-count comparison**: F0 has 2,836 params, F2 has 2,852 — a difference of exactly **16** (the `4×4` new weight block connecting `z` to the 4-class head; `hidden=16`). A negligible capacity increase.

**The important comparison (F2 vs. F0)**: macro F1 differs by **+0.0008** pooled and **−0.0010** grouped mean — both far smaller than fold-to-fold noise (grouped std ≈0.07-0.09). Per-class: Fixed +0.024, Cosine −0.019, Sloped-start −0.004, Sloped-end +0.002 — no class shows a change large enough to read as a real effect given the noise floor established across Steps 22-24.

## 10. Fold consistency

| Fold | F0 | F2 | F2−F0 |
|---|---:|---:|---:|
| 0 | 0.473 | 0.448 | −0.025 |
| 1 | 0.295 | 0.302 | +0.007 |
| 2 | 0.302 | 0.306 | +0.004 |
| 3 | 0.199 | 0.317 | **+0.118** |
| 4 | 0.348 | 0.239 | **−0.109** |

(F0 here is identical to Step 23's B1 per-fold table, since F0 reuses that exact code path.) F2 improves 3/5 folds, worsens 2/5, median Δ **+0.0042**. The two largest deltas (fold 3: +0.118, fold 4: −0.109) point in **opposite directions and nearly cancel** — exactly the "large gain in one fold is not sufficient evidence" case section 10 warns about, and here it is not even a single outlier fold driving a real aggregate effect; it is two outliers cancelling into noise.

## 11. Recording consistency

9 recordings improved, 8 worsened, median Δ **+0.0044**, **mean Δ −0.0040** (slightly negative). A coin flip, structurally identical to Step 23's B1-vs-B0 recording-level pattern (also 9/8) — reinforcing that pooled/grouped-mean gains in this feature-engineering branch are consistently concentrated in a minority of folds/recordings rather than reflecting a general effect.

## 12. Per-class attribution

Fixed improves modestly (+0.024) and Cosine degrades modestly (−0.019) — the opposite of the "useful result" pattern section 12 describes (T2/T3 up, Cosine stable). Sloped-start/Sloped-end themselves barely move (−0.004/+0.002). This is not "T2/T3 up, Cosine down" (a tradeoff) either — it is uniformly small movement in every class, consistent with §9's "no real effect" reading rather than any class-shifting tradeoff.

## 13. Prediction-frequency comparison

| | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| True | 18.2% | 69.0% | 6.5% | 6.3% |
| F0 | 19.5% | 30.4% | 25.5% | 24.6% |
| F1 | 23.7% | 20.2% | **46.9%** | 9.2% |
| F2 | 18.9% | 27.7% | 28.5% | 24.8% |

F2 does **not** clearly move closer to the true distribution than F0 — Fixed improves marginally (19.5%→18.9%, true 18.2%), but Cosine moves further from true (30.4%→27.7%, true 69.0%) and Sloped-start moves further from true too (25.5%→28.5%, true 6.5%). F1 alone shows a dramatic, Step-24-like over-prediction of Sloped-start (46.9%, over 7x its true rate) — the linear template read-out inherits template argmin's own curved-template bias even more strongly than raw argmin did.

## 14. Confusion matrices

(Full matrices in `results.json`; the interpretively relevant cells match §12/§13's findings — Cosine→Sloped-start/Sloped-end and Sloped-start/Sloped-end→Cosine remain the dominant error modes in both F0 and F2, essentially unchanged in relative size. Adding `z` does not visibly shrink this shared confusion axis; it reshuffles a small number of individual decisions without changing its overall magnitude, consistent with §9's near-zero net effect.)

## 15. Template-feature-use sanity check

**Weight magnitude** (not causal, a sanity check only): mean |weight| on the 4 `z`-columns of the final linear layer *exceeds* the mean |weight| on the 16 contour-feature columns in every fold (ratio 1.14-1.68x) — naively, the trained model assigns real magnitude to `z`.

**Zeroing `z` at test time** (same trained weights, no retraining) is the decisive test: F2 normal = 0.3115, **F2 with `z` zeroed = 0.3177** — z-zeroed is *higher* than F2 with real `z`, and also higher than F0 itself. Despite non-trivial learned weight magnitude on `z`, using the real template evidence at inference is net *harmful* relative to not using it at all. The most defensible reading (without over-interpreting raw weight magnitude, per spec's own caution): the training-time gradient signal from `z` may have had some incidental regularizing effect on the shared contour pathway, but the *feature itself*, at test time, adds noise rather than signal — the opposite of the pattern (`F2 normal > F2 z-zeroed ≈ F0`) that would support real complementary use.

## 16. Representative changed decisions

596 primitives where F0 was wrong and F2 was correct; 661 where F0 was correct and F2 was wrong — **more decisions get worse than get better**, consistent with §9's near-zero (very slightly negative in some framings) net effect.

Deterministic examples (`results.json::representative_changed_decisions`):

- **F0 wrong → F2 correct** (true Cosine, `z=[2.79, 0.00, 0.68, 0.10]`, template argmin=Cosine): F0 alone predicted Sloped-end; F2 correctly used the strong, genuinely-informative `z_cosine=0` signal to predict Cosine. This is exactly the intended complementary-use case, and it does happen.
- **F0 correct → F2 wrong** (true Cosine, `z=[0.18, 0.26, 0.77, 0.00]`, template argmin=Sloped-end): F0 alone correctly predicted Cosine from the contour; F2 was misled by `z`'s confidently-wrong signal (`z_sloped_end=0`, the template argmin itself wrong here, inheriting exactly Step 24's known Cosine-under-recognition failure mode) into predicting Sloped-end instead.

Both failure directions are real and roughly balanced in frequency (596 vs. 661) — `z` helps exactly when the template argmin happens to be right and the contour alone was not enough, and hurts when the template argmin is confidently wrong (which, per Step 24, is common for Cosine specifically) and overrides a contour signal that was already correct.

## 17. F1 interpretation

F1 (0.2107) is neither "≈ hard argmin" (0.2408) nor ">  hard argmin" — it is **meaningfully worse**, and also clearly worse than F0 (0.3107). Under the exact frozen balanced-sampling protocol, learning a 20-parameter linear decision rule over normalized template evidence alone is a worse classifier than either the deterministic argmin rule or the raw-contour CNN. This rules out the "compact geometric residual representation may be better than the raw contour representation" possibility section 17 flags as the surprising case — the opposite occurred.

## 18. Optional oracle control

| Condition | Macro F1 (pooled) | Grouped mean ± std |
|---|---:|---:|
| F0-oracle | 0.8187 | 0.8383 ± 0.0383 |
| **F1-oracle** | **0.8618** | **0.8684 ± 0.0424** |
| F2-oracle | 0.8436 | 0.8563 ± 0.0312 |

On oracle contours, the ordering **inverts relative to CREPE**: F1-oracle (template evidence alone) is the *best* of the three, exceeding even F0-oracle by +0.043, and F2-oracle sits between the two (+0.025 over F0-oracle). This distinguishes the two candidate explanations section 18 poses cleanly: template evidence is **not** fundamentally redundant information — on clean contours it is highly informative, more so alone than what the CNN extracts from the raw 64-point sequence. It is specifically **CREPE's distortion of the template residuals** (already characterized in Step 24: large systematic bias, weak margins, real-but-modest endpoint-error correlation) that destroys their value, flipping them from the *best* single feature (oracle) to actively harmful at inference time (CREPE, §15). This oracle result is reported as supporting evidence for *why* the CREPE result looks the way it does — it does not change the primary, CREPE-based outcome selection below, per spec's own instruction.

## Primary scientific question

> Do normalized canonical-template residuals provide trajectory information beyond what the balanced ContourCNN can already infer from `q(x)+dq/dx`?

**On CREPE — no.** F2 vs. F0 macro F1 differs by an amount (+0.0008 pooled, −0.0010 grouped mean) far smaller than ordinary fold-to-fold noise (±0.07-0.09 grouped std); fold consistency shows two large, opposing, cancelling swings rather than a real effect; recording consistency is a coin flip with a slightly negative mean; per-class movement is small and non-directional; prediction frequency does not clearly improve; and the decisive z-zeroed-at-test-time ablation shows the trained model does *better* without the feature it was given, despite assigning it real weight magnitude. F1 supports this: learning directly over `z` performs worse than even Step 24's simple deterministic argmin. The oracle control shows the *reason* is not that template geometry is inherently uninformative (it is the single best feature on clean contours) but that CREPE's specific noise characteristics degrade `z` enough to make it a net-negative input once fused with an already-reasonably-capable contour model.

## Outcome

**`TEMPLATE_FEATURES_REDUNDANT`**

On CREPE — the condition this outcome selection must be based on — F2 ≈ F0 within ordinary fold/recording variability (§9-11), and the z-zeroed ablation (§15) is the clean, decisive confirmation: the CNN already extracts essentially all the CREPE-usable information the template residuals represent, and supplying them explicitly adds noise rather than net signal. `TEMPLATE_FEATURES_ADD_COMPLEMENTARY_SIGNAL` and `TEMPLATE_FEATURES_REVEAL_CLASS_TRADEOFF` are ruled out by the near-zero, non-directional per-class movement (§12); `TEMPLATE_FEATURES_HURT` is not quite selected instead because F2's pooled macro F1 is not *consistently* below F0's (it is marginally above pooled, marginally below grouped-mean, i.e. a wash rather than a consistent regression) — `REDUNDANT` is the more precise description of a genuine null result.

## Decision gate

**`STOP_INCREMENTAL_CONTOUR_FEATURE_ENGINEERING`**

Per spec's own accounting, this branch has now tested fixed-time motion (Step 15/17/18), normalized contours (Step 22), velocity (Step 22 §10), class balancing (Step 23), canonical template argmin (Step 24), and template-residual fusion (Step 25) — six increasingly specific representations of the same underlying CREPE pitch-contour evidence, converging on the same ceiling (CREPE oracle-boundary four-class macro F1 ≈0.30-0.31) regardless of representation. No further small hand-engineered contour feature is recommended.

## Recommendation for Step 26

Per spec, do not implement this in Step 25 — but the concrete direction the evidence points to is explicit in the spec's own framing and directly supported by this step's findings: since six contour-only representations (learned and template-based alike) have plateaued at the same macro F1 ceiling on CREPE despite oracle contours reaching 0.80-0.86 under the identical representations, the bottleneck is very unlikely to be resolved by a seventh contour feature. Step 26 should ask the larger modeling question the spec itself names: **whether pitch alone (in any of the representations tested across Steps 21-25) is sufficient for reliable Cosine/Sloped-start/Sloped-end discrimination from CREPE, or whether trajectory classification needs additional acoustic/contextual information** (e.g. features derived from the audio directly rather than only its extracted pitch, or sequence context from neighboring trajectories) to break past the ceiling this entire branch has now independently confirmed from multiple angles.
