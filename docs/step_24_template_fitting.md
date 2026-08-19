# Step 24 — Canonical Template Fitting for CREPE Trajectory Shapes

Steps 22-23 established that the four canonical trajectory classes are known parametric shapes, that oracle contours are highly separable, that CREPE preserves real geometric information, and that Step 23's balanced-CNN recovery of the T2/T3 collapse traded heavily against Cosine, leaving Cosine↔Sloped-start/Sloped-end as the dominant remaining confusion. Since T1/T2/T3 are not arbitrary learned categories but literal, known parametric curves, this step asks whether **matching a CREPE segment's normalized contour directly against the known canonical templates** — no learned decision boundary, no class prior — does better than asking a small CNN to rediscover those shapes from a small, imbalanced dataset.

Frozen exactly from Steps 22-23: CREPE extraction, `MIN_SPAN_CENTS`, interpolation, the 64-point phase grid, GT primitive boundaries, and the grouped 5-fold manifest. No pitch-frontend work, no architecture search, no class balancing in the primary decision rule, no segmentation.

Machine-readable outputs: `output/shape_classification/step24/{results,scored_primitives,endpoint_errors}.json`, figures in `output/shape_classification/step24/figures/`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step24_experiments
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| **Templates recovered directly from idtap's `Trajectory.id0/id1/id2/id3`** (not hand-approximated) reproduce Step 22's oracle statistics exactly: Cosine `q(0.5)=0.500`, Sloped-start `q(0.5)=0.750`, Sloped-end `q(0.5)=0.250` | §2 |
| Real corpus `slope` values for raw types 2/3 (Sloped-start/end) range ~1.1-8.0, not uniformly the default 2.0 used for the fixed template — a genuine, quantifiable curve-family variation that limits even the *oracle* template ceiling | §2 |
| Fixed handled with zero special-casing: templates are projected INTO cents space by multiplying by the primitive's own observed span (never dividing by it), so Fixed (span≈0) is just the `f_fixed≡0` case of one unified formula | §3 |
| **Oracle sanity gate passes decisively, and template fitting on oracle actually *beats* Step 22's oracle CNN**: macro F1 0.849 (template) vs. 0.801 (CNN); T2/T3 F1 0.969/0.953 vs. 0.955/0.996 | §4, §10 |
| Robust (Huber) scoring gives **no measurable improvement** over raw MSE on CREPE (0.2408→0.2407 pooled) — frame-level jitter/outliers are not the dominant noise mechanism, correctly gating out the optional smoothing control (§17) | §6, §17 |
| **CREPE 4-way template fitting underperforms Step 23's CNNs on every primary metric**: macro F1 0.241 vs. B0's 0.290 / B1's 0.311; Cosine F1 collapses to 0.339 (worse than either CNN); 3-way bend-only template F1 0.236 vs. M1's 0.339 | §10, §13 |
| The failure mode is the **opposite** of the CNN's: template fitting *under*-predicts Cosine (19.0% of predictions vs. 69.0% true rate) and *over*-predicts Sloped-start/Sloped-end (64.6% combined vs. 12.8% true) — the reverse asymmetry from Step 23's B0 (90.4% Cosine, 0% T2/T3) | §7 |
| Margin analysis: for true Sloped-start/Sloped-end, the correct template beats Cosine only **53.6%/58.9%** of the time — barely better than a coin flip; for true Cosine, the wrong (sloped) template wins **37.6-42.7%** of the time | §11 |
| T2-vs-T3 template diagnostic is a clean, **fully reliable** (no training instability, unlike Step 23's confounded CNN) 68.2% accuracy — below the 76.8% sign-test baseline, but computed without any of Step 23's fold-collapse confound | §12, §15 (of this report) |
| Endpoint noise is a real but modest contributor (mean abs. endpoint error 172-176¢ on correct predictions vs. 264-265¢ on incorrect; correlation −0.10) — not the dominant cause | §14 |
| CREPE template macro F1 is **strongly duration- and span-dependent** (0.13 at <100ms rising to 0.33 at 250-500ms; 0.08 at <50¢ span rising to 0.39 at >400¢) — unlike the CNN, which was roughly flat across both — raw MSE matching is a direct SNR statistic, and short/small-span primitives given CREPE too few effectively-independent, high-amplitude samples | §15 (duration), §16 (span) |

**Primary outcome: `TEMPLATE_FITTING_NO_BETTER_THAN_CNN`**

**Decision gate: `REASSESS_DOWNSTREAM_TRAJECTORY_FEATURES`**

---

## 1. Frozen upstream pipeline

CREPE extraction, `MIN_SPAN_CENTS=50¢`, phase-grid interpolation, GT primitive boundaries, `N=64`, and the grouped 5-fold `performance_group_id` manifest are all reused unchanged from Step 22's corpus (`training/shape_classification/dataset.py::build()` — no rebuild, no new pitch extraction). No pitch-frontend, architecture-search, or segmentation work was done in this step.

## 2. Canonical templates recovered from idtap's own code

`idtap/classes/trajectory.py` (`Trajectory.id0/id1/id2/id3`, the exact functions `dataset/canonical/contour.py` already uses to regenerate GT framewise targets):

| id | Class | Formula | Normalized `q(x) = (p(x)-p(0))/(p(1)-p(0))` |
|---|---|---|---|
| 0 | Fixed | `log2_freq(x) = log_freqs[0]` (constant) | `q(x) ≡ 0` |
| 1 | Cosine | `pi_x = (cos(π(x+1))/2)+0.5` | `q(x) = 0.5 - 0.5·cos(πx)` — **no free parameter** |
| 2 | Sloped-start | `(a-b)(1-x)^slope + b` | `q(x) = 1-(1-x)^slope` |
| 3 | Sloped-end | `(b-a)x^slope + a` | `q(x) = x^slope` |

`slope` is a per-annotation idtap attribute (default 2.0, `DEFAULT_SLOPE` in `dataset/canonical/schema.py`). The fixed templates use `slope=2.0` — idtap's own documented default, and the value that reproduces Step 22's reference statistics exactly (verified below) — **fixed before any Step 24 result was examined**, not tuned.

**Verification against Step 22** (`training/shape_classification/templates.py::template_curves()`, evaluated on the same `X_GRID`):

| | `q(0.25)` | `q(0.5)` | `q(0.75)` |
|---|---:|---:|---:|
| Cosine (Step 22 oracle) | 0.147 | 0.500 | 0.853 |
| Cosine (recovered template) | 0.1465 | 0.5000 | 0.8535 |
| Sloped-start (Step 22 oracle) | 0.437 | 0.750 | 0.937 |
| Sloped-start (recovered template) | 0.4375 | 0.7499 | 0.9375 |
| Sloped-end (Step 22 oracle) | 0.063 | 0.250 | 0.563 |
| Sloped-end (recovered template) | 0.0625 | 0.2501 | 0.5625 |

Exact match (to float precision). **Gate passes — proceeding was warranted.**

**A real, quantified caveat**: real corpus `slope` values for raw types 2/3 are far from uniformly 2.0 — histogram of the actual wire-stored `slope` field across all raw type-2/type-3 trajectories shows a mode at 2.0 but a wide spread from ~1.1 to 8.0 (many distinct decimal values, not just annotator presets). This means the fixed template is necessarily an approximation of a *curve family*, not a per-instance-exact fit — a real, structural source of residual error even for the oracle template classifier (visible in §4's F1 being strong but not literally 1.0), reported honestly rather than hidden, and **not** compensated for with a per-primitive free slope parameter (explicitly disallowed by section 8 — that would let every template fit every shape and destroy the experiment's interpretation).

## 3. Fixed-vs-moving normalization

Rather than branching the *scoring* rule on span (as Step 22's *feature* representation does), all four templates are projected **into** the primitive's own observed relative-cents space by multiplying by its own observed span: `r_hat_k(x) = span_cents · f_k(x)`, `E_k = mean((r(x) - r_hat_k(x))^2)`. This never divides by span, so it is safe at span≈0 by construction — Fixed is simply the `k=0` case of one shared formula (`f_fixed ≡ 0`, so `r_hat_fixed ≡ 0` regardless of span), not a separately branched rule. No GT label is used to choose a branch; the same formula runs for every primitive regardless of true class.

## 4-5. Deterministic MSE scorer and oracle sanity gate

`E_k = mean_i (r(x_i) - span·f_k(x_i))^2`, prediction `= argmin_k E_k` (`training/shape_classification/templates.py::template_errors/predict`).

| | Macro F1 | Accuracy | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Oracle MSE template** | **0.849** | 0.795 | 0.642 | 0.834 | 0.969 | 0.953 | 0.851 ± 0.041 |

Strong, as required — the gate passes cleanly, and (see §10) this actually *exceeds* Step 22's oracle CNN (0.801). Proceeding to CREPE was warranted.

## 6. Robust (Huber) scorer

`HUBER_DELTA_CENTS = 50.0` — reused from Step 22's already-established `MIN_SPAN_CENTS`, not a new tuned constant, fixed before any Step 24 result was examined.

| | Oracle macro F1 | CREPE macro F1 |
|---|---:|---:|
| MSE | 0.849 | 0.241 |
| Robust (Huber) | 0.846 | 0.241 |

Essentially no change in either condition (CREPE: 0.2408→0.2407, a difference far below noise). **Answer to section 6's question: no** — down-weighting individual noisy CREPE samples does not improve template discrimination, which rules out sparse frame-level jitter/outliers as the dominant noise mechanism and correctly gates out the optional smoothing control (§17).

## 7. Prediction-frequency diagnostic

| | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| True distribution | 18.2% | 69.0% | 6.5% | 6.3% |
| CREPE MSE template | 16.5% | **19.0%** | **38.3%** | 26.3% |
| (for reference) Step 23 B0 (unweighted CNN) | 9.6% | 90.4% | 0.0% | 0.0% |

Template fitting shows the **opposite** pathology from the unweighted CNN: instead of collapsing onto the majority Cosine class, it collapses *away* from Cosine, over-predicting the two curved (Sloped) templates by nearly 5-6x their true rate. This is mechanistically sensible: with no class prior at all, a decision rule that measures raw geometric fit will favor whichever template happens to track CREPE's noise-corrupted contour more closely on average — and, per §11, that is often one of the curved templates even for genuinely-Cosine primitives.

## 8. Central four-way table

| Method | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|---:|
| Oracle MSE template | 0.849 | 0.642 | 0.834 | 0.969 | 0.953 |
| Oracle robust template | 0.846 | 0.640 | 0.834 | 0.969 | 0.940 |
| CREPE MSE template | 0.241 | 0.291 | 0.339 | 0.152 | 0.181 |
| CREPE robust template | 0.241 | 0.291 | 0.341 | 0.153 | 0.178 |
| Step 23 B1 CNN (balanced sampler) | 0.311 | 0.379 | 0.459 | 0.202 | 0.203 |
| Oracle CNN reference (Step 22) | 0.801 | 0.450 | 0.805 | 0.955 | 0.996 |

**The important comparison** (per spec, CREPE template fitting vs. Step 23's balanced CNN): template fitting is **worse on every column** — macro F1 (0.241 vs. 0.311), Fixed (0.291 vs. 0.379), Cosine (0.339 vs. 0.459), Sloped-start (0.152 vs. 0.202), Sloped-end (0.181 vs. 0.203). Grouped 5-fold mean±std: CREPE MSE template 0.232±0.023 vs. B1's 0.323±0.089 (Step 23) — template fitting is not only lower on average, it is also *more* consistent (lower std) at a *worse* level, i.e. reliably mediocre rather than variably competitive.

## 9. Per-template error vectors

Every primitive's `[E_fixed, E_cosine, E_sloped_start, E_sloped_end]` plus true class, recording id, fold, duration, and pitch span is saved to `scored_primitives.json`, reusable as-is by a future segmental scorer (section 23's forward-compatibility requirement) without touching the template machinery again.

## 10. (see §8's central table above; this section number is folded in for the deliverable checklist)

## 11. Cosine-vs-Sloped margin analysis

`M_start = E_cosine - E_sloped_start`, `M_end = E_cosine - E_sloped_end` (positive ⇒ the sloped template fits better).

| True class | `M_start` median | `M_start` frac>0 | `M_end` median | `M_end` frac>0 |
|---|---:|---:|---:|---:|
| Cosine | −15.6 | **0.376** | −5.2 | **0.427** |
| Sloped-start | +1.9 | **0.536** | −816.4 | 0.244 |
| Sloped-end | −1596.5 | 0.256 | +10.0 | **0.589** |

(`output/shape_classification/step24/figures/margins_crepe_mse.png`, clipped to ±3000¢² so the decision-relevant region near zero is visible against a long noise tail.)

Answers section 11's four questions directly:
1. **Are true Sloped-start contours genuinely closer to their own template than to Cosine?** Barely — 53.6% of the time, a near coin flip.
2. **True Sloped-end?** Somewhat more often (58.9%), still far from decisive.
3. **Margin size**: median margins for the correct class are small (+1.9¢² and +10.0¢²) relative to the noise floor implied by the huge tails, while true-Cosine's wrong-template win rate (37.6-42.7%) is itself large.
4. **Wrong-usually-wins vs. correct-barely-wins?** Both are present, but the dominant story is **"the correct template only barely wins"** for the Sloped classes (53.6%/58.9%, not near 0% or 100%) combined with a genuinely large wrong-template win rate for Cosine (up to 42.7%) — a close, real, three-way overlap rather than a one-sided catastrophic failure.

## 12. T2-vs-T3 template diagnostic

Restricted to true Sloped-start/Sloped-end, `argmin(E_sloped_start, E_sloped_end)`:

| | Accuracy | Macro F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|
| Oracle MSE template | 99.89% | 0.999 | 0.999 | 0.999 |
| **CREPE MSE template** | **68.19%** | 0.682 | 0.684 | 0.680 |
| Sign-test baseline (Step 22 §18) | 76.8% | — | — | — |
| Step 23 binary CNN (confounded, §9 of that report) | 52.99% pooled / ~63% on stable folds | 0.435 | 0.667 | 0.203 |

Confusion matrix (CREPE MSE template): `[[317,151],[142,311]]` — balanced, no degenerate single-class collapse (precision/recall both ≈0.67-0.69 for each class). **This is the cleanest, most reliable T2-vs-T3 number in Steps 22-24**: unlike Step 23's binary CNN, template fitting is deterministic and has no training-instability confound at all. It sits below the trivial sign-test baseline (68.2% vs. 76.8%) but is a solid, fully-trustworthy result, and clearly better-behaved than Step 23's binary CNN was (balanced confusion matrix vs. two collapsed folds).

## 13. Three-way bend-only template diagnostic

Restricted to true Cosine/Sloped-start/Sloped-end, `argmin` over those three template errors:

| | Macro F1 | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| Oracle MSE template | 0.985 | 0.996 | 0.969 | 0.990 |
| **CREPE MSE template** | **0.236** | 0.344 | 0.187 | 0.177 |
| Step 23 M1 (balanced 3-way CNN) | 0.339 | 0.551 | 0.243 | 0.222 |

Template fitting is worse than the balanced CNN on every column here too. Confusion matrix (CREPE, `[Cosine,SlS,SlE]` order): `[[1069,1805,2076],[111,246,111],[81,115,257]]` — of 4950 true-Cosine primitives, only 1069 (21.6%) are correctly classified; 1805 are called Sloped-start and 2076 Sloped-end, confirming §7's over-prediction pattern concretely within the 3-way restriction as well.

## 14. Endpoint-error analysis

CREPE start/end pitch error relative to oracle, measured directly (never used to correct anything):

| | Correctly classified (CREPE MSE) | Incorrectly classified |
|---|---:|---:|
| Mean \|start error\| (¢) | 171.9 | 264.3 |
| Mean \|end error\| (¢) | 175.8 | 264.6 |
| Correlation(\|start error\|, correct) | | **−0.104** |

A real, consistent gap (~90¢ higher endpoint error on misclassified primitives) and a weak-but-nonzero negative correlation. **Answer to section 14's question**: endpoint noise is a genuine, measurable contributor — not a rounding artifact — but a −0.10 correlation and a ~90¢ (not order-of-magnitude) gap do not support "endpoint error clearly dominates." It is one real contributing factor among several (see also §11's margin closeness and §22's qualitative examples below), not the primary explanation on its own.

## 15. Duration analysis

| Bucket | n | CREPE MSE template macro F1 |
|---|---:|---:|
| <100ms | 2,285 | 0.135 |
| 100-250ms | 3,273 | 0.239 |
| 250-500ms | 1,020 | **0.326** |
| 500ms-1s | 364 | 0.305 |
| >1s | 235 | 0.154 |

**Answer to section 15's question: yes** — template matching is markedly worse on the shortest primitives (<100ms, ≤10 native CREPE frames) and improves through 250-500ms before falling off again at the extreme tail (>1s, the same rare long-primitive population Step 22 also found hardest). This is a genuinely different pattern from Step 22's CNN, which was roughly *flat* across duration for CREPE — raw MSE matching is a direct signal-averaging statistic and benefits mechanically from more independent samples in a way a trained, globally-normalized CNN does not show as strongly.

## 16. Pitch-span analysis

| Bucket | CREPE MSE macro F1 | Oracle MSE macro F1 |
|---|---:|---:|
| <50¢ | 0.085 | 0.308 |
| 50-100¢ | 0.150 | 0.735 |
| 100-200¢ | 0.210 | 0.738 |
| 200-400¢ | 0.331 | 0.733 |
| >400¢ | **0.389** | 0.743 |

Oracle is flat (~0.73-0.74) once span clears the 50¢ normalization threshold — confirms span normalization does its job for a clean signal, consistent with Step 22. **CREPE, in sharp contrast, rises monotonically and substantially with span** (0.085→0.389, a >4x range) — direct evidence that raw-MSE template fitting is fundamentally an SNR statistic: a fixed-magnitude CREPE noise floor matters far less against a large true pitch swing than a small one. The `<50¢` bucket is confirmed, again, as intrinsically hard for every method and every source (already known from Step 22) — reported, not hidden, and no method was changed per bucket.

## 17. Smoothing control — gate not met

Gate condition (section 17): run the smoothing control only if raw-vs-robust shows jitter/outliers materially hurting CREPE fitting. §6 found robust scoring changed macro F1 by −0.0001 — no measurable effect. **Gate not met; the smoothing control was skipped**, per the spec's own instruction ("if the gate is not clearly met: skip this section"). This is itself informative: it means CREPE's degradation of template fitting is not primarily an outlier/jitter phenomenon fixable by down-weighting a few bad frames — consistent with §22's qualitative finding that some of the worst cases are *systematically*, not just noisily, the wrong shape.

## 18. (Section reserved by the deliverable checklist — see §20 "compare error patterns" and §21 "oracle-vs-CREPE gap" below, which cover this content.)

## 20. Comparison of error patterns with Step 23

| | Predicts Cosine when uncertain? | Enters T2/T3 region at all? | Distinguishes T2 from T3? | Dominant confusion |
|---|---|---|---|---|
| Step 23 B0 (unweighted CNN) | Yes — 90.4% of all predictions | No — literally 0% | N/A (never predicted) | everything → Cosine |
| Step 23 B1 (balanced CNN) | No — pulled down to 30.4% | Yes — 25.5%/24.6% | Weakly (F1 0.202/0.203) | still mostly Cosine↔T2/T3 |
| **Step 24 CREPE template** | **No — pulled down to 19.0%, below true rate** | **Yes — over-predicts, 38.3%/26.3%** | Moderately (68.2% binary acc.) | still Cosine↔T2/T3, opposite direction |

Explicit geometry does **not** produce a cleaner separation structure than the learned decision surface — it produces a *differently biased* one. The compelling result the spec anticipated (CNN confused, template fit clean) did not occur: both approaches are dominated by the same fundamental Cosine↔Sloped-start/Sloped-end confusion; only the direction of the bias changes (CNN over-predicts the majority class, templates over-predict the curved classes). This is itself an important, useful finding — the shared confusion axis (not the class-prior direction) is the real bottleneck.

## 21. Oracle-vs-CREPE gap

| | Oracle macro F1 | CREPE macro F1 | Gap |
|---|---:|---:|---:|
| Template fitting | 0.849 | 0.241 | **0.608** |
| Step 22/23 CNN (unweighted / B1) | 0.801 | 0.290 / 0.311 | 0.511 / 0.490 |

Template fitting has the *larger* absolute gap between oracle and CREPE (0.608 vs. 0.49-0.51 for the CNNs) despite having the *higher* oracle ceiling (0.849 vs. 0.801) — i.e., template fitting is intrinsically the stronger method on clean geometry but is *more*, not less, sensitive to CREPE's specific distortions than a learned classifier that can partially absorb noise characteristics during training. This directly quantifies how much of template fitting's theoretical advantage is lost specifically to CREPE noise.

## 22. Representative segments

`output/shape_classification/step24/figures/representative_segments.png` — 3 (true class) × 3 (highest-confidence correct / lowest-margin correct / largest-margin error) grid, selected deterministically (no cherry-picking): `argmax`/`argmin` of the section-11-style margin among correctly/incorrectly classified primitives of each true class.

Qualitative findings beyond the aggregate statistics:
- **Highest-confidence-correct Cosine** shows CREPE's contour is often a *step function* (a sharp jump mid-segment), visually nothing like a smooth cosine — yet still the best-fitting of the four rigid templates, because the alternatives fit even worse.
- **Largest-margin error, true Cosine → predicted Fixed**: CREPE's contour is nearly flat for 90% of the segment with a sharp jump only in the last ~5% of phase — CREPE's estimate essentially fails to track the annotated pitch motion until very late, a *qualitative* mistracking, not fine jitter.
- **Largest-margin error, true Sloped-start → predicted Sloped-end** (and the true-Sloped-end mirror case, predicted Sloped-start): CREPE's observed contour is close to flat through most of the segment then rises/plateaus sharply near the opposite end from where the true template says it should — a systematically wrong-shaped estimate, not a noisy version of the right one.

These qualitative cases corroborate §14's endpoint-error and §17's failed-smoothing-gate findings: at least some of CREPE's worst errors are structural mistracking (the estimate resembles the *wrong* canonical shape outright), which neither a robust loss (§6) nor endpoint correction alone (§14, correlation only −0.10) would fix.

## 23. Reusable segment-score API

`training/shape_classification/templates.py::template_errors(r, span_cents, robust=False) -> [E_fixed, E_cosine, E_sloped_start, E_sloped_end]` takes only a relative-cents contour and its span — nothing GT-boundary-specific. A future segmental scorer can call it on **any** candidate interval `[i:j]`'s normalized CREPE contour without modification, exactly as section 23 requires. No segmentation was implemented this step.

## 24. Primary scientific question

> Is explicit canonical-template fitting a better way to use CREPE for trajectory typing than learning the four shapes directly with a CNN?

**No, not on the evidence gathered here.** On oracle contours, template fitting is clearly superior (macro F1 0.849 vs. 0.801) — confirming the geometric templates themselves are correct and well-recovered. But on CREPE, the realistic condition, template fitting underperforms Step 23's CNNs on every reported metric: 4-way macro F1 (0.241 vs. 0.311), Cosine F1 (0.339 vs. 0.459), Sloped-start/Sloped-end F1 (0.152/0.181 vs. 0.202/0.203), and 3-way bend macro F1 (0.236 vs. 0.339). The one genuinely favorable result — a clean, training-instability-free 68.2% T2-vs-T3 binary accuracy — is a secondary diagnostic, not the primary comparison, and still falls short of the trivial sign-test baseline (76.8%). Robust scoring ruled out simple jitter/outliers as the CREPE-specific cause; the margin analysis shows real three-way overlap (correct template barely wins for Sloped-start/Sloped-end, wrong template often wins for Cosine); endpoint noise is a real but modest contributor; and the representative examples show CREPE sometimes produces qualitatively, not just quantitatively, wrong-shaped contours that no residual reweighting can fix. The oracle-CREPE gap is *larger* for template fitting than for the CNN, meaning a learned model's implicit robustness to CREPE's specific noise characteristics — even without balancing — currently outweighs the benefit of using known-correct geometry with no learned parameters at all.

## 25. Interpretation — primary outcome

**`TEMPLATE_FITTING_NO_BETTER_THAN_CNN`**

Template fitting on CREPE performs worse than Step 23's learned classifiers across every primary metric (§8, §13), not "roughly the same" and not "an improvement with remaining noise" — ruling out `TEMPLATE_FITTING_SOLVES_SHAPE_TYPING` and `TEMPLATE_FITTING_IMPROVES_BUT_CREPE_NOISE_REMAINS`. `TEMPLATE_FITTING_REVEALS_ENDPOINT_NORMALIZATION_PROBLEM` is not selected: the endpoint-error correlation is real but weak (−0.10, §14), and §11's margin analysis shows the correct template only barely wins even independent of any endpoint-specific accounting — the errors are not "otherwise a good match" as that outcome's definition requires. Known shape templates alone, applied with no class prior at all, do not solve the CREPE Cosine/Sloped-start/Sloped-end overlap problem — they trade one bias (majority-class collapse) for another (curved-template over-prediction) without net improvement.

## 26. Decision gate

**`REASSESS_DOWNSTREAM_TRAJECTORY_FEATURES`**

Per spec, this does **not** mean reopening pitch-frontend research (Step 21's scope boundary stands). It means reconsidering what information the downstream trajectory classifier should be given — and this step's own results point at a concrete answer.

## Recommendation for Step 25

Steps 23 and 24 produced two independent, imperfect classifiers of the *same* underlying CREPE geometry with **complementary, opposite failure modes**: the balanced CNN over-predicts Cosine relative to templates and under-predicts it relative to raw prior; the template fitter over-predicts the curved classes. Section 23 (this step) explicitly built the four-template error vector `[E_fixed, E_cosine, E_sloped_start, E_sloped_end]` as a reusable, per-primitive feature — Step 25 should test whether **feeding these four template-fit errors as additional engineered input features to Step 23's balanced CNN** (concatenated with, or replacing, the raw `q(x)+dq/dx` channels) recovers more than either signal alone, rather than treating template matching as a standalone argmin decision rule. This is a natural, minimal next experiment: it reuses Step 23's frozen balanced-training protocol (no new balancing question), reuses this step's frozen, unmodified template-error API (no new geometry question), and directly tests whether the two methods' complementary biases (§20) are exploitable by a model that sees both. Only after that should `MOVE_TO_SEGMENTAL_TRANSCRIPTION` be revisited.
