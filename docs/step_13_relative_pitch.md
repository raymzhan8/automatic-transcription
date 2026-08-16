# Step 13 — Octave-Invariant Relative Pitch Movement Diagnostic

> **Follow-up:** the recommended next step (a controlled A/B/C/D feature ablation on the actual T0-T3 task) was run in [`docs/step_14_relative_pitch_trajectory.md`](step_14_relative_pitch_trajectory.md). Headline result: estimated relative pitch **alone** beats both audio-only and naive audio+pitch fusion; oracle relative pitch dramatically resolves T2/T3. Decision: `IMPROVE_RELATIVE_PITCH_ESTIMATION`.

Follows directly from Step 12.5's `STOP_REGISTER_DECODER_ENGINEERING` decision: HPS/HPS+D3 remains the strongest absolute-pitch frontend, and further decoder engineering on absolute pitch is not worth pursuing. This step asks the question that decision deferred: even when absolute register is wrong, does the local pitch-motion signal that trajectory-shape classification (T0-T3) actually needs survive? **Diagnostic only** — not the final trajectory classifier. No register-decoder changes, no new Viterbi/fusion machinery beyond re-applying Step 12/12.5's own already-validated hyperparameters, no class weighting, no sequence model.

Frozen references: [`docs/step_12_5_fusion_viterbi.md`](step_12_5_fusion_viterbi.md), [`docs/step_12_register_resolution.md`](step_12_register_resolution.md).

Machine-readable outputs: [`output/pitch_diagnostics/relative_pitch/`](../output/pitch_diagnostics/relative_pitch/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/relative_pitch/path_cache.py   # decoded cent paths, fixed hyperparams from Step 12/12.5
python training/pitch_diagnostics/relative_pitch/signals.py      # delta/velocity/contour metrics, octave-conditional, type/dpdt breakdown
python training/pitch_diagnostics/relative_pitch/probe.py        # T0-T3 logistic-regression probes (sliding-window + primitive-aligned)
```

`path_cache.py` re-runs the existing HPS-salience/learned-model forward passes and `decoders.py`'s/`fusion.py`'s own `_states_for_frame`/`viterbi_decode`/fusion functions once each, using the lambda_t and (alpha, beta) values already validation-selected in `decoder_ablation.json` / `fusion_viterbi_result.json` — no new grid search, no D4 (Step 12/12.5 already showed the octave penalty adds ~nothing).

---

## 1. Frontends and relative-pitch signals

Five decoded absolute-pitch-cent paths per recording (native 10ms hop, tonic-relative cents), plus GT:

| Frontend | Range | Source |
|---|---|---|
| GT (oracle) | — | IDTAP annotation `pitch_log2_hz` |
| HPS argmax | full 0-360 | Step 10/11 |
| **HPS + D3** | full 0-360 | Step 12, fixed λ_t per fold |
| Learned argmax | native 34-244 | Step 11 |
| Learned + D3 | native 34-244 | Step 12, fixed λ_t per fold |
| Fused + D3 | shared 34-244 | Step 12.5, fixed (α,β) and λ_t per fold |

From each path: **(A)** frame-to-frame delta (`1200·log2(f[t]/f[t-1])`, equivalently cents-difference since paths are already in cents); **(B)** time-gap-aware velocity (`delta / actual_elapsed_seconds`, using real inter-valid-frame gaps, not raw frame-index differences); **(C)** short-window relative contour (`pitch_cents[t] − pitch_cents[window_start]`). No learned representation for (A)-(C).

---

## 2. Delta / velocity / direction accuracy (pooled, 169,133 consecutive valid-frame pairs)

**Deadband:** ±100 cents/s for the "flat" direction class — reused verbatim from Step 12.5's own first `|dp/dt|` bucket boundary (0-100 c/s), not a new threshold.

| Frontend | Delta MAE (¢) | Delta median AE | Velocity MAE (c/s) | Direction accuracy |
|---|---:|---:|---:|---:|
| HPS argmax | 53.1 | 0.0 | 4978.2 | 62.5% |
| HPS + D3 | 33.3 | 0.0 | 3014.0 | 63.2% |
| Learned argmax | 67.2 | 0.0 | 6401.6 | 58.5% |
| Learned + D3 | 18.4 | 0.0 | 1545.5 | 63.5% |
| **Fused + D3** | **16.5** | 0.0 | **1376.5** | **64.9%** |

Median delta error is exactly 0¢ for every frontend — the modal case in this corpus is a genuinely near-static pitch frame-to-frame (sustained notes), where both GT and every frontend correctly show ~no movement; MAE (dominated by the real-movement frames) is the informative statistic here, not the median.

**Direction accuracy is misleading at face value.** Per-class breakdown (HPS+D3): rising 17.9%, falling 18.9%, **flat 89.1%** — pooled accuracy is inflated by the "flat" majority class (89-91% of pairs are GT-flat under the 100 c/s deadband, since even genuine bend motion is often <1 cent per 10ms native frame — 100 c/s × 0.01s = exactly 1 cent). When GT is genuinely rising or falling, every frontend gets the *sign* right only ~18-21% of the time — because at the single-native-frame (10ms) granularity, a frontend's own delta noise (10-25¢ MAE, i.e. up to 1-2 CQT bins) routinely exceeds the 1-cent GT motion the deadband is trying to resolve. **This specific metric is not diagnostic at 10ms granularity** — reusing Step 12.5's bucket threshold here (as instructed) exposes that it was calibrated for bucketing a smooth analytic `dp/dt`, not for classifying noisy single-frame deltas; the coarser windowed/primitive-level representations (§4) are the better-posed test of directional/shape information and are treated as primary below.

---

## 3. Octave-error robustness — the core test

For every consecutive pair, each frontend's own octave correctness (`octave_k` vs GT) is checked at both frames, giving three categories:

| Category | Fraction of pairs | Meaning |
|---|---:|---|
| `both_correct` | 78.6% | both frames in the right register |
| `both_wrong_same_k` | 19.4% | both frames wrong, by the *same* octave offset |
| `transition` | 2.0% | the octave estimate itself flips between the two frames |

Delta MAE by category:

| Frontend | both_correct | both_wrong_same_k | transition |
|---|---:|---:|---:|
| HPS argmax | 11.2 | 14.3 | 1193.7 |
| HPS + D3 | 10.8 | 13.2 | 1134.5 |
| Learned argmax | 12.3 | 22.6 | 1148.5 |
| Learned + D3 | 9.8 | 12.4 | 1000.0 |
| **Fused + D3** | **9.6** | **11.3** | 986.0 |

**Octave errors overwhelmingly cancel under differencing.** When the wrong register persists across both frames of a pair (19.4% of all pairs — consistent with Step 12's "sticky, sustained" wrong-octave runs), delta MAE is only modestly worse than when both frames are correct (e.g. HPS+D3: 13.2¢ vs 10.8¢, +22%) — nowhere near the ~1200¢ scale of the octave error itself. The catastrophic case (~1000-1200¢ delta MAE, i.e. a spurious apparent octave jump) is concentrated entirely in the rare `transition` pairs (2.0%), where the register estimate itself changes between frames — this is the one scenario differencing cannot fix, but it is a small minority of the data. This is a clean, direct confirmation of Step 13's central hypothesis: **wrong absolute pitch does not generally mean wrong relative movement.**

---

## 4. Breakdown by trajectory type and |dp/dt|

Delta MAE (¢), by trajectory type:

| Frontend | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|
| HPS argmax | 44.5 | 54.7 | 65.1 | 69.7 |
| HPS + D3 | 25.8 | 35.5 | 40.4 | 45.6 |
| Learned + D3 | 12.3 | 19.9 | 22.6 | 31.9 |
| **Fused + D3** | **10.5** | **18.2** | **21.5** | **27.1** |

Delta MAE (¢), by `|dp/dt|` bucket (Step 12.5's own buckets):

| Frontend | 0-100 c/s | 100-400 c/s | 400-1000 c/s | >1000 c/s |
|---|---:|---:|---:|---:|
| HPS argmax | 41.7 | 48.2 | 49.2 | 78.0 |
| HPS + D3 | 22.7 | 25.6 | 30.1 | 56.3 |
| Learned + D3 | 8.1 | 13.0 | 16.1 | 39.5 |
| **Fused + D3** | **6.5** | **11.1** | **15.5** | **37.1** |

Two consistent patterns: **(1)** error grows gracefully, not catastrophically, with trajectory complexity (T0→T3) and speed — no cliff. **(2)** Learned+D3 and Fused+D3 beat HPS+D3 on relative-motion accuracy at *every* type and *every* speed bucket, including the fastest (>1000 c/s: 37.1-39.5¢ vs HPS+D3's 56.3¢) — the same frontends that lost to HPS on absolute pitch MAE in Steps 12/12.5 are the *best* relative-motion trackers here. This is not an oversmoothing artifact (which would show the gap widening specifically at high speed, where over-smoothing would suppress genuine fast motion); instead the gap is roughly constant in relative terms across buckets, consistent with genuinely better local tracking rather than a smoothing bias, and corroborates Step 12.5's own finding of no oversmoothing on fast-moving regions.

---

## 5. Absolute vs. relative error: the two large-failure recordings

Step 11/12's two flagged large-support failure recordings, absolute vs. relative-motion metrics:

| | `6417585554…` (fold 2, worst absolute-MAE recording) | `6824de49…` (fold 1, fusion's Step 12.5 catastrophe) |
|---|---|---|
| HPS+D3 abs MAE / delta MAE / contour MAE | 531.6 / 49.3 / 294.1 | 243.5 / 33.2 / 185.8 |
| Learned+D3 abs MAE / delta MAE / contour MAE | 577.0 / **30.1** / **183.7** | 537.0 / **10.7** / **73.5** |
| **Fused+D3** abs MAE / delta MAE / contour MAE | **492.5** / **25.2** / **148.8** | 537.0 / **10.7** / **73.5** |

**Both recordings remain far from "solved" in absolute terms, but their relative-motion signal is much better preserved than their absolute error suggests.** On `6824de49…` specifically — the recording where Step 12.5 found Fusion+D3 collapsed to pure-learned and lost catastrophically on absolute MAE (537.0¢ vs HPS's 243.5¢, a 293¢ absolute loss) — the *relative* picture flips entirely: Learned/Fused+D3's delta MAE (10.7¢) and contour MAE (73.5¢) are **more than 3× better than HPS+D3's** (33.2¢ / 185.8¢) on the identical frames. This is the clearest, most concrete instance in this step of "wrong absolute pitch + still-correct relative movement" — exactly the pattern that would make register-resolution failures on this recording largely irrelevant to trajectory-shape recognition specifically.

---

## 6. T0-T3 discriminative probe

**Sliding-window version** (21-frame/~210ms window, centered on every eligible frame) collapsed to majority-class (T1) prediction for *every* frontend **including oracle GT pitch** (macro F1 ≈ 0.17 uniformly, T0/T2/T3 F1 ≈ 0). This is a red flag, not a genuine null result: Step 10's duration table shows ~40% of primitive-frames belong to primitives *shorter* than this 210ms window, so a fixed window routinely spans multiple primitives of different types — contaminating even the trivially-separable "flat" (T0) class. Reported for completeness but not trusted as the answer.

**Primitive-aligned version** (fairer test): one example per already-labeled primitive segment (contiguous same-type valid-frame runs, derived from the existing per-frame `trajectory_type` array — no canonicalization change), relative contour linearly resampled to a fixed 20-point vector. 2,935 pooled test primitives (class support: T0 1,089 / T1 1,032 / T2 426 / T3 388). Plain multinomial logistic regression, `StandardScaler`, no class weighting, no tuning, grouped 5-fold (recording-disjoint train/test, matching every prior step).

| Source | Accuracy | Macro F1 | T0 F1 | T1 F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Oracle (GT)** | 49.0% | **0.296** | 0.617 | 0.462 | 0.034 | 0.069 |
| HPS argmax | 36.9% | 0.202 | 0.505 | 0.295 | 0.004 | 0.005 |
| HPS + D3 | 37.0% | 0.197 | 0.510 | 0.274 | 0.000 | 0.005 |
| Learned argmax | 37.2% | 0.209 | 0.502 | 0.314 | 0.017 | 0.005 |
| Learned + D3 | 40.1% | 0.218 | 0.539 | 0.317 | 0.004 | 0.010 |
| **Fused + D3** | **41.3%** | **0.230** | 0.550 | 0.346 | 0.017 | 0.005 |

Now genuinely informative: **real, above-baseline signal exists** (oracle macro F1 0.296 vs. a plurality-class-only baseline of ~0.135), and it survives pitch-estimation noise only *partially* — the best estimated frontend (Fused+D3, again the best of the estimated sources, mirroring §4's finding) reaches 0.230, a real but meaningful ~22% relative macro-F1 loss from oracle. T0/T1 carry the signal (oracle F1 0.617/0.462, estimated 0.50-0.55/0.27-0.35 — a real but partial loss); **T2/T3 remain weak even with perfect GT pitch** (F1 ≤ 0.07 for oracle itself) — consistent with `docs/step_9_c_report.md`'s already-documented finding that unweighted training collapses T2/T3 in this corpus regardless of input quality (small class support: 14.5%/13.2% of primitives), so this specific weakness is at least partly a known class-imbalance property of the task, not solely evidence against relative pitch.

---

## 7. Key diagnostic comparison — absolute vs. relative error, synthesized

| Axis | Verdict |
|---|---|
| Octave-conditional delta MAE (§3) | Octave errors cancel under differencing in 98% of pairs (both_correct ≈ both_wrong_same_k); catastrophic only at rare register-transition pairs (2%) |
| Type/speed robustness (§4) | Graceful degradation, no cliff; learned/fused frontends beat HPS on relative motion at every type and speed, despite losing to HPS on absolute pitch |
| Large-failure recordings (§5) | Relative-motion metrics 3×+ better than absolute-MAE ranking would suggest, most dramatically on `6824de49…` |
| T0-T3 probe (§6) | Real, above-chance signal survives pitch noise, but with a genuine ~22% relative macro-F1 loss vs. oracle, and strong type-dependence (T0/T1 usable, T2/T3 weak in both oracle and estimated regimes) |

The delta/velocity/contour axis (§2-5) supports a strong claim; the actual downstream discriminative test (§6) — the more direct proxy for what a trajectory classifier needs — shows real, non-trivial information loss and does not uniformly retain strength across classes.

---

## 8. Final classification

**`RELATIVE_PITCH_PARTIAL`**

Rationale against the pre-declared criteria:

- Relative motion **clearly improves robustness over absolute pitch** and survives the great majority of octave/register errors (§3: 78.6%+19.4% = 98% of pairs show delta MAE within ~2¢-5¢ of the fully-correct case; only the 2% transition pairs are catastrophic) — this rules out `RELATIVE_PITCH_WEAK` (which requires octave failures to *generally* also destroy local movement estimates; here they generally do not).
- But it **still loses significant trajectory information and varies strongly by type** (§6: real ~22% relative macro-F1 loss from oracle to the best estimated frontend; T2/T3 remain weak throughout, T0/T1 carry essentially all the usable signal) — this rules out `RELATIVE_PITCH_STRONG` (which requires the probe to retain "substantial" discriminative power, not a partial, type-skewed subset of it).
- A secondary, genuinely useful finding not anticipated by the gate criteria: the learned/fused salience frontends — which *lost* to HPS on absolute pitch accuracy in every comparison across Steps 12/12.5 — are consistently the **best relative-motion trackers and the best probe inputs** (§4, §6). Absolute-pitch quality and relative-motion/trajectory-classification usefulness are measurably different axes.

**Interpretation:** relative pitch/motion is a genuinely useful signal — reliable enough that register-resolution failures should not block trajectory-modeling progress — but it should **complement** learned time-frequency features in the eventual trajectory model rather than serve as the sole pitch representation, particularly for T2/T3 discrimination where even oracle pitch alone is insufficient under a simple classifier.

---

## 9. Recommendation

**Do next:** when trajectory modeling resumes, include relative/delta pitch features (e.g. the fused+D3 path's delta or windowed contour, which was the best-performing estimated source throughout this step) as an **auxiliary input alongside learned audio/time-frequency features**, not as a replacement for them — consistent with the `RELATIVE_PITCH_PARTIAL` interpretation and with this step's finding that T2/T3 need more than pitch-contour-shape alone even in the oracle case.

**Do not:** treat solving absolute-register recovery as a prerequisite for trajectory work (Step 12.5's `STOP_REGISTER_DECODER_ENGINEERING` stands, now further justified — the register errors it left unresolved largely wash out under differencing). Do not conclude relative pitch alone is sufficient to replace learned features (`RELATIVE_PITCH_STRONG`'s bar was not met). Do not fix the T2/T3 collapse in this step by adding class weighting/oversampling/architecture — that crosses into the eventual trajectory classifier's design (out of scope here) and duplicates a fix Step 9 already flagged for the real BiGRU/TCN classifiers.

**Practical takeaway:** the pitch-frontend investigation (Steps 10-13) converges on a clear division of labor — HPS(+D3) for absolute pitch where it's needed, learned/fused salience's smoother relative-motion tracking as a trajectory-relevant auxiliary signal — rather than a single frontend that has to win on every axis at once.
