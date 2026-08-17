# Step 16 — Fine-Contour Acoustic Pitch Audit

A diagnostic-only audit — **no training, no model changes, no register decoding, no fusion, no canonicalization changes.** Step 15 established `ESTIMATED_MOTION_REMAINS_BOTTLENECK`: oracle pitch motion drives trajectory macro F1 to 0.771 while every estimated-pitch representation tested (fixed φ, learned dense delta, learned salience window) tops out around 0.30-0.34. This step asks *exactly what acoustic error* is responsible, distinguishing pitch-value error, octave/register error, temporal lag, temporal smoothing, slope error, turning-point error, jitter, quantization, and harmonic/drone confusion — because each implies a different fix.

Frozen references: [`docs/step_15_learned_pitch_motion.md`](step_15_learned_pitch_motion.md), [`docs/step_13_relative_pitch.md`](step_13_relative_pitch.md).

Machine-readable outputs: [`output/pitch_diagnostics/pitch_audit/`](../output/pitch_diagnostics/pitch_audit/).

Reproduce (from repository root, `idtap` env):

```bash
python -m training.pitch_diagnostics.pitch_audit.motion
python -m training.pitch_diagnostics.pitch_audit.shape
python -m training.pitch_diagnostics.pitch_audit.salience_and_harmonics
python -m training.pitch_diagnostics.pitch_audit.counterfactual   # reuses Step 15's frozen P0 checkpoints, no retraining
python -m training.pitch_diagnostics.pitch_audit.phase_and_recording
python -m training.pitch_diagnostics.pitch_audit.visualize
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| Pitch sources audited (spec section 1) | GT parametric pitch vs. Fused+D3 estimated pitch, full native 10ms grid, 1,817,161 total frames / 169,150 valid. HPS+D3 excluded — no dense per-frame cache exists for it. |
| **Motion is severely attenuated at short timescales, recovering somewhat at longer ones** | Attenuation ratio R (median\|est\|/median\|gt\|, GT-moving frames only): **0.39** at 50ms, 0.68 at 100ms, 0.83 at 200ms |
| **The estimated path frequently doesn't move at all, even when GT is moving fast** | 77-81% of frames with GT \|velocity\| > 100 c/s show an *exactly zero* estimated frame-to-frame delta, uniformly across T0-T3 |
| **Genuine local direction reversals are almost never captured** | Turning-point recall only 34-40% (±100ms tolerance), 6-14% at ±20ms; false-turn rate 64-99%. Shape-confusion: GT rise→fall / fall→rise correctly classified only 9-13% of the time (vs. 85% for flat) |
| **Estimated-vs-GT instantaneous velocity is nearly uncorrelated** | Correlation 0.02-0.16 across all types; T2/T3 sign agreement only 0.34-0.39 (worse than a 3-way coin flip) |
| **Error concentrates sharply at primitive boundaries** — exactly where T1/T2/T3 shape is defined | 50ms-scale motion error: 61¢ MAE within ±50ms of a boundary vs. 20¢ MAE (median exactly 0) away from boundaries — a 3× gap |
| Register/octave transitions are a **real but minor** contributor | Only 26.4% of large motion errors fall within 100ms of an octave transition; explicit GT-based register correction (Q1) changes downstream macro F1 by **−0.0002** (nothing) |
| Temporal lag is **not** the driver | Best-lag alignment recovers only ~10-19% of motion MAE; GT-optimal lag correction (Q2) changes downstream macro F1 by **+0.001** (nothing) |
| Naively amplifying motion magnitude **actively hurts** | Motion-magnitude correction (Q4, ×2.54 to counter the measured 50ms attenuation) *drops* downstream macro F1 by **−0.051** — confirms the failure is staircasing/absence of motion, not uniformly-scaled-down motion a linear correction could fix |
| Underlying salience evidence is **not catastrophically absent** | Median GT rank 3-5 in the local salience window; median top-1-candidate distance from GT is exactly 1 CQT bin (16.7¢) for every type — the raw per-frame evidence is reasonably good |
| ...but **T2/T3 show meaningfully weaker salience concentration** than T0/T1 | Mean probability mass within ±100¢ of GT: 0.49-0.50 (T2/T3) vs. 0.51-0.60 (T0/T1); lower salience value at the GT bin |
| Jitter is real but **localized to T0, not the dominant failure** | In T0, estimated std (33¢) exceeds GT's own (15¢) with a 54% direction-reversal rate among the (rare, 8.4%) nonzero deltas — a real but minority, non-explanatory phenomenon |
| The `692ed7e6…` outlier (Step 14/15) is a **class-composition artifact**, not a pitch-quality problem | 91.9% T1 / 8.1% T0 / **0% T2, 0% T3**; median GT \|dp/dt\| = 0 (near-static); its absolute (128¢) and motion-error (35¢) stats are *better* than the corpus average |

**Primary diagnosis: `TEMPORAL_RESOLUTION_LIMITED`**

**Step 17 recommendation:** investigate and improve fine-motion/temporal fidelity in the pitch *estimation* pipeline — specifically the Viterbi movement-cost decoder's smoothing behavior — not register decoding, not post-hoc lag/amplitude correction, and not (primarily) the salience frontend's evidence quality.

---

## 1. Pitch sources and evaluation masks

Every audit script shares one bundle builder (`training/pitch_diagnostics/pitch_audit/common.py`) operating on the **full native 10ms grid** (not valid-only), reusing Step 13's frozen dense Fused+D3 log2Hz path and `RecordingLaneIndex`'s GT annotations directly — no new decoding, no new model. `valid` = `valid_target & frame_time < duration_s`, identical to every prior step's mask. 17 recordings, 1,817,161 total frames, 169,150 valid frames (matches every previous step exactly — a live cross-check that the bundle is correctly aligned).

---

## 2-3. Absolute vs. relative-motion error, by trajectory type

Absolute pitch error (est − GT, tonic-relative cents): MAE 349¢, median 27¢, p95 1377¢ — dominated by the long octave-error tail already characterized in Steps 10-13; **no longer the primary metric per spec section 2**, reported only as a reference point.

---

## 4. Multi-timescale Δ-error and smoothing/attenuation

| Offset | Δ-error MAE | Δ-error median | Direction agreement |
|---|---:|---:|---:|
| 10ms | 10.8¢ | 0.0¢ | 65.0% |
| 20ms | 19.5¢ | 0.2¢ | 64.9% |
| 50ms | 39.9¢ | 16.1¢ | 62.0% |
| 100ms | 61.5¢ | 16.7¢ | 59.4% |
| 200ms | 83.3¢ | 21.5¢ | 72.4% |

Direction agreement at 10-100ms hovers near 60-65% — not strongly diagnostic on its own (Step 13 already flagged this exact metric as dominated by both-flat frames at short timescales); the attenuation ratio below is the more informative signal at short scales.

**Attenuation ratio R** (median\|Δest\|/median\|Δgt\|, computed on GT-*moving* frames only — pooled R is degenerate since median GT motion is exactly 0¢ at 10-20ms, matching Steps 13-15's own finding):

| Offset | R (all types) | frac. frames with exactly-zero est delta |
|---|---:|---:|
| 10ms | undefined (est median also 0) | 78.5% |
| 20ms | undefined (est median also 0) | 62.4% |
| 50ms | **0.394** | 35.8% |
| 100ms | 0.682 | 20.8% |
| 200ms | 0.833 | 12.8% |

A clean, monotonic signature: the shorter the timescale, the more severely the estimate under-moves relative to GT — the estimated contour recovers roughly 39% of GT's true 50ms-scale motion magnitude, rising toward ~83% by 200ms. This is exactly the profile spec section 4 predicted for genuine oversmoothing.

---

## 5. Temporal-lag analysis

Cross-correlation-style lag search (20ms/k=2 delta series, ±100ms in 10ms steps), per recording per type, diagnostic only:

| Type | Median best lag | Mean MAE @ lag 0 | Mean MAE @ best lag | Improvement |
|---|---:|---:|---:|---:|
| T0 | −20ms | 10.5¢ | 8.8¢ | 1.7¢ (17%) |
| T1 | 0ms | 20.6¢ | 19.5¢ | 1.2¢ (6%) |
| T2 | −70ms | 25.4¢ | 21.5¢ | 4.0¢ (16%) |
| T3 | +15ms | 23.5¢ | 20.7¢ | 2.8¢ (12%) |

Best lags are inconsistent across types (no single systematic delay), and even at each type's own best lag the improvement is modest (6-17% relative). **Answer to section 5's question: no, small temporal alignment does not substantially close the gap.** Confirmed downstream in §14-15 (Q2).

---

## 6. Velocity/slope fidelity

GT's own analytic `dp/dt` vs. estimated frame-to-frame velocity:

| | Overall | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|
| Correlation | 0.10 | 0.02 | 0.13 | 0.12 | 0.16 |
| Sign agreement | 0.65 | 0.91 | 0.56 | 0.39 | 0.34 |
| MAE (c/s) | 1084 | 519 | 1308 | 1450 | 1794 |
| frac. est delta = 0 when \|GT vel\| > 100 c/s | 77.4% | 80.6% | 77.9% | 76.7% | 75.9% |

**Answer to section 6's question: no** — the acoustic estimator does not preserve fast melodic motion's direction or magnitude in any type. T0's high sign agreement (0.91) is a trivial consequence of both signals mostly being flat there; T1-T3's sign agreement is at or below chance for a 3-way call, and roughly 3 in 4 frames where GT is genuinely moving fast show an estimated velocity of *exactly* zero.

---

## 7. Turning-point audit

GT turn: sign change in analytic `dp/dt` with both sides exceeding the reused 100 c/s deadband (conservative, avoids counting tiny fluctuations). Estimated turn: same rule on a 50ms-smoothed estimated velocity (matching GT's own smoothness scale, not the raw 10ms difference).

| Type | GT turns | Recall @20/50/100ms | False-turn rate @20/50/100ms | Median timing error |
|---|---:|---|---|---:|
| T0 | 0 | — | 429 spurious est. turns in nominally-flat regions | — |
| T1 | 1,060 | 14% / 26% / 37% | 86% / 74% / 64% | 30ms |
| T2 | 132 | 12% / 28% / 40% | 90% / 79% / 74% | 40ms |
| T3 | 32 | 6% / 28% / 34% | 99% / 94% / 87% | 40ms |

Most GT turns are missed even at the widest tolerance tested, and most estimated "turns" don't correspond to a real GT reversal at all. T0's 429 spurious estimated turns despite zero real GT turns previews the jitter finding (§10).

---

## 8. Local-shape confusion

5-class local shape (flat / rising / falling / rise→fall / fall→rise, ±50ms window, 20¢ move threshold, identical rule applied to both sources), confusion matrix row-normalized by GT class (pooled):

| GT \\ EST | flat | rising | falling | rise→fall | fall→rise |
|---|---:|---:|---:|---:|---:|
| **flat** | 85.2% | 7.3% | 6.8% | 0.4% | 0.4% |
| **rising** | 39.4% | **48.6%** | 8.1% | 2.2% | 1.8% |
| **falling** | 42.6% | 6.9% | **47.8%** | 1.5% | 1.2% |
| **rise→fall** | 37.7% | 24.0% | 24.1% | **13.3%** | 0.9% |
| **fall→rise** | 41.3% | 21.8% | 26.6% | 1.0% | **9.4%** |

The clearest single result in this audit. Flat is recovered well (85%). Simple monotonic movement is recovered about half the time, with "smoothed into apparent flatness" as the dominant confusion (39-43%) — directly corroborating the attenuation finding at the shape level. **Genuine local direction reversals (rise→fall, fall→rise) — the shapes that most directly define T2 vs. T3 — are correctly recovered only 9-13% of the time**, overwhelmingly mistaken for flat or for one-directional movement instead. This is the most direct explanation available for why T2/T3 collapse under estimated pitch while thriving under oracle pitch (Step 15's P3: T2 F1 0.76, T3 F1 0.82).

---

## 9. Quantization / staircase audit

Run-length analysis (consecutive frames within 8¢ — half a native CQT bin — of a run's start value):

| Type | GT median run | EST median run |
|---|---:|---:|
| T0 | 23 frames (230ms) | 6 frames (60ms) |
| T1 | 1 frame (10ms) | **3 frames (30ms)** |
| T2 | 1 frame (10ms) | **3 frames (30ms)** |
| T3 | 1 frame (10ms) | **3 frames (30ms)** |

GT is a continuous parametric curve — during any bend (T1-T3) it essentially never stays within one CQT bin for two consecutive frames (median run = 1). The estimated path, by contrast, plateaus for a median of 3 frames during exactly these moving segments — a genuine, quantified staircase effect specifically in the classes that need fine motion most. (T0's own numbers are explained differently — see §10.)

---

## 10. Jitter audit

In T0 (GT-flat) regions specifically: estimated std 33¢ vs. GT's own 15¢; only 8.4% of frames show any nonzero estimated delta at all, but *among those*, the direction reverses frame-to-frame 54% of the time (T1-T3, by contrast: 18-34% reversal rate — see §9's table for context). This resolves the apparent T0 staircase paradox (§9: EST median run 6 frames, *shorter* than GT's 23): T0's estimated path is mostly correctly flat, but the rare deviations look like noise (rapid alternation) rather than smooth drift, which breaks up otherwise-long flat runs into shorter fragments. **This is a real, opposite-direction failure mode, but a minor and localized one** — it affects T0, which is already the best-recovered class, not T1-T3 where the dominant problem is the reverse (too little motion, not too much).

---

## 11. Octave-transition contribution

Only 3.6% / 6.2% / 10.5% of valid frames fall within ±50/100/200ms of an estimated octave transition. Of the 16,128 frames with "large" (>100¢) 50ms-scale motion error, only **26.4%** occur within 100ms of a transition — the large majority of large motion errors happen in regions with *stable* (if possibly wrong) register. Register transitions are a real but clearly minor contributor. **Closing this again**, exactly as Step 13 did — reconfirmed here with a direct, quantified downstream test (§14-15, Q1) showing essentially zero effect.

---

## 12. Salience evidence audit

Per-frame evidence quality in the windowed relative-salience representation (Step 15's exact cache), evaluated against GT for diagnosis only:

| | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|
| Median GT rank | 3 | 3 | 5 | 5 |
| Median top-1 distance from GT | 16.7¢ (1 bin) | 16.7¢ | 16.7¢ | 16.7¢ |
| Mean top-1 distance from GT | 30.3¢ | 40.9¢ | 50.0¢ | 53.8¢ |
| Mean coverage within ±100¢ of GT | 0.509 | 0.600 | 0.500 | 0.489 |
| Salience value at GT bin (median) | 0.095 | 0.095 | 0.061 | 0.055 |

**Answer to section 12's question: mostly present but poorly selected, with a real T2/T3-specific degradation, not a wholesale absence.** The median case is reassuring — GT rank 3-5, top-1 candidate within one bin — meaning the raw evidence is not catastrophically wrong. But means (pulled up by a heavier tail for T2/T3) and coverage are both meaningfully worse for T2/T3 than T0/T1, and Step 15 §16 already showed that *conditioning P2 on this exact ambiguity signal never revealed a regime where retaining the full distribution helped* — together these say the salience frontend contributes a real, secondary weakness for T2/T3 but is not the primary bottleneck.

---

## 13. Harmonic/drone-confusion audit

Among "large" (>200¢) absolute errors, association with simple harmonic/drone relationships (±50¢ tolerance):

| Type | n large errors | near +1 octave | near tonic | other/unexplained |
|---|---:|---:|---:|---:|
| T0 | 16,334 | **51.6%** | 11.6% | 21.0% |
| T1 | 22,202 | 40.2% | 19.1% | 35.5% |
| T2 | 4,507 | 41.3% | 12.6% | 40.0% |
| T3 | 4,046 | 25.8% | **30.4%** | 36.5% |

T0's large errors are dominated by a clean +1-octave association (consistent with static/drone-adjacent notes being octave-confused). T3 shows an elevated tonic-proximity association (30.4%, notably higher than other types) — a real, type-specific drone-confusion signature worth noting, though "other" (unexplained by any simple harmonic ratio) remains substantial for every type. These are reported as associations, not causal claims, per spec section 13.

---

## 14-15. Justified counterfactual corrections and downstream effect

Per spec section 14, only corrections the preceding audit actually supports were built. **Q3 (jitter correction) was not built** — §10 showed jitter is a secondary, T0-localized phenomenon, while the dominant failure across T1-T3 is under-motion, not excess noise; "inventing" a jitter fix would target the wrong mechanism.

| Variant | Construction | Downstream macro F1 (frozen Step 15 P0 classifier) | Δ vs. Q0 |
|---|---|---:|---:|
| Q0 — baseline | Fused+D3, unmodified | 0.3375 | — |
| Q1 — register-corrected | GT-only octave shift, within-octave error preserved | 0.3373 | **−0.0002** |
| Q2 — lag-corrected | GT-optimal global per-recording lag (§5) applied as a pure shift | 0.3385 | **+0.0010** |
| Q4 — motion-amplified | Frame-to-frame deltas × 2.54 (=1/R₅₀ₘₛ), cumulatively reconstructed | 0.2863 | **−0.0512** |

(Q0's 0.3375 matches Step 15's own pooled P0 result of 0.338 almost exactly — a live consistency check that this evaluation harness faithfully reproduces the original.)

**None of the three testable corrections closes any of the P0 (0.338) → P3 (0.771) gap.** Register and lag correction are noise-level. Motion amplification — the correction most directly targeted at the dominant attenuation finding (§4) — makes things *worse*, because scaling a signal that is mostly **exactly zero** (staircased, §9) by a constant factor does nothing to the zero frames and only amplifies whatever noise exists in the rare nonzero ones. This is important, clean negative evidence: **the fix is not a post-hoc scalar correction on the decoded path** — the missing fine motion has to be recovered at the estimation stage itself, not reconstructed afterward.

---

## 16. Primitive-phase / boundary localization

50ms-scale motion error by normalized position within the primitive, and by boundary proximity:

| Phase | beginning | early-mid | middle | late-mid | end |
|---|---:|---:|---:|---:|---:|
| MAE | 45.2¢ | 35.0¢ | 37.7¢ | 40.6¢ | 41.4¢ |

| | within ±50ms of a boundary | within ±100ms | away from any boundary |
|---|---:|---:|---:|
| MAE | **61.3¢** | 55.3¢ | 19.9¢ (median exactly 0¢) |

**Answer to section 16's question: motion error is specifically and substantially worse right at trajectory transitions** — a 3× MAE gap between near-boundary and non-boundary frames, with a mild additional U-shape (worst at the very beginning of a primitive, best in the early-middle, rising again toward the end). This aligns exactly with T2 ("sloped start") and T3 ("sloped end") being defined by behavior at precisely these least-reliable regions.

---

## 17. Recording-level analysis, including `692ed7e6…`

Investigated per spec section 17's explicit request (Step 14 and Step 15 independently flagged this recording as the sole case where oracle-pitch conditions *underperformed*):

| Property | Value |
|---|---|
| Class composition | 91.9% T1, 8.1% T0, **0.0% T2, 0.0% T3** |
| Median \|GT dp/dt\| | 0.0 c/s (near-static pitch throughout) |
| Median primitive duration | 114ms (mean 176ms — short even by this corpus's standards) |
| Absolute pitch error (this recording) | 128¢ MAE — *better* than the 349¢ corpus average |
| 50ms motion error (this recording) | 35¢ MAE — close to/better than corpus norms |

**Determined cause: class composition and GT pitch-dynamics, not acoustic pitch quality.** This recording's absolute and motion-error statistics are unremarkable-to-good relative to the rest of the corpus — its pitch is not unusually hard to estimate. What is unusual is that it contains **zero T2/T3 examples at all**, is 92% a single class (T1), and that class's own GT pitch barely moves (median velocity exactly 0). Macro F1 with only 2 of 4 classes present and one dominating so heavily is inherently unstable and not comparable to other recordings' scores — this explains the outlier without invoking any acoustic-quality hypothesis, exactly as spec section 17 required ("do not speculate beyond measurable evidence").

---

## 18. Representative visualizations

`output/pitch_diagnostics/pitch_audit/figures/` (4 panels: CQT, trajectory-type strip, GT/estimated pitch overlay, GT/estimated velocity overlay):

- `staircase_oversmoothing_example.png` — the single largest T2/T3 50ms-scale motion-error window in the corpus; shows both visible staircase steps in the estimated path and a co-occurring octave-scale offset for part of the window (large failures are often compound, not single-cause).
- `octave_transition_example.png` — a clean isolated register-transition artifact for reference.
- `good_t1_tracking_example.png` — selected as the lowest-error T1-and-moving window; shows a genuinely well-matched flat stretch alongside an adjacent badly octave-shifted, visibly staircased stretch within the same short excerpt — illustrating how uneven quality is even within a single "good" window.
- `outlier_recording_692ed7e6.png` — visual confirmation of §17: GT pitch stays in a narrow, slowly-varying band while the estimate swings roughly a full octave away for extended stretches, on a recording whose own visible dynamics are otherwise unremarkable.

Not exhaustive relative to spec section 18's full example list (turning-point miss, jitter, harmonic/drone specifically) — the quantitative audit (§4-13) covers those cases numerically; visualization was kept to a representative subset given this step's diagnostic-only scope.

---

## 19. Ranked failure-mode summary

| Failure mode | Evidence | T0 | T1 | T2 | T3 | Downstream importance |
|---|---|---|---|---|---|---|
| **Missed turning points / shape reversal** | Recall 6-40%; rise↔fall shapes correct only 9-13% of the time | n/a | high | high | high | **Very high** — directly explains T2/T3 collapse |
| **Temporal smoothing / attenuation** | R=0.39 @50ms, worsens at shorter scales | mild | high | high | high | **Very high** — root mechanism behind turning-point misses |
| **Quantization / staircase** | EST run-length 3× GT's during motion | mixed (jitter, not staircase) | high | high | high | **Very high** — same mechanism as attenuation, confirmed by Q4's failure |
| **Boundary/phase concentration** | 3× MAE near boundaries vs. elsewhere | — | high | high | high | **Very high** — spatially ties the above to T2/T3's defining regions |
| Salience evidence quality | GT rank 3-5, decent top-1 distance; T2/T3 modestly worse coverage | low | low | moderate | moderate | Moderate — secondary contributor |
| Harmonic/drone confusion | T0: 52% of large errors near +1 octave; T3: 30% near tonic | moderate | low | low | moderate | Moderate, type-specific |
| Register/octave transitions | Only 26% of large errors near a transition; Q1 gives 0 downstream benefit | low | low | low | low | **Low** — closed |
| Temporal lag | 6-17% relative improvement only; Q2 gives 0 downstream benefit | low | low | low | low | **Low** — closed |
| Jitter | Real in T0 (54% reversal rate) but localized, minority phenomenon | moderate | low | low | low | **Low** — not the dominant class's problem |

---

## 20. Primary diagnosis

**`TEMPORAL_RESOLUTION_LIMITED`**

The dominant, mutually-reinforcing evidence — severe short-timescale attenuation, exactly-zero deltas during genuine fast motion, missed turning points, staircase run-lengths, and error sharply concentrated at primitive boundaries — all describe the same underlying mechanism: **the estimator finds approximately the right region but smooths/quantizes away the short-timescale, direction-changing movement that distinguishes trajectory shapes**, "despite reasonable pitch candidates" (§12 confirms the raw salience evidence is not catastrophically absent). This is the decision criterion's exact description.

Not `SALIENCE_EVIDENCE_LIMITED` as primary — GT pitch is present and reasonably ranked in the salience window most of the time; the T2/T3-specific weakness there is real but secondary (§12), not the dominant story. Not `HARMONIC_DRONE_CONFUSION_DOMINATES` — real, type-specific associations exist (T0/octave, T3/tonic) but explain a minority of errors and were never the majority mechanism. Not `MULTIPLE_ACOUSTIC_FAILURE_MODES` — while several contributing factors exist (as any real system has), one mechanism clearly dominates the evidence and, critically, is the one mechanism whose closure was *not* achievable by the two alternative-hypothesis corrections that were tested and failed (Q1 register, Q2 lag) — register and lag are conclusively ruled out as primary, leaving temporal/fine-motion resolution as the clear leading explanation, not one of several co-equal causes.

---

## 21. Step 17 recommendation

**Investigate and improve fine-motion/temporal fidelity in the pitch estimation pipeline itself — specifically the Viterbi movement-cost decoder's smoothing behavior (Step 12) — before any further representation or fusion work.**

This follows the required combination of acoustic diagnostic **and** downstream counterfactual effect (spec section 21): the attenuation/staircase finding is both substantial (§4, §9 — R=0.39 at the clearest scale, 3× run-length inflation) and trajectory-relevant (§8, §16 — it is concentrated exactly at the shape-defining boundary regions and directly explains the turning-point recall failure that most plausibly accounts for T2/T3's collapse under estimated pitch). Register and lag were tested with the same rigor and *failed* the counterfactual bar (§14-15, Q1/Q2 ≈ 0 effect) — they are not being recommended merely because errors of those types exist, but rejected because a direct downstream test showed fixing them changes nothing.

Concretely, Step 17 should examine whether the Viterbi decoder's movement-cost penalty (tuned in Step 12 for overall pitch accuracy, not shape fidelity) is suppressing exactly the fast, reversing motion trajectory classification needs — e.g. by comparing trajectory-relevant motion fidelity of the raw framewise argmax/salience-derived path (pre-Viterbi) against the current smoothed decode, or by exploring a decoder objective that explicitly weights direction-change preservation rather than absolute-position accuracy. **Do not** revisit register/octave decoding (closed twice now, Step 13 and this step). **Do not** attempt another post-hoc correction on the existing decoded path (§14-15 showed the simplest such correction actively hurts). This step does not train anything — Step 17's design and execution are a separate decision.
