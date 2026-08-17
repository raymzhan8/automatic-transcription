# Step 19 — Pre-Decoder Fine-Motion Evidence Localization

> **Follow-up:** the recommended acoustic-representation experiment was carried out in [`docs/step_20_acoustic_frontend_bakeoff.md`](step_20_acoustic_frontend_bakeoff.md) as a controlled, frontend-only bake-off (Phase A, no learned model). Outcome: `FRONTEND_CHALLENGER_FOUND` — a shorter-context CQT (`filter_scale=0.5`, same log-frequency bin grid) substantially repairs T3's motion-discrimination pathology and turning-point degradation while leaving T0/T1/T2 acoustic rank nearly unchanged; fixed-window STFT alternatives were decisively ruled out (catastrophic low-frequency resolution, below-chance motion contrast, 90%+ sub-resolution rates for real T1-T3 motion).

Step 18 closed movement-cost (`lambda_t`) tuning: no operating point among L0/L25/L50/L100 produces a fold/recording-robust trajectory-classification improvement, and the best of them (0.340) remains nowhere near oracle pitch's 0.771. This step does **not** touch the decoder again. It asks a narrower, upstream question: **at which pre-decoder stage — acoustic representation (A), salience/candidate score (S=C), or framewise candidate selection (D0's argmax) — does fine trajectory-relevant pitch motion actually disappear?** No model is trained; every result below is either a deterministic recomputation from cached forward passes (`training/pitch_diagnostics/pitch_audit/predecoder_common.py::iter_recording_records`) or a controlled synthetic-audio test of the CQT stage alone.

Frozen references: [`docs/step_18_lambda_selection.md`](step_18_lambda_selection.md), [`docs/step_17_pre_post_viterbi_fidelity.md`](step_17_pre_post_viterbi_fidelity.md), [`docs/step_16_acoustic_pitch_audit.md`](step_16_acoustic_pitch_audit.md).

Machine-readable outputs: [`output/pitch_diagnostics/pitch_audit/predecoder_audit.json`](../output/pitch_diagnostics/pitch_audit/predecoder_audit.json), [`output/pitch_diagnostics/pitch_audit/synthetic_resolution_test.json`](../output/pitch_diagnostics/pitch_audit/synthetic_resolution_test.json), figures in `output/pitch_diagnostics/pitch_audit/figures/predecoder_*.png` and `synthetic_resolution_test.png`.

Reproduce (from repository root, `idtap` conda env — matplotlib/torch/librosa are not on the base env):

```bash
python -m training.pitch_diagnostics.pitch_audit.predecoder_audit
python -m training.pitch_diagnostics.pitch_audit.synthetic_resolution_test
python -m training.pitch_diagnostics.pitch_audit.visualize_predecoder
```

Not done, per spec: no new pitch model, no salience retraining, no CQT/frequency/temporal-resolution change, no Viterbi change, no lambda sweep, no trajectory classifier training, no audio fusion, no class weighting, no canonicalization change. All 210-candidate ranges, checkpoints, and fusion hyperparameters are the same ones frozen since Step 12.5.

---

## Executive summary

| Finding | Evidence |
|---|---|
| **Stage A already destroys most of what the decoder is later blamed for.** Acoustic rank of GT's true bin degrades T0→T3 exactly as Step 10 found (median rank 8→14→17→26), and this degradation happens *before* salience or selection ever see the frame | §4 |
| **Salience improves on stage A, it does not degrade it.** S's rank is better than A's at every type and every stratum (median 3→4→5→7 vs. A's 8→14→17→26); motion contrast M_S > M_A everywhere | §8, §6-7 — rules out `SALIENCE_SCORING_LIMITED` as primary |
| **The central causal table is decisive.** Among zero-delta failures on genuinely-moving GT frames, `acoustic_weak_or_absent` accounts for 48-61% of cases across all 4 types; `salience_lost_candidate` only 3-17% (≤4.4% for T1-T3); `present_but_not_selected` only 3-8% | §11-12 — rules out `FRAMEWISE_SELECTION_LIMITED` as primary |
| **A real, secondary quantization contributor exists but is not dominant.** 57-69% of T1-T3's true 10ms GT motion is smaller than half a candidate bin (16.7¢); this shows up as `sub_bin_quantization`, 27-38% of zero-delta causes | §13 |
| **The CQT's own analysis window is theoretically wide enough to explain this**: 998ms at the candidate range's low end (104Hz), 134ms at the high end (778Hz), ~408ms at the corpus median pitch (254.7Hz) — one to several syllables of context smeared into every frame | §2 |
| **A's evidence quality gets *worse*, not better, immediately after a real turning point for T2/T3** (mean rank 38.5→42.3 for T2, 48.5→55.5 for T3), while T1 is flat — a specific, mechanistically-consistent signature of temporal smearing that can't "catch up" after a direction change | §14 |
| **T3's acoustic evidence is at-or-below-chance for preferring the true motion over a slower counterfactual** (46.1% GT-beats-half-speed, 50.2% GT-beats-stationary) — the representation is *biased toward underestimating speed*, exactly what a smearing/low-pass window predicts | §6-7, §15 |
| **A synthetic, noise-free, no-learning sanity test reproduces this signature on pure tones.** A flat tone tracks to ~1¢ error; a curved contour (rise-then-fall) shows 17.6¢ median / up to 147¢ error concentrated at the curvature extremum, with monotonic ramps in between (~4.5-4.8¢ median, up to ~33-99¢ at the endpoints) | §18 |
| Harmonic/octave competitors are a real but minor secondary factor (~8-10% of zero-delta failures land near +1 octave), not the dominant mechanism | §17 |

**Primary diagnosis: `ACOUSTIC_REPRESENTATION_LIMITED`**

**Decision gate: `CHANGE_ACOUSTIC_REPRESENTATION`**

---

## 1. Pipeline stage identity (why S and C are one array here)

`AUDIO → [A: CQT log-magnitude] → [S: HPS+learned fusion] → [D0: framewise argmax over S] → [Viterbi] → TRAJECTORY`. In this codebase the "salience score" and the "framewise candidate score" are the *same* array: `fused_probs` from `predecoder_common.py::_fused_probs` (log-linear fusion of the deterministic HPS baseline and the learned per-frame harmonic-salience CNN, Step 12.5's frozen (α, β)) is both what Step 16 calls salience and what D0's `argmax` operates on directly (no separate re-scoring step exists between them). So the three pipeline stages the spec asks about collapse to two empirically distinguishable stages here: **A** (raw CQT magnitude, `linear_mag(cqt_log)`) and **S=C** (fused probs). D0's own behavior (whether the argmax picks the best available candidate) is evaluated separately in §9-12 as "framewise selection."

## 2. Resolution audit — not inferred from array shape

The canonical grid is 10ms (`CQT_HOP=220` samples at `SR=22050`). That is the *output* stride, not the analysis resolution. The actual analysis window is `librosa.filters.wavelet_lengths()` at each CQT bin's center frequency (`filter_scale=1`, `BINS_PER_OCTAVE=72`, `FMIN=75`):

| Candidate-range edge | Frequency | CQT wavelet length |
|---|---:|---:|
| bin 34 (`LRN_LO`) | 104.0 Hz | **998.4 ms** |
| bin 127 (corpus median pitch) | 254.7 Hz | **407.8 ms** |
| bin 244 (`LRN_HI`) | 778.1 Hz | **133.5 ms** |

Every 10ms output frame at the corpus' typical pitch is built from a window covering **~41 output frames of audio**, roughly an order of magnitude longer than the 10-40ms scale at which T1-T3's bends and turns actually happen. This single number is capable, on its own, of producing exactly the staircase/smearing signature audited below — it is the leading theoretical candidate, tested empirically in §4-17 and directly in §18.

## 3. Evaluation masks

All per-frame statistics below use `valid_target`-gated frames only (native 10ms grid, GT never used to build features, only to evaluate them — same convention as every prior step). "Moving" = `|dp/dt| > 100 ¢/s`; "near boundary" = within ±50ms of a primitive start/end (Step 16/17 convention).

## 4. Acoustic (stage A) GT-ridge visibility

1-based rank of GT's true bin within the full 360-bin `mag[:, t]` column, by type:

| Type | Median rank | Mean rank | n |
|---|---:|---:|---:|
| T0 | 8 | 19.7 | 58,178 |
| T1 | 14 | 34.1 | 84,854 |
| T2 | 17 | 37.2 | 13,746 |
| T3 | 26 | 54.7 | 12,372 |

By GT speed bucket (pooled across types):

| |dp/dt| bucket | Median A-rank | n |
|---|---:|---:|
| 0-100 c/s | 8 | 106,501 |
| 100-400 c/s | 16 | 11,228 |
| 400-1000 c/s | 19 | 17,580 |
| >1000 c/s | 31 | 32,590 |

By boundary proximity: **near = 18, away = 9** (103,676 away / 65,474 near). Both the type ordering and the reproduction of Step 10's original T0→T3 degradation (median 8→14→17→26) are a strong internal consistency check that this is measuring a real, reproducible property of stage A, not audit noise. Rank degrades monotonically with speed (8→16→19→31) and is 2x worse near primitive boundaries — both exactly what temporal smearing predicts (faster motion or a nearby discontinuity means the ~200-1000ms window mixes more disagreeing content).

## 5. GT-centered acoustic patches

(Qualitative; see §19/figures.) `predecoder_acoustic_absent.png` shows the clearest failure pattern directly: GT's true ridge (cyan) is essentially invisible in `mag` — the dominant, high-energy ridge in the frame sits ~1150-1200¢ above GT and stays flat while GT moves (a stationary drone/accompaniment partial), and D0 repeatedly locks onto that instead. `predecoder_t0_control.png` shows the opposite: GT's ridge is the dominant, sharply-defined energy band throughout, and D0 tracks it cleanly even through two small T0-internal wiggles.

## 6-7. Acoustic and salience motion contrast, counterfactual paths

Motion contrast `M = (E_GT_path − E_stationary_path) / (E_GT_path + E_stationary_path)` over 100ms (k=10) windows on GT-moving, both-valid frame pairs:

| Type | M_A median | M_A frac>0 | M_S median | M_S frac>0 | n |
|---|---:|---:|---:|---:|---:|
| T0 | 0.065 | 58.9% | 0.149 | 64.3% | 8,474 |
| T1 | 0.018 | 53.7% | 0.060 | 59.8% | 51,367 |
| T2 | 0.027 | 54.8% | 0.057 | 56.8% | 10,652 |
| T3 | **0.0006** | **50.2%** | 0.061 | 62.2% | 10,926 |

**M_S > M_A at every type** — fusion sharpens moving-vs-stationary discriminability relative to raw acoustic evidence; it does not degrade it. T3's raw acoustic contrast is a coin flip (0.0006 median, 50.2% positive) — the acoustic representation alone barely distinguishes T3's true moving path from a frozen path, before salience even runs.

Counterfactual preference (does the acoustic path along GT's actual trajectory score higher than a stationary path, or a path moving at half GT's true speed?):

| Type | GT beats stationary | GT beats half-speed |
|---|---:|---:|
| T0 | 58.9% | 58.9% |
| T1 | 53.7% | 52.2% |
| T2 | 54.8% | 49.9% |
| T3 | 50.2% | **46.1%** |

For T3, the acoustic representation prefers a **half-speed counterfactual over the true motion more often than not** (46.1% < 50%) — a specific, quantitative signature of a low-pass/smearing bias toward underestimating how fast pitch is actually moving.

## 8. Salience (S=C) GT-rank and coverage

Same rank statistic, computed over the 210-candidate `fused_probs` column instead of the full 360-bin `mag` column:

| Type | Median S-rank | Mean S-rank | n |
|---|---:|---:|---:|
| T0 | 3 | 9.2 | 58,178 |
| T1 | 4 | 12.8 | 84,854 |
| T2 | 5 | 16.4 | 13,746 |
| T3 | 7 | 21.7 | 12,372 |

By speed: 3→4→5→9 (0-100 → >1000 c/s). By boundary: near=5, away=3.

## 9. Acoustic → salience degradation

There is no degradation to report: **S is uniformly better than A** (lower rank at every type, every speed bucket, every boundary condition) and its *relative* degradation from T0→T3 is gentler than A's (S: ~2.3x from T0 to T3; A: ~3.3x). Fusion (HPS + learned harmonic-salience CNN) is doing real, positive work on top of the raw acoustic signal — it just cannot recover information that stage A never captured in the first place. This directly argues against `SALIENCE_SCORING_LIMITED` as the primary bottleneck.

## 10. GT-near candidate continuity

Vectorized run-lengths of "rank ≤ 5" (i.e. GT stays a top-5 candidate) at both stages:

| Type | A median run | A p90 | S median run | S p90 |
|---|---:|---:|---:|---:|
| T0 | 120 ms | 520 ms | 110 ms | 530 ms |
| T1 | 30 ms | 190 ms | 40 ms | 240 ms |
| T2 | 30 ms | 130 ms | 30 ms | 200 ms |
| T3 | 30 ms | 140 ms | 30 ms | 190 ms |

A and S show nearly the same continuity pattern (T0 four times longer than T1-T3) — again consistent with S inheriting, not adding to, A's limitation. A 30ms median "GT is a live top-5 candidate" run for moving types is only ~3 native frames — barely longer than the 20-30ms staircase the spec asks to explain, and considerably shorter than the ~130-1000ms CQT analysis windows computed in §2.

## 11-12. Zero-delta causal decomposition (the central table)

For frames where GT is genuinely moving fast (`|dp/dt|>100¢/s`) and D0 nonetheless emits exactly zero pitch change frame-to-frame, an ordered decision tree attributes the cause:

| Type | acoustic_weak_or_absent | salience_lost_candidate | sub_bin_quantization | present_but_not_selected | n |
|---|---:|---:|---:|---:|---:|
| T0 | 48.2% | 16.5% | 27.4% | 7.9% | 369 |
| T1 | 58.5% | 3.3% | 30.7% | 7.5% | 29,030 |
| T2 | 53.8% | 4.4% | 38.2% | 3.6% | 7,286 |
| T3 | 60.6% | 4.3% | 32.1% | 2.9% | 6,997 |

**`acoustic_weak_or_absent`** (A-rank > 20 at the failure frame or its predecessor) is the majority cause at every type, 48-61%. **`sub_bin_quantization`** (GT's true frame-to-frame motion is smaller than one candidate bin, 16.7¢) is a real, non-trivial secondary cause, 27-38%. **`salience_lost_candidate`** (S-rank > 10 despite acceptable A-rank) and **`present_but_not_selected`** (both A and S rank the true candidate well but D0 still picked something else) are minor, together never exceeding ~24% (T0) and typically under 12% (T1-T3). This is the single most decisive result in the audit: the dominant failure mode is that the acoustic evidence for the true pitch is weak or absent *before* salience or selection ever run.

## 13. Frequency-resolution vs. GT motion

Fraction of true (GT) frame-to-frame pitch change smaller than a fraction of one candidate bin (16.67¢):

| Type | frac < 0.5 bin | frac < 1 bin | frac < 2 bin | n |
|---|---:|---:|---:|---:|
| T0 | 99.5% | 99.5% | 99.6% | 57,721 |
| T1 | 68.6% | 83.3% | 94.9% | 84,354 |
| T2 | 63.8% | 83.4% | 95.7% | 13,587 |
| T3 | 57.5% | 77.5% | 93.2% | 12,237 |

For T1-T3, 57-69% of true 10ms motion is genuinely sub-bin. This is a real, quantization-driven ceiling on what any framewise system built on this 16.7¢/210-bin candidate grid could resolve at native rate — it is the correct explanation for `sub_bin_quantization`'s 27-38% share in §11-12, but it cannot explain the *majority* share held by `acoustic_weak_or_absent`, which is a magnitude/visibility problem, not a grid-resolution problem.

## 14. Turning-point temporal response

Mean A-rank in the 5 frames immediately before vs. after a real GT turning point (sign change in GT velocity, 100¢/s deadband both sides):

| Type | Before | After | Δ | n turns |
|---|---:|---:|---:|---:|
| T1 | 45.31 | 45.09 | −0.22 | 1,060 |
| T2 | 38.51 | 42.31 | **+3.79** | 132 |
| T3 | 48.49 | 55.49 | **+7.00** | 32 |
| T0 | — | — | — | 0 (no qualifying turns) |

T1's turns show no post-turn degradation (essentially flat), but T2 and T3 — the types whose defining feature *is* a slope change at a turn — show acoustic evidence getting measurably *worse* right after the turn. This is exactly the signature a wide analysis window predicts: right after a direction change, the window still straddles pre-turn content, diluting/blurring the post-turn ridge before it "catches up." It is also consistent with T3 having the single worst rank statistics throughout this audit (§4, §6-7, §8).

## 15. T2/T3-specific findings

T2 and T3 are the two types this whole investigation exists to fix, and they show the same mechanism as T1 but consistently worse, on every metric measured:

- **Acoustic ridge visibility**: median A-rank T1=14 → T2=17 → T3=26; the >1000¢/s speed bucket (dominated by T2/T3 turns) sits at median rank 31, roughly 4x T0's.
- **Motion contrast**: M_A median falls from T1's 0.018 to T2's 0.027 (comparable) but collapses to **T3's 0.0006** — a coin flip. M_S partially compensates (T3's M_S median 0.061 is actually *higher* than T1's 0.060) but T3's *raw acoustic* discriminability between "GT moved" and "GT stayed put" is essentially absent.
- **Counterfactual bias**: T2 is roughly neutral (GT beats half-speed 49.9%, borderline); **T3 actively prefers the slower counterfactual (46.1%)** — the only type where this happens.
- **Candidate continuity**: T2 and T3 both sit at the 30ms floor for both A and S, identical to T1 — continuity is not what separates T2/T3 from T1.
- **Zero-delta decomposition**: T2 has the highest `sub_bin_quantization` share (38.2%, vs. T1's 30.7% and T3's 32.1%) — consistent with T2 (sloped-start bends) containing the most genuinely-small, fast direction changes right at bend onset. T3 has the highest `acoustic_weak_or_absent` share (60.6%) and the lowest `present_but_not_selected` (2.9%) — when T3 fails, it is overwhelmingly because the evidence was never there, essentially never because a good candidate was visible and skipped.
- **Turning-point response**: T2 (+3.79) and T3 (+7.00) are the only types showing genuine post-turn degradation, and the effect is roughly twice as large for T3 as for T2, in proportion to T3's already-worse baseline rank.

Together this shows T2/T3's extra difficulty over T1 is not a new mechanism — it is the same acoustic-representation limitation, amplified by T2/T3's turning points sitting closer together in time and by T3 specifically triggering the counterfactual-preference bias that indicates active mis-tracking (preferring a slower path), not just weaker tracking.

## 16. T0 control

T0 (stable pitch) is the one condition where every stage performs well, and it quantifies how good the pipeline *can* be when the acoustic representation isn't fighting fast motion:

- Median A-rank 8 (vs. 14-26 for moving types), median S-rank 3.
- Frequency resolution: 99.5% of T0's frame-to-frame GT motion is already sub-half-bin — T0 is, by construction, mostly not asking the system to resolve fine motion at all.
- Candidate continuity: 120ms median run at rank≤5 for A (4x T1-T3's 30ms), 110ms for S.
- Motion contrast is actually the *highest* of all four types (M_A=0.065, M_S=0.149) — counterintuitive at first, but consistent: the "motion" being contrasted for T0 is small internal wobble against a background that stays put almost everywhere else, so the moving/stationary energy separation is easy even though the wobble itself is small.
- `predecoder_t0_control.png` shows this directly — a single sharp, high-energy ridge that D0 tracks cleanly through two small internal T0 wiggles, with essentially no competition from other bins.

T0's strength is not a separate, better-designed pathway — it is the *same* CQT/salience/D0 pipeline succeeding because T0's demands (slow motion, small changes, a stable dominant ridge) sit well inside the ~130-1000ms window's comfort zone, unlike T1-T3.

## 17. Harmonic/drone competitor association

For the same zero-delta failure frames as §11-12, checking whether D0's flat estimate sits near GT ± one octave (±1200¢, 50¢ tolerance):

| Type | octave +1 | octave −1 | other | n |
|---|---:|---:|---:|---:|
| T0 | 8.1% | 3.3% | 88.6% | 369 |
| T1 | 8.4% | 0.8% | 90.8% | 29,030 |
| T2 | 10.0% | 1.5% | 88.5% | 7,286 |
| T3 | 7.9% | 1.6% | 90.5% | 6,997 |

A real but clearly secondary factor: 8-10% of zero-delta failures land near an octave above GT (consistent with the strong stationary partial visible in `predecoder_acoustic_absent.png`, which is offset from GT by roughly this much), effectively none at an octave below. The large majority (88-91%) of failures are not simply "locked onto a harmonic" — they are more diffuse evidence problems, consistent with §11-12's `acoustic_weak_or_absent` describing genuinely low or spread-out acoustic energy rather than a single, identifiable competing tone winning cleanly.

## 18. Synthetic resolution sanity test

A deterministic, no-learning test of stage A alone: four synthetic pitch contours (flat 250Hz, slow linear ramp ~0.5 octave/2s, fast linear ramp ~2 octaves/2s, and a rise-then-fall sinusoid with a true curvature extremum) rendered as pure tones + 4 harmonics, passed through the exact production `cqt_log_magnitude` function (no salience — a trained, real-audio-domain model would be meaningless on pure synthetic tones, per the spec's own permission to omit it here), and the CQT's own argmax ridge compared against the known true trajectory:

| Contour | MAE (¢) | Median abs. err (¢) | Max abs. err (¢) |
|---|---:|---:|---:|
| flat | 1.0 | 1.0 | 1.0 |
| ramp_slow | 5.1 | 4.5 | 32.3 |
| ramp_fast | 8.5 | 4.8 | 99.0 |
| rise_then_fall | **29.3** | **17.6** | **147.3** |

A flat tone is tracked almost perfectly (≈1¢, i.e. bin-quantization noise only). Monotonic ramps are tracked well on average (4.5-4.8¢ median) but show large *transient* error (32-99¢) at their endpoints, where the window is asymmetrically populated. The curved contour — the only one with a genuine turning point, and the closest synthetic analogue to T2/T3 — shows by far the largest and most concentrated error (median 17.6¢, max 147¢, i.e. well over a semitone), visibly located at the curvature extremum in `synthetic_resolution_test.png`. This reproduces, under fully controlled, noise-free, real-melody-free conditions, the same qualitative signature found empirically in §14 (A's evidence gets worse specifically around turning points) and confirms it is a property of the CQT extraction itself, not of real-recording noise, competing instruments, or the learned salience model.

## 19. Representative visualizations

Five real-data cases (`output/pitch_diagnostics/pitch_audit/figures/predecoder_*.png`), selected by extremum of an already-computed diagnostic (not hand-picked for narrative):

- `predecoder_successful_moving_ridge.png` — best D0 tracking on fast T1-T3 motion: A shows a clear, followable ridge, D0 tracks it closely.
- `predecoder_acoustic_absent.png` — worst-rank zero-delta failure: GT's ridge is essentially invisible in both A and S against a dominant stationary partner ~1150-1200¢ above; D0 repeatedly locks onto that instead. The clearest single illustration of `acoustic_weak_or_absent`.
- `predecoder_present_not_selected.png` — the rare case where both A and S rank GT well but D0 still fails; illustrates how small a share of failures this mechanism explains (§11-12).
- `predecoder_t3_turning_point.png` — a T3 segment with several sharp bends: the acoustic ridge visibly blurs/widens and D0 rounds off the sharpest turn (~92.2-92.3s), consistent with §14's post-turn degradation.
- `predecoder_t0_control.png` — clean, high-confidence T0 tracking through small internal wiggles, for contrast.

Plus `synthetic_resolution_test.png` (§18), covering the idealized flat/ramp/curved cases directly.

## 20. Stage-wise information-loss table

| Diagnostic | Acoustic (A) | Salience (S=C) | Framewise selection (D0) |
|---|---:|---:|---:|
| Median GT rank (moving types, T1-T3) | 14-26 | 4-7 | N/A (selection *is* argmax of S) |
| Median GT rank, T0 vs T1-T3 ratio | ~3.3x worse | ~2.3x worse | N/A |
| Motion contrast M (median, T3) | 0.0006 (coin flip) | 0.061 (clearly positive) | N/A |
| GT-beats-half-speed counterfactual (T3) | 46.1% (below chance) | N/A (not computed at S) | N/A |
| GT-near candidate continuity (T1-T3 median) | 30ms | 30-40ms | N/A (inherits A/S continuity) |
| Zero-delta failure share attributable to this stage | **48-61%** (`acoustic_weak_or_absent`) | 3-17% (`salience_lost_candidate`) | 3-8% (`present_but_not_selected`) |
| Sub-bin quantization share (resolution, not this stage alone) | 27-38% (grid property, shared A/S/D0) | — | — |
| Degrades T0→T3 more or less than the other stage | **More** | Less (partially compensates) | — |

## 21. Ranked failure mechanisms

1. **Acoustic temporal blur** (CQT analysis window 130-1000ms vs. 10-40ms motion scale) — supported by §2 theory, §4/§6-7/§8 empirical rank/contrast degradation, §11-12's dominant `acoustic_weak_or_absent` share, §14's post-turn degradation, and §18's synthetic reproduction. **Primary.**
2. **Acoustic/candidate frequency resolution** (16.7¢ bins vs. 57-69% sub-bin true motion) — supported by §13, contributes 27-38% of zero-delta causes. **Real secondary contributor**, not separable from mechanism 1 by architecture (same CQT bin grid) but conceptually distinct (blur = temporal, quantization = frequency).
3. **Weak/absent moving fundamental vs. stationary competitors** (drone/accompaniment partials outranking a moving true pitch) — the concrete mechanism behind `acoustic_weak_or_absent` in cases like `predecoder_acoustic_absent.png`; a sub-case of mechanism 1 (the moving ridge's energy is smeared/diluted while a stationary competitor's is not).
4. **Harmonic/octave competition** — real (~8-10% of zero-delta failures near +1 octave, §17) but small and clearly secondary.
5. **Salience scoring weakness** — S is *better* than A at every stratification; ruled out as an independent failure mechanism (§8-9).
6. **Framewise selection error given adequate evidence** — 3-8% of zero-delta failures (`present_but_not_selected`); a real but minor mechanism (§11-12, §19).

## 22. Primary diagnosis

**`ACOUSTIC_REPRESENTATION_LIMITED`**

The evidence converges from four independent angles: (a) a theoretical CQT window-length calculation showing 130-1000ms smearing against 10-40ms motion; (b) empirical rank/motion-contrast statistics showing stage A degrading substantially more than stage S across type, speed, and boundary strata, with S consistently *improving* on A rather than degrading it; (c) the zero-delta causal decomposition, the audit's central table, attributing 48-61% of failures directly to weak/absent acoustic evidence versus 3-17% for salience and 3-8% for selection; and (d) a controlled synthetic sanity test reproducing the same curvature-localized error signature with no real audio, no salience model, and no learned component at all. `SALIENCE_SCORING_LIMITED` and `FRAMEWISE_SELECTION_LIMITED` are both ruled out as primary by direct measurement (S beats A everywhere; selection-given-good-evidence failures are a small minority). `RESOLUTION_QUANTIZATION_LIMITED` is real (§13, mechanism 2) but is outranked by acoustic weakness/absence as a cause (27-38% vs. 48-61% of zero-delta failures) — it is a genuine contributing factor, not the primary one. `MULTIPLE_PREDECODER_FAILURES` was seriously considered given quantization's non-trivial share, but the >2:1 to >4:1 dominance of `acoustic_weak_or_absent` over every other named cause, at every trajectory type, is decisive enough to name a single primary stage rather than split the diagnosis.

## 23. Decision gate

**`CHANGE_ACOUSTIC_REPRESENTATION`**

Both the diagnostic strength (§20-22) and the fact that salience/selection have already been shown to *not* be the bottleneck argue against `REDESIGN_SALIENCE_OBJECTIVE` or `IMPROVE_FRAMEWISE_PITCH_SELECTION`. `INCREASE_EFFECTIVE_RESOLUTION` targets the real but secondary quantization mechanism and would leave the larger `acoustic_weak_or_absent` mechanism untouched. `TARGET_HIGHEST_IMPACT_PREDECODER_FAILURE` is functionally what this gate already does, since the highest-impact failure (§21, mechanism 1) *is* the acoustic representation.

## 24. Recommendation for Step 20

Per spec, this is not a training run — it is a proposal for what to change and how to test it before committing further engineering effort. The single implicated stage is **A**, specifically its **temporal analysis window** (§2, §14, §18), not the salience model, not the decoder, and not the candidate grid's frequency spacing (a real but secondary contributor).

**Concrete next experiment (Step 20):** hold the model architecture, salience fusion, decoder, and candidate grid completely fixed, and replace only the CQT's temporal-smearing behavior with a shorter-context, comparably-invertible time-frequency front end — e.g. reduce `filter_scale` (directly shortens `wavelet_lengths()` at the cost of coarser low-frequency resolution) or substitute a fixed-window STFT/multi-resolution front end restricted to the existing 104-778Hz candidate band, chosen so the *effective* analysis window at the corpus median pitch (currently 408ms) drops to something closer to the 10-40ms scale of T1-T3 motion. Before any retraining: rerun exactly this step's §4/§6-7/§11-12/§18 diagnostics (acoustic rank, motion contrast, zero-delta decomposition, synthetic curvature test) on the new front end alone, with the existing frozen salience/decoder held out of the loop, to confirm the change actually raises GT's acoustic rank and shrinks `acoustic_weak_or_absent`'s share before spending a training run on it. Only if that diagnostic improvement is confirmed does it make sense to retrain salience/D0/D1 on the new front end and re-measure trajectory macro F1 end-to-end.

## 25. Summary

Oracle pitch achieves trajectory macro F1 = 0.771; the current estimated-pitch system tops out at 0.340 (Step 18). Decoder/lambda tuning is closed (no more headroom there, Step 18). This step traced the gap one stage further upstream and found it does not originate at salience or candidate selection — both perform reasonably well *given what stage A hands them* — but at the acoustic representation itself, whose ~130-1000ms analysis window is temporally too wide for the ~10-40ms scale of T1-T3 pitch bends, corroborated by four independent lines of evidence (theory, empirical rank/contrast statistics, the zero-delta causal table, and a synthetic no-learning sanity test) and by a real but secondary quantization contribution from the 16.7¢ candidate grid. Step 20 should target the acoustic representation's temporal resolution directly, with a cheap pre-training diagnostic check before committing to a full retrain.
