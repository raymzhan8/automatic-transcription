# Step 18 — Trajectory-Optimized Movement-Cost Selection

> **Follow-up:** the "move upstream" recommendation was carried out in [`docs/step_19_predecoder_evidence_localization.md`](step_19_predecoder_evidence_localization.md), tracing the gap one stage further to the pre-decoder pipeline. Diagnosis: `ACOUSTIC_REPRESENTATION_LIMITED` — the CQT's own analysis window (130-1000ms) is the dominant bottleneck, not salience or framewise selection (both perform reasonably well given what the acoustic stage hands them). Decision gate: `CHANGE_ACOUSTIC_REPRESENTATION`.

A closing hyperparameter ablation — **no Viterbi cost-function changes, no new lambda values, no acoustic/salience retraining, no architecture changes, no class weighting, no T2/T3-specific objective.** Step 17 established a clean, monotonic tradeoff along the existing movement-cost weight `lambda_t` (absolute pitch accuracy vs. turning-point fidelity) but trained downstream trajectory classifiers only at the two endpoints (0x, 1.0x). This step trains the two missing intermediate points and asks one narrow question: does an intermediate setting actually improve four-class trajectory transcription, using trajectory macro F1 — not any diagnostic proxy — as the sole selection criterion.

Frozen references: [`docs/step_17_pre_post_viterbi_fidelity.md`](step_17_pre_post_viterbi_fidelity.md), [`docs/step_16_acoustic_pitch_audit.md`](step_16_acoustic_pitch_audit.md).

Machine-readable outputs: [`output/pitch_motion_ablation/step18_lambda_comparison.json`](../output/pitch_motion_ablation/step18_lambda_comparison.json).

Reproduce (from repository root, `idtap` env):

```bash
python training/train_pitch_motion_ablation.py --condition P0 --pitch-variant 0.25x --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P0 --pitch-variant 0.5x  --all-folds --max-epochs 50 --patience 10
python -m training.pitch_diagnostics.pitch_audit.evaluate_lambda_downstream
```

**Data-integrity note, disclosed up front:** Step 17/18's variant-training runs all stage through the same `condition_P0/` directory (`train_pitch_motion_ablation.py`'s `train_fold` always writes there regardless of `--pitch-variant`, then the script renames it to a variant-specific directory afterward). Running `--pitch-variant D0` in Step 17 overwrote Step 15's original L100/D1-trained checkpoint *before* it could be renamed away, and no earlier copy survived — a real bug, not fixed retroactively here since fixing it now would not recover the lost files. L100's checkpoint files are gone, but Step 15's complete evaluation output (pooled, per-fold, per-recording metrics) was saved to a separate JSON at the time and is unaffected; all L100 numbers in this report are reused verbatim from that file, not re-derived. L0/L25/L50 were evaluated fresh in this step from their intact checkpoints.

---

## Executive summary

| Finding | Evidence |
|---|---|
| Pooled and grouped-mean macro F1 both show **L50 nominally highest, L0 clearly worst** | Pooled: L0 0.330 / L25 0.336 / **L50 0.340** / L100 0.338. Grouped mean: L0 0.335 / L25 0.349 / **L50 0.353** / L100 0.348 |
| **But this apparent win does not survive fold-level scrutiny** | L50 improves only **2 of 5 folds** vs. L100 (worsens 3); L25 improves 2 of 5 (worsens 3) |
| **Per-recording, the result is a minority improvement with a negative median** | L50 vs. L100: **7/17 recordings improved, 10/17 worsened**, median per-recording delta **−0.004** |
| **The entire pooled/grouped "win" traces to one fold** | Fold 0 alone contributes deltas of +0.11 to +0.15 across 4 of its 5 recordings for both L25 and L50; every other fold is flat-to-negative, fold 4 shows a clear regression (−0.08 to −0.10) |
| Same pattern holds **within T2 specifically** — not just pooled | T2 F1 for L25/L50 spikes to 0.48-0.49 in fold 0 alone vs. 0.06-0.30 in every other fold; fold 2 even shows L50's T2 *worse* than L0's |
| The known `692ed7e6…` outlier **remains decoder-insensitive**, as Step 17 found | L0/L25/L50/L100 deltas on this recording are all within ±0.004-0.03, far smaller than the fold-4 swing driven by its fold-mate `6491d48d…` |
| No setting comes remotely close to the oracle ceiling | Best pooled macro F1 across all four settings: 0.340, vs. oracle (P3, Step 15) 0.771 |

**Primary outcome: `NO_MEANINGFUL_LAMBDA_DIFFERENCE`**

**Decision gate: `FREEZE_LAMBDA_AND_MOVE_UPSTREAM`**

---

## 1-3. Four lambda conditions, training equivalence, primary metric

| Label | `lambda_t` multiplier | Status entering this step |
|---|---|---|
| L0 | 0x | Trained in Step 17 (= D0) |
| **L25** | 0.25x | **Trained in this step** |
| **L50** | 0.5x | **Trained in this step** |
| L100 | 1.0x | Trained in Step 15 (= D1, current system) |

L25/L50 used the exact Step 15/17 P0 setup: `training/train_pitch_motion_ablation.py --condition P0 --pitch-variant {0.25x,0.5x}` — identical architecture (`FramewiseConditionalTCNModel(use_pitch=True)`), identical grouped 5-fold splits/seeds, 50 epochs/patience 10, unweighted CE on `valid_target`-masked frames, identical excerpt sampler, and fold-wise φ normalization statistics re-derived from each condition's own train recordings (matching Step 17's D0 protocol exactly). No training-protocol difference between any of the four conditions. Both new runs converged normally within budget (no run hit the 50-epoch ceiling).

**Primary selection metric: grouped 5-fold trajectory macro F1** (both pooled and grouped-mean reported, per spec section 3 — neither absolute pitch MAE nor T2/T3 F1 alone was used to select).

---

## 4. Central four-lambda comparison table

| lambda | Pooled macro F1 | Grouped mean ± std | T0 F1 | T1 F1 | T2 F1 | T3 F1 | Turn recall @50ms | Abs. pitch MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0x (L0) | 0.330 | 0.335 ± 0.038 | 0.507 | 0.547 | 0.171 | 0.093 | 0.406 | 368.4¢ |
| 0.25x (L25) | 0.336 | 0.349 ± 0.062 | 0.506 | 0.556 | 0.186 | 0.095 | 0.345 | 359.7¢ |
| 0.5x (L50) | **0.340** | **0.353 ± 0.070** | 0.512 | **0.581** | **0.187** | 0.079 | 0.314 | 355.6¢ |
| 1.0x (L100) | 0.338 | 0.348 ± 0.070 | **0.547** | 0.568 | 0.150 | **0.085*** | 0.273 | **349.1¢** |

*L100's per-class figures are pooled-only (its per-fold breakdown was lost along with the checkpoint — see the data-integrity note above; the pooled numbers themselves are unaffected, reused directly from Step 15's saved output).

At face value this table alone would suggest L50 is a mild winner — see §5 for why that reading does not hold up.

---

## 5. Fold consistency

Per-fold macro F1:

| Fold | L0 | L25 | L50 | L100 |
|---|---:|---:|---:|---:|
| 0 | 0.288 | 0.360 | 0.369 | 0.248 |
| 1 | 0.314 | 0.314 | 0.314 | 0.325 |
| 2 | 0.316 | 0.323 | 0.318 | 0.322 |
| 3 | 0.368 | 0.388 | 0.391 | 0.390 |
| 4 | 0.388 | 0.361 | 0.375 | **0.457** |

Deltas vs. L100, folds improved/worsened:

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Improved / worsened** |
|---|---:|---:|---:|---:|---:|---|
| L0 − L100 | **+0.040** | −0.011 | −0.007 | −0.023 | −0.069 | 1 / 4 |
| L25 − L100 | **+0.112** | −0.010 | +0.001 | −0.002 | **−0.096** | 2 / 3 |
| L50 − L100 | **+0.121** | −0.011 | −0.005 | +0.000 | **−0.082** | 2 / 3 |

**A minority of folds improve for every non-L100 setting.** The entire pooled/grouped-mean edge for L25/L50 is generated by fold 0's large, isolated gain (+0.11 to +0.12); folds 1-2 are flat (within ±0.011, noise-level); fold 3 is a wash; fold 4 is a clear, substantial *loss* (−0.08 to −0.10) that nearly cancels fold 0's gain in the pooled aggregate. This is precisely the "tiny pooled improvement driven by one fold" pattern spec section 5 explicitly warned against checking for — and it is present.

---

## 6. Per-class (T0-T3) tradeoff

Step 17's finding (less smoothing → better T2/T3, more smoothing → better T0/T1) is visible in the *pooled* numbers (T2: 0.171→0.187 from L0→L50; T0: 0.507→0.547 from L0→L100) but **does not hold up fold-by-fold even restricted to T2 alone**:

| Fold | L0 T2 | L25 T2 | L50 T2 |
|---|---:|---:|---:|
| 0 | 0.160 | **0.491** | **0.483** |
| 1 | 0.198 | 0.163 | 0.187 |
| 2 | 0.070 | 0.107 | 0.059 |
| 3 | 0.264 | 0.267 | 0.218 |
| 4 | 0.302 | 0.191 | 0.142 |

Fold 0 alone accounts for essentially all of L25/L50's pooled T2 advantage (0.48-0.49 there vs. 0.06-0.30 everywhere else); fold 2 shows L50's T2 *below* L0's; fold 4 shows both L25 and L50 *below* L0's T2. **The per-class tradeoff Step 17 identified is real in direction (confirmed again by the monotonic diagnostic sweep, unaffected by this step's findings) but is not reliably large enough at the downstream-classifier level to produce a consistent, fold-general improvement** — the training/evaluation noise at this dataset's scale (17 recordings) is comparable to or larger than the effect.

---

## 7. Per-recording comparison

L50 vs. L100, all 17 recordings:

| | n improved | n worsened | median Δ | mean Δ |
|---|---:|---:|---:|---:|
| L50 − L100 | **7** | **10** | **−0.004** | +0.011 |

A minority of recordings improve; the median delta is slightly *negative* (the mean is pulled positive only by fold 0's four large outliers: +0.123, +0.149, +0.051, +0.108). The fifth fold-0 recording (`65b2ab70…`) actually *worsens* (−0.064) — even fold 0 is not unanimous. Fold 4's regression is driven by `6491d48d…` (−0.095), not by `692ed7e6…` (+0.004, negligible) — **`692ed7e6…` remains decoder-insensitive across every lambda tested**, exactly reconfirming Step 17's finding and ruling it out as a factor in this step's results.

---

## 8. Relating lambda to pitch MAE, turning recall, and trajectory F1

| | Prefers | Basis |
|---|---|---|
| Absolute pitch MAE | **1.0x** (L100) | Monotonic, Step 17 |
| Turning-point recall | **0x** (L0) | Monotonic, Step 17 |
| Trajectory macro F1 (pooled/grouped) | Nominally 0.5x, but **not fold/recording-robust** | This step |

The hoped-for clean result — "trajectory macro F1 peaks at an intermediate point, proving absolute pitch was the wrong tuning criterion" — is only superficially true. It is true that the pooled numbers peak at 0.5x, and it would have been a strong, useful result if that peak had been broadly supported. It is not: the peak is a single-fold artifact sitting inside noise comparable to the effect size itself. Restated honestly: **this dataset (17 recordings, 5 folds) does not have enough statistical power to distinguish these four operating points on the trajectory task**, even though it clearly *can* distinguish them on the underlying motion-fidelity diagnostics (Step 17's monotonic sweep, which pools far more, finer-grained frame-level observations than the 17-recording, class-level trajectory comparison here).

---

## 9-10. No over-tuning, and the primary outcome

Per spec section 9: no intermediate point "clearly wins" once fold/recording consistency is checked (the explicit, required check) — so the correct response is to **keep the simplest, best-supported existing setting and close lambda tuning**, not to search further values (0.3x, 0.4x, ...), and not to force a change based on a pooled number that does not survive scrutiny.

**`NO_MEANINGFUL_LAMBDA_DIFFERENCE`**

> All four conditions are effectively within experimental variation at the downstream trajectory-classification level. Movement-cost tuning does redistribute per-class performance in the direction Step 17 predicted (confirmed again, directionally, in the pooled numbers and even within T2 alone) — but not with enough consistency, at this dataset's scale, to constitute a reliable trajectory-transcription improvement. It does not explain the large estimated-vs-oracle gap (best setting: 0.340 vs. oracle 0.771).

Not `INTERMEDIATE_LAMBDA_IMPROVES_TRAJECTORIES` — that outcome requires "a clear, reasonably fold/recording-consistent macro-F1 improvement," and §5/§7 show the opposite (minority of folds and recordings improve, median recording delta negative). Not `FULL_VITERBI_STILL_BEST` — L100 is not clearly best either; L0 is clearly *worst* (lowest on every aggregate metric), but among L25/L50/L100 no setting dominates consistently. Not `NO_SMOOTHING_BEST` — L0's pooled and grouped-mean macro F1 are the lowest of the four, ruling this out unambiguously.

---

## 11. Decision gate

**`FREEZE_LAMBDA_AND_MOVE_UPSTREAM`**

Differences among L25/L50/L100 are small and inconsistent (§5, §7); L0 is ruled out as clearly worse. Per spec, **no further lambda sweep in Step 19** — this step closes movement-cost-weight tuning. The current system (`lambda_t` at its Step 12.5-selected value, i.e. L100) is retained by default since no alternative earned its place, not because it was reconfirmed as optimal.

---

## 12. Recommendation for Step 19

As anticipated going into this step (spec section 12): since the outcome is one of the three "move upstream" cases, Step 19 should investigate the **pre-decoder** temporal-resolution problem directly. Step 17 already showed that even at `lambda_t=0` (no temporal decoding at all): R50 = 0.394, moving-region run length = 2.33 frames (vs. GT's ~1), 71.1% of frames are exactly zero-delta when GT is moving fast, and downstream trajectory macro F1 is only 0.330 — nowhere near oracle's 0.771. Decoder tuning, now closed off in both directions (Step 17's redesign question and this step's retuning question), cannot plausibly close that gap on its own. The remaining, unexamined stage is the framewise acoustic/salience evidence itself — Step 16 §12 already found real per-frame degradation specifically for T2/T3 (weaker coverage, lower confidence at the true pitch) that this step's results are consistent with, not contradicted by.
