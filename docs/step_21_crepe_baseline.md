# Step 21 — Freeze CREPE as Pitch Source + Return to Trajectory Modeling

This step ends the pitch-frontend research branch (Steps 10-20: CQT/STFT
frontend design, harmonic-salience training, register/octave resolution,
Fused+D3 Viterbi decoding, movement-cost tuning, pre-decoder localization,
the acoustic frontend bake-off). **CREPE (pretrained `torchcrepe`, run on
the vocals-stem-or-source audio) is now the frozen default estimated-pitch
source**, replacing the custom CQT→salience→Viterbi pipeline as the
project's pitch frontend. No further pitch-frontend, salience, decoder, or
CQT/STFT/`filter_scale` optimization is in scope after this step — see §15.

Frozen references: [`docs/step_15_learned_pitch_motion.md`](step_15_learned_pitch_motion.md) (P0-P3 architecture/protocol, D1/oracle reference numbers), [`docs/step_20_acoustic_frontend_bakeoff.md`](step_20_acoustic_frontend_bakeoff.md) (closes the frontend-optimization branch this step formally ends).

Machine-readable outputs: [`output/pitch_diagnostics/relative_pitch/crepe_alignment_validation.json`](../output/pitch_diagnostics/relative_pitch/crepe_alignment_validation.json), [`output/pitch_motion_ablation/condition_P0_CREPE/`](../output/pitch_motion_ablation/condition_P0_CREPE/) (per-fold training + `pooled_test_evaluation.json`), reused from prior steps: [`output/pitch_motion_ablation/evaluation_result.json`](../output/pitch_motion_ablation/evaluation_result.json) (D1/oracle), [`output/pitch_diagnostics/pitch_audit/crepe_comparison.json`](../output/pitch_diagnostics/pitch_audit/crepe_comparison.json) (pre-existing raw pitch-motion diagnostic, reused unchanged for §12).

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.pitch_diagnostics.relative_pitch.validate_crepe_alignment      # §2
python training/train_pitch_motion_ablation.py --condition P0 --pitch-variant CREPE --fold 0 --tiny-overfit 4   # §6 smoke test
python training/train_pitch_motion_ablation.py --condition P0 --pitch-variant CREPE --all-folds --max-epochs 50 --patience 10   # §7
python -m training.pitch_diagnostics.pitch_audit.evaluate_crepe_downstream        # §8-11
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| CREPE integrates cleanly: 17/17 recordings, 169,150/169,150 valid-target frames covered, zero NaN/Inf, exact frame-count match against the canonical grid | §2 |
| CREPE→P0 is non-oracle: only `estimated_pitch` and the pre-existing `valid_target` gate feed the feature function; GT trajectory/boundary/phase never touch feature construction | §4 |
| Fold-wise normalization is genuinely recomputed from CREPE train-only data (not reused from D1) via the existing `estimated_pitch_override` path | §5 |
| Smoke test passes on every checklist item; full 5-fold training completes in ~13 minutes total, no tuning applied | §6-7 |
| **Downstream trajectory macro F1 is worse with CREPE, not better**: pooled 0.320 vs. D1's 0.338 (grouped 5-fold mean 0.299±0.039 vs. 0.348±0.070) | §8 |
| Fold-level: CREPE improves 2/5 folds, worsens 3/5 (median fold delta −0.011); recording-level: improves 8/17, worsens 9/17 (median delta −0.002, mean −0.035) — not a clean, uniform decline either | §9-10 |
| **The entire net decline is concentrated in one class**: T3 F1 collapses from 0.085 (D1) to 0.003 (CREPE) — essentially zero in every one of the 5 folds/17 recordings — while T0 F1 *improves* (0.547→0.576) and T2 is flat-to-slightly-better (0.150→0.155) | §11 |
| **This directly contradicts the raw pitch-motion diagnostic**: CREPE beats D1 on absolute MAE (all 4 types), Δ50/100/200ms MAE, T1-T3 turning-point recall (0.273→0.574 mean, T3 alone 0.281→0.594), velocity correlation (0.102→0.278), and shows none of D1's exact-zero staircasing (77.4%→0.003% of GT-fast frames) — CREPE is a strictly better *contour* by nearly every classic motion-fidelity metric, yet the downstream T3 classifier fails outright on it | §12 |
| CREPE does not close any of the oracle gap — it widens it slightly (pooled: −4.1% of the D1→oracle gap; grouped mean: −11.6%) | §13 |

**Primary outcome: `CREPE_WORSE_THAN_D1`**

**Per spec: CREPE remains the frozen default pitch source regardless (§1's own top-level directive); this result is reported, not acted on by reverting or reopening frontend search.**

---

## 1-3. Integration and representation

`training/train_pitch_motion_ablation.py --pitch-variant CREPE` was added mirroring the existing D0/lambda-sweep override pattern exactly: `training.pitch_diagnostics.relative_pitch.dense_crepe_path.build()` supplies `estimated_pitch_override`, and everything downstream (condition P0's `FramewiseConditionalTCNModel`, `fold_pitch_stats_for`, `compute_phi`'s fixed 10/50/100/200ms octave-unwrapped-delta representation) is untouched — the only change from the existing D1 P0 baseline is the pitch source feeding `compute_phi`. No absolute pitch, audio, CQT, salience, confidence, phase, boundary, or GT-pitch feature was added; `dense_crepe_path.py` itself (already existing, unmodified) was used as-is, including its already-defined vocals-stem-or-source fallback (§16 of the original spec — that variable was not touched here).

## 2. Alignment / coverage validation

`training/pitch_diagnostics/relative_pitch/validate_crepe_alignment.py` (new) checks recording IDs, lane IDs, frame counts, the 10ms native grid, `valid_target` coverage, and finite/missing values, recording-by-recording, against `RecordingLaneIndex` — the same index D1 and every other pitch-path variant is built from.

| Check | Result |
|---|---:|
| Recordings: canonical / CREPE cache / match | 17 / 17 / **exact match** |
| Total valid-target frames | 169,150 |
| CREPE-covered valid frames | 169,150 (**100.0000%**) |
| Missing/invalid valid frames | 0 |
| NaN/Inf on the full native grid (all 17 recordings) | 0 |
| Frame-timestamp grid | uniform 10ms, 0 deviation, all recordings |

No systematic alignment problem; nothing was dropped. Full per-recording detail in `crepe_alignment_validation.json`.

## 4. Non-oracle feature audit

Traced the CREPE→P0 path: `dense_crepe_path.build()` returns per-recording log2-Hz arrays computed purely from audio (torchcrepe forward pass, no annotation input). `train_fold` passes this as `estimated_pitch`; `FramewiseExcerptDataset`/`FullRecordingDataset` convert it to cents via `log2_hz_to_cents` (uses only `lane.fundamental_hz`, a fixed per-recording tonic constant, not derived from any trajectory annotation) and call `compute_phi(cents, valid_target)`. `valid_target` is the same annotation-derived gating mask D1's P0 has always used — it zeroes the delta feature at positions where GT itself doesn't trust the frame, but contributes no directional/type/boundary/phase information, exactly the pre-existing, already-audited D1 pattern (Step 14 spec section 6's C/D-conditions requirement). GT trajectory type is used only as the loss target (`FramewiseTypeLoss`) and for evaluation, never as a model input. **Confirmed: no GT trajectory type, primitive boundary, phase, GT pitch, or future annotation metadata reaches feature construction.**

## 5. Fold-specific normalization / leakage

`fold_pitch_stats_for("P0", split.train_recording_ids, index, estimated_pitch)` recomputes `phi_mu`/`phi_sigma` from whichever `estimated_pitch` source is passed in — for this run, CREPE's train-only recordings, per fold, not reused from D1. Confirmed directly: the smoke-test checkpoint's saved `phi_mu`/`phi_sigma` (`[0.79, 3.56, 6.75, 9.98]` / `[18.3, 71.9, 106.7, 146.8]`) differ substantially in scale from D1's own fold stats — a live check that the two sources are not sharing statistics. Grouped 5-fold CV, `performance_group_id` grouping, and train/val/test separation are unchanged (same `grouped_kfold_k5_seed42.json` manifest, same `prepare_fold`/`build_fold_split` code path D1 uses). The existing leakage check (`assert_no_split_leakage` → `check_leakage`, verifying no shared `audio_id` or `performance_group_id` across train/val/test) runs automatically inside `prepare_fold` on every fold of every training call — it ran, and passed, for all 5 CREPE folds as an unavoidable side effect of the training run in §7 completing without error.

## 6. Smoke test

`--condition P0 --pitch-variant CREPE --fold 0 --tiny-overfit 4`, 50 epochs on 4 cached excerpts:

| Check | Result |
|---|---|
| CREPE variant loads | ✓ (from cache) |
| P0 features build | ✓ |
| Dataloader | ✓ |
| Tensor shapes | ✓ (`[B, T, 4]` phi input, matches D1) |
| Loss finite | ✓ every epoch |
| Gradients update | ✓ — train macro F1 rises 0.18→0.87 over 50 epochs on the tiny cache |
| Validation runs | ✓ |
| Checkpoint save/reload | ✓ — reloaded `best.pt` (epoch 37, val macro F1 0.311) into a fresh model, state dict matches |

No hyperparameter was changed based on this result; it was discarded and the full run launched immediately.

## 7. Training protocol

Identical to the existing P0/D1 protocol (Step 15, unchanged by Steps 17/18's D0/lambda-sweep variants): `FramewiseConditionalTCNModel(use_audio=False, use_pitch=True, pitch_dim=4)`, AdamW (`lr=1e-3`, `weight_decay=1e-4`), `FramewiseTypeLoss` (unweighted CE, valid-frame-masked), batch size 8, 512 excerpts/epoch, grad clip 1.0, max 50 epochs / patience 10, seed 42 (+fold offset), same 5-fold grouped manifest. Only the pitch source changed. Full run: 5 folds in ~13 minutes wall-clock (each fold ~0.6-0.7s/epoch, matching D0/lambda-sweep run costs); no hyperparameter was tuned for CREPE specifically.

| Fold | Best val epoch | Best val macro F1 |
|---|---:|---:|
| 0 | 10 | 0.303 |
| 1 | 6 | 0.336 |
| 2 | 7 | 0.353 |
| 3 | 4 | 0.222 |
| 4 | 5 | 0.320 |

## 8. Primary comparison

Held-out **test-set** pooled macro F1 (Step 15's evaluation harness; D1/oracle reused unmodified from `evaluation_result.json`, not retrained, per spec):

| Pitch source | Macro F1 | T0 F1 | T1 F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|---:|
| D1 | 0.3375 | 0.547 | 0.568 | 0.150 | 0.085 |
| **CREPE** | **0.3199** | **0.576** | **0.546** | **0.155** | **0.003** |
| Oracle | 0.7705 | 0.733 | 0.768 | 0.759 | 0.822 |

Grouped 5-fold mean ± std:

| Pitch source | Mean macro F1 | Std |
|---|---:|---:|
| D1 | 0.3484 | 0.0703 |
| **CREPE** | **0.2985** | **0.0386** |
| Oracle | 0.7777 | 0.0610 |

Both the pooled and grouped-mean primary metric go down with CREPE, not up.

## 9. Fold consistency

| Fold | D1 | CREPE | Δ |
|---|---:|---:|---:|
| 0 | 0.2482 | 0.3391 | **+0.0909** |
| 1 | 0.3246 | 0.3132 | −0.0114 |
| 2 | 0.3224 | 0.3282 | +0.0058 |
| 3 | 0.3904 | 0.2793 | −0.1112 |
| 4 | 0.4565 | 0.2326 | **−0.2239** |

Folds CREPE improves: 2/5 (0, 2). Folds CREPE worsens: 3/5 (1, 3, 4). Median fold delta: **−0.0114**. Not a uniform decline — fold 0 improves substantially — but the two worst folds (3, 4) are large enough to pull both the pooled and grouped-mean metric down.

## 10. Recording consistency

17/17 held-out recordings, D1 vs. CREPE macro F1:

| Recording | Fold | D1 | CREPE | Δ |
|---|---:|---:|---:|---:|
| 6503e36cd9ff49d3988d0b40 | 0 | 0.2756 | 0.3887 | +0.1131 |
| 65b14e207f607fb149202019 | 0 | 0.2173 | 0.1555 | −0.0619 |
| 65b2ab707f607fb14920201a | 0 | 0.2612 | 0.2709 | +0.0098 |
| 6655f08ad5788878a197c5a5 | 0 | 0.2480 | 0.2465 | −0.0015 |
| 68f53fbf9d93a4cd2923711c | 0 | 0.2467 | 0.3762 | **+0.1295** |
| 6503e348d9ff49d3988d0b3f | 1 | 0.2086 | 0.2203 | +0.0117 |
| 6824de49abc4705438ce918b | 1 | 0.3100 | 0.2592 | −0.0508 |
| 6912841f213d07041b95a800 | 1 | 0.3834 | 0.3949 | +0.0115 |
| 6417585554a0bfbd8de2d3ff | 2 | 0.3186 | 0.3453 | +0.0267 |
| 645ff354deeaf2d1e33b3c44 | 2 | 0.3484 | 0.2795 | −0.0689 |
| 6653ce5fd5788878a197c57f | 2 | 0.2297 | 0.2521 | +0.0224 |
| 6653d349d5788878a197c580 | 2 | 0.2775 | 0.2786 | +0.0011 |
| 65e4a79cc7b694145529a3f1 | 3 | 0.3285 | 0.2223 | −0.1062 |
| 66552c6bd5788878a197c590 | 3 | 0.2951 | 0.1291 | **−0.1660** |
| 68d85d4570785f961df2499d | 3 | 0.3923 | 0.2970 | −0.0952 |
| 6491d48d608d1718e0311003 | 4 | 0.4750 | 0.3160 | **−0.1590** |
| 692ed7e6213d07041b95a80d | 4 | 0.2674 | 0.0634 | **−0.2040** |

Recordings improved: 8/17. Recordings worsened: 9/17. Median delta: **−0.0015** (essentially flat). Mean delta: **−0.0346** (pulled down by outliers). Largest outliers: `692ed7e6213d07041b95a80d` (−0.204, worst — this is the same `OUTLIER_RECORDING` flagged in `pitch_audit/common.py` since Step 16) and `66552c6bd5788878a197c590`/`6491d48d608d1718e0311003` (−0.166/−0.159, both fold 3/4). Largest gains: `68f53fbf9d93a4cd2923711c` (+0.130) and `6503e36cd9ff49d3988d0b40` (+0.113), both fold 0. No recording is dropped or removed from this table.

## 11. Per-class behavior

| Type | D1 P | D1 R | D1 F1 | CREPE P | CREPE R | CREPE F1 |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 0.498 | 0.608 | 0.547 | 0.475 | **0.731** | **0.576** |
| T1 | 0.572 | 0.564 | 0.568 | 0.588 | 0.509 | 0.546 |
| T2 | 0.219 | 0.114 | 0.150 | 0.259 | 0.111 | 0.155 |
| T3 | 0.115 | 0.068 | 0.085 | 0.082 | **0.0015** | **0.003** |

CREPE's confusion matrix (pooled test set, rows = true, cols = predicted T0/T1/T2/T3): T3 row = `[3754, 8174, 426, 18]` out of 12,372 support — 66% of true T3 frames are predicted T1, 30% predicted T0; only 18 frames (0.15%) are correctly predicted T3, in **every one of the 5 folds and 16 of 17 recordings** (per-recording T3 F1 is exactly 0.0 in 16/17 recordings; the other two are 0.058 and 0.015). This is not one bad fold or one bad recording — it is a uniform, essentially total failure to predict the rarest class (T3, ~7% of valid frames), specific to CREPE.

Per spec's framing: this is **not** "CREPE helps T2/T3 while hurting T0/T1" — it is closer to the reverse. T0 improves clearly (+0.029 F1, driven by a large recall gain, 0.608→0.731). T2 is flat-to-marginally-better (+0.005). T1 is flat-to-slightly-worse (−0.022). T3 collapses (−0.082, essentially to zero). Arithmetically, T3's collapse alone accounts for essentially all of the net pooled macro-F1 decline (−0.0176): T0's and T2's small gains almost exactly cancel T1's small loss, leaving T3 as the dominant term. **Overall macro F1 remains the primary metric, and by that metric CREPE is worse — but the mechanism is a single-class collapse, not a broad-based regression.**

## 12. Small motion diagnostic

Reused directly, unmodified, from the pre-existing `crepe_comparison.json` (built by `training/pitch_diagnostics/pitch_audit/crepe_comparison.py`, itself dated before this step — no new diagnostic infrastructure was written). This compares raw decoded/estimated pitch contours (D1 vs. CREPE) against GT using the same Step 16-17 motion-fidelity utilities:

| Metric | D1 | CREPE | Direction |
|---|---:|---:|---|
| Absolute pitch MAE (T3) | 403.6¢ | 280.5¢ | CREPE better |
| Δ50ms MAE | 39.9¢ | 37.1¢ | CREPE better |
| Δ100ms MAE | 61.5¢ | 53.8¢ | CREPE better |
| Δ200ms MAE | 83.3¢ | 71.1¢ | CREPE better |
| R50 (GT-moving attenuation ratio, →1 is unbiased) | 0.394 | 0.611 | CREPE better |
| R100 | 0.682 | 0.571 | **CREPE worse** |
| R200 | 0.833 | 0.637 | **CREPE worse** |
| Frac. exactly-zero delta when GT moving fast | 77.4% | 0.003% | CREPE far less staircased |
| Velocity correlation | 0.102 | 0.278 | CREPE better |
| Turning recall @50ms (T1-T3 mean) | 0.273 | 0.574 | CREPE far better |
| Turning recall @50ms, T3 alone | 0.281 | 0.594 | CREPE far better |
| Moving-region median run length | 3.0 frames | 1.0 frame | CREPE: no smoothing at all |

**Answer to the mandated question**: no, CREPE's better trajectory-relevant pitch motion does **not** translate into better trajectory classification — if anything the opposite, concentrated entirely in T3. CREPE dominates D1 on absolute error, short/medium-scale delta error, turning-point recall (especially T3's own turning-point recall, which nearly doubles), and velocity correlation. The one metric where D1 pulls ahead is R100/R200 (CREPE is somewhat more attenuated than D1 at the 100-200ms scale, though both are well below 1 — i.e., both still underestimate true motion magnitude at that scale, CREPE more so). The most likely mechanical explanation, stated as a hypothesis rather than a proven cause: D1's Viterbi decode is heavily staircased (77% exact-zero deltas, 3-frame median run length) — a signal texture the existing φ/normalization/TCN protocol was implicitly validated against across Steps 14-20. CREPE is the opposite: continuous, dense, almost never exactly flat, 1-frame run length. The same fixed architecture and training protocol, unchanged for this new signal texture, may simply not be well suited to it, particularly for the rarest, most motion-dependent class (T3). This is exactly the kind of question Step 21 explicitly redirects to *trajectory* modeling (§13) rather than more pitch-frontend work.

## 13. Remaining oracle gap

| Source | Pooled macro F1 | Grouped mean |
|---|---:|---:|
| D1 | 0.3375 | 0.3484 |
| CREPE | 0.3199 | 0.2985 |
| Oracle | 0.7705 | 0.7777 |

D1→CREPE change: **−0.0176** pooled / **−0.0499** grouped mean (a regression, not an improvement).
CREPE→Oracle remaining gap: 0.4506 pooled / 0.4792 grouped mean, vs. D1→Oracle's 0.4330 pooled / 0.4293 grouped mean.

CREPE does not close any of the oracle gap — it is **slightly wider** with CREPE than with D1 (pooled: gap grows by 4.1% relative to D1's own gap; grouped mean: 11.6%). This confirms the spec's own caution: better raw pitch estimation (§12) did not close the oracle gap, and in this specific architecture/protocol it made pooled/grouped downstream performance slightly worse, entirely via the T3 mechanism in §11.

## 14. Interpretation — primary outcome

**`CREPE_WORSE_THAN_D1`**

Both the pooled and grouped-mean primary metric decline with CREPE (§8), and while the decline is not a uniform fold- or recording-level regression (§9-10: 2/5 folds and 8/17 recordings actually improve), it is driven by a real, severe, and completely consistent single-class failure (§11: T3 F1 essentially zero in 16/17 recordings and all 5 folds) that arithmetically accounts for nearly all of the net decline. `CREPE_CLASS_TRADEOFF` was considered but does not fit as cleanly — that outcome anticipates offsetting gains and losses that leave the pooled metric "effectively unchanged"; here the net effect is a real, if modest, decline, not a wash, and the asymmetry (T0/T2 slightly better, T3 catastrophically worse) is closer to a single-class failure than a broad tradeoff.

Per spec: **report this result; do not begin another pitch-frontend search.** CREPE remains the frozen default pitch source per §1's own directive, independent of this specific architecture's downstream result — the open question this raises is squarely about trajectory modeling (how the existing φ/normalization/TCN protocol handles a continuous, non-staircased signal, especially for the rare T3 class), not about pitch estimation quality, which §12 already shows CREPE wins on by nearly every classic measure.

## 15. Scope boundary

No further pitch-frontend, CQT/STFT/`filter_scale`, salience, or Viterbi/decoder work is proposed or in scope after this step, regardless of §14's outcome. Vocals-stem-vs-source CREPE ablation (spec's own §16) was explicitly not touched — `dense_crepe_path.py`'s pre-existing per-recording vocals-or-source fallback was used unchanged and consistently.

## Recommendation for the next trajectory experiment

§12's disconnect — CREPE wins decisively on essentially every raw motion-fidelity metric (including T3's own turning-point recall, +111% relative) yet the downstream T3 classifier fails outright on it — points at the **trajectory input representation**, not pitch estimation, as the next question. Concretely: test whether the fixed φ(10/50/100/200ms)-delta representation and its per-fold z-score normalization, both implicitly tuned against D1's heavily-staircased signal texture (77% exact-zero deltas, 3-frame run length) across Steps 14-20, are mismatched to CREPE's continuous, always-moving signal (0.003% exact-zero, 1-frame run length) — e.g. a fixed light smoothing/denoising of the CREPE contour as a trajectory-input preprocessing choice (not a frontend change), or a T3-aware loss/capacity adjustment, before concluding CREPE cannot support better-than-D1 trajectory classification. This is a trajectory-representation question, consistent with §15's redirection, not a reopening of pitch-frontend research.
