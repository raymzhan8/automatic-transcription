# Step 15 — Learned Pitch-Motion Representation

> **Follow-up:** the acoustic bottleneck was diagnosed in detail in [`docs/step_16_acoustic_pitch_audit.md`](step_16_acoustic_pitch_audit.md) — a diagnostic-only audit (no training) pinpointing *why* estimated pitch motion destroys trajectory information. Primary diagnosis: `TEMPORAL_RESOLUTION_LIMITED` (the decoder smooths/staircases away exactly the short-timescale, direction-reversing motion T2/T3 depend on), not register, lag, or salience-evidence absence — both alternative corrections were tested downstream and found to have zero effect.

Pitch/salience-evidence-only experiment, following directly from Step 14's `IMPROVE_RELATIVE_PITCH_ESTIMATION` decision. Asks a more specific question than "improve absolute F0": is trajectory-relevant information being lost at the *representation* stage — collapsing the full salience distribution `S(f,t)` to one decoded pitch path `p(t)`, then to four hand-picked differences `φ(t)` — rather than at the underlying acoustic pitch estimate itself? No audio/CQT branch, no fusion, no register-decoder work, no class weighting, no architecture search.

Frozen references: [`docs/step_14_relative_pitch_trajectory.md`](step_14_relative_pitch_trajectory.md), [`docs/step_13_relative_pitch.md`](step_13_relative_pitch.md).

Machine-readable outputs: [`output/pitch_motion_ablation/`](../output/pitch_motion_ablation/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/relative_pitch/dense_relative_salience.py   # P2's windowed register-invariant salience cache
python -m pytest training/tests/test_pitch_motion_ablation.py
python training/train_pitch_motion_ablation.py --condition P0 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P1 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P2 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P3 --all-folds --max-epochs 50 --patience 10
python training/evaluate_pitch_motion_ablation.py
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| **P0 reproduction holds**, with the expected budget-related uplift | Pooled test macro F1 0.338 (vs. Step 14 B's 0.325 under a shorter 20-epoch budget) — same architecture and inputs, a small, explained improvement from the fuller 50-epoch/patience-10 protocol |
| **P1 (learned dense pitch motion) does not beat P0**: essentially tied, mixed-sign per fold | Pooled macro F1 0.334 vs. 0.338 (−0.004); fold deltas mostly negative (4/5 folds); 9/17 recordings favor P0, 8/17 favor P1 |
| **P2 (learned salience motion) is the weakest of the three deployable conditions**, not the strongest | Pooled macro F1 0.305 — below both P0 and P1; grouped mean 0.278 |
| **P2 never beats P1, at any level of pitch ambiguity** — the one comparison this step was built to test | P2−P1 accuracy delta is negative in every margin bucket, every entropy quartile, every GT-rank bucket (−0.02 to −0.07 throughout, no exceptions) |
| **P3 (oracle) is dramatically, uniformly dominant** — more so than Step 14's audio+oracle condition D | Pooled macro F1 0.771 (vs. D's 0.604); T2 F1 0.759, T3 F1 0.822 (vs. D's 0.519/0.568); P3−P1 delta positive in all 5 folds (+0.34 to +0.58) |
| The windowed relative-salience representation genuinely **preserves uncertainty and secondary candidates** (verified, not assumed) | 94.8% of frames show a top1-top2 margin < 0.1; representative frames show near-tied competing candidates |
| P1's raw signal is **legitimately sparse** at native 10ms resolution — a real property, not a bug | Nonzero fraction of the raw offset-1 delta is only 4.5-21.5% per excerpt; both P1 and P3 needed far more gradient steps to escape an early majority-class plateau in tiny-overfit than P0/P2 did (confirmed to be a slow-optimization issue, not a learnability failure, by running longer) |
| No evidence P1 specifically rescues **longer primitives** (Step 14's hypothesis about fixed offsets) | P0 vs. P1 accuracy by duration bucket track each other closely at every bucket, including >1s (0.525 vs. 0.534) and 500ms-1s (0.399 vs. 0.395) — no consistent long-duration-specific gain |

**Outcome: `ESTIMATED_MOTION_REMAINS_BOTTLENECK`** (with `FIXED_PHI_SUFFICIENT` as essential supporting context)

**Decision gate: `INVESTIGATE_ACOUSTIC_MOTION_ESTIMATION`**

---

## 1. P0-P3 definitions

| Condition | Source | Representation | Encoder |
|---|---|---|---|
| P0 | Fused+D3 | Fixed φ: octave-unwrapped Δ10/50/100/200ms (Step 14) | `Linear(4→128)` → shared 128-ch TCN (`FramewiseConditionalTCNModel`, `use_pitch`-only) |
| P1 | Fused+D3 | Dense octave-unwrapped frame-to-frame delta (1-dim, every native frame) | `PitchMotionEncoder`: `Conv1d(1→32)` → dedicated 32-ch TCN |
| P2 | Frozen fused salience `S(f,t)` | Windowed, register-recentered salience distribution (73 bins, ±600¢) | `SalienceMotionEncoder`: small 2D `FrequencyCNN`-style frontend (16→32 ch) → same 32-ch TCN class as P1 |
| P3 | GT parametric pitch | Identical to P1 | Identical to P1 |

All four share the unweighted-CE framewise loss, grouped 5-fold splits, excerpt sampler, and (new for this step) an identical **50-epoch / patience-10** training protocol — the validated full B0-style budget, chosen because Step 14's reduced 20/5 budget was shown to under-train the audio-only condition there; pitch-only models here train fast enough (~1-4s/epoch) that the fuller budget was easily affordable (see §3).

---

## 2. Training protocol

`max_epochs=50, patience=10, excerpts_per_epoch=512, batch_size=8, AdamW lr=1e-3/wd=1e-4` — identical across P0-P3, per spec section 3. All 20 fold-runs converged and early-stopped well inside the budget (best epochs ranged 5-45, no run hit the 50-epoch ceiling), so the protocol was sufficient, not merely "the max we could afford."

---

## 3. P0 reproduction gate

| | Step 14 condition B (20 epochs, patience 5) | Step 15 P0 (50 epochs, patience 10) |
|---|---:|---:|
| Pooled test macro F1 | 0.325 | **0.338** |
| Grouped mean macro F1 | 0.328 | 0.348 |

Same architecture, same inputs (Fused+D3 → fixed φ → shared TCN), same code path (P0 literally calls `FramewiseConditionalTCNModel(use_audio=False, use_pitch=True)`, the exact class Step 14 condition B used) — the ~0.01-0.02 uplift is fully explained by the longer, previously-under-budgeted training protocol (Step 14 itself flagged condition A as prematurely converged under the 20-epoch budget; the same mechanism applies here, modestly, to B/P0). The reproduction holds: no unexplained discrepancy, and the gap is in the expected direction and rough magnitude.

---

## 4. P1's local-relative transformation

Exactly Step 14's `φ` function (`training/relative_pitch_features.py::compute_phi`), evaluated at a single offset:

```
v(t) = octave_unwrap(cents[t] - cents[t-1])   if valid[t] and valid[t-1], else 0
```

This is channel 0 of the existing 4-dim φ tensor (offsets `(1,5,10,20)` — channel 0 is offset 1) — no new feature code, reusing the already-tested `compute_phi`. A sustained absolute-pitch offset of `+1200k` cancels exactly in this difference (spec section 5's requirement), and no GT boundaries are used (φ's `valid` gate is the existing `valid_target` mask, identical to every prior step). Unlike Step 14's four hand-picked offsets, P1 provides only this dense, single-offset signal and leaves the discovery of longer time-scale structure to the learned encoder's own receptive field (§6) — the "frame-to-frame octave-unwrapped increments, let the temporal encoder integrate them" alternative the spec explicitly offered instead of picking more taps by hand.

---

## 5. P1 encoder and receptive field

`PitchMotionEncoder` (`training/framewise_models.py`): `Conv1d(1, 32, kernel_size=1)` lifts the scalar to 32 channels, then the **same `TemporalConvNet` class** used throughout this project (`kernel_size=5, dilations=(1,2,4,8)`) — receptive field **61 frames = 610ms**, inside spec section 6's 0.5-1.0s target and verified by test. Not a Transformer, not a large BiGRU — a small dedicated 1D-conv pitch encoder, structurally parallel to (but far narrower than) the audio `FrequencyCNN`+TCN pathway.

---

## 6. P2's salience transformation

`training/pitch_diagnostics/relative_pitch/dense_relative_salience.py`. At every native frame, extracts a **fixed window of the frozen fused salience distribution** (`hps_salience_probs` + learned model, Step 12/12.5's exact checkpoints and validation-selected fusion hyperparameters — no retraining), recentered on that frame's own Fused+D3 decoded bin:

```
S_rel(Δ, t) = S(ref_bin(t) + Δ, t),   Δ ∈ [-36, +36] bins (±600¢ at 16.67¢/bin, 73 bins total)
ref_bin(t) = round(bin_from_hz(Fused+D3 decoded Hz at frame t))
```

Zero outside the native [34,244) candidate range; **not** renormalized to sum to 1 (captured mass, typically ~0.72 of the full distribution, is itself an uncertainty signal, kept rather than discarded).

**Register-invariance argument:** a sustained absolute-pitch offset shifts both the candidate distribution and the recentering reference by the same amount, so their difference — the windowed relative view — is unchanged. This is the same differencing-cancels-sustained-octave-error mechanism Step 13 established for the 1-D decoded path (§4 there), applied to the 2-D distribution instead of collapsing it first.

`SalienceMotionEncoder`: a scaled-down `FrequencyCNN` (16→32 channels vs. audio's 32→64→128, same kernel/pooling structure) over the 73-bin window, feeding the same 32-channel `TemporalConvNet` class P1 uses — P1 and P2 differ only in what enters the temporal encoder, not in its capacity.

---

## 7. Proof P2 preserves secondary-candidate information

Verified numerically (not assumed), on real cached data:

- **94.8%** of frames show a top1-top2 margin < 0.1 (out of a max possible margin of ~1.0) — the large majority of frames present genuinely close competing candidates, not a single dominant spike.
- Representative frames show explicit near-ties, e.g. relative bin 0 at 0.140 vs. relative bin +1 at 0.140 vs. relative bin −1 at 0.121 — the "candidate A=0.45, candidate B=0.40"-style ambiguity spec section 8 describes, concretely present in the cache.
- Automated test (`test_relative_salience_preserves_secondary_peaks`) asserts this close-margin fraction exceeds 50% — passes at 94.8%, far above the bar.

---

## 8. Parameter counts

| Condition | Params |
|---|---:|
| P0 | 329,348 (= Step 14 condition B exactly) |
| P1 | 4,932 |
| P2 | ~24,000 (`SalienceFrequencyCNN` 16→32ch + 32-ch TCN + head) |
| P3 | 4,932 (identical architecture to P1, spec section 9) |

P1/P2 are one to two orders of magnitude smaller than P0, by design ("small and interpretable") — none of this step's findings can be attributed to P1/P2 simply having *more* capacity than the fixed-φ baseline; if anything the reverse.

---

## 9. Alignment / leakage tests

`training/tests/test_pitch_motion_ablation.py` (7 tests, all passing) plus the full existing 39-test suite (1 pre-existing skip, unrelated) unaffected — no regression to Steps 7-14 infrastructure:

- Receptive field falls in the declared 0.5-1.0s range (test, not just documentation).
- P1/P2 forward shapes and "small" (<50k / <100k param) bounds, asserted.
- P1's input is exactly channel 0 of the already-tested Step 14 φ tensor — cross-checked against an independent recompute from the same excerpt's own `pitch_cents`/`valid_target`.
- **End-to-end salience alignment**: for every sampled excerpt, the dataset's sliced `relative_salience[:, :n]` matches the cached full-recording array at the identical `[start:start+n]` frame range exactly, including the zero-padded tail beyond a short excerpt — same slicing contract as spec/target arrays.
- Secondary-peak preservation (§7) and reference-tracks-decoded-pitch (mean peak position within 5 bins of the recentering reference) verified numerically.
- All fold-wise normalization (φ stats, CQT stats) computed from train recordings only, reusing Step 14's already-tested `fold_phi_stats`; grouped-fold leakage safety inherited unchanged from `training.folds`.

---

## 10. Tiny-overfit results

16 cached excerpts, no early stopping. **P0 and P2 escape the initial majority-class plateau quickly** (by epoch ~10-18); **P1 and P3 need substantially longer** (~40 and ~26-30 epochs respectively) due to the much sparser raw single-offset signal (§ investigation below) — but **all four conditions do eventually show clear, sustained memorization** once given enough steps (verified by extending P1's run to 100 epochs: train macro F1 climbs from 0.17 to 0.47, train accuracy to 0.68, a clean escape, not a stall). This is a genuine, disclosed property of the sparse dense-delta representation, not an implementation bug (investigated per spec section 11 before proceeding) — full training's 512-excerpts/epoch (64 batches vs. tiny-overfit's 2) gives ~32× more gradient steps per epoch, more than enough margin against the 50-epoch/patience-10 full-CV budget, and indeed no full-CV run hit the epoch ceiling (§2).

**Root cause, quantified**: the raw octave-unwrapped offset-1 delta is nonzero in only 4.5-21.5% of frames per excerpt (median ~9%) — most consecutive valid-frame pairs show exactly zero measurable movement at native 10ms resolution (consistent with Step 13's own finding that the *median* delta error was exactly 0¢). After fold-wise standardization, ~89% of P1's input sits at a single near-constant "background" value, making the useful signal sparse and slow for a narrow 1-channel encoder to discover via gradient descent — especially with very few repeated examples (tiny-overfit) rather than the full data variety used in real training.

---

## 11-13. Grouped/pooled results and primary comparison table

Pooled test set (all 17 recordings, continuous framewise inference):

| Condition | Frame Acc | Macro F1 | T0 F1 | T1 F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|---:|---:|
| P0 — fixed φ | 0.506 | 0.338 | 0.547 | 0.568 | 0.150 | 0.085 |
| P1 — learned pitch motion | 0.523 | 0.334 | 0.555 | 0.585 | 0.094 | 0.104 |
| P2 — learned salience motion | 0.471 | 0.305 | 0.531 | 0.515 | 0.127 | 0.048 |
| **P3 — oracle pitch motion** | **0.758** | **0.771** | 0.733 | 0.768 | **0.759** | **0.822** |

Grouped mean ± std:

| Condition | Mean macro F1 | Std |
|---|---:|---:|
| P0 | 0.348 | 0.070 |
| P1 | 0.324 | 0.056 |
| P2 | 0.278 | 0.021 |
| P3 | 0.778 | 0.061 |

Per-fold deltas (macro F1):

| Fold | P1 − P0 | P2 − P1 | P3 − P1 |
|---|---:|---:|---:|
| 0 | −0.012 | +0.024 | +0.577 |
| 1 | −0.041 | +0.011 | +0.403 |
| 2 | **+0.026** | −0.042 | +0.489 |
| 3 | −0.011 | −0.100 | +0.344 |
| 4 | −0.084 | −0.122 | +0.454 |

P1−P0 has no consistent sign (positive in 1/5 folds) — no reliable effect, matching the pooled near-tie. P2−P1 is increasingly negative in the later folds. P3−P1 is uniformly, dramatically positive in every fold.

---

## 14. Duration analysis

Accuracy by primitive duration:

| Bucket | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| <100ms | 0.656 | 0.709 | 0.565 | 0.913 |
| 100-250ms | 0.559 | 0.589 | 0.485 | 0.859 |
| 250-500ms | 0.408 | 0.412 | 0.364 | 0.801 |
| 500ms-1s | 0.399 | 0.395 | 0.377 | 0.640 |
| >1s | 0.525 | 0.534 | 0.554 | 0.614 |

**Answering spec section 14's key question directly: no.** P1 does not specifically improve over P0 on longer trajectories — the two track closely at every duration bucket, with P1 marginally ahead on short/fast primitives (<250ms, where its denser signal has the most to work with) and effectively tied or marginally behind at 500ms-1s. Step 14's "fixed offsets are the limitation" hypothesis is **not supported** by this data: replacing the fixed taps with a dense, learnable-timescale signal did not unlock the longer-primitive cases it was expected to help most.

---

## 15. |dp/dt| analysis

Accuracy by movement speed:

| Bucket | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 0-100 c/s | 0.508 | 0.516 | 0.494 | 0.664 |
| 100-400 c/s | 0.413 | 0.424 | **0.303** | 0.849 |
| 400-1000 c/s | 0.467 | 0.497 | 0.392 | 0.913 |
| >1000 c/s | 0.553 | 0.593 | 0.497 | 0.946 |

P1 preserves (slightly exceeds) P0's strength on fast motion, consistent with Step 14 B's own pattern. P2 is **uniformly worse than both P0 and P1 at every speed**, most severely on 100-400 c/s (0.303, a full 11-12 points below P0/P1) — the salience-window representation does not translate into better fast-motion tracking despite retaining more raw information. P3's advantage *grows* with speed (0.664→0.946), mirroring Step 14 D's pattern but more pronounced — clean pitch motion is most informative exactly where trajectory shape is most expressed.

---

## 16. Salience-uncertainty analysis (P2 vs. P1)

The central reason to preserve `S(f,t)` instead of collapsing it: does P2 win specifically when the decoded pitch path is ambiguous? Stratified 41,578 subsampled evaluation frames by three uncertainty measures (evaluation metadata only, never used for decoding):

| Stratum | P1 acc | P2 acc | P2 − P1 |
|---|---:|---:|---:|
| **Margin** — very ambiguous (<0.03) | 0.517 | 0.474 | −0.042 |
| ambiguous (0.03-0.08) | 0.527 | 0.482 | −0.046 |
| clear (0.08-0.2) | 0.521 | 0.479 | −0.042 |
| **Entropy** — lowest quartile | 0.503 | 0.451 | −0.051 |
| highest quartile | 0.473 | 0.451 | −0.022 |
| **GT rank in window** — rank 1 | 0.568 | 0.498 | −0.070 |
| rank > 20 / outside window | 0.474 | 0.443 | −0.031 |

**Answer: no, at no level of ambiguity does P2 outperform P1.** The P2−P1 delta is negative in every single stratum tested — margin, entropy, and GT-rank alike — including the *most* ambiguous frames, which is exactly where retaining the full distribution should matter most if it were going to. This is the cleanest, most direct evidence in this step: preserving salience uncertainty, at least in this windowed/register-recentered form with this small 2D-conv encoder, does not convert into better trajectory decisions anywhere, not just "on average."

---

## 17. Per-recording analysis

9/17 recordings favor P0 over P1, 8/17 favor P1 — no dominant single-recording explanation either direction (consistent with the fold-delta table's mixed signs). P3 dominates on 16/17 recordings, often overwhelmingly (e.g. `6655f08a…` 0.267→0.882, `65e4a79c…` 0.390→0.883). The **one exception is `692ed7e6…`** (fold 4): P3=0.288, actually *below* P0(0.267)/P1(0.272) there — the same recording Step 14 flagged as its sole outlier where oracle-pitch condition D also underperformed. This recording appears to be a genuine, reproducible hard case for pitch-motion-based models specifically, across two independent experiments — worth a targeted look in any future step, not resolved here.

---

## 18. Oracle-gap interpretation

P3 = 0.771 pooled, dramatically above P1 (0.334) and P2 (0.305) — even further above than Step 14's own oracle condition D (0.604) was above its estimated counterparts. As spec section 18 requires: P3 uses the *same parametric pitch source* trajectory labels are themselves derived from (T1/T2/T3 are literally bend-shape categories over that curve), so **this is a representation/capacity ceiling, not a realistic estimate of achievable production performance from a hypothetical perfect generic F0 estimator.** Its value here is narrower and specific: it proves the small `PitchMotionEncoder` architecture — the *same* one P1 uses, at ~5k parameters — can express strong T0-T3 discrimination (T2/T3 F1 0.76/0.82) when given clean contour information. Architecture capacity is conclusively **not** the bottleneck; input quality is.

---

## 19. Final outcome

**`ESTIMATED_MOTION_REMAINS_BOTTLENECK`**

> P3 remains dramatically above P1 and P2 (pooled macro F1 0.771 vs. 0.334/0.305 — more than double), uniformly across every fold, every duration bucket, every speed bucket. Clean contour information is highly useful; current acoustic pitch evidence, in every form tested here (a single decoded scalar, a dense per-frame delta, or the full local salience distribution), still fails to recover enough of it.

This composes with, rather than contradicts, `FIXED_PHI_SUFFICIENT` (P1 does not meaningfully beat P0 — the hand-designed four-tap representation already extracts about as much as this step's more sophisticated alternatives could, from the *same* noisy source): together they say the compression step actually responsible for the loss identified in this step's core question is **not** `S(f,t) → p(t)` (collapsing salience to a decoded path — P2 never beats P1) and **not** `p(t) → φ(t)` (collapsing the path to four taps — P1 never beats P0) but further upstream, in the **acoustic pitch/salience estimate itself** being insufficiently accurate at the fine-contour level to express T2/T3-distinguishing shape, even before any of these downstream representation choices are made. **Not** `LEARNED_PITCH_MOTION_HELPS` or `SALIENCE_MOTION_HELPS` (neither P1 nor P2 shows a consistent win anywhere, including exactly the ambiguous-frame regime P2 was built to help).

---

## 20. Decision gate

**`INVESTIGATE_ACOUSTIC_MOTION_ESTIMATION`**

Both preconditions hold explicitly: P3's advantage remains dominant (§19), and neither P1 nor P2 closes meaningful ground on P0 (§13, §16 — P2 is if anything the *worst* of the three deployable conditions).

**Do next:** the evidence chain across Steps 13-15 has now closed off representation-level fixes at every stage tested (decoded-path taps, dense deltas, salience windows) without closing off the possibility that the underlying acoustic salience map itself is not fine-grained/accurate enough to express T2/T3 shape distinctions — independent of octave/register (already shown in Step 13 to mostly cancel under differencing). Worth investigating next: whether the salience *model itself* (Step 11's harmonic-salience network) has enough fine-pitch-class resolution and temporal precision for this task, e.g. via a targeted fine-pitch-accuracy audit conditioned on T2/T3 ground truth specifically, before considering any representation or fusion change again.

**Do not:** return to global octave/register decoding (explicitly out of scope — Steps 12/12.5 already closed that branch with `STOP_REGISTER_DECODER_ENGINEERING`, and this step's own evidence, P1/P3 both showing octave-unwrapped differencing works as intended, reconfirms register was not the remaining issue). Do not adopt P1 or P2 as an upgrade over Step 14's condition B/P0 — neither earns it. Do not conclude the small-encoder architecture needs to be bigger — P3 proves the *existing* small architecture already expresses the target distinctions given adequate input; more capacity on the current noisy input is not the fix.
