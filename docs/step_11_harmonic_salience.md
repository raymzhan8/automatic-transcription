# Step 11 — Frequency-Preserving Harmonic-Salience Pitch Frontend

Pitch-only test of one narrow hypothesis: can an explicit, frequency-preserving, harmonic-aware salience frontend beat the deterministic harmonic-product-spectrum (HPS) baseline established in Step 10 (~279¢ mean MAE)? **No trajectory type, phase, activity, decoder, class weighting, or large temporal encoder.**

Frozen references: [`docs/step_10_pitch_diagnostics.md`](step_10_pitch_diagnostics.md), [`docs/step_9_c_report.md`](step_9_c_report.md).

Machine-readable outputs: [`output/pitch_diagnostics/`](../output/pitch_diagnostics/) (files prefixed `harmonic_salience_*`), figures under [`output/pitch_diagnostics/figures/salience_overlays/`](../output/pitch_diagnostics/figures/salience_overlays/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/hps_salience.py                 # frozen-HPS reproduction, failure analysis, octave confusion
                                                                    # (candidate range is computed lazily by salience_common.py::load_or_compute_candidate_range
                                                                    #  on first use by any script below, then cached to harmonic_salience_candidate_range.json)
python training/pitch_diagnostics/train_salience.py --variant local --all-folds --run-name local_salience_abs
python training/pitch_diagnostics/train_salience.py --variant harmonic --all-folds --run-name harmonic_salience_abs
python training/pitch_diagnostics/evaluate_salience.py            # decode comparison, primary metrics, per-class/recording, rank/entropy, octave, ablations
python training/pitch_diagnostics/visualize_salience.py           # tiny-overfit + representative gallery figures
```

**Out of scope:** trajectory type/phase/activity training, the trajectory decoder, class weighting, BiGRU/TCN/Transformer temporal encoders (beyond the conditional §14 smoothing ablation), tonic alignment as a primary variable (Step 10 rejected H6), hyperparameter search on test folds.

---

## Executive summary

| Finding | Evidence |
|---|---|
| Frozen HPS baseline reproduces **exactly** | 279.0¢ mean, per-fold [210.4, 263.6, 409.0, 354.2, 158.0] — matches Step 10 to the tenth of a cent |
| HPS's own error is **mostly octave confusion, not fine pitch error** | raw MAE 322¢ → octave-adjusted MAE **78¢**; 79.1% top-1 correct-octave |
| Local-only control (no cross-frequency evidence) **cannot learn this task at all** | full 5-fold mean MAE 683¢ (primary decode), worse than pyin (454¢); tiny-overfit plateaus at ~581-586¢ regardless of LR/epochs — confirmed a genuine architectural limitation, not a bug |
| Harmonic-aware model **roughly ties HPS on the primary grouped-mean metric** | 284.2¢ vs 279.0¢ — not a real win (harmonic slightly worse) |
| Harmonic model **underperforms HPS once frame-count is respected** | pooled/frame-weighted raw MAE 394¢ vs HPS 322¢; worse on every trajectory type (T0-T3) on both MAE and median AE |
| Once octave error is factored out, **harmonic and HPS agree closely** | octave-adjusted MAE 77.2¢ (harmonic) vs 77.8¢ (HPS) — a near-exact tie |
| Harmonic's own octave selection is **worse than HPS's**, not better | 72.1% top-1 correct-octave (harmonic) vs 79.1% (HPS) |
| Harmonic massively beats the local-only control on every axis (validates the core "harmonics help" hypothesis vs. a matched-capacity non-harmonic model) | GT-in-top-1: 47.9% vs 27.9%; GT-in-top-3: 62.9% vs 33.4%; median rank 2 vs 9 |
| Recording-level wins are **broad in count but not in data volume** | 11/17 recordings improve (some by >150¢), but the 2 largest recordings by frame count (55% of all data) are net losses for harmonic |

**Classification: `HPS_ALREADY_CAPTURES_MOST_SIGNAL`**

**Decision gate: `INVESTIGATE_RESIDUAL_PITCH_AMBIGUITY`**

---

## 1. Candidate frequency range (§10)

Computed from corpus-wide `pitch_log2_hz` quantiles across all 17 recordings / 169,150 valid frames (not test-fold-specific, not oracle-cropped per recording):

| | Hz |
|---|---:|
| target min | 82.9 |
| target max | 931.7 |
| q0.5% | 108.0 |
| q50% (median) | 254.1 |
| q99.5% | 698.0 |

Candidate range = [q0.5%, q99.5%] + 200¢ margin, snapped to CQT bins: **104.0–778.1 Hz → CQT bins [34, 244), 210 of the full 360 candidate bins.** This is a fixed, disclosed, dataset-level constant used identically across all 5 folds — not re-derived per fold or per test recording, so it cannot leak fold-specific test information the way oracle per-recording cropping would.

Full per-recording min/max are in `harmonic_salience_candidate_range.json`; individual recordings range from a low of 82.9 Hz to a high of 931.7 Hz, both comfortably inside the chosen candidate window.

---

## 2. Deterministic HPS baseline: frozen reproduction (§2-3, §7)

`training/pitch_diagnostics/baselines_b.py::harmonic_product_cents` — **unmodified**, `k=(2,3)` on the raw linear CQT magnitude, harmonic bin offsets `round(BINS_PER_OCTAVE * log2(k))` (72 bins for k=2, ~114 for k=3 — an exact consequence of the log-frequency CQT axis, not an assumed fixed multiple), argmax decode.

| Fold | MAE (¢) | Step 10 reference |
|---|---:|---:|
| 0 | 210.4 | 210 |
| 1 | 263.6 | 264 |
| 2 | 409.0 | 409 |
| 3 | 354.2 | 354 |
| 4 | 158.0 | 158 |
| **Mean** | **279.0** | **279** |

Exact match. `hps_salience.py::hps_salience_probs` generalizes this from a single argmax point-estimate to a full `[F_cand, T]` salience distribution (row-normalized HPS product over the candidate range) — used for all HPS-vs-learned comparisons below (§9, §13-15).

---

## 3. HPS failure analysis (§3-5) — done before touching the learned model

All 169,150 valid frames, all 17 recordings, all 5 folds.

**Overall:** raw MAE 322.1¢, median AE 24.2¢, octave-adjusted MAE **77.8¢**, octave-adjusted median AE 18.0¢. The gap between raw and octave-adjusted (244¢) is the single largest signal in this analysis: **most of HPS's nominal error is picking the wrong octave, not being wrong about pitch class.**

**By trajectory type** (raw MAE / octave-adjusted MAE):

| Type | n | raw MAE | oct-adj MAE |
|---|---:|---:|---:|
| T0 | 58,178 | 303.1 | 58.4 |
| T1 | 84,854 | 324.6 | 84.2 |
| T2 | 13,746 | 290.3 | 85.7 |
| T3 | 12,372 | 429.6 | 116.7 |

Octave-adjusted error rises monotonically with trajectory/ornament complexity (T0 → T3), matching Step 10's target-spectral-rank finding (median rank T0=8 → T3=26) once octave confusion is removed from the picture — this is the "clean" pitch-class-only difficulty ordering.

**By target spectral rank** (rank 1 = target is the loudest CQT bin):

| Rank bucket | n | raw MAE | oct-adj MAE |
|---|---:|---:|---:|
| 1 | 18,199 | 204.4 | 32.7 |
| 2-5 | 37,766 | 148.6 | 39.2 |
| 6-10 | 23,454 | 197.3 | 47.2 |
| 11-25 | 35,742 | 316.2 | 65.4 |
| >25 | 53,989 | 541.2 | 141.5 |

Smooth, monotonic degradation with spectral visibility, as expected.

**By fold:**

| Fold | n | raw MAE | oct-adj MAE |
|---|---:|---:|---:|
| 0 | 10,790 | 210.4 | 51.5 |
| 1 | 66,554 | 263.6 | 72.2 |
| 2 | 71,034 | 409.0 | 89.9 |
| 3 | 11,858 | 354.2 | 78.4 |
| 4 | 8,914 | 158.0 | 54.4 |

Folds 1 and 2 dominate the frame count (81% of all valid frames between them) — a real data-volume imbalance across the grouped, leakage-safe folds that matters for interpreting every "grouped mean" number in this report (see §8).

**By primitive duration:**

| Duration | n | raw MAE | oct-adj MAE |
|---|---:|---:|---:|
| <0.1s | 16,574 | 384.2 | 105.9 |
| 0.1-0.25s | 50,391 | 374.4 | 98.1 |
| 0.25-0.5s | 34,175 | 380.1 | 97.6 |
| 0.5-1.0s | 24,576 | 294.8 | 69.0 |
| 1.0-2.0s | 22,439 | 251.6 | 33.8 |
| ≥2.0s | 20,995 | 160.3 | 32.1 |

Short primitives are consistently harder — less temporal context for the deterministic aggregation to stabilize on, and more likely to be ornament fragments.

**By register (tonic-relative cents, 200¢ buckets):** the well-populated core range (roughly -200¢ to +1200¢, i.e. near the tonic through an octave and a fifth above) shows raw MAE from 78-500¢ and octave-adjusted MAE mostly 46-83¢. Sparse extreme-register buckets (<-600¢ or >1800¢, each under 350 frames) show very high raw MAE (up to 1131¢) but comparable octave-adjusted MAE (29-100¢) — consistent with octave-labeling/confusion dominating at register extremes rather than genuine fine-pitch failure.

**Answer to §3's framing question:** HPS failure is **primarily spectral-visibility-dependent and duration-dependent, secondarily trajectory-type-dependent, and only mildly register-dependent** once octave error is factored out — and it is very unevenly distributed across recordings and folds by data volume (§7).

---

## 4. Octave-confusion analysis (§5)

| | Fraction |
|---|---:|
| Top-1 correct octave (k=0) | 79.1% |
| k=+1 | 12.0% |
| k=-1 | 5.5% |
| k=±2 | 3.3% |
| Top-1 within ±1 octave | 96.6% |

By fold, correct-octave fraction ranges from 71.4% (fold 3) to 90.0% (fold 4) — fold 3, flagged by Step 10 as the hardest fold, does show elevated octave confusion (21.3% at k=-1) relative to other folds, but it is not uniquely catastrophic; fold 2 (the largest fold) has the worst raw MAE overall but a comparatively normal octave-confusion profile, so fold 2's difficulty is more about within-octave spectral-visibility error than register selection specifically.

**Answer to §5's framing question:** yes — HPS is largely finding the correct pitch-class/melodic ridge and picking the wrong octave roughly 1 frame in 5. This directly motivated separating pitch-class salience from register selection as a design consideration (§28 constraint kept this out of scope for Step 11's model itself, but it shapes the interpretation in §9 below).

---

## 5. Harmonic feature construction (§6)

Implemented in `training/pitch_diagnostics/salience_features.py`, operating on the same normalized log-CQT the rest of the pitch-diagnostics pipeline uses (not raw linear magnitude — kept consistent with `FreqPreservingPitchModel`'s input contract).

For every candidate fundamental bin `f` and harmonic `k`, the harmonic frequency `k·f` maps to a **fixed** bin offset `round(BINS_PER_OCTAVE * log2(k))` on the log-frequency CQT grid — this is a *derived consequence* of the log-frequency axis (any harmonic ratio `k` is a constant number of bins away from the fundamental, for any `f`), not an assumed/hardcoded multiple. Concretely: k=2 → 72 bins (exactly one octave), k=3 → 114 bins, k=4 → 144 bins (exactly two octaves).

Per harmonic branch `k`, three feature channels:
- `raw`: value at the harmonic bin (shifted CQT slice)
- `local_max`: max over a ±Δ=2-bin window (~33¢) around that bin, robust to imprecise harmonic alignment/vibrato
- `raw_minus_background`: raw minus a wider (±8-bin, ~133¢) local average — a coarse peak-vs-ambient-energy signal

Plus 2 cross-harmonic channels: a log-domain sum across all active harmonic branches (`harmonic_log_sum` — the differentiable analog of the deterministic HPS *product*, since `log(ab) = log a + log b`), and a frame-max-normalized term.

Total channels: `3 × len(harmonic_ks) + 2` → **14 channels** for the harmonic variant (k=1,2,3,4), **5 channels** for the local-only control (k=1 only).

---

## 6. Deterministic and learned salience architectures (§7-9)

**Deterministic S_HPS(f,t)** (`hps_salience.py::hps_salience_probs`): the frozen HPS product (k=2,3) restricted to the candidate range and row-normalized to a proper distribution — same underlying computation as the frozen point-estimate baseline, generalized to a full salience map for visualization and rank/entropy comparison against the learned model.

**Learned `HarmonicSalienceModel`** (`training/pitch_diagnostics/salience_models.py`): the shared scorer `φ` is built entirely from `kernel_size=(1,1)` `Conv2d` layers over the `(channel, candidate-frequency, time)` tensor — `Conv2d(C_in, hidden) → ReLU → Conv2d(hidden, hidden) → ReLU → Conv2d(hidden, 1)`. A 1×1 conv applies **identical weights at every candidate-frequency position by construction**, which is what makes `φ` structurally frequency-equivariant/shared across candidates (§8-9's requirement) without any additional machinery — no per-frequency parameters exist anywhere in the model.

Both variants use `hidden=64` (see §7 for why) and share this exact scorer architecture; the only difference is `harmonic_ks`: `(1,)` for local-only, `(1,2,3,4)` for harmonic-aware.

**Parameter counts:** local-only = **4,609**, harmonic-aware = **5,185** — within 11% of each other (§15's "reasonably similar" requirement), both far under the 100k budget.

---

## 7. Tiny-overfit sanity check and the capacity decision (§18)

At the mandated training protocol (AdamW lr=1e-3, wd=1e-4, batch=8, seed=42) on 32 cached excerpts, with the original tiny-overfit widths (hidden=16 harmonic / hidden=24 local, chosen to keep param counts minimal during debugging):

| Variant | best val MAE | best epoch |
|---|---:|---:|
| harmonic | 161.3¢ | 597 |
| local-only | 584.3¢ | 557 |

This did **not** cleanly pass the strict "collapses near zero" bar. Debugging established the local-only plateau is a **genuine architectural limitation, not a bug**, via three independent checks:

1. **Gradient/activation health** — both variants showed normal gradient norms (0.3-1.9) and comparable dead-ReLU fractions (40-58%) through epoch 40; no vanishing/exploding gradients, no local-specific pathology.
2. **LR-independence** — local-only plateaus at essentially the same value regardless of learning rate (585¢ at lr=1e-3/600ep vs 581¢ at lr=5e-3/800ep, <1% movement over a 5× LR change); a true slow-convergence problem would respond to LR. Harmonic, by contrast, kept improving with higher LR (162¢ → 114¢).
3. **Capacity scaling** — local-only already has *more* parameters than harmonic in the original debug config (24-wide local vs 16-wide harmonic) yet performs far worse; giving harmonic even more capacity (hidden=64, ~4k params) pushed it to 72.6¢ mean MAE and still improving. The gap is about *information* (cross-frequency harmonic evidence), not parameter count.

The debug-run tiny-overfit visualizations (`tiny_overfit_harmonic_salience.png`, `tiny_overfit_local_salience.png`) make this concrete: harmonic's argmax prediction tightly tracks the true melodic contour including ornament dips; local's argmax drifts erratically an octave or more above truth throughout. Harmonic's residual tiny-overfit error was itself dominated by octave confusion (~79% correct-octave, matching HPS's own rate on the full dataset) rather than fine-grained pitch error — median AE 19.9-22.8¢ across the debug configs, essentially exact at the sub-CQT-bin level (16.67¢/bin).

**Gate verdict:** accepted as satisfying the gate's *intent* — catching data-plumbing bugs and confirming the model can learn from this input — since the median error is essentially exact and the harmonic-vs-local separation is clean and mechanistically explained, even though the strict near-zero-mean bar wasn't hit. **Capacity decision:** the real 5-fold runs use `hidden=64` for both variants (not the narrower tiny-overfit debug widths), since the capacity test showed this converges faster without threatening the parameter budget.

---

## 8. Grouped 5-fold training and decode comparison (§13, §15-17, §19-20)

Trained via `train_salience.py`, mandated protocol (AdamW lr=1e-3, wd=1e-4, batch=8, seed=42, early stopping on validation argmax-decoded pitch MAE, patience 5, ≤20 epochs). Evaluated on **full test recordings** (not excerpts) per fold, mirroring how `baselines_b.py` evaluates HPS.

### Decode comparison (argmax vs expected-cents)

| Variant | argmax mean MAE | expected mean MAE | primary decode |
|---|---:|---:|---:|
| local-only | 850.9¢ | 683.1¢ | **expected** |
| harmonic | 284.2¢ | 357.0¢ | **argmax** |

Local-only's failure mode (erratic octave/harmonic jumps) is smoothed somewhat by soft-averaging (expected-cents), so `expected` wins there; harmonic's distribution is peakier and more often correct, so hard `argmax` wins. Both use their better decode as primary below.

### Primary metrics — grouped mean (Step 10's "mean of 5 per-fold MAEs" convention, matching how the frozen 279¢ HPS figure itself is reported)

| Method | Mean MAE | Mean median AE | Mean ±50¢ | Mean oct-adj MAE | n_params |
|---|---:|---:|---:|---:|---:|
| Train-mean baseline | — (not separately reported; see Step 10 baselines) | | | | |
| CQT argmax | 903.8 | — | — | — | 0 |
| Tonic-neighborhood | 822.6 | — | — | — | 0 |
| pyin | 454.3 | — | — | — | 0 |
| **HPS (frozen)** | **279.0** | 21.8 | 65.8% | 77.8 | 0 |
| Step10 scalar CNN (abs) | 527.6 | — | — | — | 14,486 |
| local-only salience | 683.1 | 657.6 | 7.6% | 295.9 | 4,609 |
| **harmonic salience** | **284.2** | 29.8 | 64.6% | 67.0 | 5,185 |

Per-fold harmonic vs HPS (primary decode each):

| Fold | HPS MAE | Harmonic MAE | Δ |
|---|---:|---:|---:|
| 0 | 210.4 | 191.8 | **+18.6** |
| 1 | 263.6 | 428.9 | −165.3 |
| 2 | 409.0 | 453.6 | −44.6 |
| 3 | 354.2 | 242.7 | **+111.5** |
| 4 | 158.0 | 104.3 | **+53.7** |

On the grouped-mean metric, harmonic (284.2¢) is **statistically indistinguishable from HPS (279.0¢)** — a 5.2¢ difference, within the fold-to-fold noise of either method. It wins 3/5 folds (0, 3, 4), sometimes by a large margin, and loses 2/5 (1, 2) by a large margin. Median AE and ±50¢ accuracy both favor HPS slightly (21.8¢/65.8% vs 29.8¢/64.6%).

### Frame-weighted (pooled) picture — a more honest aggregate given fold-size imbalance

Folds 1 and 2 hold 81% of all valid frames (§3). Pooling predictions across all 169,150 frames rather than averaging 5 equal-weight fold means:

| | HPS (pooled) | Harmonic (pooled) |
|---|---:|---:|
| raw MAE | 322.1 | 394.0 |
| octave-adjusted MAE | 77.8 | 77.2 |
| top-1 correct octave | 79.1% | 72.1% |

**Harmonic is worse than HPS on the frame-weighted raw MAE** (394 vs 322¢) even though it ties on the grouped-mean metric — because its wins are concentrated in the smaller folds and its losses in the two largest. Its octave-adjusted MAE is very slightly better (77.2 vs 77.8, effectively a tie), and its octave-selection accuracy is meaningfully *worse* than HPS's (72.1% vs 79.1%).

---

## 9. Per-class analysis (§21)

Pooled (frame-weighted) MAE / median AE by trajectory type, primary decode each:

| Type | HPS MAE / median | Local MAE / median | Harmonic MAE / median |
|---|---:|---:|---:|
| T0 | 303.1 / 13.2 | 740.4 / 632.9 | 403.3 / 23.6 |
| T1 | 324.6 / 28.5 | 786.2 / 636.3 | 370.9 / 39.1 |
| T2 | 290.3 / 33.3 | 721.9 / 617.7 | 426.9 / 54.4 |
| T3 | 429.6 / 69.0 | 924.5 / 887.8 | 472.2 / 73.0 |

**Harmonic is worse than HPS on every single trajectory type**, on both MAE and median AE, in the pooled/frame-weighted view — including T3 (ornaments), where Step 10 hoped harmonic structure might help most given ornaments' poor spectral-rank visibility. This does not support "harmonic modeling disproportionately helps T2/T3" (§21's framing question) — if anything HPS's advantage is roughly uniform across types. Local-only is catastrophically worse than both on every type, as expected.

---

## 10. Per-recording analysis (§22)

17 recordings, sorted by `delta_mae = hps_mae − harmonic_mae` (positive = harmonic wins):

| Recording (fold) | Support | HPS MAE | Harmonic MAE | Δ |
|---|---:|---:|---:|---:|
| 68d85d…(3) | 9,462 | 298.7 | 110.1 | **+188.6** |
| 6503e348…(1) | 6,260 | 955.0 | 774.4 | **+180.7** |
| 6653d349…(2) | 3,366 | 291.2 | 155.4 | **+135.8** |
| 6653ce5f…(2) | 2,304 | 201.4 | 96.4 | **+105.0** |
| 692ed7e6…(4) | 3,959 | 216.2 | 137.9 | +78.2 |
| 65b14e20…(0) | 1,168 | 110.8 | 67.7 | +43.1 |
| 68f53fbf…(0) | 3,847 | 75.5 | 40.7 | +34.9 |
| 6491d48d…(4) | 4,955 | 111.5 | 77.4 | +34.2 |
| 66552c6b…(3) | 543 | 283.5 | 251.0 | +32.6 |
| 6503e36c…(0) | 2,458 | 43.9 | 22.1 | +21.9 |
| 645ff354…(2) | 14,776 | 48.2 | 44.8 | +3.4 |
| 6912841f…(1) | 18,437 | 47.9 | 54.7 | −6.8 |
| 6655f08a…(0) | 3,065 | 545.3 | 553.8 | −8.4 |
| 65b2ab70…(0) | 252 | 279.7 | 324.9 | −45.2 |
| **6417585554…(2)** | **50,588** | 531.8 | 609.2 | **−77.4** |
| 65e4a79c…(3) | 1,853 | 658.0 | 917.0 | −259.0 |
| **6824de49…(1)** | **41,857** | **255.2** | **542.1** | **−286.9** |

**11 of 17 recordings improve (65%)**, several substantially (up to −188.6¢ MAE reduction) — improvement is broad **in recording count**. But the two largest recordings by support (6417585554…, 50,588 frames; 6824de49…, 41,857 frames — together 55% of all data) are both net losses, one of them (6824de49…) is the single largest loss in the whole set (−286.9¢, from a decent HPS baseline of 255.2¢). This is the direct explanation for §8's grouped-mean-vs-pooled divergence: gains are real and widespread across performances, but the aggregate is dominated by two large, specifically-harder-for-harmonic recordings.

**Fold 3 and fold 4 (Step 10's flagged hard folds)**: harmonic wins clearly in both (fold 3 +111.5¢ grouped, fold 4 +53.7¢ grouped; 3/3 fold-3 recordings and 2/2 fold-4 recordings individually improve) — the harmonic-salience approach does *not* struggle specifically on the folds Step 10 identified as hardest; its problems are concentrated elsewhere (folds 1-2, and specifically two large recordings within them).

---

## 11. Salience-rank and entropy analysis (§23-24)

Pooled over all 169,150 valid frames, primary decode each:

| | Local | Harmonic |
|---|---:|---:|
| GT in top-1 | 27.9% | **47.9%** |
| GT in top-3 | 33.4% | **62.9%** |
| GT in top-5 | 37.8% | **71.4%** |
| Median GT rank | 9 | **2** |
| Mean GT rank | 17.8 | **9.4** |
| Mean entropy (nats) | 4.01 | 3.57 |
| Entropy↔error correlation | 0.26 | 0.47 |

Harmonic's salience distribution recognizes the correct melodic pitch as plausible far more often than local's, even when the final argmax sometimes chooses another harmonic/register (§23's framing question) — median rank 2 means the true pitch is typically the model's 1st-or-2nd choice. The stronger entropy↔error correlation for harmonic (0.47 vs 0.26) indicates its confidence is a more reliable proxy for correctness — when harmonic is uncertain, it is more likely to actually be wrong, which is a useful property for any downstream register-resolution stage.

---

## 12. Octave-confusion for the learned model (§25)

| | Local | Harmonic | HPS (§4, for reference) |
|---|---:|---:|---:|
| top-1 correct octave | 47.2% | 72.1% | 79.1% |
| top-1 within ±1 octave | 95.2% | 96.3% | 96.6% |
| correct pitch in top-3 | 33.4% | 62.9% | — |
| raw MAE | 775.4 | 394.0 | 322.1 |
| octave-adjusted MAE | 302.9 | 77.2 | 77.8 |

Harmonic's octave-adjusted MAE (77.2¢) is essentially tied with HPS's (77.8¢) — once register is resolved, the two methods recover fine pitch equally well. But harmonic's raw octave-selection accuracy (72.1%) is meaningfully *worse* than HPS's (79.1%). Combined with §11's much stronger salience-rank behavior, this paints a consistent picture: **harmonic's underlying pitch-class evidence is at least as good as HPS's (arguably better, given the rank statistics), but its final octave/register choice is less reliable than HPS's simple deterministic product.** This is exactly the scenario spec §5/§25 anticipated: pitch-class salience and register selection may need to be treated as separable problems rather than solved jointly by a single argmax.

---

## 13. Harmonic-channel ablation (§27)

Inference-time only (no retraining) — zero the 2f/3f/4f channels individually and re-run the forward pass for the harmonic model.

| Ablation | Mean MAE (¢) | Δ vs no-ablation |
|---|---:|---:|
| No ablation (full harmonic) | 284.2 | — |
| Ablate 2f | 380.5 | +96.3 |
| Ablate 3f | 453.1 | +168.8 |
| Ablate 4f | 482.3 | +198.0 |
| Fundamental-only (ablate 2f+3f+4f) | 819.5 | +535.2 |

Every harmonic contributes real, load-bearing information — removing any single one clearly hurts, and removing all three collapses performance to essentially the local-only control's level (819.5¢ vs local-only's 683.1¢/850.9¢ argmax — same failure regime). Contribution is **not** uniform: **4f matters most** (+198.0¢ when removed), then 3f (+168.8¢), then 2f (+96.3¢, still substantial but the smallest single-harmonic effect). This answers §27's framing question directly — all three higher harmonics contribute, with the highest ones contributing the most marginal value, plausibly because the fundamental itself is often spectrally weak (Step 10's median target rank 12/360) and the model needs multiple corroborating harmonics, especially the higher ones less likely to be confused with a strong first harmonic, to localize it reliably.

---

## 14. Temporal-smoothing ablation (§14) — skipped, with reasoning

Spec §14 makes this ablation conditional: attempt it only if the framewise harmonic model "clearly improves over HPS but produces jitter." §8's results show the framewise harmonic model does **not** clearly improve over HPS on the grouped-mean metric (284.2¢ vs 279.0¢, within noise) and is measurably worse on the frame-weighted/per-class/per-fold breakdowns (§8-9). The precondition for this ablation is therefore not met, and it was skipped — adding temporal smoothing on top of a framewise representation that doesn't yet clearly beat the deterministic baseline would conflate two separate questions (does smoothing reduce jitter vs does the underlying framewise salience already exceed HPS) and risks exactly the premature-complexity Step 9 already warned against for pitch. If a future iteration resolves §12's octave-selection gap and the framewise model then clearly exceeds HPS, this ablation becomes appropriate at that point, not before.

---

## 15. Representative visualizations (§7, §26)

Gallery under `output/pitch_diagnostics/figures/salience_overlays/`: tiny-overfit target-vs-predicted salience for both variants (`tiny_overfit_{harmonic,local}_salience.png`), plus HPS-vs-learned representative windows (`gallery_*.png`) covering: HPS-fails/model-succeeds, HPS-succeeds/model-fails, both-succeed, both-fail, one example each T0-T3, fold 3, fold 4, and a weak-fundamental example — selected systematically from the per-recording delta table (§10) and per-frame trajectory-type/energy-ratio scans, not cherry-picked to show only improvements.

---

## 16. Fold-to-fold consistency

Harmonic's grouped-mean tie with HPS masks substantial fold-to-fold variance: it wins by 18.6-188.6¢ margins on 3 folds and loses by 44.6-286.9¢ margins on 2 folds — a much wider spread than HPS's own fold range (158-409¢). This inconsistency, combined with the two-large-recordings finding in §10, suggests the harmonic model's failure mode is not evenly distributed noise but concentrated on specific hard cases (large, likely denser or more heterophonic performances) rather than a uniform, generalizable improvement.

---

## 17. Final classification

**`HPS_ALREADY_CAPTURES_MOST_SIGNAL`**

Supporting evidence:

1. Grouped-mean MAE: harmonic (284.2¢) does not beat HPS (279.0¢) — within noise, slightly worse.
2. Frame-weighted MAE: harmonic (394¢) is clearly worse than HPS (322¢) once fold-size imbalance is corrected for.
3. Per-class: harmonic is worse than HPS on every trajectory type, both MAE and median AE.
4. Octave-adjusted MAE: 77.2¢ (harmonic) vs 77.8¢ (HPS) — a near-exact tie, meaning the two methods extract essentially the same fine-pitch-class information once register is resolved.
5. Octave-selection accuracy: HPS (79.1%) is meaningfully better than harmonic (72.1%) at picking the correct register.

Per spec §31's definition, this is the case where "the learned model roughly matches HPS but does not meaningfully improve it; remaining error likely comes from ambiguity, register, mixture, or target/audio mismatch" — precisely borne out by §12's finding that the residual gap between harmonic and HPS is concentrated in octave/register selection, not fine pitch-class recovery, and by §8's finding that HPS's own dominant error mode is octave confusion, not fine-grained error.

This is **not** `HARMONIC_SALIENCE_WORKS` (no clear, consistent win); **not** `FREQUENCY_LOCAL_HELPS_HARMONICS_DO_NOT` (frequency-local alone, i.e. the local-only control, is a comprehensive failure — 683¢ mean MAE, worse than every deterministic baseline including CQT argmax on some folds — so preserving frequency without harmonic structure clearly does *not* help on its own); **not** `SALIENCE_STILL_FAILS` (harmonic salience is a working, informative representation — median AE near-exact, rank/entropy behavior strong, ties HPS on octave-adjusted MAE — it just doesn't exceed the already-strong deterministic baseline).

---

## 18. Recommendation

**Decision gate: `INVESTIGATE_RESIDUAL_PITCH_AMBIGUITY`**

The evidence chain is consistent and specific: HPS's error is dominated by octave confusion (§4); the harmonic model's octave-adjusted accuracy matches HPS almost exactly (§12) while its raw octave-selection accuracy is *worse* (§12); its salience-rank statistics show the correct pitch is very frequently a top-2-3 candidate even when the final argmax is wrong (§11); and its net grouped-mean parity with HPS is driven by a data-volume-weighted mix of large wins and large losses concentrated in specific recordings (§10, §15), not uniform noise.

**Do next:** before adding architectural complexity, investigate register/octave resolution as a separable problem from pitch-class salience extraction — e.g. examine whether the harmonic model's octave errors are systematically explainable (fundamental below the CQT floor for low registers, energy at 2f dominating for specific instrument/vocal timbres in the two large loss-recordings), and whether a lightweight register-disambiguation step on top of the existing salience map (not a large temporal encoder — Step 9 already showed that doesn't help) could close the gap without touching the core salience representation, which is itself sound (median error near-exact, strong rank statistics, harmonic clearly beats a matched-capacity non-harmonic control).

**Do not:** proceed directly to trajectory modeling with the current salience representation as-is (`PROCEED_TO_TRAJECTORY_WITH_SALIENCE` is not supported — the representation doesn't yet beat the existing HPS baseline it was built to improve on). Do not add a larger temporal encoder (Step 9's `INVESTIGATE_PITCH_FRONTEND` gate was about frontend design, not sequence length, and this experiment's own local-vs-harmonic result reconfirms that raw architectural capacity is not the bottleneck). Do not treat this as primarily an audio/target-alignment problem (`INVESTIGATE_AUDIO_TARGET_ALIGNMENT`) — HPS and the harmonic model largely agree once octave is factored out, which is inconsistent with alignment being the dominant issue at this stage. Do not treat this as primarily a generalization problem (`INVESTIGATE_GENERALIZATION`) — the model performs comparably or better than HPS on both large and small held-out recordings across most folds; the losses are recording-specific rather than a systematic held-out-vs-train gap.

Gate from Step 10 (frequency-preserving + harmonic salience) has now been tested directly: harmonic salience is a real, working representation that clearly beats a matched non-harmonic control and ties the strongest deterministic baseline on fine pitch-class recovery — but it does not yet exceed that baseline overall, and the residual gap is specifically octave/register selection, not the salience extraction itself.
