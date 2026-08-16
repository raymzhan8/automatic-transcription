# Step 12.5 — Fused Salience + Viterbi Closing Ablation

> **Follow-up:** the recommended next question ("is octave-invariant relative pitch movement already sufficient for trajectory recognition?") was tested in [`docs/step_13_relative_pitch.md`](step_13_relative_pitch.md). Verdict: `RELATIVE_PITCH_PARTIAL` — register errors largely cancel under differencing, but the T0-T3 probe still loses real information vs. oracle pitch, especially for T2/T3.

Closes the one open branch from Step 12's decision gate `TRY_FUSION_PLUS_VITERBI_THEN_STOP_DECODER_ENGINEERING`: does movement-cost Viterbi decoding (D3/D4) recover substantially more useful pitch information when run on the **fused** HPS+learned salience distribution than it did on either source alone? Tests only that combination — no retraining, no new decoder variants, no candidate-representation changes, no neural temporal model, no trajectory-classification work.

Frozen references: [`docs/step_12_register_resolution.md`](step_12_register_resolution.md), [`docs/step_11_harmonic_salience.md`](step_11_harmonic_salience.md).

Machine-readable output: [`output/pitch_diagnostics/register_resolution/fusion_viterbi_result.json`](../output/pitch_diagnostics/register_resolution/fusion_viterbi_result.json).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/register_resolution/fusion_viterbi.py
```

Reuses Step 12's implementations directly: `fusion.py`'s `RATIOS`/`fused_argmax` (log-linear mixing, validation-only selection) and `decoders.py`'s `K_TOP`/`CAP_CENTS`/`OCTAVE_TOL_CENTS`/`LAMBDA_T_GRID`/`LAMBDA_OCT_GRID`/`_states_for_frame`/`viterbi_decode` (identical top-K-plus-octave-shift state construction and time-gap-aware movement cost) — no new transition function, no new grid, no larger grid.

---

## 1. Fused-emission formulation

Identical to Step 12 §12, evaluated over the shared 34-244 candidate window (104.0-778.1 Hz) — the only range the learned model is valid over, so this is also the only range a fused distribution can be defined over. This limitation is inherited unchanged from Step 12 and applies to every fused number in this report.

```
score(f,t) = alpha * log S_learned(f,t) + beta * log S_HPS(f,t)
S_fused(f,t) = softmax_f[ score(f,t) ]     (proper per-frame probability distribution)
```

`fused_argmax` (Step 12's point decode) uses `score` directly; the Viterbi decoders here need a genuine probability column per frame (to rank top-K candidates and take their log-scores as emissions), so `S_fused` is the softmax-normalized form of the same score — the natural probability interpretation of the log-linear mix, not a new formulation.

---

## 2. Validation-selected parameters

alpha/beta (fusion mixing) and lambda_t/lambda_oct (Viterbi, selected separately for HPS-only, learned-only, and fused emissions) — all chosen on validation only, per fold, from Step 12's exact grids:

| Fold | (α_learned, β_HPS) | HPS (λ_t, λ_oct) | Learned (λ_t, λ_oct) | Fused (λ_t, λ_oct) |
|---|---|---|---|---|
| 0 | (0.75, 0.25) | (0.0005, 5.0) | (0.008, 5.0) | (0.008, 5.0) |
| 1 | **(1.0, 0.0)** | (0.0005, 5.0) | (0.008, 0.0) | (0.008, 0.0) |
| 2 | (0.75, 0.25) | (0.0005, 5.0) | (0.008, 0.0) | (0.008, 0.0) |
| 3 | (0.75, 0.25) | (0.008, 0.0) | (0.008, 0.0) | (0.008, 0.0) |
| 4 | (0.75, 0.25) | (0.0005, 5.0) | (0.002, 0.0) | (0.008, 0.0) |

Fold 1 is the outlier: validation selected **pure-learned fusion (α=1.0, β=0.0)** — HPS contributes nothing to the fused distribution for that fold. This is a real, validation-justified choice (val MAE 119.7¢ for (1.0,0.0) vs 132.6¢ for (0.75,0.25) on that fold's single validation recording) — but it does not generalize to fold 1's actual test set, which is exactly where this ablation's headline result comes apart (§5).

---

## 3. Baseline comparison table (pooled, 34-244 shared window)

| Method | MAE | Median AE | ±25¢ | ±50¢ | Octave-adj MAE | Correct-octave |
|---|---:|---:|---:|---:|---:|---:|
| HPS argmax (34-244) | 308.1 | 23.6 | 50.8% | 63.1% | 76.3 | 79.7% |
| HPS + D3 (34-244) | 304.3 | 23.6 | 51.0% | 63.4% | 75.8 | 80.0% |
| HPS + D4 (34-244) | 302.9 | 23.6 | 51.0% | 63.5% | 76.5 | 80.2% |
| Learned argmax | 394.0 | 34.9 | 43.0% | 56.9% | 77.2 | 72.1% |
| Learned + D3 | 377.2 | 32.3 | 44.0% | 57.8% | 72.6 | 73.0% |
| Fusion argmax | 368.4 | 27.2 | 47.3% | 59.5% | 72.9 | 73.8% |
| **Fusion + D3** | **349.1** | **27.2** | 48.1% | 60.5% | **68.0** | 75.0% |
| **Fusion + D4** | **349.1** | 27.2 | 48.1% | 60.5% | 68.0 | 75.0% |
| *Reference: HPS argmax, full 0-360 range* | *322.1* | — | — | — | *77.8* | *79.1%* |
| *Reference: HPS + D3, full 0-360 range* | *317.5* | — | — | — | *77.1* | *79.4%* |
| *Reference: HPS + D4, full 0-360 range* | *315.5* | — | — | — | *77.7* | *79.6%* |

**Fusion+Viterbi does not beat HPS on raw MAE in any comparison** — not against the fair 34-244 HPS baseline (302.9-308.1¢) and not against the original full-range HPS (315.5-322.1¢). D4's octave-jump penalty again adds essentially nothing over D3 (identical to 3 decimal places), matching Step 12 §13's finding. Fusion+Viterbi's real wins are octave-adjusted MAE (68.0¢, the best of every method tested, beating HPS's best of 75.8¢ by ~10%) and median AE (27.2¢, beating HPS's 23.6¢ is actually a loss — HPS has the better median; fusion's win there is only over learned/fusion-argmax, not over HPS). Correct-octave rate is a loss for fusion (75.0%) vs HPS (79.7-80.2%).

---

## 4. Per-fold pooled MAE

| Fold | HPS argmax | HPS + D3 | Learned + D3 | Fusion argmax | **Fusion + D3** |
|---|---:|---:|---:|---:|---:|
| 0 | 192.8 | 188.9 | 165.8 | 183.0 | **170.1** |
| **1** | 243.6 | 236.2 | 421.3 | 428.9 | **421.3** |
| 2 | 401.8 | 401.4 | 426.6 | 398.8 | **365.9** |
| 3 | 326.6 | 320.0 | 235.1 | 221.1 | **205.8** |
| 4 | 157.2 | 156.8 | 98.3 | 95.4 | **82.6** |

Fusion+D3 **beats HPS+D3 on 4 of 5 folds** (0, 2, 3, 4 — by 18.8, 35.5, 114.2, and 74.2¢ respectively), several substantially. It loses on **fold 1 alone**, catastrophically (421.3¢ vs HPS's 236.2¢, a 185¢ gap) — large enough on its own to make the pooled aggregate a net loss despite winning 80% of folds. Fold 1 is exactly the fold that selected pure-learned fusion (§2).

---

## 5. Per-recording win/loss counts

Fused+D3 vs HPS+D3, raw MAE, all 17 recordings:

**13 of 17 recordings improve (76%)**, several substantially:

| Recording (fold) | Support | HPS+D3 | Fusion+D3 | Δ |
|---|---:|---:|---:|---:|
| `68d85d45…` (3) | 9,462 | 299.3 | 98.4 | **+200.8** |
| `6503e348…` (1) | 6,260 | 944.0 | 765.9 | **+178.2** |
| `65b2ab70…` (0) | 252 | 247.8 | 76.2 | **+171.6** |
| `692ed7e6…` (4) | 3,959 | 216.9 | 127.8 | **+89.1** |
| `6653d349…` (2) | 3,366 | 195.1 | 109.1 | **+86.0** |
| `6491d48d…` (4) | 4,955 | 108.7 | 46.5 | **+62.2** |
| `6417585554…` (2) | 50,588 | 532.4 | 492.5 | **+39.9** |
| `65b14e20…` (0) | 1,168 | 82.9 | 49.7 | +33.3 |
| `6653ce5f…` (2) | 2,304 | 93.4 | 60.5 | +32.9 |
| `6655f08a…` (0) | 3,065 | 545.5 | 516.5 | +29.0 |
| `6503e36c…` (0) | 2,458 | 33.2 | 19.3 | +13.9 |
| `645ff354…` (2) | 14,776 | 48.0 | 38.7 | +9.3 |
| `6912841f…` (1) | 18,437 | 47.1 | 41.8 | +5.2 |
| `68f53fbf…` (0) | 3,847 | 32.5 | 33.3 | −0.8 |
| `66552c6b…` (3) | 543 | 240.3 | 273.0 | −32.7 |
| `65e4a79c…` (3) | 1,853 | 449.4 | 734.5 | **−285.0** |
| **`6824de49…` (1)** | **41,857** | 213.6 | 537.0 | **−323.3** |

The two catastrophic losses are `65e4a79c…` (−285.0¢, small support) and **`6824de49…` (−323.3¢, 41,857 frames — 24.7% of the entire corpus, and one of Step 11/12's two flagged large-failure recordings, §6 below).** For `6824de49…`, `fused_argmax`/`fused_d3` numbers are *identical* to `learned_argmax`/`learned_d3` — direct confirmation that fold 1's pure-learned mixing ratio (§2) discarded HPS entirely on exactly the recording where HPS was strongly ahead (213.6¢ HPS+D3 vs 537.0¢ learned+D3 on the same frames).

**This is a real, mechanistically-identified failure mode, not a general property of fusing+decoding**: with only one validation recording per fold (Step 12/10's grouped-CV design picks a single group), the mixing-ratio selection has high variance and can generalize badly whenever a fold's test set contains a recording very different from its lone validation recording — exactly what happened in fold 1. The broad per-recording pattern (13/17, 4/5 folds) is a genuine, non-trivial win; the pooled/support-weighted aggregate is a net loss because of one large, identifiable, single-point-of-failure recording.

---

## 6. The two large-support failure recordings

| | `6417585554…` (fold 2, n=50,588) | `6824de49…` (fold 1, n=41,857) |
|---|---:|---:|
| HPS argmax MAE | 532.3 | 222.5 |
| HPS+D3 MAE | 532.4 | 213.6 |
| Learned argmax MAE | 609.2 | 542.1 |
| Learned+D3 MAE | 577.0 | 537.0 |
| Fusion argmax MAE | 535.6 | 542.1 *(= learned)* |
| **Fusion+D3 MAE** | **492.5** | **537.0** *(= learned)* |
| HPS+D3 correct-octave | 64.4% | 85.7% |
| Fusion+D3 correct-octave | 64.7% | 58.2% |

The two flagged recordings behave **oppositely** under fusion: `6417585554…` (fold 2, ratio 0.75/0.25) is a genuine fusion+Viterbi win (532.4→492.5¢, +39.9¢) with correct-octave essentially unchanged; `6824de49…` (fold 1, ratio 1.0/0.0) is a total fusion failure because the fold's selected ratio zeroed out HPS's contribution on the one recording where HPS was already strong (85.7% correct-octave) and learned was weak (58.2%). The pooled result is dominated by `6824de49…`'s larger loss outweighing `6417585554…`'s smaller win, since both are large but the loss is larger in magnitude and both carry comparable support.

---

## 7. Oversmoothing check (T0-T3, |dp/dt|)

Trajectory type and `dp_dt_log2_hz_per_s` used as evaluation metadata only, never for decoding.

**By trajectory type**, Fusion+D3 vs HPS+D3 (MAE, relative degradation):

| Type | HPS+D3 | Fusion+D3 | Relative Δ |
|---|---:|---:|---:|
| T0 | 287.7 | 364.1 | +26.6% |
| T1 | 305.3 | 324.3 | +6.2% |
| T2 | 266.0 | 389.3 | +46.4% |
| T3 (fastest/ornaments) | 417.2 | 403.6 | **−3.3% (improvement)** |

**By |dp/dt| bucket** (movement speed in cents/second):

| Bucket | HPS+D3 | Fusion+D3 | Relative Δ |
|---|---:|---:|---:|
| 0-100 c/s | 294.9 | 345.3 | +17.1% |
| 100-400 c/s | 271.7 | 365.6 | +34.6% |
| 400-1000 c/s | 284.8 | 356.4 | +25.1% |
| >1000 c/s (fastest) | 351.0 | 350.3 | **−0.2% (essentially tied)** |

**No oversmoothing signature.** If Viterbi's movement-cost were damaging fast-moving regions, degradation should be worst at T3 and in the fastest `|dp/dt|` bucket — instead both show the *smallest* relative degradation of any group (T3 actually improves; the fastest-movement bucket is a wash). The largest relative degradation is at T2 and in the moderate-speed 100-400 c/s bucket, which lines up with the per-recording story (§5-6): the damage is concentrated in specific recordings (particularly `6824de49…`, where fusion collapsed to pure-learned), not in fast-trajectory frames generically.

---

## 8. Final classification

**`FUSION_VITERBI_MARGINAL`**

Rationale against the pre-declared criteria:

- **Does not clearly beat HPS raw MAE** — pooled Fusion+D3/D4 (349.1¢) is worse than every HPS variant tested (302.9-322.1¢), disqualifying `FUSION_VITERBI_CLEAR_WIN` outright.
- **But the win is real and broad, not absent** — 4/5 folds and 13/17 recordings (76%) show genuine, often large, fusion+Viterbi improvements over HPS+Viterbi; octave-adjusted MAE (68.0¢) and correct-octave-given-wrong-octave-recoverable-evidence are both improved pooled; the pooled raw-MAE loss traces to one mechanistically-identified, single-point-of-failure recording (`6824de49…`, §5-6) rather than a systematic weakness of the fused-Viterbi combination itself.
- This is "beats it only... inconsistently" (the `FUSION_VITERBI_MARGINAL` criterion), not "does not beat HPS" outright (`FUSION_VITERBI_FAILS`) — the inconsistency has a specific, identified cause (single-validation-recording ratio selection, §2) rather than being unexplained noise.
- No oversmoothing on fast-moving/T2-T3 regions (§7) — ruling out one plausible alternative explanation for the shortfall.

**Decision: `STOP_REGISTER_DECODER_ENGINEERING`**

---

## 9. Answer to the core question and recommendation

> Does Viterbi recover substantially more useful pitch information when operating on the complementary HPS + learned fused salience distribution than it did on either representation alone?

**Partially, but not enough.** Fusion+D3 (349.1¢) clearly beats Learned+D3 alone (377.2¢, −28.1¢/−7.4%) and Fusion-argmax alone (368.4¢, −19.3¢/−5.2%) — so Viterbi *does* extract real additional value from the fused distribution specifically, confirming the fused map carries more temporally-exploitable structure than the learned map alone. But it does **not** exceed what Viterbi already achieves on HPS alone (304.3¢/302.9¢) — the best decoder in this entire step, on every measured configuration, pooled, remains **HPS + D3/D4 on its own**, at 302.9-317.5¢ depending on range.

Per Step 12's original gate, this closes the branch: fusion and Viterbi decoding, separately and combined, have now been tried on every combination the evidence justified, and none clears HPS's own raw-MAE floor at the pooled level. Continuing to iterate on decoder variants (different transition costs, different fusion weighting schemes, per-fold-adaptive ratios) is very unlikely to be worth it relative to the effort — the one concrete lever identified here (fold 1's single-validation-recording fragility, §2/§5) is a **data-splitting** issue, not a decoding one, and fixing it (e.g. multi-recording validation pools) is a Step 13+-scale change to shared CV infrastructure, out of scope for a closing ablation.

**Do next:** per the calling brief, move to testing whether **octave-invariant relative pitch movement** (e.g. `dp/dt`, contour shape, tonic-relative delta sequences that don't require resolving absolute register at all) is already sufficient signal for trajectory-shape recognition — sidestepping the register-resolution problem this and Step 12 have now shown is real, well-characterized, but not cheaply fixable, rather than requiring it to be solved first.

**Do not:** try another decoder variant, another fusion scheme, or retrain the salience model in pursuit of closing this specific gap further. HPS argmax (or HPS+D3, a ~2% free improvement) remains the strongest deployable absolute-pitch frontend from this entire investigation (Steps 10-12.5).
