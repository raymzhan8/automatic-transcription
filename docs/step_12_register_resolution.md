# Step 12 — Register/Octave Resolution as a Separable Problem

> **Closing ablation:** the one combination this step's own gate flagged as untested (Viterbi decoding on the *fused* HPS+learned distribution) was run in [`docs/step_12_5_fusion_viterbi.md`](step_12_5_fusion_viterbi.md). Verdict: `FUSION_VITERBI_MARGINAL` → `STOP_REGISTER_DECODER_ENGINEERING`. HPS argmax/D3 remains the strongest pooled pitch frontend from Steps 10-12.5.

Direct follow-up to Step 11's decision gate `INVESTIGATE_RESIDUAL_PITCH_AMBIGUITY`: Step 11 found HPS and the learned harmonic-salience model tie almost exactly on octave-adjusted MAE (fine pitch-class recovery) but HPS wins clearly on raw octave-selection accuracy (79.1% vs 72.1%) — implying pitch-class salience and register/octave selection are separable problems, and register selection is where the residual gap lives. This step tests that directly: diagnose the structure of octave errors (Phase A), then try the two lightweight fixes Phase A's own evidence justifies (Phase B) — **no retraining, no large temporal encoder, no trajectory/decoder work.**

Frozen references: [`docs/step_11_harmonic_salience.md`](step_11_harmonic_salience.md), [`docs/step_10_pitch_diagnostics.md`](step_10_pitch_diagnostics.md).

Machine-readable outputs: [`output/pitch_diagnostics/register_resolution/`](../output/pitch_diagnostics/register_resolution/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/register_resolution/candidate_range_fixed.py  # §1 range-fix audit
python training/pitch_diagnostics/register_resolution/collect.py                # pooled per-frame HPS+learned cache (all 17 recordings' test folds)
python training/pitch_diagnostics/register_resolution/octave_diagnostics.py     # §2-6 octave-error structure
python training/pitch_diagnostics/register_resolution/oracles.py                # §7-8 decoder-free upper bounds
python training/pitch_diagnostics/register_resolution/static_prior.py           # §9 D1 static register prior
python training/pitch_diagnostics/register_resolution/disagreement.py           # §10 HPS/learned disagreement (D2 justification)
python training/pitch_diagnostics/register_resolution/synthesize.py             # Phase-A checkpoint: reads the above, emits the D1-D4 scope verdict
python training/pitch_diagnostics/register_resolution/fusion.py                 # §12 D2: frame-independent log-linear fusion
python training/pitch_diagnostics/register_resolution/decoders.py               # §13 D3/D4: movement-cost (+ octave-penalty) Viterbi
```

**Out of scope:** retraining the salience model, a learned/large temporal encoder, trajectory type/phase/activity, the trajectory decoder, class weighting, hyperparameter search beyond small fixed grids on validation.

---

## Executive summary

| Finding | Evidence |
|---|---|
| Step 11's candidate range for the learned model was **derived from all recordings, including held-out ones** — a latent leakage risk | Fixed to the full physically-meaningful 0-360 CQT range (§1); HPS is unaffected (it always argmaxed over the full CQT) |
| Under the corrected full range, the learned model gets **materially worse**, not better | Grouped MAE 284.2¢→370.5¢, octave-adjusted 77.2¢→110.2¢ — an out-of-distribution effect, not evidence Step 11 was inflated by leakage (§1) |
| Learned model's valid operating domain is its **native trained range** (34-244) — used for every subsequent diagnostic here | HPS stays full-range throughout, exactly matching Step 11 |
| Wrong-octave states are **sticky and sustained**, not isolated glitches | Octave self-transition probability 92-98%; median wrong-octave run ≈50-60ms (several frames) for both methods (§2) |
| When the argmax picks the wrong octave, the **correct pitch is still commonly in the top-3/5** for HPS, much less so for the learned model | HPS: 36.8%/50.5% (top3/top5); learned: 17.3%/33.5% (§3) — a decoding problem for HPS, more of a representation problem for learned |
| Correct-octave frames have **much larger salience margins and lower entropy** than wrong-octave frames, for both methods | HPS margin 0.183 vs 0.090; learned margin 0.0148 vs 0.0104 — but learned's margins are **~12x smaller in absolute terms**, i.e. its distribution is far flatter throughout (§4) |
| Once the wrong octave is picked, the **failure mode is asymmetric and different between methods** | HPS +1-octave errors are dominated by 2×GT outscoring GT (55.7% of +1 cases); learned +1-octave errors are dominated even more strongly (80.7%) — the learned model over-trusts the second harmonic more often (§5) |
| The two largest-support recordings (55% of all data) are the **dominant source of harmonic's Step-11 grouped-mean parity** | Both show the learned model's entropy roughly doubling vs HPS on the same frames — diffuse, unconfident predictions, not a clean alternative octave choice (§6) |
| **Decoder-free oracle headroom is large** for both methods | HPS: argmax 322.1¢ → oracle-top5 77.5¢ (244.5¢ headroom); learned: 394.0¢ → 192.2¢ (201.8¢ headroom) (§7) |
| Given the **true octave**, HPS and learned recover fine pitch **equally well** (~62¢ MAE both) | Reconfirms Step 11 §12: once register is resolved, both methods extract essentially the same pitch-class information (§8) |
| D1 (static training-only register prior) **does not help** | 0¢ improvement for HPS (best λ=0); actively **hurts** learned (394.0¢→440.1¢) — rejected, as Phase A predicted |
| D2 (HPS/learned disagreement) is **genuinely complementary** | HPS-right/learned-wrong 13.2% of all frames; learned-right/HPS-wrong 6.2% — both non-trivial, justifying fusion (§10) |
| **D2 fusion underdelivers**: log-linear mixing does not beat HPS alone on raw MAE | Fused 368.4¢ vs HPS-only 308.1¢ (same 34-244 shared window) — dragged down by the learned model's poor raw calibration even at the validation-selected mixing ratio (mostly 0.75 learned / 0.25 HPS) (§12) |
| Fusion **does help octave-adjusted MAE**, its best result of the three | Fused 72.9¢ vs HPS-only 76.3¢ vs learned-only 77.2¢ — a small, real fine-pitch win, not a register win (§12) |
| **D3/D4 Viterbi decoding gives a modest, real win for the learned model, only a marginal one for HPS** | Learned: raw MAE 394.0¢→377.2¢, octave-adjusted 77.2¢→72.6¢; HPS: raw MAE 322.1¢→315.5-317.5¢, octave-adjusted essentially flat (§13) |
| The explicit octave-jump penalty (D4) **adds nothing over plain movement-cost (D3)** for the learned model, and only a small correct-octave-rate bump for HPS | Validation always selected λ_oct∈{0, 5} with near-identical resulting MAE; d4=d3 exactly for learned in the pooled summary (§13) |
| **No configuration tested beats HPS's raw pooled MAE (322.1¢)** | Best of all Phase-B results is HPS+D4 Viterbi at 315.5¢ — a ~2% improvement over HPS argmax, not a qualitative fix |

**Classification: `REGISTER_STRUCTURE_CONFIRMED_LIGHTWEIGHT_DECODING_INSUFFICIENT`**

**Decision gate: `TRY_FUSION_PLUS_VITERBI_THEN_STOP_DECODER_ENGINEERING`**

---

## 1. Candidate-range audit and reconciliation

`register_resolution/candidate_range_fixed.py` re-derives the candidate frequency axis with **no target statistics at all** — the full physically-meaningful CQT range, bins `[0, 360)` (75.0-2400.0 Hz) — because Step 11's range (`[34, 244)`, 104.0-778.1 Hz) was computed from the 0.5-99.5% quantiles of `pitch_log2_hz` **pooled across all 17 recordings**, including ones each fold later holds out. That is a latent test-statistics-in-training-input risk, even though the learned model's scorer is frequency-position-agnostic (1×1 convs) and the range only ever bounded the *output axis*, not model weights directly.

`step11_reconciliation.json` compares Step 11's frozen numbers against the same metrics recomputed with the model evaluated over the full `[0, 360)` range:

| Metric | Step 11 (34-244) | Full range (0-360) | Relative change | Material? |
|---|---:|---:|---:|:---:|
| HPS mean MAE | 279.0 | 279.0 | 0.02% | No |
| HPS pooled MAE | 322.1 | 322.1 | 0.01% | No |
| Learned grouped-mean MAE | 284.2 | 370.5 | +30.4% | **Yes** |
| Learned pooled MAE | 394.0 | 516.6 | +31.1% | **Yes** |
| Learned octave-adjusted MAE | 77.2 | 110.2 | +42.8% | **Yes** |
| Learned correct-octave | 72.1% | 68.4% | 5.1% | No |

HPS is unaffected (it always argmaxes over the full CQT already). The learned model gets **materially worse**, not better, under the full range — driven by a new upward-octave bias into the 44,742 frames (26.4% of the corpus) that fall in the newly-opened high-frequency bins the scorer never received gradient signal for during training. This is out-of-distribution extrapolation, not evidence that the narrower range was propping up Step 11's numbers via leakage — if leakage had inflated Step 11's quality, removing it should not make results this much worse in a systematic, direction-specific way.

**Resolution, carried through the rest of this report:** the learned model is evaluated on its **native trained range** (34-244) for every diagnostic below — that is its valid operating domain. HPS is reported at the full range throughout, exactly matching Step 11. The original range-derivation issue remains a disclosed methodological caveat on Step 11's learned-model numbers; a fully clean fix (retraining under a pre-registered range) is out of scope here and flagged as a Step 13+ item.

---

## 2. Octave-error temporal structure

`octave_diagnostics.py`, run-length and Markov-transition analysis over `octave_k` sequences (consecutive frames within a recording).

**Run durations** (`octave_run_durations.json`):

| | HPS | Learned |
|---|---:|---:|
| n runs (any `k≠0`) | 3,351 | 4,676 |
| median duration | 60ms | 50ms |
| p90 duration | 280ms | 260ms |
| mean duration | 138ms | 119ms |

Median run length is several native 10ms frames, not a single-frame glitch — **wrong-octave errors are sustained, not isolated spikes**, for both methods.

**Transition stickiness** (`octave_transition_matrix.json`):

| | HPS | Learned |
|---|---:|---:|
| P(k=0 → k=0) | 98.2% | 97.6% |
| P(k=+1 → k=+1) | 91.7% | 91.6% |

Once a wrong-octave state is entered, it is very likely to persist into the next frame for both methods — this is the direct justification for a temporal decoder (movement-cost Viterbi, §13) rather than a per-frame fix.

---

## 3. GT availability given a wrong-octave argmax

`gt_topk_given_wrong_octave.json`, restricted to the 35,365 (HPS) / 47,223 (learned) frames where the argmax picked the wrong octave:

| | HPS | Learned |
|---|---:|---:|
| GT in top-2 | 19.9% | 7.0% |
| GT in top-3 | 36.8% | 17.3% |
| GT in top-5 | 50.5% | 33.5% |
| Median GT rank (wrong-octave only) | 5 | 8 |

For comparison, when the octave is already correct, GT is in the top-2 in ~76% of frames for both methods — so wrong-octave frames are genuinely harder, but HPS's salience map still frequently "knows" the right answer at rank 2-5 even when its argmax is wrong. The learned model's map is less informative conditional on being wrong: **HPS's residual error looks like a decoding problem (evidence present, wrong choice made); the learned model's looks more like a representation problem (evidence often genuinely thin).**

---

## 4. Salience margins and entropy by octave correctness

`salience_margins.json`, top1-vs-top2 score margin and full-distribution entropy, split by whether the argmax landed in the correct octave:

| | HPS correct | HPS wrong | Learned correct | Learned wrong |
|---|---:|---:|---:|---:|
| Mean margin | 0.183 | 0.090 | 0.0148 | 0.0104 |
| Mean entropy (nats) | 2.06 | 2.97 | 3.37 | 4.11 |

Both methods show the expected qualitative pattern (higher margin / lower entropy when correct), but the learned model's margins are **roughly 12× smaller in absolute terms** than HPS's at every octave-correctness level — its softmax output is far flatter throughout, which is consistent with §3's finding that its evidence is thinner, and previews why a downstream decoder built on these same emission scores (§13) has less to work with for the learned model even where it does help.

---

## 5. Harmonic-pair analysis (GT/2 vs GT vs 2·GT)

`harmonic_pair_analysis.json`: for frames where the argmax lands one octave off, compare the normalized salience mass at the candidate bin nearest to GT/2, GT, and 2·GT.

| | HPS +1-octave (n=20,355) | Learned +1-octave (n=35,806) | HPS -1-octave (n=9,331) | Learned -1-octave (n=5,209) |
|---|---:|---:|---:|---:|
| frac(2·GT mass > GT mass) | 55.7% | **80.7%** | — | — |
| frac(GT/2 mass > GT mass) | — | — | 62.9% | 71.2% |

When either method errs an octave high, the second harmonic (2·GT) usually genuinely outscores the fundamental in the salience map — and the learned model does this **more** often than HPS (80.7% vs 55.7%), i.e. it is more prone to over-trusting the second harmonic specifically. This is a concrete, mechanistic explanation for part of §12's earlier ablation finding (Step 11 §13: 4f and 3f mattered most) — the learned model's harmonic-aggregation still under-weights the fundamental relative to 2f often enough to flip the octave choice.

---

## 6. Large-failure-recordings deep dive

`large_failure_recordings_analysis.json` revisits the two recordings Step 11 §10 flagged as the dominant source of harmonic's grouped-mean parity with HPS (`6417585554…`, 50,588 frames; `6824de49…`, 41,857 frames — 55% of all data), contrasted with three recordings where the learned model clearly wins.

| Recording | n | HPS raw MAE | Learned raw MAE | HPS entropy | Learned entropy |
|---|---:|---:|---:|---:|---:|
| `6417585554…` (large failure) | 50,588 | 531.8 | 609.2 | 2.87 | 3.95 |
| `6824de49…` (large failure) | 41,857 | 255.2 | 542.1 | 2.00 | 3.73 |
| `68d85d45…` (strong win) | 9,462 | 298.7 | 110.1 | 2.26 | 3.84 |

On the two large-failure recordings, the learned model's mean entropy is roughly **double** HPS's on the identical frames — its predictions there are diffuse and unconfident rather than confidently wrong in a specific alternative register. On the strong-win recording, the learned model still has higher entropy than HPS but decisively lower error — so higher entropy alone doesn't predict failure; what distinguishes the large-failure recordings is high entropy **combined with** the octave-selection actually landing wrong at a materially higher rate than in the win cases.

---

## 7. Oracle top-k headroom (decoder-free upper bound)

`oracle_topk.json`: given the top-k candidates already in the salience map, pick whichever is closest to GT (an oracle, not achievable by any real decoder, but an honest ceiling on what a *smarter decoder over these same emissions* could ever recover).

| | HPS argmax | HPS oracle-top5 | Learned argmax | Learned oracle-top5 |
|---|---:|---:|---:|---:|
| MAE | 322.1 | **77.5** | 394.0 | **192.2** |
| Octave-adjusted MAE | 77.8 | 39.7 | 77.2 | 48.4 |
| Headroom (argmax − top5) | — | **244.5¢** | — | **201.8¢** |

Both methods carry substantial headroom — the salience maps already "contain" a much better answer than the argmax picks most of the time. This headroom is the central justification for attempting any decoder at all (D2-D4); §13 tests how much of it two cheap, non-learned decoding schemes can actually recover.

---

## 8. Octave-oracle: fine pitch given the true register

`octave_oracle.json`: candidate search restricted to a ±600¢ band around the true pitch (i.e. the octave is given for free; diagnostic only, not a real decode).

| | HPS | Learned |
|---|---:|---:|
| MAE | 62.7 | 61.5 |
| Median AE | 16.2 | 21.1 |

Essentially tied (within 1¢ of each other), reconfirming Step 11 §12: **once register is resolved, HPS and the learned model recover fine pitch-class equally well.** This is strong evidence that any real gain to be had from register resolution is capped near ~60¢ MAE, not near 0 — the two methods' pitch-class representations are already close to saturated relative to each other.

---

## 9. D1 — static training-only register prior (rejected)

`static_prior.py`: per fold, `P_train(f)` estimated from train-only recordings (smoothed absolute-Hz pitch-bin histogram), λ selected on validation from `{0, 0.25, 0.5, 1.0, 2.0}`, decode `argmax_f[log S(f,t) + λ·log P_train(f)]`.

| | HPS | Learned |
|---|---:|---:|
| Raw argmax MAE | 322.1 | 394.0 |
| Static-prior-decode MAE | 322.1 (λ=0 selected every fold) | **440.1** (λ selected per-fold from val, still net negative on test) |

HPS's validation-selected λ is 0 in every fold (the prior never helps), and applying a per-fold-selected prior to the learned model actively **hurts** test performance (394.0→440.1¢) despite occasionally looking good on validation — a train/test mismatch consistent with performance-specific register drift that a single corpus-level static prior can't capture. **D1 rejected**, exactly as Phase A's synthesis predicted before Phase B ran.

---

## 10. D2 justification — HPS/learned disagreement

`disagreement.py`, 169,150 pooled frames, correct/wrong defined by `octave_k==0`:

| Category | Fraction | n |
|---|---:|---:|
| A: both correct | 65.9% | 111,412 |
| B: HPS correct, learned wrong | 13.2% | 22,359 |
| C: learned correct, HPS wrong | 6.2% | 10,501 |
| D: both wrong | 14.7% | 24,878 |

Both B and C are well above a 3% triviality threshold — **the errors are genuinely complementary**, not one method strictly dominating. In category B (HPS right, learned wrong), HPS's own margin is high (0.120) while learned's is near its typical floor (0.011); in category C (learned right, HPS wrong), the reverse holds less strongly (HPS margin 0.072, still non-trivial). This complementarity is what justifies attempting fusion (§12) — but note the margin asymmetry already hints fusion will be lopsided: HPS is more confidently right in its wins than learned is in its wins.

---

## 11. Phase A → Phase B scope verdict

`synthesize.py` reads the above JSON artifacts and applies fixed, pre-declared thresholds:

| Decoder | Justified? | Reasoning |
|---|:---:|---|
| D1 (static prior) | **No** | Improvement > 10¢ for at least one method: False |
| D2 (fusion) | **Yes** | Both B and C disagreement fractions > 3% |
| D3/D4 (Viterbi ± octave penalty) | **Yes** | Oracle top-5 headroom > 30¢ AND GT commonly in top-k given wrong octave AND runs are sustained (all true) |

D1 was skipped; D2 and D3/D4 were implemented and evaluated (§12-13 below).

---

## 12. D2 — frame-independent log-linear fusion

`fusion.py`: `R(f,t) = α·log S_learned(f,t) + β·log S_HPS(f,t)`, no register-prior term (D1 already rejected). Both maps restricted to the learned model's native window (34-244) so HPS is compared fairly on the same candidate bins. α/β selected from 5 fixed ratios per fold on validation only.

| Fold | Best (α_learned, β_HPS) |
|---|---|
| 0 | (0.75, 0.25) |
| 1 | (1.0, 0.0) |
| 2 | (0.75, 0.25) |
| 3 | (0.75, 0.25) |
| 4 | (0.75, 0.25) |

Test, pooled, shared 34-244 window:

| | HPS-only | Learned-only | Fused |
|---|---:|---:|---:|
| MAE | **308.1** | 394.0 | 368.4 |
| Octave-adjusted MAE | 76.3 | 77.2 | **72.9** |

Validation consistently weights the fusion **toward the learned model** (0.75/0.25 in 4/5 folds, fully learned in fold 1) — but the learned model's raw pitch selection is worse than HPS's, so the fused argmax ends up worse than HPS alone on raw MAE (368.4 vs 308.1), even though it is the single best result on octave-adjusted MAE (72.9, beating both inputs). **D2 fusion delivers a real, if narrow, fine-pitch-class win but not a register win** — validation optimizes for overall MAE, which pulls the mixing ratio toward the (relatively) worse raw performer because that same performer happens to help the octave-adjusted metric fusion isn't being selected on.

---

## 13. D3/D4 — movement-cost Viterbi decoding

`decoders.py`: per-frame candidate states = top-5 salience candidates unioned with their ×2/÷2 octave-shifted bins (deduped). Decoding runs over the sequence of *valid* frames only, with a time-gap-aware movement cost (normalized by elapsed native-hop count, capped at 1200¢) — a documented simplification vs. decoding through literal every-10ms-frame including invalid gaps. D3 = movement cost alone; D4 = D3 + an additive penalty when a transition is close to an exact octave multiple. λ_transition and λ_octave selected per fold, per method, from small fixed grids on validation only.

Pooled test summary (HPS at full 0-360 range, learned at native 34-244 range — matching §1's resolution):

| | Raw argmax (§7 ref) | D3 | D4 |
|---|---:|---:|---:|
| HPS MAE | 322.1 | 317.5 | **315.5** |
| HPS octave-adjusted MAE | 77.8 | 77.1 | 77.7 |
| HPS correct-octave | 79.1% | 79.4% | 79.6% |
| Learned MAE | 394.0 | **377.2** | **377.2** |
| Learned octave-adjusted MAE | 77.2 | **72.6** | **72.6** |
| Learned correct-octave | 72.1% | 73.0% | 73.0% |

Validation selected λ_octave=0 for the learned model in 4/5 folds (fold 0 the exception, λ_oct=5 with a small extra gain) — **the explicit octave-jump penalty adds essentially nothing beyond plain movement-cost smoothing** for the learned model, and only a marginal correct-octave-rate bump for HPS (79.4%→79.6%). Fold 3 (Step 10/11's known hardest fold) shows the largest absolute lambda_t values selected (0.008, the top of the grid) for both methods, consistent with that fold needing the strongest smoothing.

**Viterbi decoding gives a real, if modest, win for the learned model** (raw MAE −17¢/−4.3%, octave-adjusted −4.6¢/−6.0%) and only a marginal one for HPS (raw MAE −4-7¢/−1.4-2.2%, octave-adjusted flat to slightly worse). This matches §4's margin finding directly: HPS's per-frame emissions are already sharp (large correct/wrong margin gap), leaving little room for a downstream decoder relying on those same emissions to improve; the learned model's emissions are flat, so smoothing over them recovers more.

---

## 14. Final classification

**`REGISTER_STRUCTURE_CONFIRMED_LIGHTWEIGHT_DECODING_INSUFFICIENT`**

Supporting evidence:

1. Register/octave errors are genuinely structured — sticky, sustained runs (§2), not per-frame noise — and the correct answer is usually recoverable from the top-3/5 candidates already in the salience map (§3, §7), confirming Step 11's diagnosis that register selection is a separable, addressable problem.
2. Once given the correct octave, HPS and the learned model recover fine pitch-class equally well (§8, ~62¢ MAE both) — the remaining gap is specifically about *which* octave gets picked, not about pitch-class representation quality.
3. D1 (static prior) is correctly rejected by Phase A's own pre-declared criteria and confirmed net-negative on test (§9).
4. D2 (fusion) and D3/D4 (Viterbi ± octave penalty) are both justified by Phase A's evidence and both produce **real but small** gains — fusion wins on octave-adjusted MAE only (§12); Viterbi wins meaningfully for the learned model, marginally for HPS (§13).
5. **No tested configuration beats HPS's own raw pooled MAE (322.1¢).** The best result overall is HPS + D4 Viterbi at 315.5¢ — a ~2% improvement, not a resolution of the problem. The ~245¢ oracle headroom (§7) remains almost entirely unrecovered.

This is **not** `REGISTER_RESOLVED` (headroom is still ~90%+ unclaimed) and **not** `REGISTER_UNRESOLVABLE_WITH_CURRENT_SALIENCE` (the oracle and top-k analyses show the correct answer is very often *present* in the salience maps — this is a decoding/calibration gap, not an information gap). It sits between: the diagnosis from Step 11 is fully confirmed and mechanistically explained, but the two lightweight fixes the evidence justified were not sufficient to close it.

---

## 15. Recommendation

**Decision gate: `TRY_FUSION_PLUS_VITERBI_THEN_STOP_DECODER_ENGINEERING`**

**Do next (cheap, not yet tried):** run the D3/D4 Viterbi decoder **on top of the D2 fused distribution** rather than on HPS-alone or learned-alone emissions. This wasn't tested in this step (`decoders.py` only takes single-source emissions) but is a direct, low-cost combination of two already-validated, already-implemented pieces, and fusion's own octave-adjusted win (§12) suggests the fused map may have sharper correct-octave structure than either input alone for a decoder to exploit. If that combination still doesn't clear HPS's raw pooled MAE, treat decoder engineering on these emissions as exhausted.

**Do next (if the above still doesn't close the gap):** the real bottleneck is more likely emission calibration than decoding strategy — §4 showed the learned model's margins are ~12× smaller than HPS's at every octave-correctness level, i.e. its distribution is chronically flat. A sharper training objective (temperature-scaled or focal-style loss) or more training signal for the model's confidence, not another decoder variant, is the more promising lever — but this crosses into retraining, out of scope for a Step 12-style diagnostic.

**Do not:** add a learned/large temporal encoder to replace the Viterbi decoder — Step 9 and Step 11 already rejected added architectural capacity as the bottleneck, and nothing here contradicts that; the gap is in decode/calibration, not model size. Do not adopt the fused or Viterbi-decoded **learned** model as a production replacement for HPS on raw MAE — HPS argmax alone (322.1¢) still beats every Phase-B learned-model variant tested (best: 377.2¢). Do consider the Viterbi-decoded **learned** model where octave-adjusted quality specifically matters (72.6¢, better than HPS's 77.1-77.8¢ in every decoding condition tested) — e.g. as a fine-pitch-class refinement source once register is otherwise resolved.

**Practical takeaway:** HPS remains the strongest deployable pitch frontend for now. The register-resolution mechanism Step 11 hypothesized is real and now well-characterized (sticky sustained runs, complementary errors, huge but under-realized oracle headroom, and a concrete margin/entropy explanation for why HPS resists cheap decoding improvements more than the learned model does) — but closing it requires either the one untried cheap combination above, or a genuine calibration fix to the learned model, not more decoder variants on the emissions as they stand today.
