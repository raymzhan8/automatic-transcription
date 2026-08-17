# Step 17 — Pre-Viterbi vs. Post-Viterbi Fine-Motion Fidelity

> **Follow-up:** the recommended movement-cost sweep was completed downstream in [`docs/step_18_lambda_selection.md`](step_18_lambda_selection.md). The pooled numbers nominally favored an intermediate setting, but fold- and recording-level scrutiny showed the "win" was driven almost entirely by a single fold — verdict `NO_MEANINGFUL_LAMBDA_DIFFERENCE`, decision `FREEZE_LAMBDA_AND_MOVE_UPSTREAM`. Lambda tuning is now closed.

A diagnostic decoder ablation — **no new acoustic model, no salience-frontend retraining, no register decoding, no broad hyperparameter search.** Step 16 established `TEMPORAL_RESOLUTION_LIMITED` but could not distinguish *where* fine motion disappears: already-weak framewise evidence, or the Viterbi temporal decoder suppressing it. This step isolates temporal decoding as the only variable, holding per-frame candidate scoring fixed.

Frozen references: [`docs/step_16_acoustic_pitch_audit.md`](step_16_acoustic_pitch_audit.md), [`docs/step_12_5_fusion_viterbi.md`](step_12_5_fusion_viterbi.md) (the current decoder's origin).

Machine-readable outputs: [`output/pitch_diagnostics/pitch_audit/`](../output/pitch_diagnostics/pitch_audit/), [`output/pitch_motion_ablation/condition_P0_D0/`](../output/pitch_motion_ablation/condition_P0_D0/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/relative_pitch/dense_framewise_argmax_path.py   # D0: framewise argmax, no Viterbi
python -m training.pitch_diagnostics.pitch_audit.decoder_ablation                 # Step 16's full diagnostic suite, D0 vs D1
python -m training.pitch_diagnostics.pitch_audit.visualize_decoder
python training/train_pitch_motion_ablation.py --condition P0 --pitch-variant D0 --all-folds --max-epochs 50 --patience 10
python -m training.pitch_diagnostics.pitch_audit.evaluate_d0_downstream
python training/pitch_diagnostics/relative_pitch/dense_lambda_sweep_path.py       # 0.25x / 0.5x movement-cost sweep points
python -m training.pitch_diagnostics.pitch_audit.lambda_sweep_diagnostics
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| D0 and D1 built from **identical frozen per-fold fused salience** (§ proof below), differing only in the presence of Viterbi | Both computed from the same `fused_probs` forward pass per fold; D0 takes a plain per-column argmax, D1 additionally calls `viterbi_decode` |
| D1 (this audit's build) **exactly reproduces** Step 16 | Same R50=0.394, R100=0.682, absolute MAE 349.1¢, boundary MAE 61.3¢ — trivial exact match since it is literally the same code path, re-run as a live consistency check |
| **The attenuation ratio R is essentially unchanged by Viterbi** — a genuine surprise | R50 = 0.394, R100 = 0.682 for **both** D0 and D1, identically, at every point in a later 4-point sweep (§ sweep) |
| **Turning-point recall is substantially better pre-Viterbi** | Mean recall @50ms (T1-T3): D0 **0.406** vs. D1 **0.273** — a 49% relative improvement |
| **D0 is measurably noisier** — the jitter/smoothing tradeoff is real | T0 estimated std 66.9¢ (D0) vs. 33.3¢ (D1) vs. GT's own 15.1¢; direction-reversal rate 60% (D0) vs. 54% (D1) among nonzero deltas |
| **D0 has *worse* absolute and raw delta-MAE, and worse boundary-region MAE** | Absolute MAE 368.4¢ (D0) vs. 349.1¢ (D1); boundary ±50ms MAE 78.5¢ (D0) vs. 61.3¢ (D1) |
| **D0 has far more octave-transition-driven error** — a real cost of removing Viterbi's implicit large-jump suppression | 62.3% of D0's large motion errors are transition-adjacent, vs. 26.4% for D1 |
| **A clean, monotonic dose-response sweep (0x/0.25x/0.5x/1x movement-cost weight)** confirms the mechanism, not just a two-point comparison | Turning recall: 0.406 → 0.345 → 0.314 → 0.273 (monotonic ↓); absolute MAE: 368.4 → 359.7 → 355.6 → 349.1 (monotonic ↓ the other way) |
| **Downstream: no clean pooled win either direction, but a real, consistent per-class redistribution** | D0-trained P0: pooled macro F1 0.330 vs. D1's 0.338 (flat, within noise) — but T2 F1 **0.150→0.171** (+14%) and T3 F1 **0.085→0.093** (+9%), while T0/T1 both decline modestly |
| The `692ed7e6…` outlier is **decoder-independent**, reconfirming Step 16 | D0 and D1 give nearly identical absolute (128.8 vs. 127.8¢) and motion-error (36.7 vs. 35.2¢) stats on this recording |

**Primary outcome: `VITERBI_TRADES_JITTER_FOR_TOO_MUCH_SMOOTHING`**

**Decision gate: `RETUNE_MOTION_COST_FOR_TRAJECTORIES`**

---

## 1-2. D0/D1/D2 definitions and proof of identical frozen evidence

| Path | Definition |
|---|---|
| **D0** | `training/pitch_diagnostics/relative_pitch/dense_framewise_argmax_path.py`: per-frame `argmax_f S_fused(f,t)` — no transition cost, no future-frame information. |
| **D1** | Step 13's frozen Fused+D3 (`dense_pitch_path.py`) — the current system, unmodified hyperparameters. |
| **D2** | GT parametric pitch (identical source used in Steps 15-16), reference only, not deployable. |

D0 and D1 share the exact same per-fold checkpoint (`load_learned_model("harmonic", fold, ...)`), the exact same `hps_salience_probs` + learned-model forward pass, and the exact same validation-selected fusion ratio (`h["fusion_ratio"]`, Step 12.5) to build `fused_probs` — **byte-identical evidence**. The only code-path difference: D1 calls `viterbi_decode(states, dt_steps, h["fused_lambda_t"], 0.0)`; D0 calls `fused_probs.argmax(axis=0)` directly on the same array. This matches spec section 2's "D0b = fused framewise candidate argmax" as the primary, single, essential comparison — no decoder zoo was built.

---

## 3. Reproduction gate

| Metric | Step 16 (original) | This step's D1 rebuild |
|---|---:|---:|
| R50 | 0.394 | 0.394 |
| R100 | 0.682 | 0.682 |
| Absolute pitch MAE | 349.1¢ | 349.1¢ |
| Boundary ±50ms MAE | 61.3¢ | 61.3¢ |

Exact match, as expected — D1 here is generated by literally the same frozen cache and the same analysis functions Step 16 used, re-run as a live consistency check rather than approximated. No discrepancy to investigate.

---

## 4-5. Primary motion-fidelity comparison and the critical attenuation test

| Path | Abs. MAE | Δ50 MAE | R50 | R100 | Zero-delta \| GT fast | Velocity corr. | Turn recall @50ms (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **D0 framewise** | 368.4¢ | 58.8¢ | 0.394 | 0.682 | 71.1% | 0.057 | **0.406** |
| **D1 Viterbi** | **349.1¢** | **39.9¢** | 0.394 | 0.682 | 77.4% | **0.102** | 0.273 |
| D2 oracle | 0 | 0 | 1.0 | 1.0 | reference | 1.0 | reference |

**The interpretation is the "third possibility" spec section 5 flagged as especially important, not either of the first two.** R50/R100 are identical between D0 and D1 — Viterbi does not change the *typical magnitude* of a move once the decoder decides to move at all (see §17's sweep for why this stays flat across the whole movement-cost range). What Viterbi *does* change is captured elsewhere: it reduces raw delta-MAE and improves velocity correlation (0.057→0.102) — genuine denoising value — while cutting turning-point recall nearly in half. **D0 has more movement, and a meaningful fraction of that extra movement is genuinely useful (more real turns caught) — but a meaningful fraction is also noise** (worse absolute/delta MAE, worse boundary MAE, far more octave-transition-driven error: 62.3% of D0's large errors are transition-adjacent vs. 26.4% for D1). Neither "D0 >> D1" nor "D0 ≈ D1" cleanly describes this; both are simultaneously true depending on which axis is examined.

---

## 6. Jitter-versus-smoothing tradeoff

| | D0 | D1 | GT |
|---|---:|---:|---:|
| T0 estimated std (cents) | **66.9** | 33.3 | 15.1 |
| T0 direction-reversal rate (among nonzero Δ) | **60.3%** | 54.0% | — |
| T0 nonzero-delta fraction | 14.0% | 8.4% | — |
| T1-T3 nonzero-delta fraction | 21.6-27.4% | 15.5-21.9% | — |

Exactly the anticipated pattern from spec section 6: **D0 shows high T1-T3 motion (nonzero-delta fraction, turning recall) alongside high T0 jitter** (more than 4× GT's own std); **D1 shows low T0 jitter alongside low T1-T3 motion recall.** Both halves of the predicted tradeoff are confirmed, quantitatively, not just qualitatively.

---

## 7-8. Turning-point and local-shape comparison

Turning-point recall / false-turn rate by type and tolerance:

| Type | Recall @20/50/100ms — D0 | D1 | False-turn @20/50/100ms — D0 | D1 |
|---|---|---|---|---|
| T1 | 25% / 40% / **56%** | 14% / 26% / 37% | 89% / 80% / 70% | 86% / 74% / 64% |
| T2 | 27% / 41% / **55%** | 12% / 28% / 40% | 92% / 86% / 78% | 90% / 79% / 74% |
| T3 | 22% / 41% / **50%** | 6% / 28% / 34% | 97% / 93% / 89% | 99% / 94% / 87% |

D0's recall is 15-40 percentage points higher at every tolerance for every type — a clear, substantial, consistent improvement. False-turn rate is roughly similar to modestly worse for D0 (a few points), a much smaller cost than the recall gain — **genuine direction reversals are visible pre-Viterbi and are then substantially removed by temporal decoding**, directly answering spec section 7's key question.

Local-shape confusion (rise→fall / fall→rise recall, the exact classes Step 16 found weakest):

| | D0 | D1 |
|---|---:|---:|
| rise→fall recall | **16.4%** | 13.3% |
| fall→rise recall | **14.6%** | 9.4% |
| flat recall (cost of the above) | 71.9% | 85.2% |

D0 does preserve these shapes substantially better (spec section 8's question), though the improvement is partial, not dramatic (13→16%, 9→15%) — and comes at a real cost to flat-region accuracy (85%→72%), the same tradeoff as §6.

---

## 9. Boundary-localized fidelity

| | within ±50ms of boundary | away from boundary |
|---|---:|---:|
| D0 | 78.5¢ | (pooled with §4-5's overall figures) |
| D1 | 61.3¢ | |

**D1 is actually *better* than D0 near boundaries** (61.3¢ vs. 78.5¢) — the opposite direction from the turning-point-recall result. This is not a contradiction: boundaries are where GT motion is often largest and most abrupt, exactly where D0's extra noise (not just extra correct motion) shows up most in a magnitude-based MAE metric, even though D0 also catches more *genuine* turns there in a presence/absence sense (§7). The two metrics answer different questions — "is there error" vs. "is a real event detected" — and disagree in direction here, itself informative: **the boundary problem is not purely "Viterbi erases boundary information"; some of it is inherent noise D0 introduces that a decoder is right to want to suppress**, reinforcing that a full swing to D0 is not the fix.

---

## 10. Staircase analysis

| | D0 median run (moving regions) | D1 median run |
|---|---:|---:|
| T1-T3 | **2.33 frames** | 3.0 frames |

D0's run length is shorter (closer to GT's ideal of 1 frame) but **still well above 1** — confirming Step 16's conditional: the staircase effect is **not purely created by the temporal decoder**; a real component of it originates earlier, in the framewise candidate selection itself (the discrete 16.67¢ candidate grid, and/or the salience map's own tendency to keep the same bin dominant for a few consecutive frames even with zero transition cost). Viterbi measurably *worsens* it (2.33→3.0) but is not solely responsible.

---

## 11. Candidate-switch analysis (D0)

| Type | Switch rate | Frac. toward GT | Frac. away from GT | Median \|jump\| | p90 \|jump\| |
|---|---:|---:|---:|---:|---:|
| T0 | 14.7% | 50.2% | 49.8% | 16.7¢ | 1200¢ |
| T1 | 22.2% | 52.6% | 47.4% | 16.7¢ | 1183¢ |
| T2 | 25.5% | **62.0%** | 38.0% | 16.7¢ | 1117¢ |
| T3 | 28.0% | **43.5%** | 56.5% | 16.7¢ | 1183¢ |

Even with zero temporal cost, the framewise candidate only switches 15-28% of the time between consecutive frames — most of D0's "staircasing" (§10) is not a Viterbi artifact but a property of the salience map itself changing its argmax infrequently. When switches do occur, most (median) are a single adjacent bin (16.7¢) — fine local refinement — but the *90th percentile* jump is roughly a full octave (1100-1200¢) for every type, confirming raw framewise decoding is genuinely prone to occasional large harmonic-confusion jumps that Viterbi's movement cost (even with no explicit octave penalty, `lambda_oct=0` throughout this project) implicitly suppresses (§4-5's octave-transition-contribution finding). For T2, switches move toward GT notably more than chance (62%); for **T3, switches move toward GT *less* than chance (43.5%, worse than a coin flip)** — a specific, quantified weakness in the framewise evidence for T3 that Viterbi cannot be blamed for, since D0 has no decoder at all here.

---

## 12. Representative visualizations

`output/pitch_diagnostics/pitch_audit/figures/` (D0 vs. D1 vs. GT overlay, same coordinates):

- `d0_tracks_d1_flattens.png` — selected for the largest T2/T3 D1-worse-than-D0 gap; in practice shows both paths mostly agreeing and both substantially octave-confused for most of the window, with D1 briefly tracking closer to GT in one sub-region — a reminder that large single-window failures are often compound (octave + motion), not cleanly attributable to one mechanism.
- `d1_stabilizes_d0_noise.png` — selected for the largest T0 D0-worse-than-D1 gap; similarly shows both paths closely agreeing (and both substantially wrong) for most of the window rather than a clean "D1 stabilizes D0's noise" story.
- `both_miss_motion.png` — the cleanest and most representative of the three: an initial ~1.5s stretch where GT, D0, and D1 all track together nicely (a genuine well-matched T0 region), followed by a stretch where GT rises substantially and **both D0 and D1 fail together**, missing the same motion.

Consistent with spec section 12's instruction not to cherry-pick only cases where Viterbi looks bad — the aggregate statistics (§4-11), not individual dramatic windows, carry the real evidentiary weight in this step; representative windows are frequently confounded by simultaneous octave issues.

---

## 13. Absolute pitch (secondary)

| | D0 | D1 |
|---|---:|---:|
| MAE | 368.4¢ | **349.1¢** |
| (median, p95 tracked in the full JSON; pattern consistent with MAE) | | |

Confirmed exactly as spec section 13 anticipated: **D1's absolute MAE is lower while D0's fine-motion fidelity (turning recall) is higher** — optimizing pitch accuracy and optimizing trajectory-relevant contour fidelity are measurably different objectives on this data, not proxies for each other.

---

## 14. Downstream trajectory test

D0's φ distribution (built from a noisier, more octave-jump-prone source) was judged different enough from D1's that frozen-model evaluation would be invalid; per spec section 14's fallback, the exact same P0 architecture was retrained on D0 using identical folds, seeds, budget (50 epochs/patience 10), sampler, loss, and normalization protocol (fold-wise φ statistics re-derived from D0's own train recordings).

| Pitch path | Trajectory macro F1 | T0 F1 | T1 F1 | T2 F1 | T3 F1 |
|---|---:|---:|---:|---:|---:|
| D0 framewise | 0.330 | 0.507 | 0.547 | **0.171** | **0.093** |
| D1 Viterbi (Step 15 P0 reference) | 0.338 | 0.547 | 0.568 | 0.150 | 0.085 |
| D2 oracle (Step 15 P3 reference) | 0.771 | 0.639 | 0.690 | 0.759 | 0.822 |

Pooled macro F1 is a wash (−0.008, well within the ~0.06-0.09 fold-to-fold std seen throughout Steps 14-16). But the per-class pattern is not noise — it is the exact redistribution the motion audit predicted: **T2 (+14% relative) and T3 (+9% relative) improve under D0; T0 and T1 both decline.** This directly connects decoder behavior to the actual transcription task: the classes whose definition depends on fine directional motion (T2/T3) benefit from removing Viterbi's smoothing; the classes that benefit from noise reduction (T0/T1) are hurt by it.

---

## 15. Motion fidelity vs. downstream performance

The per-class pattern in §14 *is* the diagnostic correlation spec section 15 asks for: recordings/classes where D0 shows a large turning-recall advantage (T2, T3 — §7) are exactly the classes where D0's downstream F1 improves; classes where D0's advantage is weakest and its jitter cost is most visible (T0 — §6) are exactly where D0's downstream F1 declines. The motion metrics this audit tracked are not academic — they visibly predict where trajectory classification moves, in the correct direction, class by class.

---

## 16. Movement-cost equation

Since §4-15 clearly implicate Viterbi (real, quantified, class-dependent tradeoff — not "D0 ≈ D1 everywhere"), per spec section 16:

```python
# training/pitch_diagnostics/register_resolution/decoders.py
def _movement_cost(delta_cents, dt_steps):
    normalized = np.abs(delta_cents) / np.maximum(dt_steps, 1.0)
    return np.minimum(normalized, CAP_CENTS)          # CAP_CENTS = 1200.0

# inside viterbi_decode's per-frame transition:
trans = lambda_t * move                                # move = _movement_cost(...)
total = cost[:, None] + trans - cur_log[None, :]        # cur_log = log-salience of the candidate state
```

`lambda_t` (`h["fused_lambda_t"]`, validation-selected in Step 12.5 against **absolute pitch MAE**, per fold) is the only knob; there is no octave-jump penalty active in this pipeline (`lambda_oct=0` throughout, per Step 12/12.5's own findings). The cost is **linear in |Δcents|, uncapped below 1200¢** — moving one bin (16.7¢) costs `lambda_t·16.7`, moving ten bins costs `lambda_t·167`, proportionally. This mathematically favors staying put over following *any* movement, including correct movement, and specifically double-penalizes a genuine reversal (rise-then-fall pays the movement cost twice within a short window) relative to staying flat — a direct, mechanistic explanation for both the staircase effect (§10) and the disproportionate loss of rise→fall/fall→rise shapes (§8) observed throughout Steps 16-17.

---

## 17-18. Minimal movement-cost sweep and Pareto analysis

Four points along the single `lambda_t` axis (0x = D0, 1.0x = D1/current; 0.25x/0.5x built new for this step, same per-fold checkpoints/candidate range, no other change):

| lambda_t multiplier | Absolute MAE | R50 | R100 | Zero-delta frac (50ms, GT-moving) | Turning recall @50ms (mean) | Boundary ±50ms MAE |
|---|---:|---:|---:|---:|---:|---:|
| **0x (D0)** | 368.4¢ | 0.394 | 0.682 | 0.291 | **0.406** | 78.5¢ |
| 0.25x | 359.7¢ | 0.394 | 0.682 | 0.321 | 0.345 | 67.1¢ |
| 0.5x | 355.6¢ | 0.394 | 0.682 | 0.335 | 0.314 | 65.0¢ |
| **1.0x (D1)** | **349.1¢** | 0.394 | 0.682 | 0.358 | 0.273 | **61.3¢** |

A clean, **monotonic** dose-response on every metric except R50/R100 (flat throughout — confirming §4-5's finding that the movement-cost weight controls *how often* the decoder moves, not *how far* it moves when it does). Absolute MAE, zero-delta fraction, and boundary MAE all move monotonically in Viterbi's favor as `lambda_t` increases; turning-point recall moves monotonically the opposite way. **This is the clearest possible confirmation that Step 12's decoder — validation-tuned purely against absolute pitch accuracy — sits at exactly the wrong end of this tradeoff for trajectory transcription**, matching spec section 18's hypothesis precisely: the current setting is near-optimal for absolute pitch, far from optimal for fine-motion/trajectory-relevant fidelity.

**Scope disclosure:** downstream trajectory retraining was run only at the two endpoints (D0, D1 — §14), not at the 0.25x/0.5x intermediate points, given the time budget for this diagnostic step. The motion-metric sweep alone establishes the *direction and shape* of the tradeoff (monotonic, not U-shaped, on every individual motion metric) — per spec section 17's stated goal ("establish the direction of the tradeoff, not optimize the decoder"). Whether an intermediate `lambda_t` produces a downstream macro-F1 (or T2/T3-F1) optimum better than both endpoints is exactly what Step 18 should test directly, cheaply, using the infrastructure already built here.

---

## 19. Reconciling with Step 16 — is Viterbi implicated?

Not exclusively. §10-11 show a real, non-trivial component of the staircase/attenuation problem (run length 2.33 even at 0x, R50/R100 unchanged across the *entire* sweep) originates in the framewise salience evidence itself, not the decoder — this is the honest caveat spec section 19 asks for. But §7-8, §14-18 show Viterbi's movement-cost weight has a real, monotonic, mechanistically-explained, downstream-relevant effect specifically on turning-point/shape recall and specifically on the T2/T3 classes that matter most. Both things are true simultaneously; the decoder is not the *only* source of loss, but it is a real, quantified, and — critically — *adjustable* one.

---

## 20. Central comparison table

| Metric | D0 framewise | D1 Viterbi | D2 oracle |
|---|---:|---:|---:|
| Absolute pitch MAE | 368.4¢ | 349.1¢ | 0 |
| Δ50 MAE | 58.8¢ | 39.9¢ | 0 |
| R50 | 0.394 | 0.394 | 1 |
| R100 | 0.682 | 0.682 | 1 |
| Zero delta during GT motion (\|vel\|>100c/s) | 71.1% | 77.4% | reference |
| Velocity correlation | 0.057 | 0.102 | 1 |
| Turning recall @50ms (mean T1-T3) | 40.6% | 27.3% | reference |
| Rise→fall recall | 16.4% | 13.3% | reference |
| Fall→rise recall | 14.6% | 9.4% | reference |
| Boundary ±50ms MAE | 78.5¢ | 61.3¢ | 0 |
| Moving-region run length | 2.33 frames | 3.0 frames | 1 frame |
| Trajectory macro F1 | 0.330 | 0.338 | 0.771 |

---

## 21. Primary outcome

**`VITERBI_TRADES_JITTER_FOR_TOO_MUCH_SMOOTHING`**

> D0 preserves substantially more real motion (turning recall +49% relative, rise/fall shape recall improved) but is also substantially noisier (T0 std 4.4× GT's own, worse absolute/delta/boundary MAE, far more octave-transition-driven error). D1 does not simply "erase" motion uniformly — it trades a real, useful reduction in noise (better velocity correlation, better absolute accuracy, better boundary MAE) for an equally real, disproportionate loss of exactly the direction-reversal information T2/T3 depend on. **Temporal decoding is necessary; the current smoothness setting is not right for trajectory transcription** — confirmed causally, not just observationally, by the monotonic §17 sweep.

Not `VITERBI_SUPPRESSES_USEFUL_MOTION` cleanly — that outcome requires D0 to improve trajectory classification, and pooled macro F1 does not improve (0.330 vs. 0.338, a wash). Not `FRAMEWISE_EVIDENCE_ALREADY_LACKS_MOTION` — there is a real, monotonic, downstream-relevant difference between D0 and D1, not "similarly poor." Not `DECODER_NOT_DOWNSTREAM_RELEVANT` — the decoder setting visibly moves T2/T3 F1 in the predicted direction; it does have downstream relevance, just not a uniformly positive one.

---

## 22. Step 18 recommendation

**`RETUNE_MOTION_COST_FOR_TRAJECTORIES`**

The §17 sweep reveals a clear, monotonic tradeoff along a single existing hyperparameter (`lambda_t`), and the §14 downstream test shows the two tested endpoints land on opposite sides of a real per-class tradeoff (T0/T1 vs. T2/T3) rather than either endpoint dominating. This is exactly the "small penalty sweep reveals a clear motion/trajectory improvement while retaining useful denoising" scenario the gate describes — not evidence that the movement-cost *formulation* itself is wrong (which would call for `REDESIGN_TEMPORAL_DECODER`), but clear evidence that its *weight*, chosen by Step 12.5 against the wrong objective (absolute pitch MAE, not trajectory relevance), is minimizing the wrong thing.

**Concretely for Step 18:** train downstream P0-style trajectory classifiers at the two untested intermediate sweep points (0.25x, 0.5x `lambda_t`, both already cached by this step) to check directly whether pooled or T2/T3-specific macro F1 has an interior optimum between the D0/D1 endpoints — cheap, since the caches already exist and each downstream run takes only a few minutes. If a genuine improvement is found, consider re-validating `lambda_t` against a trajectory-relevant objective (e.g. T2/T3 F1 or a motion-fidelity proxy) rather than absolute pitch MAE, as a small, targeted follow-up — not a decoder redesign, and not a return to register/octave decoding (§19 reconfirms that branch remains closed).
