# Step 28 — Does Neighboring-Trajectory Context Improve Oracle-Boundary Typing?

Steps 21-27 exhausted the single-trajectory local-information branch: CREPE contour only (A0, macro F1 ≈0.311), audio only (A1, ≈0.319 pooled / 0.291 grouped), linear audio+CREPE fusion (Step 26 L0/A2, ≈0.367), and one nonlinear fusion layer (Step 27 L1, ≈0.316 — worse than L0 on every class, with a direct mechanistic diagnostic showing the hidden layer never learned to separate Cosine from Sloped-start). Every one of these conditions isolates a single trajectory's own local evidence window. Step 28 asks whether information from the immediately adjacent trajectories — still under oracle (GT) boundaries, still not a segmentation experiment — helps classify the center trajectory, without ever giving the model the neighbors' true labels (those are part of the output we're trying to predict, not a legitimate input).

Frozen exactly from Step 26: `ContourCNN`, `AcousticCNN`, CREPE pipeline, CQT cache, TRAIN-only normalization, class-balanced sampler, unweighted CE, optimizer/LR/WD/batch/epochs/patience/seeds, grouped folds. Step 27's nonlinear head is explicitly not reused (it was worse). The only new mechanism is neighbor construction and a linear classifier over the concatenated embeddings.

Machine-readable outputs: `output/shape_classification/step28/results.json`, `c1_full.json`, `c2_full.json`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step28_experiments
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| Neighbor coverage: 5,066/7,177 primitives (70.6%) have both neighbors; 873 prev-only, 873 next-only, 365 neither (a genuine gap or recording/lane edge) | §3 |
| Leakage check: 11,878 neighbor pairs verified, **0** cross-recording violations — neighbor construction cannot leak across grouped-fold splits by construction | §7 |
| C0 reproduces Step 26 L0/A2 exactly: pooled macro F1 0.3668 | §5 |
| **C1 (±1 full context) clearly beats C0**: pooled 0.3668→0.3957 (+0.029), grouped mean 0.3500±0.0770→0.4151±0.0920 (+0.065) | §8 |
| **Sloped-start improves without sacrificing Cosine** — the specific failure mode of Steps 26-27: Sloped-start F1 0.112→0.162 (+45% relative), Cosine F1 0.745→**0.787** (both precision *and* recall up, not a tradeoff) | §8, §10 |
| Fixed also improves (0.376→0.417); Sloped-end dips slightly (0.233→0.217) — three of four classes better, one marginally worse | §8 |
| Fold consistency: 4/5 folds improve (median Δ+0.025); fold 3 shows a large +0.253 swing that amplifies but does not solely drive the aggregate — the other four folds alone still average +0.018 | §12 |
| Recording consistency: 9/17 recordings worsen (mildly, −0.02 to −0.12) vs. 8/17 improve (substantially, several +0.08 to **+0.34**) — right-skewed, not one lucky recording | §13 |
| **T2 recovery/breakage is dramatically healthier than Step 27's**: of 295 Cosine-mistaken-for-Sloped-start cases, C1 recovers 48 (16.3%, vs. L1's 3.4%); of 3,597 correct Cosines, C1 breaks 498 (13.8%, vs. L1's 28.4%) — an order-of-magnitude better trade ratio | §14 |
| C2 (neighbor-pitch-only) also beats C0 (0.3668→0.3705 pooled, 0.3500→0.3964 grouped mean) but clearly less than C1 — neighbor audio adds something pitch alone doesn't, especially for Cosine (C2's Cosine F1 actually *declines* slightly, 0.745→0.726) | §8 |
| Context-position ablation: both neighbors together beats either alone at test time, but forcibly zeroing one side of a jointly-trained model is an out-of-distribution intervention (like Steps 26-27's modality-zeroing) — not a fair standalone read of "prev-only" or "next-only" value | §15 |
| Descriptive-only transition matrix confirms real corpus structure exists (P(Cosine\|Cosine)=0.865, P(Cosine\|Sloped-start)=0.477) — context has something to work with | §16 |
| Oracle-neighbor-label ceiling (O-context, explicitly non-deployable): pooled 0.4449, grouped mean 0.4547 — a real ceiling above C1, but nowhere near oracle-pitch's 0.819; C1 already captures roughly a third of the pooled gap between C0 and this ceiling using only observable embeddings | §17 |
| Correctly-classified Sloped-start examples have ~51% larger mean previous-neighbor pitch displacement than Sloped-start examples C0 mistakes for Cosine (184.8¢ vs. 122.1¢, n=39 vs. 295) — small-n but directionally consistent with prev-context mattering more than next | §18 |

**Primary outcome: `NEIGHBOR_CONTEXT_ADDS_SIGNAL`**

**Decision gate: `MOVE_TO_SEQUENCE_MODEL`**

---

## 1-2. Anti-leakage rule and frozen center model

C1/C2 never see a neighbor's true canonical type — only its own CREPE contour and CQT audio patch, run through the same encoder used for the center trajectory. GT labels are the loss target for the center primitive only. The oracle-neighbor-label diagnostic (§17/O-context) is the one deliberate exception, run and reported separately, explicitly non-deployable. Boundaries remain GT throughout (segmentation is not being tested here).

## 3. Triplet construction

Ordering is by `start_s` within `(recording_id, lane_id)`, not by `primitive_id` or `seq` — `primitive_id` numbering is sequential only over *kept* primitives, so two numerically-consecutive primitives can still straddle a masked/skipped raw trajectory (a type-7/12/13 krintin/silent/slide segment that never became a primitive at all). The actual gap distribution across all 7,159 consecutive same-lane pairs is sharply bimodal — 82.8% are truly contiguous (<1ms, matching how composite trajectories are decomposed into back-to-back segments), then a long tail starting well past 20ms (p90 = 141ms, up to 1,505s). A 20ms adjacency threshold sits cleanly in the gap between these two populations and is used throughout.

| Coverage | Count | % |
|---|---:|---:|
| Both neighbors | 5,066 | 70.6% |
| Previous only | 873 | 12.2% |
| Next only | 873 | 12.2% |
| Neither | 365 | 5.1% |

Missing previous/next trajectories were never dropped — encoded via a zero-vector embedding (computed from a placeholder input, then multiplied by a 0/1 presence mask, so a missing neighbor contributes exactly zero gradient to the shared encoder) plus an explicit mask feature the linear head can condition on.

## 5-6. C0 architecture and reproduction

C0 = Step 26 L0/A2, reused unchanged (not retrained): `[h_pitch_i;h_audio_i] → Linear(32,4)`. Pooled macro F1 reproduces exactly at 0.3668.

## 6-7. C1/C2 architecture and grouped-fold safety

- **C1**: one shared encoder (`pitch_net` + `AcousticCNN`, architecturally identical to Step 26's, applied independently to previous/center/next) → `[e_prev; e_center; e_next; prev_mask; next_mask] → Linear(98,4)`. No hidden layer, no recurrence, no attention. 7,676 params (vs. C0's 7,412).
- **C2**: center keeps full `[h_pitch;h_audio]` (32-dim); neighbors contribute pitch-only (`h_pitch_prev`, `h_pitch_next`, 16-dim each) → `Linear(66,4)`. 7,548 params.

Leakage: neighbor construction only ever pairs primitives sharing a `recording_id` (grouped folds keep whole recordings/performance-groups on one side of a split), so cross-split leakage is structurally impossible — verified directly rather than assumed: 11,878 neighbor pairs checked, 0 cross-recording violations.

## 8. Primary result table

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std | Accuracy | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 local | 0.3668 | 0.376 | 0.745 | 0.112 | 0.233 | 0.3500±0.0770 | 0.596 | 7,412 |
| **C1 local + prev/next** | **0.3957** | 0.417 | **0.787** | **0.162** | 0.217 | **0.4151±0.0920** | 0.631 | 7,676 |
| C2 local + neighbor pitch only | 0.3705 | 0.386 | 0.726 | 0.180 | 0.191 | 0.3964±0.0593 | 0.564 | 7,548 |
| Oracle pitch reference (Step 25 F0-oracle) | 0.8187 | 0.646 | 0.813 | 0.899 | 0.916 | 0.8383±0.0383 | 0.779 | 2,836 |

C1 vs. C0 (the primary comparison): +0.0289 pooled, +0.0651 grouped mean — both clearly positive, and by a larger margin than any single fold-noise swing seen in Steps 26-27's null/negative results.

## 9. Reading the desired pattern

The hoped-for pattern (§9 of the spec) was: Cosine stable, Sloped-start up substantially, Sloped-end stable/up, Fixed stable, macro F1 up. What actually happened: Cosine **up** (not just stable — a genuine bonus), Sloped-start up substantially, Fixed up, Sloped-end down slightly (−0.016). Three of four classes improve; only Sloped-end gives back a small amount. This is the first step in the whole audio/fusion arc (Steps 26-27) where the target class improves *and* the majority class also improves, rather than one buying the other.

## 10. Per-class precision/recall

| Class | C0 P/R/F1 | C1 P/R/F1 | C2 P/R/F1 |
|---|---|---|---|
| Fixed | 0.366/0.394/0.376 | 0.415/0.419/0.417 | 0.379/0.394/0.386 |
| Cosine | 0.765/0.727/0.745 | **0.798/0.777/0.787** | 0.787/0.673/0.726 |
| Sloped-start | 0.173/0.083/0.112 | 0.133/0.205/0.162 | 0.160/0.205/0.180 |
| Sloped-end | 0.164/0.400/0.233 | 0.170/0.298/0.217 | 0.128/0.377/0.191 |

Cosine improves on *both* precision and recall under C1 — not a recall-for-precision trade. Sloped-start's F1 gain is the more familiar shape (recall up substantially, precision down somewhat) but nets positive, and unlike Steps 26-27 it does not come at Cosine's expense.

## 11. Confusion matrices

Rows = true, columns = predicted, order [Fixed, Cosine, Sloped-start, Sloped-end]:

**C0**
```
[ 462,  623,   33,  188]
[ 588, 3597,  118,  647]
[  49,  295,   39,   85]
[  51,  185,   36,  181]
```

**C1**
```
[ 449,  541,   95,  221]
[ 312, 3845,  433,  360]
[  35,  259,   96,   78]
[  49,  173,   96,  135]
```

The Cosine↔Sloped-start axis does not simply shrink: Sloped-start→Cosine falls (295→259) but Cosine→Sloped-start actually rises (118→433). Both classes' own diagonal improves simultaneously (Cosine 3,597→3,845; Sloped-start 39→96) — C1 is not cleanly separating the two so much as becoming generally more willing to predict both "moving-with-curvature" types instead of Fixed/Sloped-end, and coming out ahead on net because both diagonals grow. Reported directly rather than described as a clean resolution.

## 12. Fold consistency

| Fold | C0 | C1 | C1−C0 |
|---|---:|---:|---:|
| 0 | 0.3258 | 0.3504 | +0.0246 |
| 1 | 0.2783 | 0.2975 | +0.0192 |
| 2 | 0.3419 | 0.3966 | +0.0546 |
| 3 | 0.3060 | **0.5588** | **+0.2527** |
| 4 | 0.4980 | 0.4724 | −0.0257 |

4 improved / 1 worsened, median Δ+0.0246. Fold 3's swing is large and deserves scrutiny — excluding it, the remaining four folds still average +0.018, so the result is not solely a fold-3 artifact, but fold 3 is doing a lot of the grouped-mean's lift and should be read as "context helps, amplified unusually strongly in one fold" rather than "context helps uniformly by ~0.06."

## 13. Recording consistency

8/17 recordings improved (several substantially: +0.076 to **+0.338**), 9/17 worsened (all mildly: −0.019 to −0.119). Median Δ−0.015, mean Δ+0.047 — the median being negative while the mean is clearly positive reflects a right-skewed distribution (many small losses, several large wins), not a single outlier recording carrying the result: 8 different recordings show real gains.

## 14. T2 error recovery/breakage

| | Step 27 L1 vs L0 | **Step 28 C1 vs C0** |
|---|---:|---:|
| Set A (predicted Cosine, true Sloped-start) | 295 | 295 |
| Recovered | 10 (3.4%) | **48 (16.3%)** |
| Set B (correct Cosine) | 3,597 | 3,597 |
| Broken | 1,023 (28.4%) | **498 (13.8%)** |

An order-of-magnitude healthier trade: nearly 5x more recoveries, on roughly half the breakage rate.

## 15. Context-position ablation

| | both | prev only | next only |
|---|---:|---:|---:|
| C1 macro F1 | 0.3957 | 0.3302 | 0.3130 |
| C1 Sloped-start F1 | 0.162 | 0.127 | 0.117 |
| C2 macro F1 | 0.3705 | 0.3079 | 0.2991 |
| C2 Sloped-start F1 | 0.180 | 0.135 | 0.154 |

Both neighbors together clearly beats either alone. Note both single-sided numbers fall *below* C0 (0.367) — this is expected and not a fair read of "how much does prev/next alone help": zeroing one side of a model trained expecting both present is an out-of-distribution intervention at test time, the same caveat that applied to Steps 26-27's modality-zeroing checks. It shows the model uses both sides, not how a model trained with only one side would perform. Previous context consistently edges out next context in both conditions — consistent with §18's descriptive finding below.

## 16. Transition diagnostic (descriptive only, TRAIN labels, never a model input)

Fold 0 TRAIN, `P(type_i | type_{i-1})`:

| prev \\ next | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| Fixed | 0.336 | 0.403 | 0.119 | 0.142 |
| Cosine | 0.080 | **0.865** | 0.032 | 0.022 |
| Sloped-start | 0.300 | **0.477** | 0.109 | 0.114 |
| Sloped-end | 0.414 | 0.290 | 0.106 | 0.190 |

Cosine is highly self-persistent (86.5% of Cosines follow a Cosine), and Sloped-start is followed by Cosine nearly half the time — real, nonuniform structure exists for context to draw on, consistent with C1/C2 finding usable signal.

## 17. Optional oracle-neighbor-label ceiling (O-context) — not a deployable condition

Center local embedding + one-hot TRUE previous type + one-hot TRUE next type → `Linear(40,4)`:

| | Pooled macro F1 | Grouped mean |
|---|---:|---:|
| C0 | 0.3668 | 0.3500 |
| C1 (observable) | 0.3957 | 0.4151 |
| **O-context (oracle labels, not deployable)** | **0.4449** | **0.4547** |
| Oracle pitch (Step 25) | 0.8187 | 0.8383 |

Even perfect knowledge of neighboring types only reaches ≈0.44-0.45 — real headroom above C1, but nowhere close to the center's own oracle-pitch ceiling (0.82-0.84). C1, using only observable audio/pitch embeddings, already captures roughly a third of the pooled gap between C0 and this theoretical ceiling (0.029 of 0.078) — meaningful room for a better *observable* context model remains, but neighbor-type information alone, even perfectly known, would never fully close this task's gap.

## 18. Neighbor similarity analysis

Comparing C0's Sloped-start→Cosine errors (n=295) against C0's correctly-classified Sloped-start (n=39), mean absolute CREPE pitch displacement of the neighbor:

| | Prev displacement | Next displacement |
|---|---:|---:|
| Errors (true SlS, predicted Cosine) | 122.1¢ | 132.9¢ |
| Correct Sloped-start | **184.8¢** | 122.6¢ |

Correctly-classified Sloped-start examples have a ~51% larger previous-neighbor pitch swing than the ones C0 confuses with Cosine; next-neighbor displacement is similar between the two groups. Small-n (39 correct examples) — directional, not conclusive — but consistent with §15's finding that previous context carries more weight than next context for this specific class.

## Primary outcome: `NEIGHBOR_CONTEXT_ADDS_SIGNAL`

C1 clearly improves on C0 (pooled +0.029, grouped mean +0.065), with reasonable — not perfect — fold consistency (4/5 folds positive) and a right-skewed but net-positive recording consistency (8/17 recordings gain substantially). Most importantly, Sloped-start improves without sacrificing Cosine, which both improve together for the first time in this arc — the specific pattern Steps 26-27 were never able to produce. The T2 recovery/breakage ratio is an order of magnitude healthier than Step 27's nonlinear-fusion result. This is not a flawless result (Sloped-end dips slightly, fold 3 and several individual recordings are doing outsized work, and the Cosine↔Sloped-start confusion axis doesn't cleanly shrink so much as both classes' correct-prediction counts rise together) — but it clears the bar the spec set for this outcome category.

## Decision gate: `MOVE_TO_SEQUENCE_MODEL`

Immediate observable context clearly helps. A future step should test a small sequence model over several trajectory embeddings (not necessarily much larger than ±1, but structured to actually model sequence rather than a single concatenated linear read), informed by two things this step establishes: (1) the oracle-neighbor-label ceiling (≈0.45) shows there's headroom above C1's ≈0.40 worth pursuing with a better *observable* context model, and (2) previous-context consistently outweighs next-context for Sloped-start specifically, which a sequence model (rather than a symmetric concatenation) could exploit more naturally than C1's fixed-position linear head.

## Recommendation for Step 29

Test a small sequence model (e.g., a short GRU/TCN over a window of trajectory embeddings, or a slightly wider ±2 window with the same linear-probe discipline before escalating architecture) — motivated directly by this step's own diagnostics: the position-ablation and neighbor-similarity results both suggest asymmetric (recency-weighted) context matters more than symmetric concatenation, which a fixed 3-slot linear layer cannot express but a sequence model naturally can. Keep the same discipline this arc has maintained throughout: one clear architectural change at a time, fold/recording consistency checked before any result is trusted, and GT neighbor labels kept out of any deployable condition.
