# Step 29 — Can an Order-Aware Trajectory Sequence Model Use Context Better Than Linear Neighbor Features?

## 0. Pre-run gate

Step 28's outcome (`docs/step_28_neighbor_context.md`, `output/shape_classification/step28/results.json`): `NEIGHBOR_CONTEXT_ADDS_SIGNAL`. C1 (linear ±1 context) clearly beat C0 (local-only) — pooled macro F1 0.3668→0.3957, grouped mean 0.3500±0.0770→0.4151±0.0920, 4/5 folds improved. This is **Situation A** (observable context helped) — proceeding directly to Step 29 as specified, no stop condition triggered.

## 1. Frozen trajectory representation

`e_i = [h_pitch_i ; h_audio_i] ∈ R^32`, from the identical `ContourCNN`/`AcousticCNN` pair used throughout Steps 26-28, frozen CREPE source, oracle boundaries. No new trajectory features (no template residuals, duration, span, CREPE confidence, derivatives, or transition probabilities) beyond the presence mask Step 28 already required.

## 2. Why this differs from the earlier framewise BiGRU

Step 9's BiGRU operated on 10ms acoustic frames within a single trajectory, predicting framewise labels — it tested whether recurrence over raw acoustic time helps *local* pitch/audio modeling, and performed poorly there. Step 29's BiGRU operates on a sequence of length 3 of already-compressed, whole-trajectory embeddings (`e_{i-1}, e_i, e_{i+1}`) — a completely different timescale (symbolic/trajectory-level, not 10ms-frame-level) and a completely different question (does *order* among neighboring trajectories carry information a flat concatenation loses). The earlier result is not evidence against this one.

## 3-4. Sequence construction, window

Reuses Step 28's exact triplet construction (`step28_neighbors.build_neighbor_map`, `step28_train.build_triplet_arrays`) unchanged — oracle primitive ordering, same 20ms adjacency-gap threshold, never crossing recording/lane boundaries or invalid-target gaps. Fixed ±1 window only (no ±2, no window sweep) — isolating linear-vs-sequence treatment of the *same* information, per the step's own design.

## 5. Missing-neighbor handling

Each sequence token is `x_j = [e_j ; present_j] ∈ R^33`. Missing neighbors get a zero embedding (computed from a placeholder, then multiplied by `present_j=0`, so zero gradient reaches the encoder) with an explicit presence bit — identical mechanism to Step 28, reused rather than reimplemented.

## 6-10. Conditions and architecture

- **S0** = Step 26 L0 / Step 28 C0, reused unchanged (not retrained). Pooled macro F1 0.3668 (reproduces exactly).
- **S1** = Step 28 C1 (linear ±1 context), reused unchanged — the actual cached result, not whichever Step 28 condition scored best.
- **S2-context**: one shared encoder (identical `SharedEncoder`, imported directly from `step28_model.py` rather than re-duplicated, so architecture is guaranteed identical) applied to all three positions → tokens `[e_j;present_j]` → `BiGRU(input_size=33, hidden_size=16, num_layers=1, bidirectional=True)` → output at the **center** position only (`h_center ∈ R^32`) → `Linear(32,4)`. No second layer, dropout, attention, or MLP head.
- **S2-center-only**: identical architecture and parameter count, but both neighbor slots are forced to the zero/absent token regardless of real data availability — the capacity-matched blind control (section 8).
- Encoders trainable end-to-end in both S2 conditions, matching Step 28 C1 (never frozen, per section 10's explicit instruction not to diverge on this).

| Condition | Params |
|---|---:|
| S0 | 7,412 |
| S1 | 7,676 |
| S2-center-only | 12,308 |
| S2-context | 12,308 |

## 11. The two comparisons, answered directly

**Comparison A (S2-context vs. S1)**: ambiguous. Pooled macro F1 favors S1 (0.3957 vs. 0.3706, S1 higher by 0.025); grouped mean favors S2-context (0.4151 vs. 0.4307, S2 higher by 0.016); fold-level median delta is essentially zero (+0.0008, 3 folds improved/2 worsened); recording-level median delta is small but positive (+0.003, 11/17 improved). No clean signal in either direction — reported as a wash, not resolved in S2's favor.

**Comparison B (S2-context vs. S2-center-only)**: real and reasonably consistent. Grouped mean +0.086 (0.3444→0.4307), 4/5 folds improve (median Δ+0.028), 12/17 recordings improve (median Δ+0.039). Real neighbor content clearly does something beyond adding GRU parameters — this rules out "pure capacity" as the explanation for whatever S2-context achieves.

Put together: **the sequence model demonstrably uses real neighbor information (Comparison B), but that does not translate into a clear, robust advantage over the much simpler linear baseline (Comparison A).** The BiGRU isn't capacity-inflated noise, it's just not obviously better at this task than concatenation-plus-linear.

## 12. Primary result table

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std | Accuracy | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 local | 0.3668 | 0.376 | 0.745 | 0.112 | 0.233 | 0.3500±0.0770 | 0.596 | 7,412 |
| S1 linear ±1 context | **0.3957** | 0.417 | **0.787** | 0.162 | **0.217** | 0.4151±0.0920 | 0.631 | 7,676 |
| S2-center-only BiGRU | 0.3751 | 0.342 | 0.784 | 0.089 | 0.285 | 0.3444±0.0694 | 0.638 | 12,308 |
| S2-context BiGRU | 0.3706 | 0.376 | 0.764 | **0.173** | 0.169 | **0.4307±0.0957** | 0.617 | 12,308 |
| Oracle pitch reference (Step 25) | 0.8187 | — | — | 0.899 | 0.916 | 0.8383±0.0383 | 0.779 | 2,836 |
| O-context ceiling (Step 28, not deployable) | 0.4449 | — | — | — | — | 0.4547 | — | — |

## 13. Reading the pattern honestly

The hoped-for pattern was Sloped-start up, Cosine stable/up, macro F1 up. What happened: Sloped-start up marginally (0.162→0.173), **via precision up but recall down** (0.205→0.162) — not the "precision and recall improving together" pattern section 14 was looking for. Cosine down modestly (0.787→0.764). Macro F1 direction depends on which metric (pooled down, grouped mean up). This is a small, mixed result, not a resolution of the Cosine↔Sloped-start tension.

## 14-15. Per-class and Cosine↔T2 table

| Metric | S1 | S2-context | Δ |
|---|---:|---:|---:|
| Cosine precision | 0.798 | 0.774 | −0.025 |
| Cosine recall | 0.777 | 0.756 | −0.021 |
| Cosine F1 | 0.787 | 0.764 | −0.023 |
| Sloped-start precision | 0.133 | 0.184 | +0.051 |
| Sloped-start recall | 0.205 | 0.162 | −0.043 |
| Sloped-start F1 | 0.162 | 0.173 | +0.011 |

## 16. Confusion matrices and T2 recovery/breakage

Rows = true, columns = predicted, order [Fixed, Cosine, Sloped-start, Sloped-end]:

**S1**
```
[ 449,  541,   95,  221]
[ 312, 3845,  433,  360]
[  35,  259,   96,   78]
[  49,  173,   96,  135]
```

**S2-context**
```
[ 540,  607,   71,   88]
[ 845, 3740,  190,  175]
[  67,  295,   76,   30]
[ 115,  193,   76,   69]
```

Of 259 cases S1 mistook as Cosine (true Sloped-start), S2-context recovers **13 (5.0%)**. Of 3,845 correct S1 Cosines, S2-context breaks **523 (13.6%)**. For comparison, Step 28's own C1-vs-C0 recovery rate was 16.3% (recovering 48/295) at a similar breakage rate (13.8%) — S2-context's recovery rate is worse than S1's own improvement over S0 was, on a comparable breakage cost. This is a weaker trade than the step immediately before it, not a stronger one.

## 17. Fold consistency

| Fold | S0 | S1 | S2-center | S2-context | S2x−S1 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.3258 | 0.3504 | 0.4622 | 0.4901 | +0.1397 |
| 1 | 0.2783 | 0.2975 | 0.2725 | 0.2982 | +0.0008 |
| 2 | 0.3419 | 0.3966 | 0.3785 | 0.3363 | −0.0603 |
| 3 | 0.3060 | 0.5588 | 0.2863 | 0.4850 | −0.0738 |
| 4 | 0.4980 | 0.4724 | 0.3223 | 0.5441 | +0.0717 |

3 improved / 2 worsened vs. S1, median Δ≈0. Notably, fold 3 — the fold that drove much of S1's own large advantage over S0 (Step 28's own flagged outlier) — is *worse* under S2-context than S1. The sequence model does not reproduce S1's biggest single-fold win.

## 18. Recording consistency

vs. S1: 11/17 improved, 6/17 worsened, median Δ+0.003, mean Δ+0.026. vs. S2-center-only: 12/17 improved, 5/17 worsened, median Δ+0.039, mean Δ+0.070. Consistent with the fold-level read: real vs. capacity-matched-blind is a clearer win than real-context vs. linear-baseline.

## 19. Temporal-order swap diagnostic

Same trained S2-context weights, previous/next swapped at test time, no retraining:

| | Normal | Swapped |
|---|---:|---:|
| Macro F1 | 0.3706 | 0.3504 |
| Sloped-start F1 | 0.173 | **0.102** |
| Sloped-end F1 | 0.169 | 0.186 |

A real, meaningful drop overall (−0.021) and specifically for Sloped-start (−0.071, a large relative fall) — evidence the model is using the previous/next distinction as an ordered relationship, not just "extra content regardless of position," consistent with Step 28's own finding that previous-context matters more than next-context for this class. Sloped-end moves slightly the other way (+0.017) — small and not over-interpreted here (T2/T3's defining geometry remains within-trajectory; this diagnostic only asks whether surrounding motion affects how easy that geometry is to read, and the T2 result says yes, directionally).

## 20. T2 vs. T3

| | S1 | S2-context |
|---|---:|---:|
| Sloped-start F1 | 0.162 | 0.173 |
| Sloped-end F1 | 0.217 | 0.169 |

Sloped-start marginally better under S2, Sloped-end worse — no clean, uniform "context helps both boundary-adjacent classes" story.

## 21. Relation to Step 28's oracle-neighbor ceiling

O-context (true neighbor *labels*, not deployable): pooled 0.4449, grouped mean 0.4547. Both S1 (0.3957/0.4151) and S2-context (0.3706/0.4307) sit meaningfully below this. The gap is real, but S2-context does not close it any more convincingly than S1 does — if anything S1 is closer on the pooled metric. The ceiling shows there's headroom in principle; it does not show that *this* sequence model is the way to reach it.

## 22. Leakage audit

S2's only inputs are the neighbor's own CREPE-derived and audio-derived embeddings plus the presence mask — never a GT class, never the O-context diagnostic's one-hot labels (which remain a separate, explicitly non-deployable condition reused from Step 28, not retrained here). Neighbor construction is recording-local by the same construction verified in Step 28; no additional leakage surface is introduced by the sequence framing.

## Primary outcome: `SEQUENCE_MODEL_PARTIALLY_HELPS`

Not `SEQUENCE_MODEL_USES_CONTEXT_EFFECTIVELY` — that requires S2 to *clearly* beat both S1 and S2-center-only with reasonably consistent, tradeoff-free gains; Comparison A doesn't clear that bar (pooled favors S1, T2's gain is precision-for-recall not both-improving, Cosine gives back ground). Not `SEQUENCE_CAPACITY_NOT_CONTEXT` either — Comparison B is real and fairly consistent, so the mechanism is doing more than adding parameters. What actually happened sits between: S2 improves some things (grouped mean, recording-consistency count, Sloped-start F1 marginally, and demonstrably uses real order per the swap diagnostic) while giving back others (pooled macro F1, Cosine, T2 recall specifically, and a worse recovery/breakage trade than S1's own improvement over S0) — small, inconsistent gains with a real if modest tradeoff remaining, exactly `SEQUENCE_MODEL_PARTIALLY_HELPS`'s definition.

## Decision gate: `FREEZE_LINEAR_CONTEXT_TYPER`

Step 28's linear context clearly helped; this step's sequence model, at nearly double S1's parameter count, does not clearly improve on it. Per this project's standing discipline (Steps 18, 25, 27 all declined to bank an unclear or inconsistent win), added architectural sophistication is not adopted without a robust payoff. Freeze S1 (Step 28's linear ±1 context) as the best oracle-boundary trajectory typer produced so far; do not adopt the BiGRU.

## Recommendation for Step 30

Two threads, not one: (1) the O-context ceiling (≈0.45) sits meaningfully above even S1, and both the swap diagnostic and Step 28's position-ablation agree that *order* and *direction* carry real information (previous context specifically for Sloped-start) — a future step could test whether a longer or differently-structured context window closes more of that gap, but only after a clean argument for why ±1 with a different treatment would do better than ±1 with a BiGRU just did. (2) Equally justified: this result, combined with Step 27's nonlinear-fusion failure and now this step's sequence-model non-win, suggests the oracle-boundary trajectory-typing branch may be approaching diminishing returns from architecture changes generally — the next highest-leverage question may not be modeling at all, but whether boundary/segmentation errors in a real (non-oracle) pipeline dominate practical performance regardless of which typer sits on top. Recommend scoping Step 30 explicitly before committing further oracle-boundary architecture effort.
