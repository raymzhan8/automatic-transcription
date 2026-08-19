# Step 27 — Can Nonlinear Audio–Pitch Interaction Recover Sloped-Start Without Sacrificing Cosine?

Step 26 established that audio adds real, heavily-used information beyond CREPE's pitch contour (linear fusion, L0/A2, clearly beats CREPE-only: macro F1 0.3668 vs. 0.3107), but the gain is substantially a majority-class decision-boundary shift: Cosine recall rose 33.1%→72.7% while Sloped-start recall collapsed 49.6%→8.3%. The unresolved question was never "does audio help" — it does. It was whether that specific tradeoff is an artifact of the fusion head being purely linear, which can only add independent weighted contributions from the two embeddings and cannot express interactions like "this acoustic pattern implies Cosine only when the pitch contour simultaneously looks like this." Step 27 tests exactly one richer mechanism — one small nonlinear hidden layer — nothing else.

Frozen exactly from Step 26: CREPE extraction, GT boundaries, CQT parameters and feature cache, TRAIN-only per-bin audio normalization, `N=64`, pitch normalization, `ContourCNN`/`AcousticCNN` architectures, grouped folds, class-balanced sampler, unweighted CE, optimizer/LR/WD/batch/epochs/patience/seeds, checkpoint criterion. Encoders remain trainable end-to-end in both conditions (Step 26's own protocol, not frozen for this comparison). The only architectural variable is the fusion head.

Machine-readable outputs: `output/shape_classification/step27/results.json`, `output/shape_classification/step27/l1_full.json`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step27_experiments   # trains L1, then the full comparison (A0/L0 reused from Step 26)
```

---

## Executive summary

| Finding | Evidence |
|---|---|
| L0 reproduces Step 26 A2 exactly: macro F1 0.3668, grouped mean 0.3500±0.0770 (reused unchanged, not retrained) | §2 |
| L1 adds 464 params (7,412→7,876) for a single `Linear(32,16)→ReLU→Linear(16,4)` head | §5 |
| **L1 is worse than L0 on every single class**: Fixed 0.376→0.366, Cosine 0.745→0.633, Sloped-start 0.112→0.093, Sloped-end 0.233→0.173; pooled macro F1 0.3668→0.3163 (−0.0505) | §7 |
| **Cosine's gain was not preserved**: Cosine recall fell 0.727→0.547 (−0.180) under L1 — the nonlinear head lost ground on the class L0 had actually fixed, while Sloped-start recall stayed pinned at ~0.07-0.08 | §12 |
| Fold consistency: 2/5 improve, 3/5 worsen, median Δ≈0 (essentially flat-to-negative, not one bad fold driving it) | §9 |
| Recording consistency: 8/17 improve, 9/17 worsen, mean Δ−0.039 | §10 |
| **The mechanistic diagnostic explains why**: standardized mean difference between true-Cosine and true-Sloped-start hidden activations `z` — 0 of 16 dimensions reach even a medium effect size (\|d\|>0.5), mean\|d\|=0.199. The nonlinear layer never learned to represent these two classes distinctly | §16 |
| **Recovery/breakage is decisive**: of 295 cases L0 wrongly called Cosine (true Sloped-start), L1 recovers only 10 (3.4%). Of 3,597 cases L0 got right as Cosine, L1 breaks 1,023 (28.4%) | §17 |
| Fold 0 (Step 26's largest regression, A0 0.473→L0 0.326) is **not fixed** — L1 scores 0.324, and its Sloped-start F1 drops to exactly **0.000** (from L0's already-poor 0.135) | §19 |
| One real but minor exception: L1 beats L0 on Sloped-start F1 for trajectories >500ms (0.301→0.367 at 500ms-1s, 0.087→0.238 at >1s) — small samples (364, 235), not enough to offset the broad-based regression elsewhere | §18 |

**Primary outcome: `NONLINEAR_FUSION_HURTS`**

**Decision gate: `INVESTIGATE_SEQUENCE_CONTEXT`**

---

## 1-4. Frozen upstream, L0/L1 architecture, trainable encoders

Pitch branch: frozen CREPE → GT boundaries → `q(x), dq/dx` → `ContourCNN` → `h_pitch ∈ R^16`. Audio branch: same oracle-boundary CQT patch → same TRAIN-only per-bin normalization → `AcousticCNN` → `h_audio ∈ R^16`. Both trainable end-to-end in both conditions (Step 26's protocol, not altered for this comparison — section 4's explicit instruction).

- **L0** (`training/shape_classification/step26_model.py::FusionModel`, reused unchanged): `[h_pitch;h_audio] → Linear(32,4)`.
- **L1** (`training/shape_classification/step27_model.py::NonlinearFusionModel`, new): `[h_pitch;h_audio] → Linear(32,16) → ReLU → Linear(16,4)`. `hidden=16` fixed before any result was examined (section 5); no dropout/batchnorm/layernorm/residual/attention/gating/bilinear/modality-specific projection/second hidden layer.

## 2. L0 reproduction

`L0 pooled macro_f1=0.3668` (reference 0.3668), `grouped mean=0.3500±0.0770` (reference 0.3500±0.0770) — exact match, since L0 is Step 26's cached A2 result reused unchanged, not retrained. Interpreting L1 against a byte-identical baseline.

## 5. Parameter counts

| Model | Params |
|---|---:|
| L0 | 7,412 |
| L1 | 7,876 |
| Δ | +464 |

## 7. Primary result table

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 CREPE only | 0.3107 | 0.379 | 0.459 | 0.202 | 0.203 | 0.3234±0.0893 | 0.363 |
| L0 linear fusion | 0.3668 | 0.376 | 0.745 | 0.112 | 0.233 | 0.3500±0.0770 | 0.596 |
| **L1 nonlinear fusion** | **0.3163** | 0.366 | 0.633 | 0.093 | 0.173 | **0.3396±0.0932** | 0.494 |
| Oracle pitch reference (Step 25 F0-oracle) | 0.8187 | 0.646 | 0.813 | 0.899 | 0.916 | 0.8383±0.0383 | 0.779 |

L1 vs. L0 (the primary comparison): **every single class is worse**, not a tradeoff where one class gains at another's expense — a broad regression.

## 8. On macro F1 weighting

Macro F1 gives all four classes equal weight regardless of prevalence; the comparisons above and in §12 are read directly per-class, not attributed to Cosine's majority share.

## 9-10. Fold and recording consistency

| Fold | L0 | L1 | L1−L0 |
|---|---:|---:|---:|
| 0 | 0.3258 | 0.3239 | −0.0020 |
| 1 | 0.2783 | 0.3234 | +0.0451 |
| 2 | 0.3419 | 0.2516 | **−0.0903** |
| 3 | 0.3060 | 0.2813 | −0.0247 |
| 4 | 0.4980 | 0.5178 | +0.0198 |

2 improved / 3 worsened, median Δ≈−0.002, mean Δ (=grouped-mean delta) −0.010. Not one fold driving a false aggregate — the regression (fold 2) and the small gains (folds 1, 4) are each real and roughly offsetting, landing on a flat-to-slightly-negative overall picture. Recording consistency: 8/17 improved, 9/17 worsened, median Δ−0.013, mean Δ−0.039 — no small subset of recordings is propping up an otherwise-negative result; it's negative throughout.

## 11-12. Per-class metrics and the central Cosine↔Sloped-start table

| Metric | L0 | L1 | Δ |
|---|---:|---:|---:|
| Cosine precision | 0.765 | 0.752 | −0.014 |
| Cosine recall | 0.727 | **0.547** | **−0.180** |
| Cosine F1 | 0.745 | 0.633 | −0.112 |
| Sloped-start precision | 0.173 | 0.136 | −0.036 |
| Sloped-start recall | 0.083 | 0.071 | −0.013 |
| Sloped-start F1 | 0.112 | 0.093 | −0.019 |

This directly falsifies the primary hypothesis (section 6): the ideal pattern was Cosine stable/Sloped-start up. What actually happened is Cosine down substantially and Sloped-start also (marginally) down — the worst of both, not a resolved tradeoff and not even a re-pointed one.

## 13. Confusion matrices

Rows = true, columns = predicted, order [Fixed, Cosine, Sloped-start, Sloped-end]:

**L0**
```
[ 462,  623,   33,  188]   Fixed
[ 588, 3597,  118,  647]   Cosine
[  49,  295,   39,   85]   Sloped-start
[  51,  185,   36,  181]   Sloped-end
```

**L1**
```
[ 689,  504,   22,   91]   Fixed
[1460, 2707,  169,  614]   Cosine
[ 130,  249,   33,   56]   Sloped-start
[ 178,  142,   18,  115]   Sloped-end
```

L1 does not reduce the Cosine↔Sloped-start confusion axis in either direction — Sloped-start→Cosine actually falls slightly (295→249) but Cosine→Sloped-start rises (118→169), and the dominant new effect is Cosine mass moving to **Fixed** instead (588→1,460) — a different, new confusion the nonlinear head introduced rather than a cleaner resolution of the original one.

## 14. Prediction frequencies

| | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| True | 18.2% | 69.0% | 6.5% | 6.3% |
| L0 | 16.0% | 65.5% | 3.1% | 15.3% |
| L1 | 34.2% | 50.2% | 3.4% | 12.2% |

L1's predicted-Sloped-start rate is essentially unchanged from L0 (3.1%→3.4%, both far below the true 6.5%) — the model still almost never predicts this class. What moved is Fixed, nearly doubling in predicted share (16.0%→34.2%) at Cosine's expense — a new operating-point shift unrelated to the hypothesis being tested.

## 15. Modality-zeroing sanity check

Same trained L1 weights, no retraining: normal macro F1 0.3163, audio-zeroed 0.1556, pitch-zeroed 0.3071. Both branches still contribute (audio strongly, pitch mildly) — the same qualitative pattern as Step 26's L0, so L1's failure isn't explained by the model collapsing onto one modality.

## 16. Interaction diagnostic — the mechanistic explanation

Standardized mean difference (Cohen's-d-style) between the hidden fusion activation `z ∈ R^16` for held-out true-Cosine vs. true-Sloped-start test examples, pooled across folds (4,950 Cosine / 468 Sloped-start examples):

- Mean |standardized difference| across all 16 dimensions: **0.199**
- Dimensions with |diff| > 0.5 (medium effect): **0 / 16**
- Dimensions with |diff| > 1.0 (large effect): **0 / 16**

This is the most direct evidence available for *why* L1 doesn't help: the nonlinear layer specifically built to enable Cosine/Sloped-start interactions never learned an internal representation that separates them. Not "separates them a little" — no dimension clears even a conventionally modest effect size. Whatever information would distinguish these two classes, this architecture and training signal did not induce the hidden layer to encode it.

## 17. Difficult-example recovery/breakage

- **Set A** (L0 wrongly predicts Cosine, true Sloped-start): 295 cases. L1 recovers **10** (3.4%).
- **Set B** (L0 correctly predicts Cosine): 3,597 cases. L1 breaks **1,023** (28.4%).

This is the starkest single number in this step: L1 fixes almost none of the errors it was designed to fix, while breaking more than a quarter of what was already working. Deterministic highest-L1-confidence examples in both directions are in `results.json` under `recovery_breakage.top_recovered`/`top_broken`.

## 18. Duration analysis

| Bucket | L0 Sloped-start F1 | L1 Sloped-start F1 |
|---|---:|---:|
| <100ms (n=2,285) | 0.000 | 0.000 |
| 100-250ms (n=3,273) | 0.049 | 0.039 |
| 250-500ms (n=1,020) | 0.172 | 0.088 |
| 500ms-1s (n=364) | 0.301 | **0.366** |
| >1s (n=235) | 0.087 | **0.238** |

The one genuine bright spot: L1 clearly beats L0 on the two longest, smallest-n buckets. Reported honestly rather than ignored — but at n=364/235 (a small fraction of the corpus) this doesn't offset the broad regression across the other 92% of examples, and the direction is notably opposite Step 26's own finding that audio helps *short* trajectories most. Diagnostic only; no duration-specific model was built.

## 19. Fold 0 — the largest known regression

| Condition | Macro F1 | Fixed F1 | Cosine F1 | Sloped-start F1 | Sloped-end F1 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.4730 | 0.569 | 0.574 | 0.440 | 0.310 |
| L0 | 0.3258 | 0.238 | 0.780 | 0.135 | 0.151 |
| L1 | 0.3239 | 0.508 | 0.676 | **0.000** | 0.111 |

Nonlinear fusion does not fix Step 26's worst-case fold — overall macro F1 is essentially unchanged (0.3258→0.3239), and Sloped-start specifically goes from already-poor (0.135) to exactly zero.

## 24. Primary scientific question, answered directly

**No.** One small nonlinear audio-pitch interaction layer does not recover Sloped-start, and it does not preserve Cosine's improvement either — it makes both worse, along with Fixed and Sloped-end. The interaction diagnostic (§16) shows this isn't a near-miss: the hidden layer shows no meaningful separation between the two classes at all, on any of its 16 dimensions.

## 25. Outcome: `NONLINEAR_FUSION_HURTS`

L1 is clearly worse than L0 overall (macro F1 −0.0505, every per-class F1 down) and no more stable (fold consistency 2 improved/3 worsened; recording consistency 8/17 improved, mean Δ negative). Not a borderline call.

## 26. Decision gate: `INVESTIGATE_SEQUENCE_CONTEXT`

We have now tested, under oracle boundaries, in this exact order: pitch-only (A0), audio-only (Step 26 A1), linear local multimodal fusion (L0), and nonlinear local multimodal fusion (L1). All four isolate a single trajectory's own local evidence window. T2 remains difficult under every one of them. Per Step 27's own stopping rule, this is the final local-fusion experiment — the next source of information to test is neighboring-trajectory or longer-sequence context, not a deeper architecture over the same local evidence.

## Recommendation for Step 28

Do not propose a deeper MLP, gated fusion, attention, calibration, or larger encoders. Move to sequence/neighboring-trajectory context: does information from adjacent trajectories (previous/next type, transition structure, or a short sequence model spanning several trajectories) help distinguish Sloped-start from Cosine in a way no single trajectory's own local audio+pitch evidence has been able to, across five independent attempts now (this step, Step 26's A0/A1/A2, and the two Step 20 Phase B frontends before that).
