# Step 8 — Experiment B1: Joint Pitch Supervision

Same TCN encoder as Experiment B0, plus a linear pitch head trained on fold-standardized tonic-relative cents. **No BiGRU, phase, class weighting, or segmentation.**

Reproduce:

```bash
python training/train_framewise.py \
  --config training/configs/b1_tcn_type_pitch.json --tiny-overfit 32 --max-epochs 100 --fold 0

python training/train_framewise.py \
  --config training/configs/b1_tcn_type_pitch.json --all-folds

for i in 0 1 2 3 4; do
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b1_tcn_type_pitch/fold_$i/best.pt
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b0_tcn_type_only/fold_$i/best.pt
done

python training/compare_b0_b1.py
```

Paired numbers: [`output/framewise_runs/b1_tcn_type_pitch/b0_vs_b1.json`](../output/framewise_runs/b1_tcn_type_pitch/b0_vs_b1.json).

---

## 1. Architecture and parameter count

```text
CQT [B,1,360,400]
  → FrequencyCNN
  → TCN dilations [1,2,4,8]
  → h_t [B,T,128]
       ├─ Linear(128→4)  type_logits
       └─ Linear(128→1)  pitch_pred_standardized
```

Verified `count_params(FramewiseTCNModel(predict_pitch=True)) == **434,629**` (B0 was 434,500). Training refuses to start if the count differs.

Encoder input is still **normalized CQT only**. Tonic is used only to convert `pitch_log2_hz` → cents and to invert standardization at eval.

---

## 2. Sanity tests

| Gate | Result |
|------|--------|
| Param count 434,629 | Pass |
| Forward shapes `[B,T,4]` and `[B,T]` | Pass |
| `denorm(std(cents))` identity | Pass (atol 1e-4) |
| Pitch stats unit `cents`; train IDs only | Pass |
| Tiny overfit (32 cached excerpts, 100 epochs) | **Pass** — train macro F1 **0.980**, train pitch MAE **97 cents** (from 347), `L_type` 0.034, `L_pitch` 0.026 |

Pitch is learnable on a frozen excerpt set. Type does not collapse under `λ=1`.

---

## 3. Fold-specific pitch normalization

Recomputed from train `valid_target` frames as tonic-relative cents (`1200 × (log2 f − log2 tonic)`). Step 7 `pitch_stats.npz` files were in **log2_hz** and were overwritten.

| Fold | μ_train (cents) | σ_train (cents) |
|------|----------------:|----------------:|
| 0 | 273.7 | 545.7 |
| 1 | 258.7 | 501.7 |
| 2 | 311.4 | 551.7 |
| 3 | 265.2 | 535.8 |
| 4 | 240.1 | 528.4 |

Same μ/σ applied to val and test. Val/test recordings never enter the stats.

---

## 4. Training configuration (identical to B0 except pitch)

| Setting | Value |
|---------|-------|
| Sampler | Valid-anchor 4 s, 512 excerpts/epoch, batch 8 |
| Optimizer | AdamW 1e-3, wd 1e-4, grad clip 1.0 |
| Seed | 42 |
| Early stop | val **macro F1**, patience 10, max 50 |
| Type loss | unweighted CE, `ignore_index=-1` |
| Pitch loss | Smooth L1 on standardized cents |
| `λ_type`, `λ_pitch` | **1.0, 1.0** (not tuned) |

Documented diffs vs B0: pitch head, cents target, joint loss, extra logs. No hyperparameter search.

Sampler audits match B0 style (`excerpts_per_recording`, class counts, T1 provenance, `valid_frame_fraction`).

---

## 5. Type / pitch loss curves

Initial train losses (epoch 1) sit in the same order of magnitude:

| Fold | L_type | L_pitch |
|------|-------:|--------:|
| 0 | 1.09 | 0.48 |
| 1 | 1.13 | 0.40 |
| 2 | 1.16 | 0.34 |
| 3 | 1.07 | 0.41 |
| 4 | 1.12 | 0.41 |

Pitch loss falls through training (typical late-train `L_pitch` ≈ 0.07–0.16) while type loss also falls. Pitch does **not** drown type optimization. Val pitch MAE on the 1-group val split stays large (~350–1500 cents); that split is too small to read pitch generalization from val MAE.

CSV logs: `output/framewise_runs/b1_tcn_type_pitch/fold_<i>/train_log.csv`.

### Validation macro F1 (early-stop metric)

| Fold | B0 val F1 | B1 val F1 |
|------|----------:|----------:|
| 0 | 0.265 | 0.253 |
| 1 | 0.332 | 0.298 |
| 2 | 0.370 | 0.352 |
| 3 | 0.192 | 0.249 |
| 4 | 0.344 | 0.290 |
| **mean** | **0.301** | **0.288** |

B1 val type F1 is slightly lower on 4/5 folds. Training still overfits type (train F1 → 0.5–0.9) as in B0.

---

## 6. Gradient diagnostics

Once per epoch, two backward passes on shared CNN+TCN parameters (no GradNorm/PCGrad).

Epoch-1 cosine similarity of `g_type` vs `g_pitch` is near **0** (−0.15 to +0.08): the two tasks are not strongly aligned or anti-aligned at the start. `||g_type||` and `||g_pitch||` are comparable (often 0.4–0.8). `λ=1` is a reasonable first scale.

---

## 7. Pitch metrics per held-out fold

Trivial baseline: always predict training-fold **mean cents**.

| Fold | MAE | Median AE | %±10 | %±25 | %±50 | Mean-pitch baseline MAE |
|------|----:|----------:|-----:|-----:|-----:|------------------------:|
| 0 | 396 | 292 | 1.7 | 4.4 | 9.7 | **260** |
| 1 | 483 | 429 | 1.4 | 3.4 | 6.8 | **478** |
| 2 | 473 | 443 | 1.0 | 2.6 | 5.2 | **423** |
| 3 | 1163 | 1190 | 0.3 | 0.7 | 1.2 | **1084** |
| 4 | **432** | 371 | 1.8 | 4.3 | 8.0 | 573 |

Held-out pitch is weak: on 4/5 folds the model **does not beat** predicting the training mean. Fold 4 is the exception (432 vs 573). Fold 3 is an octave-scale failure, consistent with a small/shifted test set.

Full-recording predicted contours: `fold_<i>/eval/pitch_contours/*.npz` (17 files). No decoder.

---

## 8. Pitch by trajectory type (MAE cents)

| Fold | T0 | T1 | T2 | T3 |
|------|---:|---:|---:|---:|
| 0 | 341 | 438 | 522 | 332 |
| 1 | 490 | 470 | 520 | 472 |
| 2 | 457 | 483 | 428 | 485 |
| 3 | 1198 | 1132 | 1107 | 1193 |
| 4 | 333 | 504 | 433 | 316 |

No class is accurately reconstructed. T2 is not uniquely worse except fold 0.

---

## 9. Pitch by T1 provenance (MAE cents, folds with support)

T6-derived T1 dominates support. Raw T1 MAE is often **higher** than T6 (folds 0, 1, 4). Pitch supervision does not preferentially reconstruct “simple” T1.

---

## 10. B1 framewise type metrics (test)

| Fold | Acc | Macro F1 | T0 rec | T1 rec | T2 rec | T3 rec |
|------|----:|---------:|-------:|-------:|-------:|-------:|
| 0 | 0.489 | 0.274 | 0.55 | 0.67 | 0.00 | 0.00 |
| 1 | 0.395 | 0.238 | 0.48 | 0.44 | 0.03 | 0.03 |
| 2 | 0.386 | 0.263 | 0.64 | 0.34 | 0.17 | 0.04 |
| 3 | 0.496 | 0.292 | 0.71 | 0.57 | 0.03 | 0.00 |
| 4 | 0.521 | 0.275 | 0.30 | 0.72 | 0.02 | 0.06 |

**Unweighted mean test macro F1: 0.268** (B0 re-eval: 0.232). Confusion matrices in each `eval_summary.json`.

---

## 11. B1 trajectory-level type metrics

Same aggregation as B0 (mean softmax over GT primitive interval).

| Fold | Traj acc | Traj macro F1 |
|------|---------:|--------------:|
| 0 | 0.584 | 0.258 |
| 1 | 0.460 | 0.231 |
| 2 | 0.367 | 0.230 |
| 3 | 0.474 | 0.269 |
| 4 | 0.698 | 0.270 |

Mean traj acc **0.517**, mean traj macro F1 **0.252**.

---

## 12. Paired B0 vs B1 (same test recordings, same eval code)

B0 checkpoints were **re-evaluated** with the B1 metric suite (trajectory macro F1, confusion pairs). No B0 retraining.

### Frame macro F1 (Δ = B1 − B0)

| Fold | B0 | B1 | Δ |
|------|---:|---:|--:|
| 0 | 0.352 | 0.274 | **−0.078** |
| 1 | 0.224 | 0.238 | +0.013 |
| 2 | 0.268 | 0.263 | −0.004 |
| 3 | 0.132 | 0.292 | **+0.160** |
| 4 | 0.185 | 0.275 | **+0.090** |
| mean | 0.232 | 0.268 | **+0.036** |
| median Δ | | | +0.013 |
| folds improved | | | **3 / 5** |

Exploratory Wilcoxon on n=5 Δ F1: p=0.44. Do not treat as significant.

### Trajectory macro F1 Δ

mean **+0.027**, median **−0.001**, improved **2 / 5**.

Trajectory *accuracy* is noisy (fold 0 +0.22, fold 2 −0.25) because majority T1 can inflate accuracy when the model collapses toward T0/T1.

---

## 13. T1 ↔ T2/T3 confusion

Pitch supervision does **not** cleanly reduce bend-class confusions.

- Fold 0: T2 and T3 recall go to **0**; almost all minority frames become T1. T1→T2/T3 counts drop because the model stops predicting T2/T3 at all.
- Fold 4: T2 recall falls 0.60 → 0.02.
- Fold 2: T2 recall rises 0.04 → 0.17, but T1→T2/T3 counts increase.

Mean Δ T2 recall **−0.102**; mean Δ T3 recall **−0.087**. Only 2/5 folds improve each.

---

## 14. T1 provenance: B0 vs B1

Inconsistent across folds. Fold 0 raw T1 recall jumps 0.11 → 0.90 (with T2/T3 collapse). Folds 1–3 mostly **lower** T1 recall after pitch is added, including T6-derived T1. Pitch does not specifically rescue T6 boundaries.

---

## 15–16. Type accuracy vs pitch error

Correct type predictions have only **slightly** lower pitch error than incorrect ones (e.g. fold 1: 472 vs 491 cents MAE; fold 2: 447 vs 489). Fold 0 is essentially tied (394 vs 399).

Pitch-error buckets are dominated by large errors; there is no sharp “accurate pitch ⇒ accurate type” regime on the test set. Association is weak, not causal.

---

## 17. Per-recording results

See `fold_<i>/eval/eval_summary.json` → `per_recording`. Fold variance is large because each test fold is 2–5 recordings. Fold 3’s type-F1 gain and pitch failure are the same small test set.

---

## 18. Visualizations

34 PNGs under `output/framewise_runs/b1_tcn_type_pitch/fold_<i>/eval/figures/` (0 s and 30 s windows): CQT, GT/pred type, GT/pred pitch (cents), valid mask, softmax, |pitch error|. Mixed success/failure windows are included; pitch overlays are often a smoothed, offset contour rather than a tight match.

UMAP skipped (not in `requirements.txt`). Shared-embedding PCA omitted; quantitative results already show weak type–pitch coupling.

---

## 19. Fold-to-fold variation

B1 test macro F1 range **0.238–0.292** is tighter than B0’s **0.132–0.352**. Gains concentrate on B0’s worst folds (3 and 4). Fold 0, B0’s best type fold, **regresses** and loses T2/T3.

---

## 20. Q1–Q12

**Q1. Frame macro F1?** Mean Δ **+0.036**, 3/5 folds up. Not a reliable win; Wilcoxon n=5 is n.s.

**Q2. Trajectory macro F1 / accuracy?** Traj macro F1 mean Δ **+0.027** (2/5). Traj accuracy moves both ways.

**Q3. T2/T3 recall?** **No** on average (Δ recall −0.10 / −0.09).

**Q4. T1↔T2/T3 confusion?** **No** systematic reduction. Several folds collapse minority classes into T0/T1.

**Q5. Pitch accuracy?** Held-out MAE typically **400–480 cents** (~4–5 semitones); usually **no better than predicting train-mean pitch**. Tiny-overfit MAE 97 cents shows capacity, not generalization.

**Q6. Pitch by class?** All types are similarly poor; T2 not uniquely failing.

**Q7. T1 provenance?** No stable pattern that pitch helps raw T1 more than T6 (or vice versa).

**Q8. Correct type ⇒ lower pitch error?** Only a small gap; not a strong association.

**Q9. Regularize or interfere?** Mild interference on val type F1 (0.288 vs B0 0.301). Test F1 mean is slightly higher because B0’s worst folds improved. Train type still overfits. Pitch loss does not dominate.

**Q10. Consistency?** Improvements are **fold-driven** (especially folds 3–4), not uniform.

**Q11. λ_pitch = 1?** Yes for loss/gradient scale (`L_type` ~ 1, `L_pitch` ~ 0.3–0.5 at start; grad norms comparable; cosine ~ 0). It does **not** yield useful pitch generalization.

**Q12. Proceed to C?** **Yes.** See gate.

---

## 21. Decision gate

### **PROCEED_TO_C**

B1 is a **stable** type+pitch TCN: sanity gates passed, 434,629 params verified, 5-fold CV completed, eval/compare artifacts written. Pitch supervision is **not** a clear type-recognition upgrade (T2/T3 stay weak; held-out pitch is near a mean baseline). That is an experimental result, not a pipeline failure.

Experiment C should swap **only** TCN → 1-layer BiGRU, keeping type+pitch and the same folds/sampler/loss, as specified in Step 6.5.

**Not chosen**

- `INVESTIGATE_MULTITASK` — type did not collapse; overfit works; λ=1 is scaled reasonably. High val pitch MAE is expected with a 1-group val set.
- `RECONSIDER_PITCH_TARGET` — the target is learnable on memorized excerpts; held-out error is a generalization/architecture issue, not proof that parametric cents are unusable.

Do not add weighted CE or change TCN width before C.
