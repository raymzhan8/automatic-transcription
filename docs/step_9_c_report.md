# Step 9 — Experiment C: BiGRU vs TCN

Same data, CQT, 4 s excerpts, cents pitch, λ=1, grouped 5-fold CV, sampler, and losses as Experiment B1. The only intended change is **TCN → 1-layer bidirectional GRU**. **C is offline** (full-excerpt bidirectional context). It is not a causal or real-time model.

A C win would **not** by itself mean “longer context helps”: parameter count, inductive bias, and bidirectionality also change. In this run C did **not** win.

Reproduce:

```bash
python training/train_framewise.py \
  --config training/configs/c_bigru_type_pitch.json --tiny-overfit 32 --max-epochs 100 --fold 0

python training/train_framewise.py \
  --config training/configs/c_bigru_type_pitch.json --all-folds

for i in 0 1 2 3 4; do
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/c_bigru_type_pitch/fold_$i/best.pt
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b1_tcn_type_pitch/fold_$i/best.pt
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b0_tcn_type_only/fold_$i/best.pt \
    --output-dir output/framewise_runs/b0_tcn_type_only/fold_$i/eval
done

python training/compare_b1_c.py
```

Paired numbers: [`output/framewise_runs/c_bigru_type_pitch/b1_vs_c.json`](../output/framewise_runs/c_bigru_type_pitch/b1_vs_c.json). Overlays: `output/framewise_runs/c_bigru_type_pitch/b1_vs_c_figures/`.

**Out of scope (not run):** class weighting, phase, Experiment D, decoder, CQT/excerpt/λ retuning, B1 retraining.

---

## 1. Exact Model C architecture

```text
CQT [B,1,360,400]
  → FrequencyCNN          (same as B1)
  → BiGRU 128 hidden, 1 layer, bidirectional
  → h_t [B,T,256]
       ├─ Linear(256→4)  type_logits
       └─ Linear(256→1)  pitch_pred_standardized
```

Encoder input is still **normalized CQT only**. Packing uses **padding lengths**, not `valid_target`: annotation-masked frames stay in GRU context and only gate CE / SmoothL1.

---

## 2. Verified parameter count

`count_params(FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)) == **305,221**`. Training refuses to start if the count differs. All five CV folds logged `n_params: 305221`.

B1 was 434,629 (TCN). C is smaller despite full-excerpt recurrence.

---

## 3. Padding / masking

- `lengths = (~padding_mask).sum(dim=1)` in `collate_excerpts`, `collate_variable_length`, train, and eval.
- `pack_padded_sequence` + `pad_packed_sequence(..., total_length=T)` so logits stay **T=400** even when every item in the batch is shorter.
- Invalid annotation frames are **not** dropped from the GRU.
- Tests: param count, T=400 in/out, mixed lengths, packed prefix matches truncated GRU on the same CNN embeddings, unpacked bidirectional GRU **leaks** padding into the prefix.

---

## 4. Tiny-overfit results

`--tiny-overfit 32 --max-epochs 100 --fold 0` (cached excerpts).

| Gate | Result |
|------|--------|
| Param count 305,221 | Pass |
| Packed shapes `[B,400,4]` / `[B,400]` | Pass |
| Train type macro F1 | **1.000** (epoch 64); epoch 1 was 0.199 |
| Train pitch MAE | **43 cents** (from 352 at epoch 1); epoch 100 still 71 |
| `L_type` / `L_pitch` at best F1 | 0.006 / 0.005 |

C can memorize type and pitch on a frozen excerpt set (stronger tiny-overfit pitch than B1’s 97 cents). **Stop-if-fail did not trigger.**

---

## 5. Training configuration (identical to B1 except encoder)

| Setting | Value |
|---------|-------|
| Sampler | Valid-anchor 4 s, 512 excerpts/epoch, batch 8 |
| Optimizer | AdamW 1e-3, wd 1e-4, grad clip 1.0 |
| Seed | 42 |
| Early stop | val **type** macro F1, patience 10, max 50 |
| Type loss | unweighted CE |
| Pitch loss | Smooth L1 on fold-standardized cents |
| `λ_type`, `λ_pitch` | **1.0, 1.0** |
| Fold CQT / pitch stats | reused; same train IDs ⇒ same cents μ/σ as B1 |

Documented diffs vs B1: `architecture: bigru`, 305,221 params, `h_t` is 256-d, packing, shared-grad module is `freq_cnn`+`gru`.

---

## 6. C learning curves

Val type macro F1 (early-stop metric):

| Fold | Best epoch | C val F1 | B1 val F1 |
|------|----------:|---------:|----------:|
| 0 | 9 | 0.259 | 0.253 |
| 1 | 6 | 0.307 | 0.298 |
| 2 | 30 | 0.338 | 0.352 |
| 3 | 4 | 0.172 | 0.249 |
| 4 | 24 | 0.352 | 0.290 |
| **mean** | | **0.286** | **0.288** |

C val F1 matches B1 on average. Train type F1 at the selected checkpoint is **0.41–0.93** (often higher than B1): C overfits type at least as hard.

Epoch-1 `L_type` ≈ 1.01–1.06, `L_pitch` ≈ 0.30–0.42. Shared-grad cosine on `freq_cnn`+`gru` is near 0 (−0.05 to +0.08). Pitch loss falls in training; val pitch MAE on the 1-group val split stays large.

CSV logs: `output/framewise_runs/c_bigru_type_pitch/fold_<i>/train_log.csv`.

---

## 7. Framewise type metrics (held-out test)

| Fold | Acc | Macro F1 | T0 rec | T1 rec | T2 rec | T3 rec |
|------|----:|---------:|-------:|-------:|-------:|-------:|
| 0 | 0.414 | 0.170 | 0.057 | 0.998 | 0.00 | 0.00 |
| 1 | 0.473 | 0.203 | 0.035 | 0.946 | 0.07 | 0.00 |
| 2 | 0.243 | 0.205 | 0.165 | 0.277 | 0.64 | 0.08 |
| 3 | 0.451 | 0.244 | 0.760 | 0.382 | 0.00 | 0.00 |
| 4 | 0.509 | 0.274 | 0.285 | 0.707 | 0.04 | 0.07 |

**Unweighted mean test macro F1: 0.219** (B1 0.268, B0 0.232). C is the worst of the three on mean type F1.

---

## 8. Trajectory-level type metrics

Mean softmax over GT primitive intervals (same aggregation as B0/B1).

| Fold | Traj acc | Traj macro F1 |
|------|---------:|--------------:|
| 0 | 0.592 | 0.186 |
| 1 | 0.626 | 0.222 |
| 2 | 0.256 | 0.178 |
| 3 | 0.432 | 0.241 |
| 4 | 0.669 | 0.255 |

Mean traj acc **0.515**, mean traj macro F1 **0.216** (B1 traj macro F1 0.252). Trajectory accuracy can rise when the model collapses to majority T1; macro F1 does not.

---

## 9. Pitch metrics (held-out)

| Fold | MAE | Median AE | %±10 | %±25 | %±50 |
|------|----:|----------:|-----:|-----:|-----:|
| 0 | 598 | 533 | 0.5 | 1.4 | 2.8 |
| 1 | 533 | 507 | 1.3 | 3.3 | 6.2 |
| 2 | 476 | 427 | 1.0 | 2.5 | 5.1 |
| 3 | 1290 | 1318 | 0.1 | 0.3 | 0.6 |
| 4 | 431 | 412 | 0.6 | 1.6 | 3.8 |

Held-out pitch is still multi-semitone error. Fold 3 remains an octave-scale failure.

---

## 10. Pitch vs mean-pitch baseline

Trivial baseline: always predict training-fold **mean cents** (same μ as B1).

| Fold | C MAE | Baseline MAE | C − baseline |
|------|------:|-------------:|-------------:|
| 0 | 598 | **260** | +338 |
| 1 | 533 | **478** | +55 |
| 2 | 476 | **423** | +53 |
| 3 | 1290 | **1084** | +206 |
| 4 | **431** | 573 | **−142** |

C **beats the mean baseline on 1/5 folds** (fold 4 only), same pattern as B1. On the other four folds the BiGRU is worse than predicting the training mean.

---

## 11. Pitch by trajectory type (MAE cents)

| Fold | T0 | T1 | T2 | T3 |
|------|---:|---:|---:|---:|
| 0 | 512 | 678 | 732 | 495 |
| 1 | 544 | 524 | 557 | 433 |
| 2 | 469 | 483 | 448 | 465 |
| 3 | 1342 | 1246 | 1193 | 1341 |
| 4 | 339 | 481 | 481 | 445 |

No class is reconstructed. T2 is not uniquely the failure mode.

---

## 12. Pitch / type by T1 provenance

T6-derived T1 still dominates support. On folds 0–1, C’s T1 recall on T6 is ~0.95–1.0 because the model **collapses to T1**, not because T6 geometry is solved. Fold 2 T6 T1 recall **falls** vs B1 (0.33 → 0.28). Raw T1 is not preferentially rescued.

---

## 13. Direct paired B1 vs C (Δ = C − B1)

Same test recordings, same eval code. B1 checkpoints were **re-evaluated**, not retrained.

### Frame macro F1

| Fold | B1 | C | Δ |
|------|---:|--:|--:|
| 0 | 0.274 | 0.170 | **−0.104** |
| 1 | 0.238 | 0.203 | −0.035 |
| 2 | 0.263 | 0.205 | −0.059 |
| 3 | 0.292 | 0.244 | −0.048 |
| 4 | 0.275 | 0.274 | −0.001 |
| mean | 0.268 | 0.219 | **−0.049** |
| median Δ | | | −0.048 |
| folds improved | | | **0 / 5** |

Exploratory Wilcoxon n=5 on Δ F1: p=0.0625. Do not treat as confirmatory; the sign is uniformly non-positive.

### Trajectory macro F1

mean Δ **−0.035**, median **−0.028**, improved **0 / 5**.

### Pitch MAE (negative Δ would mean C better)

mean Δ **+76 cents**, median **+49**, improved **1 / 5** (fold 4, −1 cent). `%±25` and `%±50` improved on **0 / 5** folds.

Per-class recall Δ (mean): T0 **−0.277**, T1 **+0.116**, T2 **+0.100**, T3 **+0.002**. The T1/T2 “gains” are collapse artifacts (see §16), not balanced recognition. T0 F1 improved on **0 / 5** folds (mean Δ **−0.215**).

---

## 14. B0 / B1 / C summary

Context only. **B0 vs C is not a single-variable contrast** (type-only TCN vs type+pitch BiGRU).

| Fold | B0 F1 | B1 F1 | C F1 | B1 pitch MAE | C pitch MAE | Baseline MAE |
|------|------:|------:|-----:|-------------:|------------:|-------------:|
| 0 | 0.352 | 0.274 | 0.170 | 396 | 598 | 260 |
| 1 | 0.224 | 0.238 | 0.203 | 483 | 533 | 478 |
| 2 | 0.268 | 0.263 | 0.205 | 473 | 476 | 423 |
| 3 | 0.132 | 0.292 | 0.244 | 1163 | 1290 | 1084 |
| 4 | 0.185 | 0.275 | 0.274 | 432 | 431 | 573 |
| mean | 0.232 | 0.268 | **0.219** | | | |

C does not recover B0’s best type fold (0) and does not keep B1’s modest mean F1 bump.

---

## 15. T2 / T3 confusion

Extra flows T2→T0/T1/T3 and T3→T0/T1/T2 (plus existing T1↔T2/T3).

- Folds 0 and 3: C predicts **no T2/T3**. Almost all minority frames go to T1 (fold 0: T2→T1 1090/1100, T3→T1 1033/1035).
- Fold 1: T2 mostly → T1 (6571 vs B1 2939); T3 recall goes to 0.
- Fold 2: opposite collapse — C predicts **55% T2**. T2 recall 0.17 → 0.64 but T2 F1 stays ~0.11 (precision collapse). T3→T2 jumps 1143 → 4614.
- Fold 4: closest to B1; T2/T3 still near zero.

The BiGRU does **not** systematically reduce bend-class confusion relative to the TCN.

---

## 16. Predicted-class distribution / collapse

Mean predicted frame share across folds:

| Model | T0 | T1 | T2 | T3 | entropy (per fold) |
|-------|---:|---:|---:|---:|--------------------|
| GT (fold 0 example) | 0.41 | 0.39 | 0.10 | 0.10 | — |
| B0 | 0.23 | 0.61 | 0.09 | 0.07 | 0.28–1.29 |
| B1 | 0.44 | 0.50 | 0.04 | 0.02 | 0.55–1.12 |
| C | 0.21 | 0.64 | 0.12 | 0.02 | 0.43–0.86 |

Fold 0 C: **97.5% T1**, mean softmax entropy **0.43** vs B1 **1.12** — more confident and more collapsed. Fold 2 C: 55% T2. Collapse direction is fold-specific; neither is a calibrated four-class predictor.

---

## 17. Boundary-distance analysis (eval only, GT primitives)

Buckets on valid frames: 0–20 / 20–50 / 50–100 / 100–250 / >250 ms.

C’s frame macro F1 is **lower than B1 in almost every bucket** on folds 0–3. There is **no** C advantage near boundaries (where longer bidirectional context might have helped) or far from them. Fold 4 is mixed and small. Boundary proximity is not the main C-vs-B1 difference; class collapse is.

---

## 18. Primitive-duration analysis

Trajectory metrics by duration: &lt;100 ms / 100–250 / 250–500 / 500 ms–1 s / &gt;1 s.

Short primitives often have **high accuracy, low macro F1** for both models (majority T1). C does not preferentially win long primitives (&gt;1 s), where a full-excerpt recurrent encoder might have been expected to help. Fold 2 C is worse in every duration bucket. Fold 1 C raises accuracy on short/mid durations via T1 collapse, not via better T2/T3.

---

## 19. Pitch by duration / recording

Pitch MAE stays hundreds of cents in every duration bucket. Fold 0 C is **worse than B1 in all five duration buckets** (~350–490 → ~540–660). Fold 4 C is slightly better on &gt;500 ms (478 → 400 on &gt;1 s) but still far from usable. Per-recording MAE on fold 0: every test recording is worse under C (e.g. 208→433, 340→652, 475→817 cents).

Longer primitives do not unlock pitch for the BiGRU.

---

## 20. Type correctness vs pitch error

Buckets include 100–300 / 300–600 / &gt;600 cents.

On several C folds, **incorrect-type frames have lower pitch error than correct-type frames** (fold 0: 540 vs 680 MAE; fold 2: 466 vs 506). That is the opposite of a “shared representation” story and is consistent with type collapse onto a class whose pitch happens to sit nearer the predicted contour.

Fine error buckets are dominated by 300–600 and &gt;600 cents. There is no sharp “accurate pitch ⇒ accurate type” regime.

---

## 21. Train / val / test generalization gaps

At the **selected** checkpoint (best val type F1):

| Fold | C train F1 | C val F1 | C test F1 | C train pitch MAE | C test pitch MAE |
|------|----------:|---------:|----------:|------------------:|-----------------:|
| 0 | 0.613 | 0.259 | 0.170 | 207 | 598 |
| 1 | 0.694 | 0.307 | 0.203 | 202 | 533 |
| 2 | 0.932 | 0.338 | 0.205 | 146 | 476 |
| 3 | 0.411 | 0.172 | 0.244 | 234 | 1290 |
| 4 | 0.828 | 0.352 | 0.274 | 143 | 431 |

Type: large train≫val≫test gaps (except fold 3, where val is the weakest split). Pitch: train MAE ~140–230 cents is already weak vs tiny-overfit 43 cents; test is another 300–1000 cents worse. C memorizes type more aggressively than it learns a transferable pitch contour.

---

## 22. Per-recording analysis

Each test fold is 2–5 recordings. Fold 0 (5 recordings): C frame F1 is worse on **every** recording vs B1. Fold 2’s T2-heavy collapse is a recording-mix effect, not a stable encoder property. Fold 3 remains the small, shifted test set that breaks pitch for both B1 and C.

See `per_recording` in each `eval_summary.json` and `b1_vs_c.json`.

---

## 23. Representative B1-vs-C visualizations

28 overlays under `output/framewise_runs/c_bigru_type_pitch/b1_vs_c_figures/` (low/mid/high F1 recording × 0 s and 30 s per fold): GT type, B1 pred, C pred, GT/B1/C pitch, valid mask, GT primitive boundaries.

Typical pattern: C type traces a flatter majority class (often T1) than B1; pitch overlays are slowly varying and offset from GT by hundreds of cents for both models.

Per-model eval figures remain in `fold_<i>/eval/figures/`. Full-recording contours: `fold_<i>/eval/pitch_contours/*.npz` (now include logits).

---

## 24. Fold-to-fold variation

C test macro F1 range **0.170–0.274** is similar in width to B1 (**0.238–0.292**) but shifted down. Collapse mode is unstable: T1-only (folds 0, 1, 3) vs T2-majority (fold 2) vs near-B1 (fold 4). Pitch fold 3 is an outlier for both models (~1200–1300 cents). Any claim about “BiGRU vs TCN” has to be read as **this 17-recording grouped CV**, not as a stable architecture ranking.

---

## 25. Likely bottleneck hypotheses (if C fails)

C failed on both tasks relative to B1. Hypotheses, **not** follow-up experiments that were run:

1. **Pitch frontend / target** — CQT log-mag + linear pitch head cannot recover tonic-relative cents on held-out performances; error stays near or worse than the train-mean baseline. Candidates: frontend (CQT/CNN), tonic/shift, drone leakage, or the parametric pitch target itself. Tiny-overfit shows the loss is wired.
2. **Class supervision** — unweighted CE on a T1-heavy frame prior; T2/T3 remain near-zero except when the model over-predicts them.
3. **Generalization / corpus size** — 17 performances, grouped folds; C’s extra sequential capacity overfits type (train F1 up to 0.93) without a transferable contour.
4. **4 s bidirectional context is the wrong inductive bias** for this label process (local TCN already had ~610 ms); packing is correct, so this is not a padding bug.

These are alternatives, not a ranked next experiment.

---

## 26. Four-way outcome

### **NEITHER_IMPROVES**

- Type: mean Δ frame macro F1 **−0.049**, **0/5** folds improved; traj macro F1 **0/5**.
- Pitch: mean Δ MAE **+76 cents** (C worse), **1/5** improved by a 1-cent fold-4 noise; still fails the mean-pitch baseline on 4/5 folds.

Not `TYPE_AND_PITCH_IMPROVE`, `TYPE_IMPROVES_PITCH_DOES_NOT`, or `PITCH_IMPROVES_TYPE_DOES_NOT`.

A full-excerpt bidirectional recurrent encoder did **not** outperform the finite-context TCN under matched data, losses, and folds.

---

## 27. Final recommendation

### **INVESTIGATE_PITCH_FRONTEND**

Pitch remains extremely poor on **both** B1 and C and still fails the training-mean baseline on 4/5 folds. Swapping TCN for BiGRU did not create a usable contour, so the next bottleneck to isolate is the pitch **frontend/target** (CQT representation, tonic-relative cents, drone/accompaniment, octave/shift), not another temporal encoder or phase head.

**Not chosen**

- `PROCEED_TO_CONTEXT_ANALYSIS` — C did not clearly improve type or pitch, so there is no C-win to attribute to context.
- `INVESTIGATE_CLASS_SUPERVISION` — T2/T3 are still poor, but this rec requires pitch to have become reasonable first.
- `INVESTIGATE_GENERALIZATION` — train/test gaps are real, but the distinguishing B1-vs-C fact is that **neither** encoder beats a constant pitch predictor on held-out audio.
- `PROCEED_TO_D` — not an allowed Step 9 gate; phase is not the assumed next experiment.

Do not add class weighting, change CQT, widen the BiGRU, or train Experiment D as an automatic next step from this result.
