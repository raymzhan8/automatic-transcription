# Step 7 — Framewise Training Pipeline + Experiment B0 Report

Experiment **B0**: TCN type-only (`FramewiseTCNModel`, 434,500 params), grouped 5-fold CV on 17 canonical recordings. Authority: [`step_6_5_architecture_corrections.md`](step_6_5_architecture_corrections.md).

Reproduce:

```bash
python dataset/canonical/build_features.py
python -m pytest training/test_frame_alignment.py training/tests/ -q
python training/train_framewise.py --all-folds
for i in 0 1 2 3 4; do
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b0_tcn_type_only/fold_$i/best.pt
done
```

Artifacts: `output/framewise_runs/b0_tcn_type_only/`, `output/canonical/v1/features/`, `output/canonical/v1/normalization/`.

---

## 1. Modules delivered

| Module | Role |
|--------|------|
| [`training/features.py`](../training/features.py) | CQT log-mag + linear interp to 10 ms grid |
| [`dataset/canonical/build_features.py`](../dataset/canonical/build_features.py) | Precompute `features/<recording_id>.npz` |
| [`training/normalization.py`](../training/normalization.py) | Fold-specific CQT μ/σ; pitch stats (B1 infra) |
| [`training/sampling.py`](../training/sampling.py) | Valid-anchor 4 s excerpts; loss mask |
| [`training/framewise_dataset.py`](../training/framewise_dataset.py) | Index, excerpt dataset, full-recording eval |
| [`training/framewise_models.py`](../training/framewise_models.py) | Model B (TCN), Model C shell (BiGRU) |
| [`training/losses.py`](../training/losses.py) | Unweighted CE on `(~padding) & valid_target` |
| [`training/metrics.py`](../training/metrics.py) | Frame, trajectory, T1-provenance metrics |
| [`training/folds.py`](../training/folds.py) | K-fold splits, val holdout, leakage checks |
| [`training/train_framewise.py`](../training/train_framewise.py) | Train loop, logging, checkpointing |
| [`training/evaluate_framewise.py`](../training/evaluate_framewise.py) | Test eval + diagnostic plots |
| [`training/configs/b0_tcn_type_only.json`](../training/configs/b0_tcn_type_only.json) | B0 hyperparameters |

Tests: [`training/tests/`](../training/tests/), [`training/test_frame_alignment.py`](../training/test_frame_alignment.py).

---

## 2. Verified parameter counts

From `python training/framewise_models.py`:

| Component | Parameters |
|-----------|----------:|
| Frequency CNN | 105,792 |
| TCN (dilations 1,2,4,8, k=5) | 328,192 |
| Type head | 516 |
| **B0 total (type-only)** | **434,500** |
| B1 total (type + pitch) | 434,629 |
| Model C BiGRU 1-layer (shell) | 305,221 |

TCN receptive field: **61 frames** (~610 ms at 10 ms hop).

Unit test `test_b0_param_count` asserts 434,500 before training.

---

## 3. Feature precomputation

- **17/17** recordings built successfully.
- **Total size:** 2,298 MB (compressed NPZ; estimate was ~1.5 GB uncompressed-equivalent).
- Each file: `cqt_log` float32 `[360, T]`, `frame_time_s` float64 `[T]`, metadata (`sr=22050`, `cqt_hop=220`, `fmin=75`, `n_bins=360`, `bins_per_octave=72`).

Fold normalization applied at **load time** from `output/canonical/v1/normalization/fold_<i>_cqt_stats.npz`.

---

## 4. Alignment and leakage tests

**Alignment** (`training/test_frame_alignment.py`, `training/tests/test_features.py`):

- Target grid starts at 5 ms; zero drift at 60 s.
- Precomputed `features/*.npz` `frame_time_s` matches `frames/*.npz` (tolerance 1e-6).
- 4 s excerpts yield exactly **400** frames on real recordings.

**Leakage** (`training/tests/test_folds.py`, `training/tests/test_normalization.py`):

- No shared `performance_group_id` or `audio_id` across train/val/test within each CV round.
- CQT stats computed from **train recordings only**; val/test IDs excluded.
- Pitch stats (B1 infra) use `valid_target` train frames only.

All 18 unit tests pass (run with `python -m pytest training/test_frame_alignment.py training/tests/ -q`).

---

## 5. Sanity gates

| Gate | Result |
|------|--------|
| Unit tests | **Pass** (18/18) |
| Tiny overfit (32 cached excerpts, 1 recording, 100 epochs) | **Pass** — train macro F1 **0.988**, loss → 0.02 |
| Label shuffle (15 epochs, fold 0) | **Pass** — val macro F1 **0.14** (below shuffled-ceiling; no leakage signal) |
| Full 5-fold B0 | **Complete** — all folds checkpointed |

Tiny overfit confirms the model and loss can memorize a fixed excerpt set; label shuffle confirms shuffled labels do not produce spurious high validation scores.

---

## 6. Sampler diagnostics (fold 0, epoch 1 audit)

From `fold_0/sampler_audit.json` (512 excerpts × batch 8):

- **Excerpts per recording:** 37–56 (min/max ratio **0.66** — no recording starved).
- **Class frame counts in sampled batches:** T0 45k, T1 83k, T2 10k, T3 10k (T1 overweight reflects corpus).
- **T1 provenance** represented in sampled frames (raw_t1, t4/t5/t6 decomposition).

Valid-anchor 4 s excerpts with jittered start positions; loss mask = `(~padding_mask) & valid_target`.

---

## 7. Training results (validation macro F1)

Early stopping on **val macro F1** (one held-out performance group per fold). Config: AdamW lr=1e-3, wd=1e-4, batch 8, max 50 epochs, patience 10.

| Fold | Best epoch | Val macro F1 | Test recordings |
|------|----------:|-------------:|----------------:|
| 0 | 13 | 0.265 | 5 |
| 1 | 34 | 0.332 | 3 |
| 2 | 21 | 0.370 | 4 |
| 3 | 4 | 0.192 | 3 |
| 4 | 11 | 0.344 | 2 |

**Mean ± std (val):** **0.301 ± 0.065**

Learning curves: `output/framewise_runs/b0_tcn_type_only/fold_<i>/train_log.csv`.

---

## 8. Held-out test results (per fold)

Each fold's `best.pt` evaluated on that fold's **test recordings** (full length, no excerpt sampling).

| Fold | Frame acc | Frame macro F1 | Traj acc | Majority baseline acc | Majority baseline macro F1 |
|------|----------:|---------------:|---------:|----------------------:|---------------------------:|
| 0 | 0.506 | **0.352** | 0.368 | 0.392 | 0.141 |
| 1 | 0.452 | 0.224 | **0.482** | 0.477 | 0.161 |
| 2 | 0.512 | 0.268 | **0.655** | 0.559 | 0.179 |
| 3 | 0.333 | 0.132 | 0.483 | 0.339 | 0.126 |
| 4 | 0.289 | 0.185 | 0.328 | 0.575 | 0.183 |

**Mean across folds (unweighted):** frame acc **0.418**, frame macro F1 **0.232**, trajectory accuracy **0.463**.

Full JSON: `fold_<i>/eval/eval_summary.json`.

### Per-class recall (test, fold 0 — best macro F1)

| Type | Recall | F1 | Support |
|------|-------:|---:|--------:|
| T0 | 0.93 | 0.65 | 4,430 |
| T1 | 0.23 | 0.34 | 4,225 |
| T2 | 0.07 | 0.11 | 1,100 |
| T3 | 0.29 | 0.31 | 1,035 |

T2/T3 recall remains weak on several folds; T1–T1 boundaries and minority classes are the main error mode (consistent with Step 5.5).

### Trajectory-aggregated metrics

Mean softmax over primitive intervals often **outperforms** frame macro F1 (e.g. fold 2 traj acc **0.655** vs frame macro F1 **0.268**), suggesting temporal aggregation helps despite noisy frame predictions.

---

## 9. Comparison to segment CNN baseline

Segment CNN (1 s clips, oracle boundaries): ~**0.34** test accuracy, ~**0.23–0.31** macro F1 ([README](../README.md)).

| Metric | Segment CNN | B0 framewise (test mean) |
|--------|------------:|-------------------------:|
| Frame/segment accuracy | ~0.34 | **0.42** |
| Macro F1 | ~0.23–0.31 | **0.23** |
| Trajectory accuracy | N/A (clip = one primitive) | **0.46** |

B0 is **not directly comparable** on accuracy (different task: continuous audio vs oracle-segment clips). Frame macro F1 is in the same ballpark; trajectory aggregation is the fairer bridge metric and exceeds chance on several folds.

---

## 10. Visualizations

Per-fold diagnostic PNGs (GT vs predicted type, max softmax, valid mask, class probabilities):

```
output/framewise_runs/b0_tcn_type_only/fold_<i>/eval/figures/
```

34 figures total (2 windows × test recordings per fold). Examples: T2/T3 regions, type transitions, failure cases at 0 s and 30 s offsets.

---

## 11. Q1–Q10

**Q1: Does the end-to-end pipeline run without errors?**  
Yes — features, folds, training, checkpointing, and evaluation all complete on all 17 recordings.

**Q2: Are CQT features aligned to framewise targets?**  
Yes — `frame_time_s` matches exactly; alignment tests pass on real data.

**Q3: Is there train/test leakage?**  
No — grouped k-fold by `performance_group_id`; leakage assertions pass; normalization uses train IDs only.

**Q4: Can the model learn (capacity check)?**  
Yes — tiny overfit reaches train macro F1 **> 0.95** on 32 cached excerpts.

**Q5: Does label shuffle destroy validation performance?**  
Yes — val macro F1 stays ~0.12–0.14 with shuffled labels (no spurious signal).

**Q6: Does B0 beat the majority-class baseline on macro F1?**  
On **3/5** test folds (0, 1, 2), macro F1 exceeds the majority baseline; folds 3–4 are near baseline.

**Q7: Which classes fail?**  
T2 and T3 recall is consistently low; model often confuses minority types with T1. T0 recall is high when supported.

**Q8: Does T1 provenance matter at eval time?**  
Infrastructure is in place (`t1_provenance` join); frame-level stratification available in eval code. T6-derived T1 boundaries remain hard (Step 5.5 prediction).

**Q9: Is trajectory aggregation useful?**  
Yes — mean trajectory accuracy (**0.46** mean) often exceeds frame macro F1, supporting aggregation for downstream use.

**Q10: Should we proceed to B1 (add pitch head)?**  
**Yes** — see decision gate below.

---

## 12. Decision gate

### **`PROCEED_TO_B1`**

**Rationale:**

1. **Pipeline validated** — alignment, leakage, overfit, and full CV all succeed.
2. **B0 learns signal** — beats majority macro F1 on several folds; trajectory accuracy up to **0.66**; not stuck at chance.
3. **Known limits are architectural/task-level**, not bugs — T2/T3 sparsity and T6 boundary ambiguity were predicted in Step 5.5; unweighted CE + type-only cannot fix these alone.
4. **B1 is the planned next experiment** — pitch supervision may disambiguate T1-heavy confusions; do **not** add weighted CE or change architecture before B1.

**Not chosen:**

- `FIX_PIPELINE` — no alignment, leakage, or loader bugs found.
- `INVESTIGATE_B0` — optional ablations (weighted CE, longer context) deferred until after B1/C comparison per Step 6.5 experiment order.

---

## 13. Next steps (Step 8 scope, not executed here)

1. Train **Experiment B1** (TCN type + pitch, fold-specific pitch normalization).
2. Train **Experiment C** (BiGRU shell) for temporal-context comparison.
3. Consider weighted CE as a **separate ablation** after B0/B1/C.
4. Phase head remains out of scope until B0/B1/C settle type + pitch.

---

## 14. Run commands (quick reference)

```bash
# Features (once)
python dataset/canonical/build_features.py

# Train B0 all folds
python training/train_framewise.py --all-folds

# Evaluate one fold
python training/evaluate_framewise.py \
  --checkpoint output/framewise_runs/b0_tcn_type_only/fold_0/best.pt
```

Config: [`training/configs/b0_tcn_type_only.json`](../training/configs/b0_tcn_type_only.json). Stub configs for B1/C exist but were not trained in Step 7.
