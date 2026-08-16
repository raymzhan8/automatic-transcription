# Step 6.5 — Architecture Corrections Before Implementation

Revisions to the Step 6 framewise model specification. **No training. No dataset loader yet.**

Supersedes the technical details in [`step_6_framewise_model_design.md`](step_6_framewise_model_design.md) where they differ. Step 6 remains useful background; this document is the implementation authority.

Reference implementations (param counting + alignment tests only):

- [`training/framewise_models.py`](../training/framewise_models.py)
- [`training/test_frame_alignment.py`](../training/test_frame_alignment.py)

---

## 1. Exact parameter counts

Instantiated in PyTorch via `sum(p.numel() for p in model.parameters() if p.requires_grad)`.

### Shared frontend

| Component | Parameters |
|-----------|----------:|
| Frequency CNN | **105,792** |

Architecture: Conv(1→32, 7×3) → MaxPool(4×1) → Conv(32→64, 5×3) → MaxPool(4×1) → Conv(64→128, 3×3) → AdaptiveMaxPool(1, None).

### Temporal encoders

| Component | Parameters |
|-----------|----------:|
| TCN (4 layers, k=5, dilations [1,2,4,8], C=128) | **328,192** |
| BiGRU 1-layer (128 hidden, bidirectional) | **198,144** |
| BiGRU 2-layer (128 hidden, bidirectional, dropout 0.3) | **494,592** |

### Heads

| Head | TCN model (128-d) | BiGRU model (256-d) |
|------|------------------:|--------------------:|
| Type (→4) | **516** | **1,028** |
| Pitch (→1) | **129** | **257** |

### Totals

| Model | Parameters |
|-------|----------:|
| **Model B — TCN (type + pitch)** | **434,629** |
| **Model C — BiGRU 1-layer (type + pitch)** | **305,221** |
| Model C — BiGRU 2-layer (type + pitch) | 601,669 |

**Note:** The TCN stack is parameter-heavy (328k) because four full-width dilated Conv1d(128→128) layers. The 1-layer BiGRU model is **smaller** despite longer context. Recommend **1-layer BiGRU** as the first Experiment C model (§5).

Experiment B0 (type-only) uses the same TCN backbone; pitch head can be omitted or frozen — parameter count differs only by the 129-dim pitch linear layer.

---

## 2. Feature / target timing convention

### Problem (Step 6)

Canonical targets use frame **centers**:

```text
t_target[k] = (k + 0.5) × 0.01 s   →   5 ms, 15 ms, 25 ms, ...
```

([`dataset/canonical/frames.py`](../dataset/canonical/frames.py) `frame_centers()`.)

Librosa CQT native timestamps (`frames_to_time`, hop=220, sr=22050) start at:

```text
0 ms, 9.977 ms, 19.955 ms, ...
```

This introduces a **~5 ms systematic offset** plus **~0.023 ms/frame drift** (period 9.977 ms ≠ 10.000 ms). Nearest-neighbor index matching would bake in an undocumented shift harmful to boundary diagnostics.

### Approved convention (single grid)

**All supervision, model outputs, and evaluation use the canonical target grid only:**

```text
t[k] = (k + 0.5) × HOP_S,   HOP_S = 0.01 s exactly
```

### Feature extraction rule

1. Compute CQT log-magnitude at **sr = 22050**, **hop_length = 220** (analysis frontend; not the supervision clock).
2. Record librosa native sample times `t_native[j]` for each CQT column.
3. **Linearly interpolate** each frequency bin onto every target center `t[k]`.
4. Model input at index `k` aligns with target index `k` — **no nearest-neighbor**, no index offset.

```text
feature[k] = interp1d(t_native, CQT[:, j], t[k])
```

### Residual drift after interpolation

| Duration | Drift on output grid |
|----------|---------------------|
| 1 s | **0 ms** (by construction) |
| 10 s | **0 ms** |
| 1 min | **0 ms** |

Drift exists only on the internal CQT lattice (~2.7 ms over 1 s if used directly); interpolation onto `t[k]` removes it from the model I/O.

### Timestamp alignment test

[`training/test_frame_alignment.py`](../training/test_frame_alignment.py):

| Test | Asserts |
|------|---------|
| `test_target_frame_centers_start_at_5ms` | Grid starts at 5 ms |
| `test_interpolated_features_share_target_timestamps` | Feature rows map 1:1 to `frame_centers()` |
| `test_native_cqt_times_are_not_used_as_supervision_grid` | Documents librosa 0 ms origin |
| `test_no_drift_over_one_minute_on_output_grid` | Last center at 59.995 s for 60 s audio |

Tolerance: **1e-6 s**. Run in CI before training:

```bash
python -m pytest training/test_frame_alignment.py -q
```

---

## 3. TCN-first experimental ordering

Step 6 incorrectly prioritized BiGRU. **Correct order:**

| Priority | Model | Question |
|----------|-------|----------|
| First | **TCN (Model B)** | Does framewise prediction work with **local** temporal context (~610 ms)? |
| Second | **BiGRU (Model C)** | Does **longer** bidirectional context beat local TCN? |

The experimental variable between B1 and C is **only** the temporal encoder (TCN ↔ BiGRU). Frontend, embedding dim (128), heads, losses, normalization, sampling, splits, and excerpt length stay fixed.

```text
local temporal context  (TCN, RF ≈ 610 ms)
        vs
full-excerpt context    (BiGRU, both directions over entire valid sequence)
```

---

## 4. TCN receptive field (exact)

### Architecture

Four Conv1d layers, **kernel = 5**, **dilations = [1, 2, 4, 8]**, symmetric padding `(k−1)/2 × dilation`, **no temporal downsampling**, channels = 128 throughout.

### Formula (odd kernel, symmetric padding, same length)

```text
RF = 1 + Σ_i (k − 1) × d_i
   = 1 + 4 × (1 + 2 + 4 + 8)
   = 61 frames
```

### In milliseconds

| Clock | RF |
|-------|---:|
| Target grid (10 ms/frame) | **610 ms** |
| Native CQT hop 220 @ 22050 Hz | 608.6 ms |

Three layers `[1, 2, 4]` → RF = 29 frames = **290 ms** — **too small**. Four layers `[1, 2, 4, 8]` → **610 ms** — within the 500–700 ms target.

Optional residual connections in Step 7 implementation are allowed but do not change RF.

---

## 5. BiGRU specification and recommendation

| Variant | Parameters (full model) | Context |
|---------|------------------------:|---------|
| **1-layer BiGRU, 128 hidden** | **305,221** | Full padded excerpt, both directions |
| 2-layer BiGRU, 128 hidden | 601,669 | Same, more capacity |

**Recommend 1-layer BiGRU for Experiment C** on the small corpus (~17 performances). Escalate to 2-layer only if underfitting is clear.

```text
TCN context  = finite, ≈610 ms local
BiGRU context = entire valid excerpt (4 s default), bidirectional
```

---

## 6. Controlled experiment sequence

| Exp | Setup | Purpose | Key comparison |
|-----|-------|---------|----------------|
| **A** | Existing segment CNN, oracle interval → type | Reference ceiling | — |
| **B0** | Model B, **type only**, ordinary CE | Framewise type works at all? | vs chance / A aggregated |
| **B1** | Model B, **type + pitch**, same weights otherwise | Pitch supervision helps type? | **B0 vs B1** |
| **C** | Model C (1-layer BiGRU), type + pitch | Longer context helps? | **B1 vs C** |
| **D** | Best of B1/C + phase head | Phase helps segmentation? | vs best B1/C |

**Do not run D until B/C are understood.**

Change **one axis** per experiment. Same audio stem, grouped splits, excerpt length, CQT norm, pitch fold stats.

---

## 7. Loss — ordinary cross-entropy first

For B0, B1, C:

```text
L_type = CrossEntropyLoss(ignore_index=-1)   # unweighted
```

Monitor macro-F1, per-class precision/recall/F1, confusion matrix.

If the model collapses to T0/T1, run a **separate class-balance ablation**:

```text
ordinary CE  vs  inverse-frequency weighted CE
```

Do **not** mix architectural changes with reweighting in the first pass.

Pitch (B1, C): Smooth L1 on fold-standardized cents (§8).

---

## 8. Fold-specific pitch normalization

**Never** standardize per batch.

Per cross-validation fold:

1. Convert targets to **cents above tonic**: `1200 × log2(f / fundamental_hz)`.
2. On **training valid frames only**, compute `μ_train`, `σ_train`.
3. Train on `(pitch_cents − μ_train) / σ_train`.
4. Apply the **same** `μ_train`, `σ_train` to validation and test frames in that fold.
5. Save `{μ_train, σ_train, fundamental_hz}` in fold metadata.
6. At evaluation, **denormalize** predictions before cent metrics.

Validation/test frames must not influence normalization statistics.

---

## 9. Tonic assumption (explicit task definition)

**v1 task:**

```text
(audio waveform, known recording tonic) → framewise trajectory type + pitch
```

Tonic / `fundamental_hz` comes from IDTAP raga metadata — **given**, not predicted.

This is **not** fully blind audio-only transcription. Future work may estimate tonic first; out of scope for Experiments A–D.

Pitch head predicts **tonic-relative cents**; convert to Hz only for reporting using the recording's known fundamental.

---

## 10. Sampling diagnostics (Step 7)

Keep the valid-anchor excerpt sampler from Step 6. Log **per epoch** (or each N steps):

| Diagnostic | Purpose |
|------------|---------|
| Excerpts sampled per recording | Detect recording dominance |
| Valid frames sampled per recording | Detect supervision imbalance |
| Valid frames per class | Detect class collapse in sampling |
| Valid T1 frames per provenance (raw/T4/T5/T6) | Detect T6-dominated batches |

Do **not** change sampling weights yet — make the distribution observable first.

---

## 11. Excerpt duration recommendation

| Duration | TCN (610 ms RF) | BiGRU | Verdict |
|----------|-----------------|-------|---------|
| 2 s | Sufficient local context | Short sequence | Too few frames/batch diversity |
| **4 s** | Comfortable | Full 4 s bidirectional context | **Default** |
| 8 s | Wasteful for TCN | Longer context | Unfair if only C uses longer |

**Recommendation: fixed 4 s excerpts for B0, B1, and C.**

- 400 target frames per excerpt at 10 ms.
- Enough masked + valid context around anchors.
- Keeps B1 vs C comparison fair (BiGRU context = 4 s, not 8 s).
- Lower CQT cost than 8 s.

Implementation supports variable length + padding; **first experiments use fixed 4 s only**. No random 4–12 s in initial runs.

---

## 12. CQT normalization

**Do not** use per-clip min–max, magma RGB, or PNG resize.

Compute on training fold only (after interpolation to target grid):

| Strategy | Pros | Cons |
|----------|------|------|
| Global scalar mean/std | Simple | One loud bin affects all |
| **Per-frequency-bin mean/std** | Stabilizes drone harmonics across pitch range | 360×2 stats to store |

**Recommend per-bin normalization** (shape `[360]` for μ and σ), computed from **training valid frames'** interpolated CQT values (or all interpolated frames in training excerpts — document which in Step 7; prefer **all frames in training excerpts** for stable statistics, including masked regions, since CQT is unsupervised).

Apply fold μ/σ to val/test. No validation/test leakage.

---

## 13. Encoder input audit (no target leakage)

### Allowed inputs

- Interpolated CQT features (normalized)
- Padding mask (batch machinery)
- Optional: known tonic for pitch denormalization at loss/eval only — **not** concatenated to encoder unless explicitly ablated later

### Forbidden encoder inputs

- `valid_target`, `trajectory_type`, `pitch_*`, `phase`
- `primitive_id`, canonical boundaries
- `source_raw_type`, T1 provenance
- Any future ground-truth segmentation

Provenance and primitive metadata are **evaluation joins only** (via `primitive_id` in `.npz`).

---

## 14. Fair comparison with segment CNN

Different training paradigms:

| Model | Input |
|-------|-------|
| Segment CNN (A) | Oracle 1 s clip → one type |
| Framewise (B/C) | Continuous excerpt → frame types |

### Fair aggregated comparison

On held-out **canonical primitive intervals** (same performance groups):

```text
1. Collect frame type logits inside [start_s, end_s)
2. p = mean(softmax(logits)) over valid frames in interval
3. predicted_type = argmax(p)
4. Compare to canonical_type
```

**Use mean softmax probabilities** (not hard mode vote) as the primary aggregation — smoother for T2/T3 minority classes.

Report **both**:

- Framewise metrics (frame accuracy, macro-F1, pitch MAE)
- Trajectory-aggregated metrics (comparable to Experiment A)

Pitch: mean predicted cents vs primitive mean target cents per interval.

---

## 15. Provenance-specific evaluation

Join frame → `primitive_id` → `primitives.json` → `rule_applied` / `source_raw_type`.

Report T1 metrics separately for:

| Provenance | Rule |
|------------|------|
| raw T1 | `keep` + `source_raw_type==1` |
| T4 | `decompose_4` |
| T5 | `decompose_5` |
| T6 | `decompose_6` |

Not a model input. Essential because **~92%** of T1 frames are T6-origin.

---

## 16. Final Model B specification

```text
audio (22.05 kHz mono, 4 s excerpt)
    ↓
CQT log-mag → linear interp to t[k]=(k+0.5)×10 ms → per-bin fold norm
    ↓  [B, 1, 360, 400]
Frequency CNN (105,792 params)
    ↓  [B, 128, 400]
TCN: 4× Conv1d k=5, dilations [1,2,4,8] (328,192 params)
    ↓  [B, 128, 400]
h_t [B, 400, 128]
   / \
type [B,400,4]   pitch [B,400]  (standardized cents)
```

| Property | Value |
|----------|-------|
| Receptive field | 61 frames = **610 ms** |
| Total parameters | **434,629** (645 without pitch head for B0) |
| Excerpt length | **4 s** (400 frames) |
| Type loss (B0/B1) | Ordinary CE |
| Pitch loss (B1 only) | Smooth L1 on fold-standardized cents |

---

## 17. Final Model C specification

Replace TCN with **1-layer BiGRU** (128 hidden, bidirectional); **everything else identical**.

```text
audio → same CQT + interp + norm → [B, 1, 360, 400]
    ↓
same Frequency CNN → [B, 128, 400]
    ↓
BiGRU (1 layer, 128 hidden, bi) → [B, 400, 256]
   / \
type [B,400,4]   pitch [B,400]
```

| Property | Value |
|----------|-------|
| Context | **Full 4 s excerpt**, bidirectional |
| Total parameters | **305,221** |
| Excerpt length | **4 s** (same as B) |

Use `pack_padded_sequence` with `padding_mask`; losses use `~padding_mask & valid_target`.

---

## 18. Remaining risks before Step 7

| Risk | Mitigation |
|------|------------|
| TCN has more params than 1-layer BiGRU | Monitor val macro-F1 vs train; compare fairly by val curves, not param count alone |
| 17 performances | Grouped 5-fold CV primary |
| T6-dominated T1 | Provenance-stratified metrics |
| Ordinary CE ignores T2/T3 | Watch per-class recall; separate weighting ablation |
| Tonic metadata required | Document task scope clearly |
| CQT per-bin stats noisy with few frames | Compute from all training excerpt frames, not valid-only |
| 4 s excerpt may limit BiGRU advantage | Accept for fair B1 vs C; optional 8 s later for C-only follow-up |
| Interpolation smoothing | Accept; prefer over systematic 5 ms offset |

---

## 19. Scientific structure preserved

```text
B0  →  Does framewise type prediction work?

B0 vs B1  →  Does pitch supervision help type recognition?

B1 vs C   →  Does longer sequential context beat local TCN?

best(B1,C) vs D  →  Does auxiliary phase help?
```

Step 7 should implement **Model B and Model C shells**, alignment tests, fold metadata, and logging — then run A → B0 → B1 → C → D in order.
