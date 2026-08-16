> **Update (Step 6.5):** Parameter counts, timing alignment, TCN-first experiment order, and Model B/C specs were corrected in [`docs/step_6_5_architecture_corrections.md`](step_6_5_architecture_corrections.md). That document supersedes conflicting technical details below.

# Step 6 — Framewise Temporal Model Architecture Design

Design specification for the first continuous-audio model mapping waveform to canonical four-primitive framewise targets. **No model was trained. No code was implemented.**

Prerequisites: Step 5 framewise targets and Step 5.5 boundary audit ([`docs/step_5_5_boundary_learnability_report.md`](step_5_5_boundary_learnability_report.md)).

---

## Executive summary

**Baseline v1 recommendation:** a small **CQT → frequency-CNN → BiGRU** encoder with dual heads for **4-class trajectory type** and **tonic-relative pitch (cents)**, trained on **8 s padded excerpts** aligned to the existing **10 ms** target grid, evaluated with **grouped 5-fold cross-validation** on `performance_group_id`.

```text
audio [B, T_samples]
    ↓  CQT, hop ≈ 10 ms, F = 360
spec  [B, 1, 360, T_frames]
    ↓  2D CNN (pool frequency only)
embed [B, 128, T_frames]
    ↓  BiGRU (2 × 128, bidirectional)
h_t   [B, T_frames, 256]
   / \
  /   \
type  pitch
[B,T,4] [B,T]
```

Estimated parameters: **~450k** (vs ~390k for the existing segment CNN).

Phase is **excluded** from baseline v1; reserved for Experiment D.

---

## 1. Audit of the existing isolated-trajectory CNN

Source: [`training/models.py`](../training/models.py), [`training/spec_dataset.py`](../training/spec_dataset.py), [`training/train_cnn.py`](../training/train_cnn.py), [`dataset/export_denoised_cnn_dataset.py`](../dataset/export_denoised_cnn_dataset.py).

### Task formulation

```text
known 1 s trajectory segment (pre-cut clip)
    →
single trajectory class (5-way: T0–T3 + silent)
```

The model never sees continuous audio, never predicts pitch, and never predicts framewise labels.

### Audio preprocessing

| Parameter | Value |
|-----------|------:|
| Sample rate | 22,050 Hz |
| Clip duration | 1.0 s (padded/truncated) |
| Channels | mono |
| Source | raw / denoised / vocals WAV stems |

### Spectrogram representation

| Parameter | Value |
|-----------|------:|
| Transform | librosa CQT magnitude |
| `fmin` | 75 Hz |
| `fmax` (implicit) | 2,400 Hz |
| `bins_per_octave` | 72 |
| `n_bins` | 360 |
| `hop_length` | 512 → **~23.2 ms/frame** |
| Log scaling | log10 magnitude, zeros replaced by minimum nonzero |
| Display | per-clip min–max → **magma RGB** uint8 |
| PNG size | **176 × 360** (time stretched 4× bilinear) |

### Normalization

Per-clip min–max normalization to [0, 1] before colormap. **Not** comparable across clips in absolute amplitude; acceptable for fixed-length segment classification, problematic for variable-length framewise regression.

### CNN architecture (`TrajectoryCNN`)

| Layer | Output shape (NCHW) |
|-------|---------------------|
| Conv2d(3→32, 3×3) + BN + ReLU | [B, 32, H, W] |
| MaxPool2d(2) | [B, 32, H/2, W/2] |
| Conv2d(32→64, 3×3) + BN + ReLU | |
| MaxPool2d(2) | [B, 64, H/4, W/4] |
| Conv2d(64→128, 3×3) + BN + ReLU | |
| MaxPool2d(2) | [B, 128, H/8, W/8] |
| Conv2d(128→256, 3×3) + BN + ReLU | |
| AdaptiveAvgPool2d(1,1) | [B, 256, 1, 1] |
| Dropout(0.3) + Linear(256→C) | [B, C] |

Input tensor: **[B, 3, 360, 176]** (RGB PNG).  
**~390k parameters** (4-class head slightly smaller than 5-class).

### Pooling

Aggressive **2×2 max pooling** on both frequency and time, then global average pool → one vector per clip. Temporal structure inside the 1 s window is largely collapsed.

### Classification head

Single linear layer → softmax over C classes (default C=5 including `silent`; four-class runs drop silent via `--exclude-labels`).

### Loss

Weighted cross-entropy. Class weights = `N / (C × count_c)` computed on the training fold ([`compute_class_weights`](../training/train_cnn.py)).

### Class balancing

Inverse-frequency loss weighting only. No balanced sampling. Macro-F1 on validation for early stopping.

### Train/validation/test split

| Mode | Strategy |
|------|----------|
| Default `--split-by label` | Stratified 80/10/10 by class |
| Recommended `--split-by piece_id` | Group by recording (`piece_id`), ~80/10/10 pieces |
| Seed | 42 |
| Epochs / patience | 50 / 10 |
| Batch size / LR | 16 / 1e-3 |
| Optimizer | Adam |
| Augmentation | Mild ColorJitter on RGB only |

**Leakage note:** `piece_id` grouping is necessary but **not identical** to canonical `performance_group_id` (multiple transcriptions can share audio). Framewise training must use canonical grouped splits (§19).

### Current performance

From [`README.md`](../README.md) A/B runs on ~17 recordings, ~4.8k–5k segment examples:

| Metric | Range across raw/denoised/vocals |
|--------|----------------------------------|
| Test accuracy | ~0.34 |
| Test macro-F1 | ~0.23–0.31 |

Near chance for 5 classes; modestly above chance for 4-class. Denoising has not shown a consistent benefit on this task.

### Implications for framewise model

The baseline proves that **segment-level type classification from CQT is hard** even with oracle segmentation. The framewise model must solve a strictly harder problem (continuous audio + pitch). Comparisons should use **aggregated framewise predictions over GT intervals** (§19), not raw segment accuracy alone.

---

## 2. Constraints from Step 5.5

| Finding | Architectural consequence |
|---------|---------------------------|
| Frame balance 34.6 / 50.0 / 8.1 / 7.3% | Monitor macro-F1; do not rely on accuracy |
| 92.3% of T1 frames from Type 6 | Report T1 metrics by provenance |
| 10 ms grid adequate (99.5% T1 primitives ≥ 3 frames) | Keep 10 ms targets |
| Many T1\|T1 boundaries, zero type change | Encoder must model **local pitch motion** |
| Phase uncorrelated with T1 geometry | **Exclude phase from baseline v1** |
| T6 internal boundaries: smooth pitch, dynamics differ | Temporal context required; pitch head helps |

---

## 3. Input audio representation

### Candidates compared

| Representation | Pitch resolution | Time @ 22.05 kHz | Drone robustness | Cost | 10 ms alignment | Baseline match |
|----------------|-----------------:|-----------------:|------------------|------|-----------------|----------------|
| **CQT log-mag** | Excellent (72 bins/oct) | tunable via hop | Moderate (harmonics visible) | Medium | **hop=220 → 9.98 ms** | Strong (same fmin, bins) |
| Mel spectrogram | Good | tunable | Moderate | Low | hop=220 works | Weaker vs baseline |
| STFT linear | Poor log spacing | tunable | Moderate | Low | hop=220 works | Different from baseline |
| Raw waveform encoder | Learned | sample-level | Unknown | High | needs striding | No comparability |

### Recommendation for v1: **CQT log-magnitude, single channel**

Match the baseline CQT definition:

```python
fmin=75, n_bins=360, bins_per_octave=72, sr=22050
hop_length=220        # 220/22050 ≈ 9.98 ms ≈ target hop
```

**Changes from baseline export (intentional):**

1. **`hop_length=220`** instead of 512 — aligns CQT frames to the 10 ms target grid (512 ms hop was for 1 s PNG export only).
2. **Single-channel log10 magnitude** — no per-clip magma RGB normalization; use **fixed corpus statistics** (mean/std from training performances) for stability across variable-length excerpts.
3. **Keep full time dimension** — no PNG resize/stretch.

**Why not waveform?** Dataset is small (~17 performances); a waveform front-end adds parameters and training difficulty without proven benefit for log-spaced pitch targets.

**Why not mel?** CQT already works in the baseline and matches logarithmic pitch semantics.

---

## 4. Temporal context / receptive field

### Primitive duration context (Step 5.5)

| Statistic | Value |
|-----------|------:|
| Median primitive | 132 ms |
| T1 median | 109 ms |
| p95 primitive | 726 ms |

### Candidate receptive fields

| Context | Covers | Verdict |
|---------|--------|---------|
| ±50 ms (100 ms) | < 1 median primitive | Too local for T2/T3 sloped shapes |
| ±100 ms (200 ms) | ~1.5× T1 median | Minimum useful |
| **±250 ms (500 ms)** | ~2–4× T1 median, part of p95 | **Good v1 default** |
| ±500 ms (1 s) | full p95 | Useful but increases excerpt edge effects |
| ~1 s+ | musical phrase scale | Defer to v2 |

### Recommendation

- **Training excerpt length:** **8 s** (default), uniform random **4–12 s** optional after v1 stabilizes.
- **BiGRU** over the full padded excerpt provides **bidirectional** context ≥ excerpt length (effective global within window).
- **Effective v1 receptive field:** within-excerpt **full 8 s backward and forward** via BiGRU; local inductive bias from CNN kernels (~30–50 ms) + recurrent integration.

For the **temporal-CNN ablation (Experiment B)**, stack three `Conv1d` layers with kernel 5 and dilations `[1, 2, 4]` on the CNN embedding sequence → receptive field ≈ **61 frames ≈ 610 ms** without recurrence.

---

## 5. Architecture comparison

| Criterion | Temporal CNN | **CRNN (CNN+BiGRU)** | Conformer |
|-----------|:------------:|:--------------------:|:---------:|
| Dataset size (~17 perf) | Good | **Good** | Risky (overfit) |
| 10 ms resolution | Easy | **Easy** | Needs careful subsampling |
| Local pitch motion | Dilated conv | **CNN + recurrence** | Good but heavy |
| Parameters | ~250k | **~450k** | ~2M+ |
| Overfitting risk | Low | **Moderate** | High |
| Debugging | Easy | **Moderate** | Hard |
| Variable-length | Easy | **Easy** | Easy |
| Extensibility | Limited context | **Good** | Best long-term |

### Recommendation

**Primary v1: CNN + BiGRU (Experiment C).**  
**Ablation: Temporal CNN with dilated conv1d (Experiment B).**  
**Defer Conformer** until CRNN demonstrates framewise training works.

---

## 6. CNN + BiGRU evaluation

### Why BiGRU is justified

| Need | CNN alone | + BiGRU |
|------|-----------|---------|
| Same-type T1\|T1 boundaries | Weak (needs long RF stacking) | **Integrates velocity context** |
| Sloped T2/T3 shapes | Needs wide dilated RF | **Sequence memory** |
| Drone-dominated background | Local spectral only | **Temporal consistency** |
| Offline transcription | N/A | Bidirectional OK |

### Why not auto-accept

BiGRU adds ~200k parameters and can overfit on 17 performances. Experiment B (no GRU) isolates whether recurrence is necessary.

---

## 7. Temporal resolution preservation

### Target grid

```text
frame_time_s[k] = (k + 0.5) × 0.01 s
```

### Encoder grid (recommended)

```text
CQT hop = 220 samples @ 22050 Hz → 9.98 ms/frame
T_cqt ≈ T_targets   (within ±1 ms; exact alignment at load time)
```

### Network time stride

```text
T_in  (CQT frames)
  → CNN: pool ONLY frequency axis → T_mid = T_in
  → BiGRU: no downsampling      → T_out = T_in
```

**No temporal max-pooling.** Frequency max-pool `(8, 1)` twice: 360 → 45 → ~6, then `AdaptiveMaxPool2d((1, None))` → **[B, C, 1, T]**.

### Alignment to 10 ms targets

At dataset load time:

1. Compute CQT frame center times: `t_cqt[k] = k × hop / sr`.
2. For each target frame at `t_target`, nearest-neighbor index `k = round(t_target × sr / hop)`.
3. Store alignment indices in the batch (or precompute on export).

Alternative: interpolate target sequences to CQT times — nearest-neighbor is sufficient for v1 given ~10 ms match.

---

## 8. Exact layer specification (recommended v1)

### Hyperparameters

| Symbol | Value |
|--------|------:|
| F (CQT bins) | 360 |
| CQT hop | 220 samples (~9.98 ms) |
| Input channels | 1 |
| CNN channels | 32 → 64 → 128 |
| BiGRU hidden | 128 per direction |
| BiGRU layers | 2 |
| Dropout | 0.3 (CNN + GRU inter-layer) |
| Type classes | 4 (T0–T3) |

### Layer-by-layer

```text
Input waveform:
  x: [B, N_samples]

CQT frontend (non-learned, librosa):
  spec: [B, 1, 360, T]

Block 1:
  Conv2d(1, 32, kernel=(7,3), padding=(3,1))
  BatchNorm2d(32), ReLU
  MaxPool2d((4,1))                     # 360 → 90 bins, time unchanged

Block 2:
  Conv2d(32, 64, kernel=(5,3), padding=(2,1))
  BatchNorm2d(64), ReLU
  MaxPool2d((4,1))                     # 90 → 22 bins

Block 3:
  Conv2d(64, 128, kernel=(3,3), padding=(1,1))
  BatchNorm2d(128), ReLU
  AdaptiveMaxPool2d((1, None))         # → [B, 128, 1, T]

Squeeze + transpose:
  e: [B, T, 128]

Temporal encoder:
  BiGRU(input=128, hidden=128, layers=2, bidirectional=True, dropout=0.3)
  h: [B, T, 256]

Heads (per time step):
  type_logits = Linear(256, 4)(h)      # [B, T, 4]
  pitch_pred  = Linear(256, 1)(h)      # [B, T]
```

### Parameter estimate

| Component | ~Parameters |
|-----------|------------:|
| CNN blocks | 120k |
| BiGRU (2-layer, bi) | 530k |
| Heads + BN | 5k |
| **Total** | **~450–650k** |

If overfitting appears in Step 7 training, reduce to **1-layer BiGRU** (~350k total).

### Temporal CNN ablation (Experiment B)

Replace BiGRU with:

```text
Conv1d(128, 128, kernel=5, dilation=1, padding=2)
Conv1d(128, 128, kernel=5, dilation=2, padding=4)
Conv1d(128, 128, kernel=5, dilation=4, padding=8)
→ [B, T, 128] → same heads (Linear 128→4, 128→1)
```

~280k parameters total.

---

## 9. Variable-length batching

### Padding strategy

| Tensor | Shape | Pad value |
|--------|-------|-----------|
| Waveform | `[B, N_max]` | 0 |
| CQT spec | `[B, 1, 360, T_max]` | log-floor constant |
| Type targets | `[B, T_max]` | -1 (ignore) |
| Pitch targets | `[B, T_max]` | nan |
| `valid_target` | `[B, T_max]` | False |
| `padding_mask` | `[B, T_max]` | True = padded |

### Two masks (do not conflate)

| Mask | Meaning | Used for |
|------|---------|----------|
| **`padding_mask`** | Batch padding beyond excerpt end | GRU `pack_padded_sequence`; exclude from all losses |
| **`valid_target`** | No canonical supervision (silent, masked types) | Exclude from type/pitch loss; **encoder still sees audio** |

Loss mask = `~padding_mask & valid_target`.

### GRU handling

Use `pack_padded_sequence` with true sequence lengths (in CQT frames). Pad outputs back to `T_max` for frame-aligned heads.

---

## 10. Training excerpt sampling (design only)

### Goals

- Sufficient valid supervision per batch
- Surrounding masked/audio context preserved
- No split leakage
- Avoid class-imbalance collapse

### Recommended sampler

```text
1. Choose recording r from train performance groups (uniform over recordings)
2. Choose lane ℓ (default 0:0 for v1 primary lane)
3. Choose anchor frame index k where valid_target[k] == True
4. Set excerpt window [t_k - W/2, t_k + W/2], default W = 8 s, clip to audio bounds
5. Load waveform + targets for that window
6. Accept if valid_frame_fraction ≥ 0.15 (tunable)
7. If rejected, retry (max 20 attempts)
```

### Parameters

| Parameter | v1 value |
|-----------|----------|
| Default excerpt duration | **8 s** |
| Variable length | optional 4–12 s uniform after baseline works |
| Min valid fraction | **15%** |
| Lane | `0:0` primary; extend to all lanes later |
| Recording filter | training `performance_group_id` only |

### Do not

- Drop majority-masked recordings entirely (they provide real context)
- Sample frames uniformly (wastes batch budget on long silent spans)
- Oversample T1 without reporting — if done, log provenance-specific metrics

---

## 11. Pitch target representation

### Options

| Formulation | Pros | Cons |
|-------------|------|------|
| **A. Absolute log2 Hz** | Direct from targets | Wide range across ragas/performers |
| **B. Cents above tonic** | Musically meaningful; narrower range | Requires tonic at inference |
| C. Log2 ratio to tonic | Linear in log domain | Same metadata dependency |

### Recommendation: **B. Cents above tonic**

```text
pitch_cents = 1200 × log2(f / fundamental_hz)
```

Compute at load time from `pitch_log2_hz` in `.npz` and `raga.fundamental_hz` in the recording doc.

**Tonic at inference:** For this project, raga/tonic is **known metadata** (same as IDTAP annotation context). Treat tonic as **given input metadata**, not a predicted quantity, for v1. Document that fully blind transcription would eventually need tonic estimation — out of scope.

Store both in evaluation logs for conversion back to Hz.

---

## 12. Type head

Output: **`[B, T, 4]`** logits for T0–T3 only (no silent class).

### Loss

**Inverse-frequency weighted cross-entropy** on valid frames — same principle as baseline [`compute_class_weights`](../training/train_cnn.py), computed on **training-frame counts** (not primitive counts):

```text
w_c = N_valid / (4 × count_c)
L_type = CrossEntropy(type_logits, type_target, weight=w, ignore_index=-1)
```

### Why not focal / aggressive balancing

Frame distribution (34.6 / 50.0 / 8.1 / 7.3%) is imbalanced but not extreme. Inverse-frequency weighting is the simplest baseline-consistent choice. Monitor **macro-F1** and **per-class recall** for T2/T3.

---

## 13. Pitch head

Output: **`[B, T]`** scalar cents.

### Loss

**Smooth L1 (Huber)** on valid frames:

```text
L_pitch = SmoothL1Loss(pitch_pred, pitch_target_cents)
```

Evaluate in **cents** (interpretable). SmoothL1 reduces sensitivity to occasional annotation endpoint spikes vs pure MSE.

### Metrics

| Metric | Notes |
|--------|-------|
| MAE (cents) | Primary |
| Median AE (cents) | Robust |
| % within ±10 / ±25 / ±50 cents | Threshold accuracies |
| MAE by trajectory type | Especially T1 vs T0 |

Do **not** add dp/dt loss in v1.

---

## 14. Multitask objective

```text
L = λ_type × L_type + λ_pitch × L_pitch
```

Both terms averaged over **valid, non-padded** frames only.

### Initial weights

**λ_type = 1.0, λ_pitch = 1.0** with pitch targets scaled to ~zero mean, unit variance on the training corpus (standardize cents per batch or globally).

### Scale matching rationale

- Standardized cents puts pitch error on a scale comparable to cross-entropy (~1–3 nats).
- If training logs show `L_pitch >> L_type` consistently, reduce λ_pitch or increase type weight.
- If type loss dominates and pitch MAE stalls, increase λ_pitch to 2–5.

### Diagnostics

| Symptom | Action |
|---------|--------|
| Type acc ↑, pitch MAE flat | Increase λ_pitch |
| Pitch MAE ↓, macro-F1 flat | Increase λ_type or check class weights |
| Both stall | Reduce model size / add regularization |

---

## 15. Phase ablation (Experiment D — design only)

Add after baseline B/C succeeds:

```text
phase_head: Linear(256, 1) + Sigmoid → [B, T]
L_phase = MSE(phase_pred, phase_target) on valid frames
L = L_type + λ_pitch L_pitch + λ_phase L_phase
```

### Recommended ablation protocol (most interpretable)

**Uniform phase loss** on all valid frames with **λ_phase = 0.25** (auxiliary, per Step 5.5 Option B).

Report separately:

- All types
- T1 provenance breakdown
- Same-type boundary proxy metrics (diagnostic, not trained)

Do **not** downweight T6-origin T1 in the first ablation — keep one clean comparison first; stratified weighting is a follow-up.

---

## 16. Evaluation plan

### A. Frame-level type

- Accuracy, macro-F1
- Per-class precision / recall / F1
- Confusion matrix

### B. T1 by provenance

Separate metrics for frames where ground-truth primitive origin is:

- raw T1
- Type 4 decomposition
- Type 5 decomposition
- Type 6 decomposition

(Provenance from `primitives/*.json` joined via `primitive_id` in `.npz`.)

### C. Pitch

- MAE / median AE in cents (valid frames)
- % within ±10 / ±25 / ±50 cents
- By type and by T1 provenance

### D. Per-trajectory aggregation (fair CNN comparison)

For each ground-truth canonical primitive interval `[start, end)`:

```text
type_pred = mode(frame_type_predictions)
type_vote = argmax(mean(softmax logits))
pitch_pred = mean(frame_pitch_predictions)
```

Compare `type_pred` to `canonical_type` → **trajectory-level type accuracy** comparable to segment CNN.

Compare pitch MAE aggregated per primitive.

### E. Boundary diagnostics (no training objective)

On validation recordings, measure whether changes in predicted type, pitch, or finite-difference pitch recover canonical boundaries. Report as analysis only (Step 5.5 oracle-style, but on model outputs).

---

## 17. Controlled experiment sequence

| Exp | Model | Task | Answers |
|-----|-------|------|---------|
| **A** | Existing segment CNN | GT segment → type | Current ceiling with oracle segmentation |
| **B** | CQT + temporal CNN | continuous → type + pitch | **Q1:** does framewise training work? |
| **C** | CQT + CNN + BiGRU | continuous → type + pitch | **Q2:** does sequence context help? |
| **C′** | Best of B/C, type-only head | continuous → type | **Q3:** pitch supervision effect |
| **D** | Best of B/C + phase head | + auxiliary phase | **Q4:** phase ablation |

Change **one axis per experiment**. Same splits, same audio stem (start with **denoised** or **raw** — pick one and hold fixed), same excerpt sampler.

### Fair comparison against segment CNN (§19)

| Comparison | Fair? |
|------------|-------|
| Segment CNN type acc vs framewise **aggregated** type acc on same GT intervals | **Yes** |
| Segment CNN type acc vs framewise **frame** accuracy | **No** (different tasks) |
| Segment CNN (5-class with silent) vs framewise 4-class | **No** unless silent mapped consistently |
| Pitch metrics vs segment CNN | **N/A** (segment CNN has no pitch head) |

---

## 18. Split strategy / leakage

Use canonical grouped splits from [`dataset/canonical/splits.py`](../dataset/canonical/splits.py):

- Atomic unit: **`performance_group_id`**
- Never split frames, trajectories, or excerpts across groups

### Recommendation

**Primary: grouped 5-fold cross-validation** (`grouped_kfold_by_performance`, k=5, seed=42).

Fixed 60/20/20 split is acceptable for debugging but **high variance** with ~17 recordings.

Do not rerun splits in Step 6 — consume existing manifests under `output/canonical/v1/splits/`.

---

## 19. Silence / activity

- `valid_target == False` regions: **no type/pitch loss**
- Encoder **may** process masked audio as context
- **No fifth class** for silence
- Do not infer acoustic inactivity from IDTAP Silent trajectories

---

## 20. Key risks and assumptions

| Risk | Mitigation |
|------|------------|
| Small corpus (17 recordings) | Grouped k-fold; small model; early stopping on macro-F1 |
| Drone/accompaniment dominance | Denoised/vocals stem ablation later; start one stem |
| T6-inflated T1 metrics | Provenance-stratified evaluation |
| T1\|T1 segmentation without phase | Monitor boundary diagnostics; Experiment D |
| Tonic metadata dependency | Document; cents→Hz uses recording fundamental |
| CQT hop vs 10 ms mismatch | hop=220 (~9.98 ms); nearest-neighbor align |
| Overfitting BiGRU | Drop to 1 layer; increase dropout |
| Log-magnitude vs RGB baseline | Accept representation shift; note in Experiment A comparison |

---

## 21. Proposed baseline (concise)

```text
audio (22.05 kHz mono)
    ↓
CQT log-magnitude [1, 360, T], hop=220 (~10 ms)
    ↓
3× Conv2d blocks (freq pool only) → [128, T]
    ↓
2-layer BiGRU (128 hidden, bidirectional) → [256, T]
    ↓
h_t every ~10 ms
   / \
  /   \
type CE (weighted)   pitch SmoothL1 (cents vs tonic)
[B, T, 4]            [B, T]
```

| Dimension | Value |
|-----------|------:|
| F | 360 |
| CNN channels | 128 |
| H (BiGRU out) | 256 |
| Excerpt | 8 s padded |
| Targets | 4-class type + cents above tonic |
| Loss | λ_type=1, λ_pitch=1 |
| Split | grouped 5-fold |
| Params | ~450k |

---

## 22. Next step (Step 7 — not started)

Implementation order when approved:

1. `training/framewise_dataset.py` — excerpt loader from `.npz` + audio refs
2. `training/framewise_model.py` — architecture above
3. `training/train_framewise.py` — multitask loop with dual masks
4. Evaluation script with T1 provenance breakdown
5. Run Experiments B → C → C′ → D

**Do not begin until this design is reviewed.**
