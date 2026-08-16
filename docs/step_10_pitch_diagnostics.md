# Step 10 — Pitch Learnability and Frontend Diagnostic

Pitch-only diagnosis of why Experiments B1 and C fail to generalize tonic-relative cents. **No B0/B1/C retraining. No phase, class weighting, larger temporal encoder, or decoder.**

Frozen references: [`docs/step_8_b1_report.md`](step_8_b1_report.md), [`docs/step_9_c_report.md`](step_9_c_report.md).

Machine-readable outputs: [`output/pitch_diagnostics/`](../output/pitch_diagnostics/).

Reproduce (from repository root, `idtap` env):

```bash
python training/pitch_diagnostics/cnn_audit.py
python training/pitch_diagnostics/analyze_a.py
python training/pitch_diagnostics/baselines_b.py
python training/pitch_diagnostics/pyin_only.py
python training/pitch_diagnostics/stems_f.py

python training/pitch_diagnostics/train_pitch.py --head scalar --run-name scalar_abs --tiny-overfit 32 --max-epochs 100 --fold 0
python training/pitch_diagnostics/train_pitch.py --head scalar --run-name scalar_abs --all-folds --max-epochs 20
python training/pitch_diagnostics/train_pitch.py --head bins --run-name bins_abs --all-folds --max-epochs 20
python training/pitch_diagnostics/train_pitch.py --head scalar --tonic-align --run-name scalar_tonic --all-folds --max-epochs 20
python training/pitch_diagnostics/train_pitch.py --head scalar --run-name scalar_within --within-recording --fold 0 --max-epochs 20
```

**Out of scope:** trajectory multitask, Experiment D, canonicalization/label changes, overwriting `output/framewise_runs/{b0,b1,c}_*`.

---

## Executive summary

| Finding | Evidence |
|---------|----------|
| FrequencyCNN **destroys pitch location** | F: 360→90→22→**1** at `AdaptiveMaxPool2d((1,None))`; shift cosine ≈0.98 |
| Annotated pitch is **often not the CQT peak** | target/max mean 0.41, median rank **12**/360 |
| Classic **drone lock is not** the B1/C failure mode | preds less concentrated at tonic than GT |
| Fold 3 is mostly **octave + distribution shift** | B1 raw 1163¢ → octave-adj **256¢**; train↔test Wasserstein ≈900 |
| Strongest non-learned baseline is **harmonic product** (~279¢), not argmax (~904¢) or pyin (~454¢) |
| Frequency-preserving scalar helps **modestly** vs B1/C (3/5 beats mean baseline; fold 4 306 vs B1 432) but does **not** solve pitch |
| Bin CE and tonic-aligned CQT do **not** beat absolute scalar |
| Within-recording (valid-frame split) **352¢** vs grouped mean **528¢** — H3 partial |
| Cached stems do **not** unlock argmax pitch |

**Classification: `MIXED`** (frequency collapse + weak/salient harmonics in mixture + cross-performance / fold-3 shift + partial target–audio mismatch).

**Next step:** redesign the shared frontend around **frequency-local + harmonic salience** (no global F-collapse; HPS-style aggregation is already the best baseline), and audit annotation↔ridge alignment on high-error recordings—**not** “train a bigger BiGRU / Experiment D.”

---

## Competing hypotheses (H1–H6)

| ID | Claim | Verdict |
|----|-------|---------|
| **H1** | FrequencyCNN destroys pitch location | **Supported** (necessary but not sufficient) |
| **H2** | Predictions lock to drone/tonic/harmonics | **Not supported** as primary (no tonic concentration; HPS helps salience, not “always tonic”) |
| **H3** | Within-recording OK, grouped held-out bad | **Partially supported** (within 352¢ beats baseline; grouped mean 528¢) |
| **H4** | Scalar cents harder than pitch-bin CE | **Rejected** (bins worse: 583 vs 528 mean MAE) |
| **H5** | Parametric IDTAP contour ≠ acoustic ridge | **Supported as co-factor** (visibility + pyin/HPS still hundreds of ¢ on many folds) |
| **H6** | Absolute CQT vs tonic-relative target | **Rejected as primary** (tonic-align mean 543 ≥ absolute 528) |

---

## 1. FrequencyCNN frequency-dimension audit

CQT geometry (unchanged): `fmin=75`, `n_bins=360`, `bins_per_octave=72` → **16.67 cents/bin**, 5 octaves to 2400 Hz.

| Layer | F | T | C |
|-------|--:|--:|--:|
| input | 360 | T | 1 |
| Conv 7×3 | 360 | T | 32 |
| MaxPool (4,1) | **90** | T | 32 |
| Conv 5×3 | 90 | T | 64 |
| MaxPool (4,1) | **22** | T | 64 |
| Conv 3×3 | 22 | T | 128 |
| AdaptiveMaxPool (1, None) | **1** | T | 128 |

Explicit pitch position is lost after the adaptive pool. Random-input frequency shifts of 4/8/16 bins yield embedding cosine ≈ **0.986 / 0.981 / 0.990**. Full table: [`output/pitch_diagnostics/cnn_audit.json`](../output/pitch_diagnostics/cnn_audit.json).

---

## 2. Target overlays

CQT log-mag with GT pitch overlays under [`output/pitch_diagnostics/figures/overlays/`](../output/pitch_diagnostics/figures/overlays/) (T0–T3, high/low pitch, drone-heavy vs clean, B1 good/bad). Example paths:

- `type_T0_6417585554a0bfbd8de2d3ff.png` … `type_T3_…`
- `b1_good_6503e36cd9ff49d3988d0b40.png`, `b1_bad_68d85d4570785f961df2499d.png`
- `drone_heavy_65b14e207f607fb149202019.png`, `clean_6503e36cd9ff49d3988d0b40.png`

Pred vs GT cents histograms: [`figures/histograms/`](../output/pitch_diagnostics/figures/histograms/).

---

## 3–4. Target spectral visibility and vs tonic

Source: [`visibility.json`](../output/pitch_diagnostics/visibility.json) (169,150 valid frames).

| Statistic | Mean | Median (p50) |
|-----------|-----:|-------------:|
| target_energy / max_energy | 0.414 | 0.305 |
| target ±2 bins / max | 0.543 | 0.447 |
| target / tonic energy | 194† | 2.60 |
| rank among 360 bins | 30.9 | **12** |

†Mean of target/tonic is heavy-tailed (rare near-silent tonic bins); median ≈2.6 is the useful summary.

**By type (median rank):** T0 8, T1 14, T2 17, T3 **26** — ornaments sit farther from the spectral peak.

**B1 good vs bad recordings** (target/max means on curated lists): good ≈0.49–0.69, bad ≈0.18–0.46 — weaker visibility co-occurs with worse B1 pitch, but is not a clean binary gate.

**Answer:** the annotated melodic target is **often visible but rarely the strongest bin**. A pure argmax frontend cannot recover it.

---

## 5. Octave-adjusted error (B1 / C)

Source: [`octave_errors.json`](../output/pitch_diagnostics/octave_errors.json). Fold-3 support-weighted:

| Model | Fold 3 raw MAE | Octave-adj MAE (`k∈[-2,2]`) |
|-------|---------------:|----------------------------:|
| B1 | 1163 | **256** |
| C | 1290 | **310** |

Nearest-octave mass on fold 3 is dominated by `k=-1` (~70–77%). Other folds also shrink under octave adjustment but remain multi-semitone. Fold 3 is primarily an **octave / register** failure on top of a large train↔test cents shift (§6).

---

## 6. Train / test pitch distributions

Source: [`pitch_distributions.json`](../output/pitch_diagnostics/pitch_distributions.json).

| Fold | Train mean ¢ | Test mean ¢ | Wasserstein (train↔test) |
|------|-------------:|------------:|-------------------------:|
| 0 | 274 | 208 | 193 |
| 1 | 259 | 284 | 130 |
| 2 | 311 | 217 | 169 |
| **3** | **265** | **1165** | **900** |
| 4 | 240 | 763 | 528 |

Fold 3 (and to a lesser extent 4) holds out performances whose annotated register sits far above the training mean — consistent with the octave story and with trivial mean-baseline MAE ≈1084¢ on fold 3.

---

## 7. Simple CQT pitch baselines (+ pyin)

Valid frames only; same cents metrics as B1. Source: [`baselines.json`](../output/pitch_diagnostics/baselines.json), [`pyin.json`](../output/pitch_diagnostics/pyin.json).

| Method | Fold MAEs (¢) | Mean |
|--------|---------------|-----:|
| CQT argmax | 797, 812, 1412, 844, 653 | **904** |
| Tonic-neighborhood `[tonic, tonic×4]` | 885, 785, 1304, 559, 580 | **823** |
| Harmonic product (×2,×3 on log-F) | 210, 264, 409, 354, 158 | **279** |
| librosa.pyin (hop 220, fmin 75, fmax 2400; valid segments) | 362, 355, 220, 1104, 231 | **454** |

pyin was run only around valid-target segments (full concerts are tens of minutes). Unvoiced frames filled with tonic. Historical `f0s.ipynb` 60¢ gate still fails on the **current** canonical cents target.

**Interpretation:** raw peak-picking is useless; **harmonic aggregation** is the best non-learned signal and often beats B1/C. pyin is better than argmax but still hundreds of cents on average → supports mixture / H5, not “just add a bigger net.”

---

## 8. Standard pitch-only diagnostic model

```text
CQT [B,1,360,T]
  → Conv2d 1→16 (5×3), BN, ReLU, MaxPool (2,1)   # F: 360→180
  → Conv2d 16→32 (5×3), BN, ReLU
  → 1×1 mix + Conv (1×5) on time
  → Linear over F'=180 → scalar  OR  → n_bins logits
```

**No `AdaptiveMaxPool2d((1,None))`.** Parameter counts: scalar **14,486**; bins (~120 classes) ~37k (asserted &lt;80k). Protocol: grouped 5-fold, seed 42, 4 s excerpts, valid-frame loss, AdamW 1e-3, wd 1e-4, batch 8, ≤20 epochs, patience 5, select on **val pitch MAE**. Checkpoints only under `output/pitch_diagnostics/runs/`.

Tiny-overfit (32 excerpts): train MAE 351 → **~64¢** ([`tiny_overfit_scalar_abs.json`](../output/pitch_diagnostics/tiny_overfit_scalar_abs.json)) — loss/targets not broken.

---

## 9. Diagnostic C — frequency-preserving scalar (absolute CQT)

Run: `scalar_abs`.

| Fold | Scalar MAE | Mean-pitch baseline | Beats? | B1 | C |
|------|-----------:|--------------------:|:------:|---:|--:|
| 0 | 353.8 | 259.7 | No | 396 | 598 |
| 1 | 470.2 | 477.6 | **Yes** | 483 | 533 |
| 2 | 417.6 | 422.9 | **Yes** | 473 | 476 |
| 3 | 1090.5 | 1083.8 | No | 1163 | 1290 |
| 4 | **306.0** | 573.4 | **Yes** | 432 | 431 |
| **mean** | **527.6** | — | **3/5** | ~589 | ~666 |

Promising relative to B1/C on several folds (especially fold 4), but **does not** beat the train-mean baseline on most folds convincingly, and fold 3 remains unusable. Frequency preservation helps; it does not alone make pitch learnable across performances.

---

## 10. H4 — scalar vs pitch-bin CE (same trunk, absolute CQT)

Train cents clipped to train 0.5–99.5% quantiles; width **25¢** chosen after reporting candidates (10/20/25). Fold 0: span ≈3000¢ → **120** classes, mean support/bin ≈1241, 0 empty bins.

| Fold | Bins MAE | Scalar MAE |
|------|---------:|-----------:|
| 0 | 440.7 | **353.8** |
| 1 | 645.6 | **470.2** |
| 2 | **416.3** | 417.6 |
| 3 | **969.6** | 1090.5 |
| 4 | 443.5 | **306.0** |
| **mean** | **583.2** | **527.6** |

Bins beat baseline on 3/5 folds but **lose to scalar overall**. H4 rejected.

---

## 11. Diagnostic D — tonic-aligned vs absolute CQT

Same scalar model; CQT shifted so `fundamental_hz` maps to a fixed reference bin (**pad/crop, no wrap**). Run: `scalar_tonic`.

| Fold | Tonic-align | Absolute |
|------|------------:|---------:|
| 0 | **343.1** | 353.8 |
| 1 | 685.2 | **470.2** |
| 2 | **370.9** | 417.6 |
| 3 | **854.8** | 1090.5 |
| 4 | 462.8 | **306.0** |
| **mean** | **543.4** | **527.6** |

Tonic alignment helps fold 3 (register) but **hurts** fold 1/4 enough that the mean is worse. H6 is not the primary bottleneck.

---

## 12. Diagnostic E — within-recording vs grouped

Best grouped config: **absolute scalar** (`scalar_abs`, mean test MAE **527.6¢**, beats mean baseline on **3/5** folds).

Within-recording protocol: train on the **first 70% of valid frames** per recording, test on the **last 30%** (annotation order), all 17 recordings. Absolute-time late windows were tried first and left **13/17 recordings with zero late valid frames**; archived under `runs/scalar_within_abs_time/` and discarded as the primary E metric.

| Split | Test MAE (¢) | Mean-pitch baseline | Beats baseline? | Support |
|-------|-------------:|--------------------:|:---------------:|--------:|
| Within (`scalar_within`) | **352.4** | 477.6 | **Yes** | 50,753 (17/17 recs) |
| Grouped `scalar_abs` mean | 527.6 | (per fold) | 3/5 folds | — |
| Grouped fold 0 (held-out groups) | 353.8 | 259.7 | No | 10,790 |

Per-recording within MAE spans ~79–800¢ (worst: `692ed7e6` 800¢; best short pieces ~80–160¢). Train MAE reached ~190¢ with best val ~331¢ — the model can exploit **same-recording timbre/register**, then still lose ~150¢+ on the late valid slice.

**H3:** **partially supported.** Within-recording clearly beats its own mean baseline and the grouped CV **mean**, but absolute error remains multi-semitone and is not uniformly “easy” on every recording. Cross-performance generalization is a real bottleneck; it is not the only one (visibility / HPS / fold-3 octave remain).

---

## 13. Drone-locking analysis

Source: [`drone_lock.json`](../output/pitch_diagnostics/drone_lock.json).

Mean fraction of frames within ±50¢ of tonic:

| | B1 pred | C pred | GT |
|--|--------:|-------:|---:|
| near tonic | ~0.053 | ~0.063 | ~0.154 |

Predictions are **less** tonic-concentrated than ground truth. Fractions near ±1200 / ±700 are also low for preds. **Reject classic drone-lock** as the explanation for B1/C pitch failure.

---

## 14. Harmonic-energy analysis

Per recording, energy near `2f,3f,4f` relative to `f` (inside CQT range): median recording-means of `2f/f` ≈ **1.8**, `3f/f` ≈ **0.97**, `4f/f` ≈ **0.44** (means are heavy-tailed). Combined with HPS ≫ argmax, **fundamental-weak / harmonic-strong** frames are common enough that a frequency-local head should aggregate harmonics—not only the annotated bin.

---

## 15. Stem comparison (Diagnostic F)

Coverage: **17/17** recordings have source + denoised + vocals under `output/denoised/` ([`stem_coverage.json`](../output/pitch_diagnostics/stem_coverage.json)). Subset of 4 vocal pieces re-extracted CQT on the 10 ms grid ([`stems.json`](../output/pitch_diagnostics/stems.json)); no full feature-cache rebuild; no stem neural CV.

| Recording | Argmax MAE source | denoised | vocals | target/max source→vocals |
|-----------|------------------:|---------:|-------:|--------------------------|
| 64175855… | 1700 | 1701 | 1540 | 0.217→0.264 |
| 645ff354… | 711 | 711 | **1633** | ~0.57 |
| 6491d48d… | 1033 | 1032 | 1083 | 0.437→0.486 |
| 6503e348… | **234** | 234 | **3011** | 0.465→0.889† |

†Vocals can raise energy at the target bin while **destroying** argmax pitch (spurious peaks). Stems do **not** consistently unlock recoverable pitch → weak support for pure `SOURCE_SEPARATION_PROBLEM`.

---

## 16. External F0 baseline

`librosa.pyin` (already in the project; no new deps). Mean MAE **454¢** (§7). Does **not** substantially beat a well-tuned spectral baseline (HPS 279¢) and remains far from usable transcription. Supports mixture/alignment hypotheses more than “our temporal model is uniquely broken.”

---

## 17–18. Per-fold and per-recording MAE

Per-fold tables: §§7, 9–11. Per-recording pitch MAE for baselines / neural runs live in each run’s `fold_*/result.json` and in `baselines.json` / `pyin.json` `per_recording` blocks. B1/C per-recording contours remain under `output/framewise_runs/{b1,c}_*/fold_*/eval/pitch_contours/` (frozen).

---

## 19. Representative success / failure plots

- Overlays: §2.
- Histograms of pred vs GT cents for B1/C: `figures/histograms/fold*_*.png`.
- Fold 3 histograms show the octave-scale offset clearly (`fold3_b1_*`, `fold3_c_*`).

---

## 20. Evidence for / against H1–H6

Summarized in the opening table. Short form:

1. **H1 yes** — collapse is real; fixing it helps but does not solve.
2. **H2 no** — not tonic-locked.
3. **H3 partial** — within 352¢ vs grouped mean 528¢; still multi-semitone.
4. **H4 no** — bins worse.
5. **H5 yes as co-factor** — visibility + pyin/HPS errors.
6. **H6 no as primary** — tonic-align not a clear win.

---

## 21. Final classification

**`MIXED`**

Dominant ingredients:

1. **Frontend frequency collapse** (H1) — must be fixed in any shared encoder.
2. **Salience / harmonic structure in mixture** — HPS ≫ argmax; stems don’t cleanly save argmax.
3. **Cross-performance + register shift** (fold 3/4) — octave ambiguity and distribution mismatch.
4. **Partial target–audio mismatch** (H5) — parametric contour often not the spectral peak; external F0 also poor.

Not a pure `TONIC_EQUIVARIANCE_PROBLEM` or pure `SOURCE_SEPARATION_PROBLEM`.

---

## 22. Recommendation (next modeling step)

**Do next:** redesign the pitch frontend to (a) **preserve frequency position through the CNN** (drop `AdaptiveMaxPool2d((1,None))`), and (b) add a **lightweight harmonic / salience path** inspired by the HPS baseline (still no large temporal model). Parallel: spot-check high-error recordings where target rank is terrible (annotation vs acoustic ridge).

**Do not:** train Experiment D / bigger BiGRU / phase / class weighting / decoder yet. Scalar absolute CQT is the best of the Step 10 neural variants, but it is not “solved.”

Gate from Step 9 (`INVESTIGATE_PITCH_FRONTEND`) stands; the frontend investigation now has a concrete direction: **frequency-preserving + harmonic salience**, not tonic-shift alone.
