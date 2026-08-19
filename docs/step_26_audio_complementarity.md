# Step 26 — Does Audio Add Information Beyond CREPE for Oracle-Boundary Trajectory Typing?

Steps 21-25 tested the pitch-contour-only branch exhaustively: fixed-time deltas, normalized phase contours, velocity, class-balanced training, canonical template argmin, template residuals, and contour+template fusion. Step 25 closed that branch on a null result — template residuals add nothing CREPE's own `q(x)+dq/dx` doesn't already carry (macro F1 0.3107 → 0.3115, indistinguishable). CREPE contour alone plateaus around macro F1 ≈0.31 on oracle boundaries, while the *same* representations on clean oracle pitch reach ≈0.82. Canonical trajectory geometry is highly informative; CREPE-derived pitch contour does not preserve enough of it.

This step asks a different, larger question, not another contour-feature variant: **given the correct trajectory boundaries, does the original audio contain trajectory-type information that CREPE's pitch contour discards?** GT `start_s`/`end_s` boundaries are used throughout — this remains an isolated trajectory-*typing* question, not a segmentation or context question.

Frozen exactly from Steps 22-25: CREPE extraction, `MIN_SPAN_CENTS`, phase normalization, `N=64`, `q(x)`/`dq/dx`, `ContourCNN`, grouped 5-fold manifest, canonical labels, Step 23 B1's training protocol (class-balanced sampler, unweighted cross entropy, Adam lr=1e-3/wd=1e-4, batch 32, max 100 epochs / patience 15, seed 42+fold).

Machine-readable outputs: `output/shape_classification/step26/results.json` (plus one `{a0,a1,a2,a4}_full.json` per condition, each with full per-fold predictions).

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.shape_classification.step26_features        # acoustic patch cache
python -m training.shape_classification.step26_experiments a0   # CREPE only (reused)
python -m training.shape_classification.step26_experiments a1   # audio only
python -m training.shape_classification.step26_experiments a2   # CREPE + audio fusion
python -m training.shape_classification.step26_experiments a4   # optional: oracle + audio
python -m training.shape_classification.step26_experiments      # analysis pass (no retraining)
```

(Each condition is trained and saved independently — deliberately, since a single combined run in this environment was killed mid-A2 by the platform's background-process limit. Nothing was lost: A0/A1 had already checkpointed, and A2/A4 restarted cleanly from the split entry points.)

---

## Executive summary

| Finding | Evidence |
|---|---|
| A0 reproduces Step 23 B1 / Step 25 F0 exactly: macro F1 0.3107, grouped mean 0.3234±0.0893 | §7 |
| A1 (audio only, 4,580 params) is **not independently strong**: pooled macro F1 0.3186 (barely above A0) but grouped mean 0.2911±0.0496 — *below* A0's grouped mean — and it collapses Sloped-start almost entirely (recall 7.9%) | §7 |
| **A2 (CREPE+audio, 7,412 params) clearly beats A0** on both pooled (0.3668 vs. 0.3107, Δ+0.0561) and grouped-fold mean (0.3500±0.0770 vs. 0.3234±0.0893, Δ+0.0266) | §7, §11 |
| But the gain is substantially a **decision-boundary shift toward the majority class**: Cosine recall 33.1%→72.7% (F1 +0.286) while Sloped-start recall **collapses 49.6%→8.3%** (F1 −0.090); total Cosine↔sloped cross-confusion mass drops (2,714→1,245) almost entirely because Cosine stops being mis-called sloped, not because sloped classes become more separable from Cosine | §14, §15 |
| Fold consistency is real but uneven: 3/5 folds improve (up to +0.150), 2/5 worsen (fold 0 by −0.147 — A0's *best* fold); A2's biggest gains land on A0's worst folds | §12 |
| Recording consistency: 10/17 improved, 7/17 worsened, median Δ+0.033, mean Δ+0.031 — not driven by one outlier recording | §13 |
| **Fusion-usage sanity check is unambiguous**: zeroing the audio embedding at test time (same trained A2 weights) collapses macro F1 to 0.1602 — *below* A0 and A1 both — while zeroing the pitch embedding only drops it to 0.3346; the trained head's weight magnitude favors audio 1.5-2.6x over pitch across folds. Audio is genuinely, heavily used | §20 |
| Duration buckets: audio helps most on short trajectories (<250ms: A0 0.18-0.25 → A2 0.24-0.28) and is flat-to-worse beyond 500ms (>1s: 0.241→0.209) | §18 |
| Pitch-span buckets: A2 beats A0 in every bucket, largest gains at 50-200 cents (+0.12-0.13) — but this pattern is confounded with the same Cosine-recall effect, not necessarily independent confirmation | §19 |
| CREPE-ambiguity strata do **not** confirm the "audio helps most where CREPE is ambiguous" hypothesis: gains are *smaller* in the most-ambiguous quartile (+0.046) than the least-ambiguous one (+0.120) | §17 |
| A4 (oracle+audio) only marginally beats A3 (oracle alone): 0.8264/0.8460±0.0254 vs. 0.8187/0.8383±0.0383 (Δ+0.008 both metrics) — audio adds far more on top of noisy CREPE than on top of already-clean pitch geometry | §7 |
| Representative changed decisions concretely confirm the tradeoff: 3 of the top-4 highest-confidence A0-correct→A2-wrong cases are true Sloped-start/Sloped-end flipped to Cosine | §21 |

**Primary outcome: `AUDIO_REVEALS_CLASS_TRADEOFF`**

**Decision gate: `INVESTIGATE_MULTIMODAL_FUSION`** (with an explicit caveat on gate fit — see §28)

---

## 1-2. Frozen task and CREPE branch

Same canonical primitives, labels (T0=Fixed, T1=Cosine, T2=Sloped-start, T3=Sloped-end), and GT `start_s`/`end_s` as Steps 22-25 — no predicted boundaries, no neighboring-trajectory labels. A0 is literally `training.shape_classification.step23_train.run_condition(records, "crepe", "shape_velocity", (0,1,2,3), FOUR_CLASS_NAMES, balancing="sampler")`, unchanged from Step 25's F0. Reference reproduced exactly (see §7).

## 3. Existing acoustic pipeline: what was inspected, what was chosen

Two existing acoustic paths were inspected before writing any new architecture:

- **Legacy CNN pipeline** (`training/models.py::TrajectoryCNN`, `training/spec_dataset.py`): a 2D CNN (Conv2d 3→32→64→128→256, BN+ReLU+MaxPool×3, `AdaptiveAvgPool2d`, `Dropout(0.3)`, `Linear(256,5)`) over pre-rendered **176×360 RGB spectrogram PNGs**, itself downstream of `export_denoised_cnn_dataset.py`'s fixed-1-second-clip rendering pipeline and a 5-class scheme (incl. `silent`) that predates the T0-T3 canonical scheme. **Rejected**: adapting it to variable-duration oracle primitives means rebuilding the whole clip-rendering step — exactly the "spectrogram architecture project" the spec rules out.
- **CQT frontend** (`training/features.py`, used unmodified throughout Steps 6-20): `SR=22050`, `CQT_HOP=220` (10ms), `FMIN=75`, `N_BINS=360`, `BINS_PER_OCTAVE=72`, `filter_scale=1` (librosa default — the config every trained model in this repo has actually used; Step 20's `filter_scale=0.5` challenger is audited only, its own Phase B retrain not yet started, so it is not an "already-tested" config for a trained classifier). `interpolate_cqt_to_target_grid` already implements exactly the per-bin interpolation this step needs. **Chosen.**

No architecture comparison was run between them — the CQT path was selected on adaptation cost alone, before any accuracy result existed.

## 4-5. Fixed acoustic representation and normalization

One CQT is computed per recording (`librosa.cqt`, unmodified `training/features.py::cqt_log_magnitude`); each primitive's own `[N_BINS=360, 64]` log10-magnitude patch is read off by interpolating the native CQT time axis onto the primitive's own phase grid `t = start_s + x·(end_s−start_s)`, `x = linspace(0,1,64)` — byte-identical in spirit to `contours.crepe_contour`'s own interpolation, just per-frequency-bin instead of scalar. **No waveform time-stretching.** Cached once to `output/shape_classification/step26_audio_cache.pkl` (7,177 patches, ~458MB, one per canonical primitive across all 17 recordings).

Normalization: per-frequency-bin mean/std computed from the fold's TRAIN primitives only (mirrors `training/normalization.py::compute_cqt_stats`'s per-bin convention, applied to this primitive-patch cache rather than the framewise feature cache). Test/val statistics and labels never touch normalization. Grouped folds unchanged (`grouped_kfold_k5_seed42.json`).

## 6. Acoustic encoder architecture (reported before training)

`AcousticCNN` (`training/shape_classification/step26_model.py`): mirrors `framewise_models.FrequencyCNN`'s conv pattern, shrunk to `ContourCNN`'s own scale, with an added final pool over time (this is a one-embedding-per-primitive segment encoder, not a framewise one):

```
Conv2d(1,8,(7,3)) → BN → ReLU → MaxPool(4,1)
Conv2d(8,16,(5,3)) → BN → ReLU → MaxPool(4,1)
Conv2d(16,16,(3,3)) → BN → ReLU
→ mean over (freq, time)  →  [B, 16]
```

No architecture search, no CNN-vs-GRU-vs-Transformer comparison, no pretrained audio model, no attention, no receptive-field tuning.

| Model | Composition | Params |
|---|---:|---:|
| A1 `AudioOnlyModel` | `AcousticCNN` + `Linear(16→4)` | 4,580 |
| A2/A4 `FusionModel` | `ContourCNN.net`-identical pitch branch (16) + `AcousticCNN` (16) → `Linear(32→4)` | 7,412 |

A2's fusion is exactly `[h_pitch ; h_audio] → one Linear(32,4)` — no fusion MLP, no attention, no cross-modal interaction layer, mirroring Step 25 F2's `extra_dim` pattern exactly (audio embedding standing in for Step 25's template-feature vector).

## 7. Primary result table

| Condition | Macro F1 (pooled) | Fixed | Cosine | Sloped-start | Sloped-end | Grouped mean ± std | Accuracy | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 CREPE only | 0.3107 | 0.379 | 0.459 | 0.202 | 0.203 | 0.3234±0.0893 | 0.363 | 2,836 |
| A1 audio only | 0.3186 | 0.308 | 0.682 | 0.114 | 0.169 | 0.2911±0.0496 | 0.524 | 4,580 |
| **A2 CREPE+audio** | **0.3668** | 0.376 | 0.745 | 0.112 | 0.233 | **0.3500±0.0770** | 0.596 | 7,412 |
| A3 oracle pitch (Step 25 F0-oracle, not retrained) | 0.8187 | 0.646 | 0.813 | 0.899 | 0.916 | 0.8383±0.0383 | 0.779 | 2,836 |
| A4 oracle+audio (optional) | 0.8264 | 0.572 | 0.817 | 0.925 | 0.992 | 0.8460±0.0254 | 0.772 | 7,412 |

A2 vs. A0 is the primary comparison: **+0.0561 pooled, +0.0266 grouped mean** — a real, positive effect by both metrics. A4 vs. A3 (+0.0077 both metrics) is far smaller: audio compensates substantially for what CREPE's noisy pitch estimate loses, but adds comparatively little on top of already-clean oracle pitch geometry — consistent with Steps 15-19's own finding that the acoustic *pitch-estimation* pipeline, not the trajectory classifier, is where information is lost.

## 8. A4 interpretation

A4 ≈ A3 (Δ+0.008, within the same range as fold-to-fold noise elsewhere in this step) — case "audio mainly compensates for information CREPE loses," not "audio contains genuinely additional information beyond pitch geometry itself." This is consistent with, and strengthens, the A2-vs-A0 reading: audio's value here looks like it is substituting for CREPE's specific failure mode, not adding an independent signal on top of good pitch information.

## 9. Training objective

Unchanged from Step 23 B1 / Step 25: class-balanced TRAIN sampling (`torch.multinomial` over inverse-frequency weights), unweighted cross entropy, same optimizer/epoch budget/patience/seed scheme/checkpoint criterion (best val macro F1), same grouped folds. Not reopened.

## 10. Capacity control

A2 (7,412 params) is not artificially matched to A0 (2,836) — reported, not equalized, per spec. Two partial guards were used instead: both branches kept deliberately small (16-dim pooled embeddings each), fusion is a single linear layer, and the head-weight-magnitude diagnostic (§20) shows audio's mean |weight| exceeds pitch's by 1.5-2.6x across folds — a large asymmetry, but this reflects the fused model's actual reliance pattern (confirmed independently by the zeroing ablation), not evidence that the comparison is purely a parameter-count artifact. The class-tradeoff pattern in §14-15 is a more informative caveat than the raw parameter delta: a bigger model that mostly reallocates errors between two classes is a different (and less clean) finding than one that improves overall separability, regardless of param count.

## 11-13. Fold and recording consistency

| Fold | A0 | A1 | A2 | A2−A0 |
|---|---:|---:|---:|---:|
| 0 | 0.4730 | 0.3870 | 0.3258 | **−0.1472** |
| 1 | 0.2952 | 0.2671 | 0.2783 | −0.0169 |
| 2 | 0.3020 | 0.2740 | 0.3419 | +0.0399 |
| 3 | 0.1986 | 0.2446 | 0.3060 | +0.1074 |
| 4 | 0.3484 | 0.2829 | 0.4980 | **+0.1496** |

3 improved / 2 worsened, median Δ+0.0399, mean Δ+0.0266 (matches the grouped-mean delta exactly, as it must). This is **not** Step 18's "one fold wearing a pooled-average disguise" pattern — the sign split is real and roughly balanced in count, and the mechanism is legible: A2's largest gains land on A0's two *worst* folds (3, 4), while its one large loss lands on A0's single *best* fold (0). Audio's contribution looks fold-dependent in a way that partially tracks how much CREPE was already struggling, not a single-fold artifact — but the −0.147 swing on fold 0 is large enough that "3/5 folds improve" alone should not be over-read as uniform.

Recording consistency: **10/17 recordings improved, 7/17 worsened**, median Δ+0.033, mean Δ+0.031, range −0.191 to +0.218. Multiple recordings show large gains (+0.218, +0.175, +0.139, +0.103, +0.103) rather than one outlier carrying the mean.

## 14. Per-class attribution — the central finding

| Class | A0 P / R / F1 | A2 P / R / F1 | ΔF1 |
|---|---:|---:|---:|
| Fixed | 0.366 / 0.392 / 0.379 | 0.402 / 0.354 / 0.376 | −0.003 |
| Cosine | 0.750 / 0.331 / 0.459 | 0.765 / **0.727** / **0.745** | **+0.286** |
| Sloped-start | 0.127 / **0.496** / 0.202 | 0.173 / **0.083** / 0.112 | **−0.090** |
| Sloped-end | 0.127 / 0.497 / 0.203 | 0.164 / 0.400 / 0.233 | +0.030 |

This is the spec's own "less compelling result" pattern, mirrored: Cosine recall rises sharply while Sloped-start recall collapses (49.6%→8.3%) rather than both improving together. Sloped-end's own F1 improves modestly via a precision gain that outweighs a recall loss (49.7%→40.0%). Fixed is essentially flat.

## 15. Confusion matrices — does audio shrink the Cosine↔sloped axis?

Rows = true, columns = predicted, order [Fixed, Cosine, Sloped-start, Sloped-end]:

**A0**
```
[ 512,  400,  178,  216]   Fixed
[ 745, 1638, 1327, 1240]   Cosine
[  69,   83,  232,   84]   Sloped-start
[  73,   64,   91,  225]   Sloped-end
```

**A2**
```
[ 462,  623,   33,  188]   Fixed
[ 588, 3597,  118,  647]   Cosine
[  49,  295,   39,   85]   Sloped-start
[  51,  185,   36,  181]   Sloped-end
```

Cross-confusion mass between Cosine and the two sloped classes actually *drops* in total (2,714→1,245 combined off-diagonal count) — but almost entirely because **Cosine stops being mis-called sloped** (Cosine→Sloped-start+Sloped-end: 2,567→765), not because sloped examples become more separable from Cosine. In the other direction, **Sloped-start→Cosine roughly triples in rate** (17.7%→63.0% of true Sloped-start) and Sloped-end→Cosine nearly triples (14.1%→40.8%). The axis hasn't been resolved so much as re-pointed: A0 mistook Cosine for sloped shapes constantly; A2 mistakes sloped shapes for Cosine constantly. Net macro F1 improves because Cosine is the 69%-majority class, so fixing its recall moves the pooled/grouped numbers more than breaking Sloped-start's hurts them — exactly the asymmetry macro-averaging is supposed to guard against, and exactly why per-class inspection (not just the headline number) drives this step's outcome call.

## 16. Prediction frequencies

| | Fixed | Cosine | Sloped-start | Sloped-end |
|---|---:|---:|---:|---:|
| True | 18.2% | 69.0% | 6.5% | 6.3% |
| A0 | 19.5% | 30.4% | 25.5% | 24.6% |
| A1 | 30.8% | 56.9% | 2.5% | 9.8% |
| A2 | 16.0% | 65.5% | 3.1% | 15.3% |

A0's balanced-sampler training pushes predictions toward a roughly even split across classes despite the true 69%-Cosine imbalance — over-predicting both sloped classes well above their true rate (25.5%/24.6% vs. true 6.5%/6.3%). A2 moves sharply back *toward* the true (imbalanced) distribution on Cosine (65.5%, much closer to 69%) while Sloped-start prediction volume nearly vanishes (3.1%, now *under*-predicted relative to its true 6.5% rate, having been over-predicted by A0). This is a healthier-*looking* distribution in one sense (closer to true marginals) but it is the same phenomenon as §14-15 seen from a different angle, not new evidence.

## 17. CREPE-ambiguity stratification

Global quartiles of the Step 24 template-fit margin (second-best − best template MSE on CREPE's own contour), fixed before looking at any A0/A2 comparison:

| Stratum | n | A0 macro F1 | A2 macro F1 | Δ |
|---|---:|---:|---:|---:|
| Q1 (most ambiguous) | 1,794 | 0.294 | 0.340 | +0.046 |
| Q2 | 1,794 | 0.293 | 0.315 | +0.022 |
| Q3 | 1,794 | 0.257 | 0.370 | +0.113 |
| Q4 (least ambiguous) | 1,795 | 0.258 | 0.379 | +0.120 |

This does **not** confirm the hypothesized pattern ("audio helps most where CREPE's own geometry is ambiguous"). Gains are smaller in Q1/Q2 than Q3/Q4 — if anything the opposite of the naive expectation. Reported as a non-confirming diagnostic, not reframed to fit.

## 18-19. Duration and pitch-span analysis

**Duration buckets (macro F1):**

| Bucket | A0 | A1 | A2 |
|---|---:|---:|---:|
| <100ms | 0.178 | 0.229 | 0.237 |
| 100-250ms | 0.247 | 0.281 | 0.276 |
| 250-500ms | 0.385 | 0.314 | 0.383 |
| 500ms-1s | 0.404 | 0.337 | 0.386 |
| >1s | 0.241 | 0.304 | 0.209 |

Audio's edge over A0 is concentrated in the shortest trajectories (<250ms) and disappears or reverses beyond 500ms — consistent with Step 24's finding that CREPE has few reliable frames on short primitives, and with audio being able to see fine structure a sparse pitch trace can't.

**Pitch-span buckets, moving primitives only (macro F1):**

| Bucket | A0 | A1 | A2 |
|---|---:|---:|---:|
| <50c | 0.209 | 0.260 | 0.256 |
| 50-100c | 0.129 | 0.219 | 0.263 |
| 100-200c | 0.150 | 0.255 | 0.273 |
| 200-400c | 0.195 | 0.274 | 0.289 |
| >400c | 0.255 | 0.168 | 0.320 |

A2 beats A0 in every bucket. **Caveat**: these buckets still include Cosine as the majority class among movers, so part of this pattern is plausibly the same majority-class-recall effect documented in §14-15 rather than fully independent confirmation that audio specifically rescues low-SNR CREPE geometry — flagged honestly rather than presented as clean corroboration.

## 20. Fusion-usage sanity check

Same trained A2 weights, no retraining, evaluated three ways:

| Variant | Macro F1 |
|---|---:|
| A2 normal | 0.3668 |
| A2, audio embedding zeroed | **0.1602** |
| A2, pitch embedding zeroed | 0.3346 |

Audio clearly contributes (normal ≫ audio-zeroed, a 0.207 collapse — well below even A0 or A1 alone, which is expected: the fused head's weights are calibrated jointly and don't degrade gracefully to either single-branch model). Pitch also contributes, more mildly (normal > pitch-zeroed by 0.032). Head-weight magnitude ratio (audio:pitch) ranges 1.49-2.61x across folds, consistently favoring audio. Both branches are used; audio is used more heavily.

## 21. Representative changed decisions

Deterministic: highest-confidence changed predictions, no cherry-picking. All four top examples in each direction happen to come from the same recording (`6912841f213d07041b95a800`), which produced unusually high-confidence A2 predictions — noted rather than hidden.

**A0 wrong → A2 correct** (2,507 total): all four examples are true Cosine, short primitives (67-97ms) that A0 called Sloped-start/Sloped-end and A2 correctly calls Cosine at >0.99 confidence.

**A0 correct → A2 wrong** (835 total): 3 of 4 are true Sloped-start/Sloped-end flipped to Cosine at >0.98 confidence — a direct, concrete instance of the §14-15 tradeoff, not just a confusion-matrix abstraction. The fourth is a true-Fixed primitive (span −4.1c, essentially flat) flipped to Cosine.

## 22. What audio might be adding

Not established here, and not investigated further per spec: possible candidates include harmonic structure, timbre, vocal articulation, onset/offset behavior, amplitude envelope, or acoustic evidence of pitch movement CREPE mistracks. Step 26 establishes that complementary information exists and is heavily used by the trained fusion head — not which cue is responsible.

## 23. Leakage check

`assert_no_split_leakage` re-run for all 5 folds: no shared `audio_id`, no shared performance group, in any fold (`no_shared_audio_id=True`, `no_shared_group=True`, 0 entries in `audio_ids_in_multiple_splits`/`groups_in_multiple_splits` for every fold). Class distribution varies substantially by recording (some recordings are near-exclusively Cosine/Fixed with very few sloped examples) — a real dataset-imbalance limitation worth keeping in mind when interpreting per-recording deltas, not something to correct here.

## 24-25. Scope discipline

No neighboring-trajectory context, no transition matrix, no sequence model, no boundary context — isolating within-segment information only, per spec. No return to contour feature engineering (no new derivatives, no new template variant, no CREPE confidence feature) — that branch stays closed per Step 25.

## 26. Interpretation — which case?

Section 26's cases are framed around A1's strength. A1 is genuinely weak (grouped mean 0.2911, *below* A0's 0.3234, and it fails Sloped-start almost completely, recall 7.9%) while A2 is clearly better than A0 on both headline metrics — structurally closest to **Case B** ("A1 weak, A2 clearly better than A0 — genuine complementarity conditioned on CREPE"). But Case B's text does not anticipate the specific failure mode found here: the complementarity is real (per §20's zeroing check) but a large share of its headline benefit is a majority-class decision-boundary shift, not uniform-across-classes improvement. That distinction is exactly what separates a clean Case B from the outcome category chosen below.

## 27. Primary outcome: `AUDIO_REVEALS_CLASS_TRADEOFF`

Chosen over `AUDIO_ADDS_COMPLEMENTARY_INFORMATION` deliberately. The literal text of `AUDIO_REVEALS_CLASS_TRADEOFF` ("degrades others enough that macro F1 does not clearly improve") doesn't match exactly — macro F1 *did* clearly and consistently improve, on both pooled and grouped metrics, across a majority of folds and recordings. But `AUDIO_ADDS_COMPLEMENTARY_INFORMATION`'s own explicit exclusion — "the improvement is not merely a class-tradeoff" — is the clause this result fails. Sloped-start recall falling from 49.6% to 8.3%, concretely visible in both the confusion matrices (§15) and representative examples (§21), is a real, severe degradation of one class in exchange for a large gain in the majority class. A single macro-F1 number moving in the right direction, on its own, does not establish that audio genuinely helps the classifier tell trajectory shapes apart — it partly does, and partly just re-weights which shape the model defaults to.

## 28. Decision gate: `INVESTIGATE_MULTIMODAL_FUSION`

None of the four gates fits cleanly, and that mismatch is reported rather than papered over:

- `FREEZE_AUDIO_PLUS_CREPE_TYPER` is wrong — typing is not "substantially healthier" when one class's recall has collapsed to 8%.
- `INVESTIGATE_MULTIMODAL_FUSION`'s literal precondition ("A1 independently strong") is not met — A1 is weak.
- `INVESTIGATE_SEQUENCE_CONTEXT`'s precondition ("local audio adds no clear complementary information") is also not accurate — §20's zeroing check shows audio is heavily used and does add signal.
- `REASSESS_TRAJECTORY_SUPERVISION` is too strong a claim — three of four classes (Fixed, Cosine, Sloped-end) show real, usable local signal; only Sloped-start looks close to unrecoverable from local evidence alone across every condition tested here (A0 recall 49.6% but precision only 12.7%; A1 recall 7.9%; A2 recall 8.3%).

`INVESTIGATE_MULTIMODAL_FUSION` is selected as the closest fit in spirit rather than letter: the problem this step surfaced is structural to the fusion mechanism, not to whether useful information exists in either modality. A single global linear head, forced to find one decision boundary that both raises Cosine recall and (as a side effect) suppresses Sloped-start, is a plausible and cheap thing to fix directly — e.g. a class-aware calibration or one small hidden layer, still far short of an architecture search — before concluding anything stronger about audio's value or moving to a much larger-scope hypothesis like sequence context.

## Recommendation for Step 27

Test whether a minimally richer (but still simple, still no architecture search) fusion head — e.g. one hidden layer between `[h_pitch;h_audio]` and the classifier, or per-class output calibration — recovers Sloped-start without sacrificing Cosine's gain, before either freezing this fusion or escalating to sequence/context modeling. If a richer head still can't decouple the two, that would be much stronger evidence that the tradeoff is inherent to the *information* available (audio genuinely conflates Cosine and Sloped-start locally) rather than an artifact of this step's deliberately minimal fusion mechanism — and would then more cleanly justify moving to `INVESTIGATE_SEQUENCE_CONTEXT`.
