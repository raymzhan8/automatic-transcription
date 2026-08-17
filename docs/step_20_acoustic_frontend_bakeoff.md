# Step 20 — Acoustic Frontend Temporal-Resolution Bake-Off (Phase A)

Step 19 localized the primary pitch-estimation bottleneck to stage A (the acoustic representation itself): the CQT's own analysis window (130-1000ms across the candidate band) is far wider than the 10-40ms scale of T1-T3 pitch bends, and salience/framewise selection were shown to perform reasonably well *given what stage A hands them*. This step is a controlled, frontend-only bake-off (no learned model, no salience, no decoder) testing whether an alternative time-frequency representation gives a better temporal-localization / pitch-resolution tradeoff than the current CQT, before spending any training run on it.

Frozen references: [`docs/step_19_predecoder_evidence_localization.md`](step_19_predecoder_evidence_localization.md), [`docs/step_18_lambda_selection.md`](step_18_lambda_selection.md).

Machine-readable outputs: [`output/pitch_diagnostics/pitch_audit/frontend_bakeoff.json`](../output/pitch_diagnostics/pitch_audit/frontend_bakeoff.json) (17-recording real-data audit), [`output/pitch_diagnostics/pitch_audit/frontend_synthetic.json`](../output/pitch_diagnostics/pitch_audit/frontend_synthetic.json) (no-learning synthetic benchmark), figures in `output/pitch_diagnostics/pitch_audit/figures/{synthetic_resolution_test,frontend_compare_*}.png`.

Reproduce (from repository root, `idtap` conda env):

```bash
python -m training.pitch_diagnostics.pitch_audit.frontend_bakeoff     # ~9 min, full 17-recording corpus
python -m training.pitch_diagnostics.pitch_audit.frontend_synthetic   # seconds, no audio needed
python -m training.pitch_diagnostics.pitch_audit.frontend_visualize   # seconds, 3 matched real-data windows
```

Not done in Phase A, per spec: no learned-model training, no salience retraining, no candidate-grid spacing change, no Viterbi/lambda change, no canonicalization change, no trajectory-classifier training. Every metric below is a deterministic property of a frontend's own raw magnitude spectrum.

---

## Executive summary

| Finding | Evidence |
|---|---|
| **A0 reproduces Step 19 exactly** (T0/T1/T2/T3 median rank 8/14/17/26, mean ranks byte-identical) | §3 |
| **STFT (A2a/b/c) and the multi-res combination (A3) are decisively ruled out**, despite superficially attractive raw rank numbers | §6-14 — the coarse linear-Hz grid means 90-99.9% of real T1-T3 motion is literally sub-resolution (§14), motion-contrast discrimination falls *below chance* for short/medium windows (§10), and the synthetic benchmark shows worse curvature-tracking AND catastrophic low-frequency error (up to 325.8¢/bin at 104Hz) than even the unmodified CQT |
| **CQT `filter_scale` reduction (A1a=0.5, A1b=0.25) is the only family that helps** without wrecking low-frequency resolution — bin spacing stays a constant 16.67¢ regardless of `filter_scale`, so the low-frequency floor is structurally protected in a way STFT cannot match | §2, §8 |
| **A1a (filter_scale=0.5) is a genuine, broad Pareto improvement**: T3's motion-contrast pathology is largely fixed (GT-beats-half-speed 48.3%→55.8%), T3 turning-point degradation nearly halves (+7.00→+4.26¢), T1-T3 continuity improves, while T0/T1/T2 raw acoustic rank barely moves (8→9, 14→14, 17→17) | §9-16 |
| **A1b (filter_scale=0.25) pushes further on T3 and matches/slightly beats A0 on T2-turn-specific degradation, but at a broader cost**: T0/T1/T2 raw rank all regress (8→13, 14→18, 17→21), T0 continuity shrinks (120ms→70ms) — closer to "another tradeoff" than a clean win | §9-16, §19 |
| **A1a is selected as the sole Phase A challenger (A\*)** | §19 |

**Phase A outcome: `FRONTEND_CHALLENGER_FOUND`**

**Selected challenger: `A1a_cqt_fs0.5`** (CQT, `filter_scale=0.5`, all other parameters identical to production)

---

## 1. Frontend definitions

| Name | Kind | Parameters |
|---|---|---|
| A0 (control) | CQT | `filter_scale=1` (current production, byte-identical) |
| A1a | CQT | `filter_scale=0.5` |
| A1b | CQT | `filter_scale=0.25` |
| A2a | STFT | `win_length=1024` (46.4ms), `n_fft=1024` |
| A2b | STFT | `win_length=2048` (92.9ms), `n_fft=2048` |
| A2c | STFT | `win_length=4096` (185.8ms), `n_fft=4096` |
| A3 | multi-res | untrained per-frame max of two z-normalized STFTs sharing one FFT grid (`n_fft=4096`): short `win_length=1024` (zero-padded to 4096) + long `win_length=4096` |

All parameters preserved from production: `SR=22050`, `BINS_PER_OCTAVE=72`, `FMIN=75`, candidate band 104.04-778.10Hz (unchanged), hop=220 samples (10ms) for every frontend including STFT (so every representation lands on the identical canonical frame grid). Implementation: `training/pitch_diagnostics/pitch_audit/frontends.py`.

A3's combination rule is deliberately simple and untrained (spec's own requirement): each of the two STFTs is normalized to its own per-frame max, then combined via elementwise `max` — whichever window has stronger relative evidence at a given time-frequency point wins. Zero-padding the short window's FFT to 4096 gives it the SAME nominal bin grid as the long window (so A2c and A3 share identical nominal Hz/bin) without pretending the short window's true resolving power improved — this distinction (nominal bin spacing vs. actual resolving power) is made explicit in §2.

## 2. Effective analysis window and resolving power (not inferred from array shape)

Per-bin CQT wavelet length via `librosa.filters.wavelet_lengths` (actual analysis window, computed independently of hop/array shape):

| Frontend | Window @104Hz | @255Hz (median) | @778Hz | Nominal freq. resolution | Hop |
|---|---:|---:|---:|---|---:|
| A0 (fs=1) | 998.4 ms | 407.8 ms | 133.5 ms | 16.67¢/bin (constant) | 10ms |
| A1a (fs=0.5) | 499.2 ms | 203.9 ms | 66.8 ms | 16.67¢/bin (constant) | 10ms |
| A1b (fs=0.25) | 249.6 ms | 102.0 ms | 33.4 ms | 16.67¢/bin (constant) | 10ms |
| A2a (46ms) | 46.4 ms (constant) | 46.4 ms | 46.4 ms | 21.53 Hz/bin = **325.8¢@104Hz** / 140.3¢@255Hz / 47.3¢@778Hz | 10ms |
| A2b (93ms) | 92.9 ms | 92.9 ms | 92.9 ms | 10.77 Hz/bin = **170.5¢@104Hz** / 71.6¢@255Hz / 23.8¢@778Hz | 10ms |
| A2c (186ms) | 185.8 ms | 185.8 ms | 185.8 ms | 5.38 Hz/bin = 87.4¢@104Hz / 36.2¢@255Hz / 11.9¢@778Hz | 10ms |
| A3 (46+186ms) | (46.4, 185.8) ms | same | same | 5.38 Hz/bin nominal grid (**shared with A2c** — the short component's zero-padding does NOT improve its true resolving power, only its sampling density) | 10ms |

**The central structural fact this table establishes**: CQT `filter_scale` reduction shortens the analysis window *without changing nominal frequency resolution at all* (16.67¢/bin is fixed by `bins_per_octave`, independent of `filter_scale` — only the wavelet's true frequency-discriminating power/selectivity narrows). STFT windows short enough to help temporal localization at 778Hz are catastrophically coarse at 104Hz, because a fixed Hz/bin translates to far more cents at low frequencies (cents ≈ 1731 × Δf/f). This is the CQT's structural advantage and the reason its family, not STFT's, is the only one that survives Phase A.

## 3. A0 reproduction check

| | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|
| Step 19 median A-rank | 8 | 14 | 17 | 26 |
| This step's A0, median rank | 8 | 14 | 17 | 26 |
| This step's A0, mean rank | 19.686788820516348 | 34.0813279279704 | 37.21846355303361 | 54.68194309731652 |

Exact match (mean ranks agree to full float precision) — A0 reuses the cached production CQT array (`index._features[rid]["cqt_log"]`) rather than recomputing it, so this is a genuine byte-for-byte reproduction, not just a similar re-derivation. Reproduction passes; proceeding.

## 4-5. Representation comparability

Rank is computed against each frontend's own FULL native bin grid (not the 210-bin candidate-only subset), exactly matching Step 19's A-rank definition — this is what makes the A0 reproduction in §3 exact. Because native bin counts differ hugely across frontends (CQT: 360 bins; STFT n_fft=4096: 2049 bins), every rank is also reported as **normalized rank percentile** (`rank / n_bins`) for cross-frontend comparability, per spec §5/§9. GT position within each frontend's native grid is computed analytically from its own Hz→bin mapping (log for CQT, linear for STFT) — no representation is forced onto the CQT's grid.

**A critical interpretive caveat, confirmed by §10 and §14 below**: raw and even normalized rank can look deceptively good for a coarse, frequency-blurred representation, because a wide bin naturally absorbs more of a competitor's spectral leakage into the *same* bin as GT, inflating GT's own apparent rank without indicating real discriminative power. This is exactly the "trivial solution" spec §13 warns about, and it is why the STFT frontends' good-looking rank numbers in §9 are overturned by §10 and §14.

## 6. Effective resolution table

See §2 (combined for both frontend families per spec's suggested table format).

## 7. Synthetic sanity benchmark (255Hz, corpus median pitch)

Same Step 19 signals (flat / slow ramp / fast ramp / rise-then-fall), identical synthesis, run through every frontend's deterministic feature extraction only:

| Frontend | Flat med AE | Slow ramp MAE (med/max) | Fast ramp MAE (med/max) | Rise→fall MAE (med/max) | Turn-region MAE |
|---|---:|---:|---:|---:|---:|
| A0 | 0.06¢ | 5.14 (4.49/30.4) | 8.38 (4.82/100.1) | 28.12 (15.64/146.2) | 9.59 |
| **A1a** | **0.06¢** | 4.45 (4.28/16.7) | 6.29 (4.42/83.4) | 9.16 (5.92/88.0) | 5.16 |
| A1b | 0.15¢ | 4.32 (4.21/15.2) | 4.80 (4.28/38.1) | **4.99 (4.41/38.0)** | **4.38** |
| A2a | 24.96¢ | 35.73 (35.20/83.5) | 38.97 (33.03/130.7) | 41.16 (35.05/114.6) | 46.95 |
| A2b | 24.96¢ | 17.94 (17.54/41.6) | 19.61 (16.45/67.7) | 20.07 (17.24/55.2) | 24.83 |
| A2c | 11.49¢ | 9.05 (8.81/20.6) | 10.27 (8.53/43.3) | 11.74 (9.30/95.6) | 11.22 |
| A3 | 11.49¢ | 9.17 (8.81/34.5) | 11.00 (8.53/125.7) | 10.65 (8.87/74.8) | 11.22 |

Both CQT variants beat every STFT/multi-res variant on curvature-tracking (rise→fall) despite STFT windows spanning a comparable or longer duration (e.g. A2c's 185.8ms window still loses to A1a's *effective* ~204ms-at-255Hz window, 9.16¢ vs 11.74¢ MAE) — confirming the CQT's constant-cents-per-bin structure, not window duration alone, is what matters. A2a (the shortest, most "temporally localized" STFT) is actually the *worst* frontend on curvature-tracking (41.16¢ MAE) — shortening a linear-frequency window without fixing its coarse Hz/bin actively hurts.

## 8. Low-frequency stress test

Flat-tone and fast-ramp error at four representative fundamentals (synthetic only — the trained salience model is real-audio-domain and excluded, per spec's own permission; the deterministic CQT/STFT extraction is exactly what's under test):

| Frontend | 110Hz flat | 110Hz ramp (median, excl. octave-edge tail) | 150Hz flat | 255Hz flat | 500Hz flat |
|---|---:|---:|---:|---:|---:|
| A0 | 3.62¢ | 8.38¢ | 0.00¢ | 0.06¢ | 1.03¢ |
| **A1a** | **3.62¢** | **5.63¢** | 0.00¢ | 0.06¢ | 1.03¢ |
| A1b | 3.85¢ | 5.39¢ | 0.08¢ | 0.15¢ | 1.03¢ |
| A2a | 37.13¢ | 77.70¢ | 8.43¢ | 24.96¢ | 16.48¢ |
| A2b | 37.13¢ | 38.56¢ | 8.43¢ | 24.96¢ | 16.48¢ |
| A2c | 37.13¢ | 20.26¢ | 8.43¢ | 11.49¢ | 2.24¢ |

*(A0/A1a/A1b's ramp MAE at 110Hz shows a large mean driven by a p95/max ≈1200¢ tail — an octave-doubling artifact when the synthetic sweep's low end drops near/below FMIN=75Hz and the fundamental's own energy briefly falls out of the represented band; median is reported instead since it excludes this known synthetic-benchmark edge case, unrelated to the temporal-resolution question under test.)*

**Answer to the mandated question ("does shortening temporal context fix bends by destroying low-frequency pitch discrimination?"): no, not for the CQT family.** A1a's low-frequency flat-tone error is *identical* to A0's (3.62¢ at 110Hz, 0.06¢ at 255Hz) because bin spacing is untouched by `filter_scale`; A1b shows a barely-measurable regression (3.62→3.85¢, 0.06→0.15¢). STFT frontends, in sharp contrast, show catastrophic low-frequency error (37.13¢ at 110Hz, roughly 10-35x worse than either CQT variant) that does NOT meaningfully improve even at their longest window (186ms) — confirming their linear-frequency structure is the wrong shape for this candidate band regardless of window length.

## 9. Real-data acoustic GT-rank

Median rank by type, full 17-recording corpus (raw / normalized-percentile in parentheses):

| Frontend | T0 | T1 | T2 | T3 | >1000c/s bucket |
|---|---:|---:|---:|---:|---:|
| A0 | 8 (2.2%) | 14 (3.9%) | 17 (4.7%) | 26 (7.2%) | 31 (8.6%) |
| **A1a** | 9 (2.5%) | 14 (3.9%) | 17 (4.7%) | **23 (6.4%)** | 25 (6.9%) |
| A1b | 13 (3.6%) | 18 (5.0%) | 21 (5.8%) | 27 (7.5%) | 25 (6.9%) |
| A2a | 5 (1.0%) | 7 (1.4%) | 7 (1.4%) | 10 (1.9%) | 8 (1.6%) |
| A2b | 6 (0.6%) | 8 (0.8%) | 10 (1.0%) | 13 (1.3%) | 13 (1.3%) |
| A2c | 8 (0.4%) | 13 (0.6%) | 16 (0.8%) | 22 (1.1%) | 24 (1.2%) |
| A3 | 16 (0.8%) | 21 (1.0%) | 24 (1.2%) | 35 (1.7%) | 29 (1.4%) |

STFT/multi-res frontends show the best-looking raw and normalized rank of all — this is the blurring artifact flagged in §5, decisively overturned by §10 and §14. Within the CQT family: **A1a matches A0 exactly on T1/T2, costs one rank position on T0 (8→9), and clearly improves T3 (26→23)** — the type Step 19 identified as most severely broken. A1b regresses noticeably on T0/T1/T2 (8→13, 14→18, 17→21) while barely moving T3 (26→27, actually flat-to-worse in raw rank despite its narrower `>1000c/s` bucket number, 25 — reflecting that A1b's benefit at high speed comes with broader cost elsewhere on T2/T3's overall rank distribution).

## 10. Motion contrast (moving-vs-stationary and moving-vs-half-speed discrimination)

`frac_positive` = fraction of GT-moving 100ms windows where the true path's energy exceeds a stationary path's; `T3 beats-half-speed` is the counterfactual most directly diagnostic of Step 19's T3 pathology (a below-chance value there means the representation prefers a *slower* interpretation of the true motion):

| Frontend | T0 | T1 | T2 | T3 | T3 GT-beats-half-speed |
|---|---:|---:|---:|---:|---:|
| A0 | 58.9% | 53.7% | 54.8% | 50.2% | 48.3% |
| **A1a** | 61.9% | 59.5% | 57.3% | **61.6%** | **55.8%** |
| A1b | 61.7% | 63.3% | 58.1% | **65.8%** | **60.1%** |
| A2a | 41.3% | 39.1% | 34.2% | 34.9% | 32.2% |
| A2b | 49.2% | 49.3% | 44.7% | 45.5% | 42.7% |
| A2c | 57.1% | 57.3% | 53.4% | 53.7% | 49.8% |
| A3 | 56.1% | 58.3% | 52.7% | 56.2% | 51.5% |

**This is the most decisive metric against STFT**: A2a and A2b's `frac_positive` sits *below 50%* for every single trajectory type — their own energy metric prefers the wrong (stationary) interpretation of genuinely-moving GT more often than not. Both CQT variants clear this bar by a wide margin and substantially repair Step 19's specific T3 finding (GT-beats-half-speed was 48.3%, effectively at-or-below chance; A1a raises this to 55.8%, A1b to 60.1%).

## 11. Turning-point response

Mean rank in the 5 frames before vs. after a real GT turn, T2/T3 (Step 19's reference: A0 T2=+3.79, T3=+7.00 — a positive delta means acoustic evidence gets *worse* right after a turn):

| Frontend | T2 before→after (Δ) | T3 before→after (Δ) |
|---|---|---|
| A0 | 38.51→42.31 (**+3.79**) | 48.49→55.49 (**+7.00**) |
| A1a | 34.58→39.36 (+4.78) | 38.23→42.49 (**+4.26**) |
| **A1b** | 38.67→42.23 (**+3.56**) | 35.73→38.23 (**+2.50**) |
| A2c | 37.74→53.58 (+15.83) | 39.86→40.51 (+0.65) |
| A3 | 42.26→43.42 (+1.16) | 38.84→41.17 (+2.32) |

A1b gives the largest, cleanest reduction in T3's post-turn degradation (7.00→2.50, a 64% cut) and is the only CQT variant that also improves T2's degradation slightly (3.79→3.56); A1a improves T3 substantially (7.00→4.26) but *worsens* T2's turn-specific degradation (3.79→4.78). A2c's drastically worse T2 number (+15.83) is a clean confirmation that a long, linear-frequency window smears real turns badly, as predicted.

## 12. Boundary-localized acoustic evidence

Median rank near (±50ms) vs. away from primitive boundaries:

| Frontend | Near | Away | Gap (near−away) | Ratio |
|---|---:|---:|---:|---:|
| A0 | 18 | 9 | 9 | 2.0x |
| **A1a** | 16 | 10 | 6 | 1.6x |
| A1b | 20 | 15 | 5 | 1.3x |

A1a narrows the boundary gap with minimal absolute cost to either side; A1b narrows the *ratio* further but at a higher absolute cost on both near and away rank (consistent with the broader T0-T2 regression already seen in §9).

## 13. GT-ridge continuity

Median run length at rank≤5, moving types (Step 19 reference: A0 flat at 30ms for T1/T2/T3):

| Frontend | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|
| A0 | 120ms | 30ms | 30ms | 30ms |
| **A1a** | 90ms | **40ms** | 30ms | 30ms |
| A1b | 70ms | **50ms** | **40ms** | **40ms** |

Both CQT variants trade some of A0's very long, "easy" T0 continuity for improved T1-T3 continuity; A1b trades more of each in both directions. (STFT continuity numbers are excluded here since §5/§10/§14 already establish they reflect frequency blur, not genuine tracking stability.)

## 14. Frontend-only causal proxy

Classification of GT-moving frames (`|dp/dt|>100¢/s`) into `sub_resolution_movement` (true motion smaller than this frontend's own local bin width), `strong` (rank≤5), `weak` (rank≤20), `ambiguous` (rank>20), same n per type across all frontends (identical GT frame set):

| Frontend | T3 strong | T3 weak | T3 ambiguous | T3 sub-resolution | T0 ambiguous |
|---|---:|---:|---:|---:|---:|
| A0 | 1.6% | 4.4% | 21.7% | 72.2% | 22.0% |
| **A1a** | 2.3% | 5.2% | 20.2% | 72.2% | 26.3% |
| A1b | 3.1% | 4.9% | 19.8% | 72.2% | 30.7% |
| A2a | 0.1% | 0.0% | 0.0% | **99.8%** | 4.8% |
| A2c | 0.8% | 1.4% | 3.3% | **94.4%** | 15.6% |

`sub_resolution_movement` is identical across CQT variants (72.2% for T3, since bin width is unchanged by `filter_scale`) but explodes to 94-99.8% for STFT — for these frontends, the overwhelming majority of real T1-T3 motion is literally below one native bin's width, so no salience or selection mechanism built on top could ever recover it. Among CQT variants, `strong` share rises and `ambiguous` share falls monotonically with more aggressive `filter_scale` reduction (best on A1b), but T0's `ambiguous` share also rises monotonically (22.0%→26.3%→30.7%) — the clearest, most direct quantification of the T0-vs-T2/T3 tradeoff.

## 15. T2/T3 critical comparison

| Frontend | T2 rank | T3 rank | T2 motion-contrast frac+ | T3 motion-contrast frac+ | T2 turn Δ | T3 turn Δ |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 17 | 26 | 54.8% | 50.2% | +3.79 | +7.00 |
| **A1a** | 17 | **23** | 57.3% | 61.6% | +4.78 | **+4.26** |
| A1b | 21 | 27 | 58.1% | **65.8%** | **+3.56** | **+2.50** |

Neither variant wins outright on every T2/T3 metric. A1b is better on both turn-degradation numbers and on T3 motion-contrast; A1a is better on T3 rank and does not regress T2 rank at all (A1b costs 4 rank positions on T2, 17→21). §19 resolves this tension.

## 16. T0 stability control

| Frontend | T0 rank | T0 continuity | T0 synthetic flat error (255Hz) | T0 causal `ambiguous` |
|---|---:|---:|---:|---:|
| A0 | 8 | 120ms | 0.06¢ | 22.0% |
| **A1a** | 9 | 90ms | 0.06¢ | 26.3% |
| A1b | 13 | 70ms | 0.15¢ | 30.7% |

A1a's T0 cost is small and consistent across every metric; A1b's is roughly 2-3x larger on every metric. Neither is "catastrophic" in isolation, but A1b's T0 cost compounds with its T1/T2 rank cost (§9) in a way A1a's does not.

## 17. Visualization

Three matched real-data windows, A0 vs. A1a, identical time/frequency crops (`frontend_compare_*.png`), reusing exactly the windows Step 19 already selected for its own visualizations:

- **T3 turning-point segment** (`645ff354deeaf2d1e33b3c44`, 91.2-95.2s): A1a's ridge is visibly thinner/sharper through the sharp downward dip near 92.2-92.3s and the small bumps near 91.8s and 94.6-94.8s — a modest but real visual confirmation of the quantitative turn-response improvement.
- **Acoustic-absent failure** (`6824de49abc4705438ce918b`, 194.3-198.2s, Step 19's own worst-case example): **A1a does NOT fix this case** — GT remains nearly invisible against a strong stationary competitor ~1150-1200¢ above in both A0 and A1a. This is an honest negative result: the frontend change addresses temporal smearing, not cases where the true melodic voice is simply weak relative to a stationary drone/accompaniment partner (a different mechanism, more SNR/source-separation-shaped than resolution-shaped).
- **T0 control** (`6503e36cd9ff49d3988d0b40`, 2.7-6.7s): visually indistinguishable between A0 and A1a — consistent with §16's small measured T0 cost.

## 18. Central frontend comparison table

| Metric | A0 | **A1a** | A1b | A2c (best STFT) |
|---|---:|---:|---:|---:|
| T0 GT rank | 8 | 9 | 13 | 8 |
| T1 GT rank | 14 | 14 | 18 | 13 |
| T2 GT rank | 17 | 17 | 21 | 16 |
| T3 GT rank | 26 | **23** | 27 | 22 |
| >1000c/s GT rank | 31 | 25 | 25 | 24 |
| T3 GT>stationary | 50.2% | 61.6% | **65.8%** | 53.7% |
| T3 GT>half-speed | 48.3% | 55.8% | **60.1%** | 49.8% |
| T2 turn degradation | +3.79 | +4.78 | **+3.56** | +15.83 |
| T3 turn degradation | +7.00 | +4.26 | **+2.50** | +0.65* |
| Boundary rank gap (50ms) | 9 | 6 | **5** | — |
| Synthetic flat error (255Hz) | 0.06¢ | **0.06¢** | 0.15¢ | 11.49¢ |
| Synthetic rise→fall MAE | 28.12¢ | 9.16¢ | **4.99¢** | 11.74¢ |
| Low-freq (110Hz) flat error | 3.62¢ | **3.62¢** | 3.85¢ | 37.13¢ |

*A2c's low T3 turn-degradation number is not a genuine strength — its rank is already so uniformly poor/blurred (§9, §14: 94.4% sub-resolution) that there is little room left to degrade further around a turn specifically; interpreted alongside §10/§14 it does not indicate real turning-point fidelity.

Reading the table as a whole: **A2c/every STFT-family frontend is eliminated by the combination of catastrophic low-frequency error, below/near-chance motion contrast, and 90%+ sub-resolution rates** — its apparently good rank and turn-degradation numbers are artifacts, not genuine strengths (§5, §10, §14). Within the surviving CQT family, A1a is the more broadly balanced Pareto point; A1b pushes further on T3/turn-specific metrics at a broader cost to T0/T1/T2.

## 19. Phase A decision

Per spec's priority order — (1) T2/T3 turning-point fidelity, (2) moving-pitch GT rank, (3) moving-vs-stationary/half-speed discrimination, (4) low-frequency pitch preservation, (5) T0 stability — A1b nominally wins priority (1) outright (only variant improving *both* T2 and T3 turn-degradation vs. A0), while A1a wins or ties priorities (2), (4), and (5), and the two are close on (3) (A1b slightly ahead on T1/T3, roughly tied elsewhere).

This is exactly the tension spec §16/§26 anticipates ("A improves T2/T3 but hurts T0/T1... then we found another tradeoff rather than a better operating point"). Resolving it: A1b's T2/T3 turn-degradation win is real, but it arrives packaged with a broader, monotonic cost across T0, T1, *and* T2's own raw rank (§9: 8→13, 14→18, 17→21) and T0 continuity (§13: 120ms→70ms) — a wider-reaching regression than A1a's, which costs only one rank position on T0 (8→9) while leaving T1/T2 literally unchanged. A1a already captures the majority of the practically important benefit — Step 19's single most severe, specific finding was T3's near/below-chance motion discrimination (GT-beats-half-speed 48.3%, i.e. the representation actively preferred a *slower* reading of the true motion), and A1a substantially repairs this (55.8%) without broadening the cost beyond T0. A1b's extra gain on T3 (60.1%) and T2-turn-specific fidelity (+3.56 vs A1a's +4.78) is real but does not clearly outweigh its added cost across three of four types on the priority-2 criterion (raw rank) and priority-5 criterion (T0 stability).

**Phase A outcome: `FRONTEND_CHALLENGER_FOUND`**

**Selected challenger: `A1a_cqt_fs0.5`** — CQT with `filter_scale=0.5`, all other parameters unchanged from production. A1b is documented as a viable, more aggressive alternative that was considered and not selected; it remains available as a fallback if A1a's Phase B results are weaker than expected.

---

## Phase B status

Per spec, Phase B (retrain the salience CNN and the P0 trajectory classifier on A1a's frontend, re-run the full pre-decoder audit including the zero-delta causal decomposition, and re-measure trajectory macro F1 end-to-end against oracle) is gated on Phase A selecting exactly one challenger, which it now has. Phase B is a substantially larger compute commitment than Phase A: it requires (a) building a full A1a feature cache and fold-wise CQT normalization statistics, (b) retraining the harmonic-salience CNN across 5 folds, (c) recalibrating HPS/learned fusion on the new salience scale using the frozen train/validation protocol, (d) rebuilding D0\*/D1\* dense paths, (e) re-running the Step 19-style pre-decoder audit, and (f) retraining the P0 trajectory classifier across 5 folds — realistically multiple hours of sequential training, distinct from Phase A's ~10 minutes of deterministic feature computation. This report stops at the Phase A checkpoint the spec itself defines ("Run Phase B ONLY if Phase A selects one clear challenger") to confirm the challenger and compute-budget commitment before starting that training run.
