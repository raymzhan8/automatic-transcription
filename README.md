# Automatic Transcription — Pitch-Trajectory Classification

Research pipeline built on [IDTAP](https://idtap.org) (Swara Studio) transcriptions of Indian classical music. It answers two questions:

1. **Can a CNN classify the *shape* of a pitch trajectory** (fixed note vs. bend vs. sloped-start bend vs. sloped-end bend) from a 1-second CQT spectrogram?
2. **Does audio cleanup help?** Every dataset is exported in three variants — raw, denoised, and vocals-only — so classifiers and silence detectors can be compared arm-for-arm on identical examples.

Everything is driven by existing IDTAP annotations. No new hand-labeling is required.

---

## Project layout

Source folders are named after the pipeline stage they belong to. `inventory/` and `dataset/`
are shared by two pipelines described below: the active **canonical/framewise pipeline**
(numbered Steps 1–14) and the earlier, still-runnable **CNN spectrogram-classification
pipeline** (lettered steps a–g).

```text
.
├── inventory/      # Steps 1, 3 — what annotations exist, how do they behave structurally?
│   ├── survey_trajectories.py           # Fast per-class candidate tally
│   ├── inspect_idtap_trajectories.py    # Full annotation inventory CSVs
│   ├── trajectory_inventory.py          # Inventory extraction + validation
│   ├── idtap_io.py                      # Typed IDTAP client wrappers
│   ├── dataset_utils.py                 # Label map + trajectory candidate iteration
│   └── debug_idtap.py                   # IDTAP API/auth introspection
├── dataset/        # CNN pipeline steps a-b — build spectrogram datasets
│   ├── export_denoised_cnn_dataset.py   # Denoise/separate audio, export variants
│   ├── combine_cnn_metadata.py          # Rebuild combined metadata.csv
│   └── canonical/  # Step 2 — recording-level canonical dataset
│       ├── schema.py                    # Single source of truth: constants, enums, ids
│       ├── build.py                     # Fetch + cache raw, emit one JSON per recording
│       ├── pitch.py                     # Raw and derived Pitch blocks
│       ├── timing.py                    # Absolute times from stored offsets + assertions
│       ├── transitions.py               # Per-lane consecutive-pair records
│       ├── canonicalize.py              # Additive Type-0 merge overlay
│       ├── audio_refs.py                # audio.files: relpaths, properties, sha256
│       ├── coverage.py                  # Coverage block and its loud assertions
│       ├── index_tables.py              # Flat CSV projections of the documents
│       ├── performance_groups.py        # Union-find performance grouping
│       ├── overrides.json               # Hand-maintained grouping escape hatch
│       ├── splits.py                    # Grouped 60/20/20 and k-fold manifests
│       └── verify_roundtrip.py          # Proves raw rebuilds idtap Trajectory contours
├── training/       # CNN pipeline step c — the CNN classifier
│   ├── spec_dataset.py                  # PyTorch dataset + label config
│   ├── models.py                        # TrajectoryCNN
│   ├── train_cnn.py                     # Training harness
│   └── predict_cnn.py                   # Inference on PNGs
├── denoise/        # CNN pipeline steps d-e — the raw/denoised/vocals experiment
│   ├── run_ab_train_eval.py             # Aligned A/B training across variants
│   └── compare_misclassified_raw_denoised.py  # Side-by-side error analysis
├── silence/        # CNN pipeline steps f-g — silent/non-silent detection
│   ├── silence_detection.py             # RMS + librosa.split segmenters
│   ├── silence_audio_prep.py            # Resolves which audio stem to segment
│   └── eval_silence_detection.py        # Tune + evaluate against IDTAP labels
├── notebooks/      # Exploratory + QA notebooks (see "Notebooks")
├── docs/
│   └── label_definitions.md             # Human annotation guide for the four shapes
├── legacy/         # Superseded code, kept for reference
├── output/         # Generated audio, datasets, results (gitignored)
├── README.md
└── requirements.txt
```

### Running scripts and the import convention

Run everything **from the repository root**, invoking scripts by path:

```bash
python silence/eval_silence_detection.py --audio-variant denoised
```

Each stage folder is a Python package. Entry-point scripts put the repository root on
`sys.path` and then use absolute imports across stages, for example:

```python
from dataset.export_denoised_cnn_dataset import process_piece_audio
from training.spec_dataset import SpecImageDataset
```

Because that bootstrap resolves the root from `__file__`, the scripts also work when
invoked from another working directory. `output/` must stay at the repository root: the
generated metadata CSVs store image and clip paths relative to it.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `vocalprep` dependency (Mel-Roformer denoise + karaoke separation) downloads roughly 1 GB of model weights the first time it runs.

Authenticate with IDTAP once; tokens are then reused from secure storage:

```python
from idtap import login_google, SwaraClient

login_google()
client = SwaraClient()
```

If the IDTAP API surface looks different from what the scripts expect, run `python inventory/debug_idtap.py` — it prints the runtime structure of the installed package (with credentials redacted) so you can confirm which methods exist.

---

## Label scheme

Labels come straight from IDTAP trajectory IDs. The live scheme has **five classes**:

| Label | IDTAP ID | IDTAP name |
|-------|----------|------------|
| `0` | 0 | Fixed |
| `1` | 1 | Bend: Simple |
| `2` | 2 | Bend: Sloped Start |
| `3` | 3 | Bend: Sloped End |
| `silent` | 12 | Silent |

Two extra rules are applied during export (`iter_labeled_segments` in `export_denoised_cnn_dataset.py`):

- **Composite trajectories are decomposed** using the trajectory's own `dur_array` to place segment boundaries: ID 4 → `[2, 1]`, ID 5 → `[1, 3]`, ID 6 → a run of `1`s. Only the first segment's label is currently exported per trajectory.
- **IDs 7–11 and 13 are skipped** (krintin, slide, vibrato, and friends) and logged to `skipped.csv` with a reason, so they can be audited later.

[`docs/label_definitions.md`](docs/label_definitions.md) describes each shape in prose — start/middle/end behavior, positive examples, ambiguous cases, and exclusion criteria. Read it before interpreting confusion matrices.

Classes can be dropped at train time with `--exclude-labels silent`, which renumbers the label map automatically.

---

## Spectrogram representation

Every example is a 1-second window rendered identically across variants:

| Parameter | Value |
|-----------|-------|
| Sample rate | 22050 Hz |
| Transform | CQT, magnitude, log10 |
| Frequency range | 75 – 2400 Hz |
| Bins per octave | 72 (360 bins total) |
| Hop length | 512 (44 frames/second) |
| Colormap | magma, min-max normalized per clip |
| Exported PNG | 176 × 360 px |

Clips shorter than one second are zero-padded; longer ones are truncated. Both the PNG and the source WAV are written so you can listen to any example.

---

## Canonical / framewise pipeline (Steps 1–16)

This is the active research track: build a continuous, lossless canonical representation of each transcription, then train framewise models on it. Step numbers here match `docs/step_N_*.md` exactly, so a doc's own title always agrees with the section that links it (`docs/step_4_5_canonical_trajectory_target.md` is linked from Steps 3 and 4, `docs/step_12_5_fusion_viterbi.md` from Step 12.5, and so on). The earlier, still-runnable CNN spectrogram-classification pipeline is described afterward, in its own unnumbered section.

### Step 1 — Dataset audit

Before building anything, inspect what IDTAP actually has: annotation coverage, audio availability, and (if the API surface ever looks different from what the scripts expect) the runtime shape of the installed `idtap` package — see [Setup](#setup) for `inventory/debug_idtap.py`.

```bash
python inventory/inspect_idtap_trajectories.py                    # all accessible transcriptions
python inventory/inspect_idtap_trajectories.py --max-pieces 10    # quick sample
python inventory/inspect_idtap_trajectories.py --piece-id PIECE_ID
```

Writes to `output/inventory/`:

- `trajectory_inventory.csv` — one row per annotation, with absolute start/end times, pitches, log-frequencies, instrument, raga, tonic, and performer
- `trajectory_counts.csv` — counts and duration statistics per IDTAP type
- `failed_transcriptions.csv` — transcriptions that could not be loaded, with error type

Validation warnings (mismatched trajectory/start-time arrays, non-positive durations, non-finite timings, inconsistent type names) are printed but never fatal. The extraction and validation logic lives in `inventory/trajectory_inventory.py`.

**Main question: what data do we actually have?**

### Step 2 — Raw → derived dataset

The CNN spectrogram-classification pipeline described later on this page flattens IDTAP into fixed 1-second clips, so anything that did not fit the window is gone. This step is the opposite, and is meant to become the foundation everything else reads from: **one JSON document per recording**, audio left continuous and unsegmented, with every downstream product — individual clips, clips with context, framewise labels, candidate trajectory chains — computed at read time from what is really an interval index over the audio.

```bash
python dataset/canonical/build.py                          # the 17 exported recordings
python dataset/canonical/build.py --recording-ids ID1 ID2   # limit to specific recordings
python dataset/canonical/build.py --refresh-raw             # re-fetch from IDTAP
python dataset/canonical/build.py --no-index                # skip the CSV projections
```

Each `client.get_piece()` response is cached verbatim and gzipped under `raw_api/`, so only the first build touches the network and no field we did not model is ever lost. Recordings that fail are logged to `failed.csv` and skipped.

**The raw-versus-derived guarantee.** Every trajectory carries two blocks. `raw` is a faithful transcription of the IDTAP wire JSON — `id`, `pitches` (swara/oct/raised/log_offset only), `dur_tot`, `dur_array`, `slope`, `vib_obj`, `fund_id12`, articulations, automation, vowels and consonants — plus `wire_keys_present`, which records exactly which keys existed on the wire, so drift between old and new transcriptions is auditable rather than silently normalized. `derived` holds everything computed: absolute `start_s`/`end_s`, frequencies, cents, sargam, control points, and every `is_*`/`has_*` flag. Nothing computed leaks into `raw`, which is what lets a consumer regenerate continuous f0 targets with `Trajectory.compute(x, log_scale=True)` without re-fetching anything.

That claim is proved rather than asserted:

```bash
python dataset/canonical/verify_roundtrip.py                 # every trajectory
python dataset/canonical/verify_roundtrip.py --sample-per-type 25
```

It rebuilds an `idtap` `Trajectory` from each `raw` block, builds a reference straight from the cached wire JSON, and compares shape parameters and contours over a dense grid. All 5,209 trajectories across the ten types present (0–7, 12, 13) currently agree to exactly 0.0 in log2 Hz.

Two more structural choices worth knowing before reading a document. Type-0 canonicalization is **additive**: merging consecutive same-pitch Fixed trajectories produces a sibling `canonicalization` overlay of units referencing raw indices, and the trajectory array is never collapsed, so `view="raw"` and `view="canonical"` are both always available. And nothing is called silence — gaps are described with Allen-interval vocabulary (`gap_s`, `interval_relation`), and the only silence-flavored field is `is_silent_annotation`, which records that IDTAP asserted trajectory id 12. The `coverage` block surfaces how much of each recording that covers, because several recordings are over 95% "Silent" simply because nobody transcribed them.

Splits live outside the data, so re-splitting never rewrites a recording:

```bash
python dataset/canonical/splits.py                                        # both manifests
python dataset/canonical/splits.py --min-non-silent-fraction 0.4 --name-suffix _dense
python dataset/canonical/splits.py --seed 7 --k 10
```

The atomic unit is the **performance**, never the recording. Twelve audioIDs in the corpus carry two or three separate transcriptions and one performance was uploaded segmented across three audioIDs, so splitting on `recording_id` puts byte-identical audio on both sides of a train/test boundary. `performance_groups.py` settles grouping once by union-find over shared `audioID`, a narrow `(soloist, raga, title)` match, and `overrides.json`; the builder stamps the answer onto every document, and the splitter refuses to run on a recording without one. Two manifests are written per run — a grouped 60/20/20 and a grouped 5-fold — and both carry per-split trajectory-type histograms and non-silent duration so imbalance is visible. Nothing is reshuffled to equalize class counts. The leakage assertions are recomputed on every run rather than hardcoded: if any `audio_id` or any performance group appears in two splits, the run raises rather than writing the manifest.

Everything lands under `output/canonical/v1/`:

```text
output/canonical/v1/
├── recordings/<recording_id>.json      # canonical document, the source of truth
├── raw_api/<recording_id>.json.gz      # verbatim get_piece() response
├── index/
│   ├── recordings.csv                  # one row per recording (17)
│   ├── trajectories.csv                # one row per trajectory (5,209)
│   └── transitions.csv                 # one row per consecutive pair (5,189)
├── splits/<split_name>.json            # split manifests, separate from the data
└── failed.csv                          # recordings that did not build
```

The three CSVs are pure projections, deletable and regenerable from `recordings/*.json`. They are written with the stdlib `csv` module because `pandas` currently fails to import in this environment.

To read the findings rather than the schema, open [`notebooks/canonical_dataset_findings.ipynb`](notebooks/canonical_dataset_findings.ipynb), which computes all of the above live from `output/canonical/v1/` and plots an annotated timeline with reconstructed pitch contours. It reads from disk only.

**Main question: can we reliably turn IDTAP into a dataset without losing information or leaking performances?**

### Step 3 — Trajectory inventory / structural analysis

Before designing a canonical target vocabulary, check which raw IDTAP types actually carry usable signal and how they behave structurally: counts, durations, transitions between consecutive trajectories, composite-type internal boundaries, adjacency, and how much of each recording is silence.

```bash
python inventory/survey_trajectories.py
```

Prints per-class candidate counts (total and with-audio) against the four target primitives, a full tally of every IDTAP trajectory type present, and how many transcriptions loaded or failed. Nothing is downloaded.

```bash
python dataset/canonical/analyze_step_4_5.py
```

Recomputes the structural statistics reported in §2–5 of [`docs/step_4_5_canonical_trajectory_target.md`](docs/step_4_5_canonical_trajectory_target.md) — raw boundary statistics, Type-0 canonicalization behavior, simple-vs-composite trajectory equivalence, and the three competing notions of "boundary" — against the canonical dataset built in Step 2. `dataset/canonical/index_tables.py`'s `transitions.csv` (one row per consecutive pair, 5,189 rows) is the underlying per-pair data these statistics summarize.

**Main question: what exactly are we trying to predict, and what structure exists in the annotations?**

### Step 4 — Canonical representation design

Turns the structural analysis above into a target vocabulary: the T0–T3 primitive classes, how each raw IDTAP type maps or decomposes into them, and how adjacent trajectories are reconciled at a shared boundary.

See [`docs/step_4_5_canonical_trajectory_target.md`](docs/step_4_5_canonical_trajectory_target.md) §6–10 for the proposed canonicalization rules, the boundary-prediction-head recommendation, the dense pitch target, and the transformation pipeline from raw trajectory to primitive.

```bash
python dataset/canonical/build_primitives.py
python dataset/canonical/verify_contours.py
```

`decompose.py`, `primitives.py`, and `boundary_geometry.py` implement the mapping/decomposition and boundary-geometry rules the doc specifies; `build_primitives.py` applies them to every recording under `output/canonical/v1/primitives/`, and `verify_contours.py` checks the primitives still reconstruct the original pitch contour.

**Main question: how should the complicated IDTAP representation become a clean ML target?**

### Step 5 — Canonical framewise targets

Rasterize the primitives onto a 10 ms timeline — `trajectory_type`, parametric pitch, phase, `valid_target`, provenance — and verify the result before designing anything that trains on it.

```bash
python dataset/canonical/build_frames.py
python dataset/canonical/validate_targets.py
python dataset/canonical/visualize_targets.py
```

Outputs land under `output/canonical/v1/frames/` and `figures/step5/`. See [`docs/step_5_framewise_targets_report.md`](docs/step_5_framewise_targets_report.md) for schemas, statistics, and validation results.

Before designing the model, audit whether canonical boundaries are actually inferable from pitch geometry and measure class balance after Type 6 decomposition:

```bash
python dataset/canonical/analyze_step_5_5.py
python dataset/canonical/visualize_boundaries.py
```

Outputs: `output/canonical/v1/step_5_5_analysis.json` and `figures/step5_5/`. See [`docs/step_5_5_boundary_learnability_report.md`](docs/step_5_5_boundary_learnability_report.md) for corrected duration statistics, boundary analysis, and the target recommendation carried into Step 6.

**Main question: can we express the transcription continuously as framewise targets without destroying the original annotation?**

### Step 6 — Model/experiment architecture design

Specify (but do not yet train) the first continuous-audio architecture:

```text
audio → CQT → frequency-CNN → BiGRU → type + pitch heads @ 10 ms
```

See [`docs/step_6_framewise_model_design.md`](docs/step_6_framewise_model_design.md) for the baseline CNN audit, representation choice, layer specification, loss design, evaluation plan, and controlled experiment sequence (Experiments A–D).

**Before implementation**, read [`docs/step_6_5_architecture_corrections.md`](docs/step_6_5_architecture_corrections.md) for exact parameter counts, CQT/target timing alignment, TCN-first experiment order (B0→B1→C→D), and finalized Model B/C specs.

Verify alignment tests:

```bash
python training/framewise_models.py          # exact parameter counts
python -m pytest training/test_frame_alignment.py -q
```

**Main question: what is the simplest controlled ML architecture for testing the framewise formulation?**

### Step 7 — Framewise training pipeline + Experiment B0

Build CQT features, train the TCN type-only model (Experiment B0) with grouped 5-fold CV:

```bash
# Precompute aligned CQT features (once)
python dataset/canonical/build_features.py

# Unit tests + alignment
python -m pytest training/test_frame_alignment.py training/tests/ -q

# Sanity: tiny overfit + label shuffle
python training/train_framewise.py --tiny-overfit 32 --max-epochs 100 --fold 0
python training/train_framewise.py --label-shuffle --max-epochs 15 --fold 0

# Full 5-fold B0
python training/train_framewise.py --all-folds

# Evaluate best checkpoint on test fold
python training/evaluate_framewise.py \
  --checkpoint output/framewise_runs/b0_tcn_type_only/fold_0/best.pt
```

See [`docs/step_7_b0_report.md`](docs/step_7_b0_report.md) for results and the B1 go/no-go decision.

**Main question: does continuous audio contain enough signal for a framewise model to predict T0–T3 at all?**

### Step 8 — Experiment B1 (joint pitch supervision)

Same TCN as B0, plus a pitch head trained on fold-standardized tonic-relative cents:

```bash
python training/train_framewise.py \
  --config training/configs/b1_tcn_type_pitch.json --tiny-overfit 32 --max-epochs 100 --fold 0

python training/train_framewise.py \
  --config training/configs/b1_tcn_type_pitch.json --all-folds

for i in 0 1 2 3 4; do
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b1_tcn_type_pitch/fold_$i/best.pt
  python training/evaluate_framewise.py \
    --checkpoint output/framewise_runs/b0_tcn_type_only/fold_$i/best.pt \
    --output-dir output/framewise_runs/b0_tcn_type_only/fold_$i/eval
done

python training/compare_b0_b1.py
```

See [`docs/step_8_b1_report.md`](docs/step_8_b1_report.md).

**Main question: does learning pitch jointly help the model distinguish trajectory shapes?**

### Step 9 — Experiment C (BiGRU vs TCN)

Same data, cents pitch, λ=1, folds, and sampler as B1; only the temporal encoder changes (1-layer bidirectional GRU, **offline**):

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

See [`docs/step_9_c_report.md`](docs/step_9_c_report.md).

### Step 10 — Pitch learnability / frontend diagnostics

Pitch-only. Does **not** retrain B0/B1/C. Writes under `output/pitch_diagnostics/` and [`docs/step_10_pitch_diagnostics.md`](docs/step_10_pitch_diagnostics.md).

```bash
python training/pitch_diagnostics/cnn_audit.py
python training/pitch_diagnostics/analyze_a.py
python training/pitch_diagnostics/baselines_b.py
python training/pitch_diagnostics/pyin_only.py
python training/pitch_diagnostics/stems_f.py

# Standard frequency-preserving pitch-only model (scalar absolute CQT)
python training/pitch_diagnostics/train_pitch.py \
  --head scalar --run-name scalar_abs --tiny-overfit 32 --max-epochs 100 --fold 0
python training/pitch_diagnostics/train_pitch.py \
  --head scalar --run-name scalar_abs --all-folds --max-epochs 20

# Same trunk: bin CE, tonic-aligned CQT, within-recording time split
python training/pitch_diagnostics/train_pitch.py \
  --head bins --run-name bins_abs --all-folds --max-epochs 20
python training/pitch_diagnostics/train_pitch.py \
  --head scalar --tonic-align --run-name scalar_tonic --all-folds --max-epochs 20
python training/pitch_diagnostics/train_pitch.py \
  --head scalar --run-name scalar_within --within-recording --fold 0 --max-epochs 20
```

See [`docs/step_10_pitch_diagnostics.md`](docs/step_10_pitch_diagnostics.md).

### Step 11 — Frequency-preserving harmonic-salience pitch frontend

Pitch-only test of a learned, harmonic-aware salience frontend against Step 10's frozen HPS baseline. Writes under `output/pitch_diagnostics/` (files prefixed `harmonic_salience_*`) and [`docs/step_11_harmonic_salience.md`](docs/step_11_harmonic_salience.md).

```bash
python training/pitch_diagnostics/hps_salience.py
python training/pitch_diagnostics/train_salience.py --variant local --all-folds --run-name local_salience_abs
python training/pitch_diagnostics/train_salience.py --variant harmonic --all-folds --run-name harmonic_salience_abs
python training/pitch_diagnostics/evaluate_salience.py
python training/pitch_diagnostics/visualize_salience.py
```

See [`docs/step_11_harmonic_salience.md`](docs/step_11_harmonic_salience.md).

### Step 12 — Register/octave resolution diagnostics

Direct follow-up to Step 11's decision gate: diagnoses why the learned salience model loses to HPS on octave selection specifically, then tests two lightweight decoders (frame-independent fusion, movement-cost Viterbi) the diagnostics justify. No retraining. Writes under `output/pitch_diagnostics/register_resolution/` and [`docs/step_12_register_resolution.md`](docs/step_12_register_resolution.md).

```bash
python training/pitch_diagnostics/register_resolution/candidate_range_fixed.py
python training/pitch_diagnostics/register_resolution/collect.py
python training/pitch_diagnostics/register_resolution/octave_diagnostics.py
python training/pitch_diagnostics/register_resolution/oracles.py
python training/pitch_diagnostics/register_resolution/static_prior.py
python training/pitch_diagnostics/register_resolution/disagreement.py
python training/pitch_diagnostics/register_resolution/synthesize.py
python training/pitch_diagnostics/register_resolution/fusion.py
python training/pitch_diagnostics/register_resolution/decoders.py
```

See [`docs/step_12_register_resolution.md`](docs/step_12_register_resolution.md).

### Step 12.5 — Fused salience + Viterbi closing ablation

Closes Step 12's decision gate: runs the one untested combination (movement-cost Viterbi decoding on the fused HPS+learned salience distribution, instead of on either source alone). No retraining, no new decoder variants. Writes `output/pitch_diagnostics/register_resolution/fusion_viterbi_result.json` and [`docs/step_12_5_fusion_viterbi.md`](docs/step_12_5_fusion_viterbi.md).

```bash
python training/pitch_diagnostics/register_resolution/fusion_viterbi.py
```

See [`docs/step_12_5_fusion_viterbi.md`](docs/step_12_5_fusion_viterbi.md) — verdict `FUSION_VITERBI_MARGINAL`, decision `STOP_REGISTER_DECODER_ENGINEERING`.

### Step 13 — Octave-invariant relative pitch movement diagnostic

Diagnostic-only follow-up: does local pitch-motion (frame-to-frame delta, time-gap-aware velocity, windowed relative contour) survive register/octave errors well enough to support T0-T3 trajectory classification without solving absolute-pitch recovery first? No retraining, no new decoder machinery beyond re-applying Steps 12/12.5's own validated hyperparameters, no class weighting, no sequence model. Writes under `output/pitch_diagnostics/relative_pitch/` and [`docs/step_13_relative_pitch.md`](docs/step_13_relative_pitch.md).

```bash
python training/pitch_diagnostics/relative_pitch/path_cache.py
python training/pitch_diagnostics/relative_pitch/signals.py
python training/pitch_diagnostics/relative_pitch/probe.py
```

See [`docs/step_13_relative_pitch.md`](docs/step_13_relative_pitch.md) — verdict `RELATIVE_PITCH_PARTIAL`: octave errors mostly cancel under differencing, but the T0-T3 probe still loses real information vs. oracle pitch (especially T2/T3), so relative pitch should complement rather than replace learned audio features going forward.

### Step 14 — Relative-pitch-augmented trajectory classification (A/B/C/D ablation)

Controlled feature ablation on the actual `audio → T0/T1/T2/T3` task: one shared TCN classifier, four input conditions (audio only / estimated relative pitch only / audio+estimated / audio+oracle). No pitch-frontend redesign, no register-decoder work, no class weighting, no architecture search. Writes under `output/relative_pitch_ablation/` and [`docs/step_14_relative_pitch_trajectory.md`](docs/step_14_relative_pitch_trajectory.md).

```bash
python training/pitch_diagnostics/relative_pitch/dense_pitch_path.py
python training/train_relative_pitch_ablation.py --condition A --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition B --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition C --all-folds --max-epochs 20 --patience 5
python training/train_relative_pitch_ablation.py --condition D --all-folds --max-epochs 20 --patience 5
python training/evaluate_relative_pitch_ablation.py
python training/visualize_relative_pitch_ablation.py
```

See [`docs/step_14_relative_pitch_trajectory.md`](docs/step_14_relative_pitch_trajectory.md) — outcome `RELATIVE_PITCH_ONLY_COMPETITIVE` (estimated pitch alone beats both audio-only and naive audio+pitch fusion), decision `IMPROVE_RELATIVE_PITCH_ESTIMATION`.

### Step 15 — Learned pitch-motion representation

Pitch/salience-evidence-only experiment: is trajectory information lost at the decoded-pitch-path stage, at the four-fixed-offset `φ` stage, or is the underlying acoustic pitch/salience estimate itself the bottleneck? Four conditions — fixed φ (P0, reproduces Step 14 B), a learned dense-delta pitch-motion encoder (P1), a learned register-invariant salience-motion encoder (P2), and an oracle-pitch version of P1 (P3). No audio branch, no fusion, no register decoding, no class weighting. Writes under `output/pitch_motion_ablation/` and [`docs/step_15_learned_pitch_motion.md`](docs/step_15_learned_pitch_motion.md).

```bash
python training/pitch_diagnostics/relative_pitch/dense_relative_salience.py
python training/train_pitch_motion_ablation.py --condition P0 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P1 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P2 --all-folds --max-epochs 50 --patience 10
python training/train_pitch_motion_ablation.py --condition P3 --all-folds --max-epochs 50 --patience 10
python training/evaluate_pitch_motion_ablation.py
```

See [`docs/step_15_learned_pitch_motion.md`](docs/step_15_learned_pitch_motion.md) — outcome `ESTIMATED_MOTION_REMAINS_BOTTLENECK` (neither learned representation beats the fixed φ baseline; oracle pitch dominates even more than Step 14's audio+oracle condition), decision `INVESTIGATE_ACOUSTIC_MOTION_ESTIMATION`.

### Step 16 — Fine-contour acoustic pitch audit

Diagnostic-only audit (no training, no model changes): exactly how does estimated acoustic pitch motion differ from the oracle contour in ways that destroy T0-T3 information? Separates pitch-value error from octave/register error, temporal lag, temporal smoothing, slope/turning-point error, jitter, quantization, and harmonic/drone confusion, then tests the audit-justified corrections downstream using Step 15's frozen P0 classifier. Writes under `output/pitch_diagnostics/pitch_audit/` and [`docs/step_16_acoustic_pitch_audit.md`](docs/step_16_acoustic_pitch_audit.md).

```bash
python -m training.pitch_diagnostics.pitch_audit.motion
python -m training.pitch_diagnostics.pitch_audit.shape
python -m training.pitch_diagnostics.pitch_audit.salience_and_harmonics
python -m training.pitch_diagnostics.pitch_audit.counterfactual
python -m training.pitch_diagnostics.pitch_audit.phase_and_recording
python -m training.pitch_diagnostics.pitch_audit.visualize
```

See [`docs/step_16_acoustic_pitch_audit.md`](docs/step_16_acoustic_pitch_audit.md) — primary diagnosis `TEMPORAL_RESOLUTION_LIMITED` (the decoder smooths/staircases away short-timescale, direction-reversing motion; register- and lag-correction counterfactuals both tested at ≈0 downstream effect), recommendation: investigate the Viterbi movement-cost decoder's smoothing behavior in Step 17.

---

## CNN spectrogram-classification pipeline (early approach)

This was the original approach: classify the *shape* of a 1-second CQT spectrogram directly, without building a continuous framewise representation first. It's still fully runnable and independent of the canonical/framewise pipeline above, but reuses that pipeline's Step 1/Step 3 annotation survey rather than repeating it, so it starts at export.

### a. Export the raw spectrogram dataset

This step currently lives in **`test.ipynb`** rather than a standalone script. Run its cells in order; the multi-piece export cell downloads audio for up to `MAX_EXPORT_PIECES` transcriptions (default 20), skips pieces without an `audioID` or without any non-silent trajectory, and writes:

```text
output/cnn_dataset/<piece_id>/images/<traj_index>_<label>.png
output/cnn_dataset/<piece_id>/clips/<traj_index>_<label>.wav
output/cnn_dataset/<piece_id>/metadata.csv
output/cnn_dataset/<piece_id>/skipped.csv
output/cnn_dataset/all/metadata.csv      # combined across pieces
output/cnn_dataset/all/skipped.csv
```

Raw source audio lands in `output/<piece_title>_<piece_id>.wav`, which the later steps locate by the `_<piece_id>.wav` suffix. Keep those files — every downstream variant is re-rendered from them.

If `all/metadata.csv` is ever lost or a per-piece `metadata.csv` is missing, rebuild the combined file from the image filenames:

```bash
python dataset/combine_cnn_metadata.py --cnn-dir output/cnn_dataset
```

### b. Export denoised and vocals variants

```bash
python dataset/export_denoised_cnn_dataset.py
```

For each piece found under `output/cnn_dataset/`, this script:

1. Locates the raw WAV in `output/`.
2. Reads the transcribed track's instrument and routes accordingly. Only `Vocal_M` / `Vocal_F` pieces get karaoke separation; instrumental pieces (Sitar, Sarangi, …) are denoised only, and any stale vocals directory for them is removed.
3. Denoises the full track, then runs separation **on the denoised mix** (not the raw one).
4. Caches stems in `output/denoised/<piece_id>/` as `<base>.denoised.wav`, `<base>.noise.wav`, `<base>.vocals.wav`, `<base>.accompaniment.wav`, so reruns are cheap.
5. Re-renders the exact same trajectory windows into `output/cnn_dataset_denoised/` and `output/cnn_dataset_vocals/`, then combines metadata.

Useful flags:

```bash
python dataset/export_denoised_cnn_dataset.py --piece-ids ID1 ID2      # limit to specific pieces
python dataset/export_denoised_cnn_dataset.py --skip-denoise           # reuse cached stems only
python dataset/export_denoised_cnn_dataset.py --vocals-only            # skip the denoised variant
python dataset/export_denoised_cnn_dataset.py --force-vocal-separation # re-run separation, ignore cache
python dataset/export_denoised_cnn_dataset.py --vocal-model MODEL.ckpt --denoise-model MODEL.ckpt
```

Full-track processing takes several minutes per recording on CPU. Failures are printed per piece and skipped, so one bad recording does not stop the batch.

### c. Train a single model

```bash
python training/train_cnn.py \
  --metadata output/cnn_dataset/all/metadata.csv \
  --split-by piece_id \
  --run-name baseline-raw
```

Key options:

| Flag | Purpose |
|------|---------|
| `--split-by piece_id` | Group splits by recording so no performance leaks across train/test. Falls back to a stratified label split (with a warning) if fewer than 3 pieces exist. |
| `--split-by label` | Stratified split — only appropriate for single-recording experiments. |
| `--exclude-labels silent` | Drop classes and renumber the label map. |
| `--train-ratio` / `--val-ratio` | Default 0.8 / 0.1, remainder is test. |
| `--epochs` / `--patience` | Default 50 epochs, early stopping after 10 epochs without validation macro-F1 improvement. |
| `--batch-size` / `--lr` / `--seed` | Default 16 / 1e-3 / 42. |

Class weights are computed from the training fold to offset imbalance. Augmentation is intentionally limited to mild color jitter — flipping or cropping a spectrogram would destroy the pitch contour being classified.

Each run writes to `output/cnn_runs/<slug>/`:

```text
best_model.pt              # checkpoint with embedded class_names + label_to_idx
history.csv                # per-epoch loss/accuracy/macro-F1
split_info.json            # split strategy, sizes, seed, class config
classification_report.txt
confusion_matrix.png
misclassified.png
summary.json               # best epoch, test accuracy, test macro-F1
```

Run inference on any PNGs with the saved checkpoint:

```bash
python training/predict_cnn.py \
  --checkpoint output/cnn_runs/baseline-raw/best_model.pt \
  output/cnn_dataset/<piece_id>/images
```

### d. Run the A/B comparison

```bash
python denoise/run_ab_train_eval.py                         # all three variants
python denoise/run_ab_train_eval.py --variants raw denoised # subset
python denoise/run_ab_train_eval.py --seed 7 --epochs 80
```

This is what makes the comparison fair. Before training, it inner-joins the variant metadata on `(piece_id, traj_index)` so every arm sees exactly the same examples, writing aligned copies to `output/ab_aligned_metadata/metadata_{raw,denoised,vocals}.csv`. Alignment always runs against *all* available variants even when you train a subset, so row keys stay consistent across arms.

It then invokes `training/train_cnn.py` once per variant with a shared seed and `--split-by piece_id`, and collects results into `output/ab_denoise_runs/ab_comparison.csv`.

### e. Inspect the errors

```bash
python denoise/compare_misclassified_raw_denoised.py \
  --run-dir output/ab_denoise_runs/ab-denoised \
  --error-patterns "0->1" "3->1" \
  --max-examples 12
```

Reloads the denoised model, reconstructs the identical test split (same seed, same grouping), collects predictions, filters to the confusion patterns you name, and renders side-by-side raw-vs-denoised spectrograms for exactly those clips. The defaults target Fixed and Sloped-End being predicted as Simple Bend. Output goes to `output/misclassified_comparisons/` as one PNG grid per pattern plus `misclassified_summary.csv`.

The point is to see whether denoising erased the visual cue the model needed.

### f. Silence detection baseline

A separate task on the same data: segment full recordings into silent / non-silent regions, using the `silent` IDTAP labels as ground truth so no new annotation is needed.

```bash
python silence/eval_silence_detection.py                          # raw audio (default)
python silence/eval_silence_detection.py --audio-variant denoised
python silence/eval_silence_detection.py --audio-variant all      # raw + denoised + vocals
python silence/eval_silence_detection.py --audio-variant denoised --skip-denoise
```

Two interchangeable detection methods live in `silence_detection.py`:

- **`rms`** — frames are silent when RMS falls below `threshold_db` *relative to that recording's peak*. Swept over −50 … −25 dB.
- **`librosa_split`** — calls `librosa.effects.split` and inverts the non-silent intervals. `top_db` swept over 20 … 60.

Both merge adjacent same-label frames, drop runs shorter than `min_duration`, and fill gaps so the segmentation always tiles the whole recording. `label_interval()` converts a segmentation into a per-clip verdict by majority overlap, which is what makes it comparable to the 1-second IDTAP clip labels.

The harness splits **pieces** 60/20/20, tunes each method's threshold on the validation pieces only, then reports test-split metrics. Results per variant land in `output/silence_detection/<variant>_baseline/`:

```text
param_sweep_rms.csv                  # metrics at every threshold
param_sweep_librosa.csv
comparison.csv                       # test metrics for both methods
per_piece_metrics.csv
confusion_matrix_rms.png
confusion_matrix_librosa_split.png
summary.json                         # split membership, chosen params, stem paths used
```

With `--audio-variant all`, a `cross_variant_comparison.csv` is also written at `output/silence_detection/`.

Other flags: `--force-vocal-separation` (run karaoke on instrumental pieces too), `--skip-idtap` (do not query IDTAP for instrument routing), `--sr`, `--hop-length`, `--min-duration`, `--seed`.

### g. Listen to the results

Open `test_silence_detection.ipynb`, set `PIECE_ID`, `AUDIO_VARIANT`, and `METHOD` in the config cell, and run it. It loads the tuned parameters from the matching `summary.json`, plots the waveform with silent regions shaded gray and non-silent shaded yellow, then plays back each detected section so you can hear whether the labels are right.

---

## Output map

| Path | Contents |
|------|----------|
| `output/<title>_<piece_id>.wav` | Raw downloaded source audio |
| `output/denoised/<piece_id>/` | Cached denoise / separation stems |
| `output/cnn_dataset/` | Raw-variant spectrograms, clips, metadata |
| `output/cnn_dataset_denoised/` | Denoised variant |
| `output/cnn_dataset_vocals/` | Vocals variant (vocal pieces only) |
| `output/ab_aligned_metadata/` | Row-aligned metadata for the A/B arms |
| `output/misclassified_comparisons/` | Raw-vs-denoised error grids |
| `output/silence_detection/` | Silence detection sweeps, metrics, confusion matrices |
| `output/inventory/` | Annotation inventory + counts + failures CSVs |
| `output/canonical/v1/` | Canonical recording documents, verbatim IDTAP cache, index CSVs, split manifests |
| `output/canonical/v1/features/` | Precomputed CQT log-magnitude on 10 ms grid |
| `output/framewise_runs/` | Framewise TCN/BiGRU training runs |
| `output/pitch_diagnostics/` | Step 10 pitch-frontend diagnostics (audits, baselines, pitch-only runs) |
| `output/cnn_runs/` | Standalone training runs |
| `output/ab_denoise_runs/` | A/B runs + `ab_comparison.csv` |
| `.cache/recordings/` | Full recordings cached by the legacy `build_dataset.py` |

Everything under `output/`, `.cache/`, and all audio is gitignored. Regenerate locally after cloning.

---

## Notebooks

All notebooks live in `notebooks/` and resolve the repository root themselves, so they work whether Jupyter is launched from `notebooks/` or from the repository root.

| Notebook | Purpose |
|----------|---------|
| `f0s.ipynb` | The go/no-go decision behind the whole design. Sweeps `librosa.pyin` configs, compares extracted F0 against IDTAP ground-truth contours in cents, and applies preregistered thresholds (abandon above 60 cents mean MAE or below 55% voiced). This is why the pipeline classifies spectrogram *images* rather than tracked pitch curves. |
| `test.ipynb` | CQT/spectrogram prototype; **also the current raw dataset exporter** (CNN pipeline step a). |
| `test_denoise.ipynb` | Qualitative and quantitative denoising comparison: listen to raw vs. denoised vs. vocals clips, plus RMS, high-frequency energy in dB, and spectral MSE per trajectory. |
| `test_silence_detection.ipynb` | QA loop for CNN pipeline step f — shaded waveform plots plus playback of each detected section. |
| `canonical_dataset_findings.ipynb` | The findings report for Step 2. Seven findings computed live from `output/canonical/v1/`: which annotation fields are actually populated, the gap-free tiling of the timeline, transcription density versus the `Silent` label, a real annotated timeline with pitch contours reconstructed from the stored shape parameters, the raw-versus-derived round-trip proof, the Type-0 canonicalization overlay, and grouped-split leakage. Reads from disk only — no IDTAP access needed. |

---

## Current dataset status

Exported from 17 transcriptions (11 of which are vocal recordings eligible for separation):

| Variant | Rows | Pieces | `0` | `1` | `2` | `3` | `silent` |
|---------|------|--------|-----|-----|-----|-----|----------|
| raw | 4859 | 17 | 1297 | 1351 | 448 | 432 | 1331 |
| denoised | 5057 | 17 | 1361 | 1437 | 468 | 439 | 1352 |
| vocals | 3706 | 11 | 901 | 1160 | 303 | 370 | 972 |

Classes `2` (Sloped Start) and `3` (Sloped End) are roughly 3× rarer than the others, which is why class weighting and macro-F1 (rather than accuracy) are used throughout. Note that the variants have different row counts — this is expected, and exactly why the CNN pipeline's A/B comparison step (step d) aligns them before training.

---

## Known issues

- **Trajectory classification is still near chance.** Latest A/B runs sit around 0.34 test accuracy and 0.23–0.31 macro-F1 across all three variants, with no clear winner. Denoising has not yet demonstrated a benefit; see `ab_comparison.csv` and the per-run `classification_report.txt` for current numbers.
- **Silence detection barely fires on raw and denoised audio.** Silent-class recall is 3–7%, so accuracy near 0.75 just reflects the majority class. The vocals stem is the clear exception and the most promising lead: silent-class recall jumps to 0.64 (`rms`) and 0.73 (`librosa_split`), with F1 around 0.37–0.39.
- **The raw and denoised silence baselines report identical metrics** down to full float precision, even though `summary.json` confirms different stems were loaded. Since the vocals arm does differ, this looks specific to the raw/denoised pair rather than a wiring problem across the board — worth confirming before reading anything into their comparison.
- **Only the first segment of a composite trajectory is exported.** `iter_labeled_segments` computes all segments, but `trajectory_export_label` returns `segments[0][0]`, so decomposed labels for IDs 4/5/6 are not yet fully used.

---

## Superseded paths

These files reflect earlier designs and are **not** part of the live pipeline. They are kept for reference:

- `legacy/build_dataset.py` — a four-class (`trajectory_1`…`trajectory_4`) WAV-clip dataset written to `data/audio/`, capped at 100 examples per class, duration-filtered to 0.15–5.0s. Predates the spectrogram approach. Its label map and candidate iteration live on in `inventory/dataset_utils.py`, which `survey_trajectories.py` still uses.
- `legacy/validate_metadata.py` and `legacy/create_clips.py` — operate on a third, hand-annotated schema (`example_id`, `recording_file`, `confidence`, `notes`) intended for manual annotation. Useful if you ever return to hand-labeling.
- `legacy/trajectory_labels.yaml` — placeholder for an IDTAP-ID-to-class mapping that was never filled in; the live mapping lives in `training/spec_dataset.py` and `dataset/export_denoised_cnn_dataset.py`.
- `legacy/trajectory-classification-README.md` — documents the original "Step 1A / Step 1B" inventory-then-hand-annotate workflow, from when training lived in its own subproject. Step 1A (the inventory export) is still current and described above; Step 1B describes the hand-annotation schema that was never adopted. Its paths refer to the old layout.
- `legacy/trajectory-classification-requirements.txt` — the old subproject's dependency list, now a subset of the root `requirements.txt`.
- `legacy/test.py` — earliest IDTAP smoke test.

Note that one active script still depends on `inventory/dataset_utils.py`, so that file stayed in `inventory/` rather than moving to `legacy/`.
