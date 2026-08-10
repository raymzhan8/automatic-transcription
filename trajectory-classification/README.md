# Trajectory Classification — Step 1: Dataset Preparation

This subproject organizes and validates annotated pitch-trajectory examples from Indian classical music recordings.

- **Step 1A** exports an inventory of existing IDTAP trajectory annotations.
- **Step 1B** covers manual metadata validation and clip generation from local recordings.

Feature extraction, pitch tracking, and model training are not implemented yet.

## Project layout

```text
trajectory-classification/
├── configs/
│   └── trajectory_labels.yaml   # Future four-class IDTAP mapping (unfilled)
├── data/
│   ├── recordings/              # Full-length source audio files
│   ├── clips/                   # Generated annotation clips
│   └── metadata.csv             # One row per annotated example
├── outputs/
│   ├── trajectory_inventory.csv # Detailed IDTAP trajectory export
│   ├── trajectory_counts.csv    # Summary counts by IDTAP type
│   └── failed_transcriptions.csv
├── scripts/
│   ├── inspect_idtap_trajectories.py
│   ├── validate_metadata.py
│   └── create_clips.py
├── src/
│   ├── idtap_io.py
│   └── trajectory_inventory.py
├── label_definitions.md
├── requirements.txt
└── README.md
```

## Step 1A: IDTAP trajectory inventory

IDTAP already contains manually annotated pitch trajectories. Step 1A inspects those existing annotations through the IDTAP Python API and exports a complete inventory without creating new labels or downloading audio.

### Authenticate with IDTAP

Install dependencies:

```bash
pip install -r requirements.txt
```

Authentication generally follows this pattern:

```python
from idtap import login_google, SwaraClient

login_google()
client = SwaraClient()
```

`SwaraClient()` also attempts to reuse stored tokens when available. The inventory scripts never log authentication tokens or private credentials.

The installed API uses:

- `client.get_viewable_transcriptions()` to list accessible transcriptions
- `client.get_piece(piece_id)` to load one transcription
- `Piece.from_json(...)` to convert the JSON into an IDTAP `Piece`

### Run the inventory script

From the project root (`trajectory-classification/`):

```bash
python scripts/inspect_idtap_trajectories.py
python scripts/inspect_idtap_trajectories.py --max-pieces 10
python scripts/inspect_idtap_trajectories.py --piece-id YOUR_PIECE_ID
```

The script writes:

- `outputs/trajectory_inventory.csv` — one row per trajectory annotation
- `outputs/trajectory_counts.csv` — counts and duration stats by IDTAP trajectory type
- `outputs/failed_transcriptions.csv` — transcriptions that could not be loaded

The inventory preserves the original IDTAP trajectory ID and human-readable name (for example, `0` / `Fixed`, `12` / `Silent`). It includes all trajectory types, including fixed and silent annotations.

Absolute start times come from `piece.traj_start_times(inst=..., string_idx=...)`. End times are computed as:

```text
trajectory_end_time = trajectory_start_time + trajectory_duration
```

Validation warnings are printed when trajectory IDs, names, durations, or timing arrays look inconsistent. Processing continues when individual transcriptions fail.

### Define machine-learning classes later

The four machine-learning trajectory classes are **not** decided in code. After reviewing `outputs/trajectory_inventory.csv`, fill in [`configs/trajectory_labels.yaml`](configs/trajectory_labels.yaml) to map IDTAP trajectory IDs into four classes. ID 12 (`Silent`) is listed as excluded by default.

Audio clip generation from the inventory is deferred to the next step. Step 1B below still covers the local recording / metadata workflow.

## Step 1B: Local recordings and clip generation

## Where full recordings belong

Place source audio files in [`data/recordings/`](data/recordings/). Each file name must match the `recording_file` column in `metadata.csv` exactly (for example, `my_recording.wav`).

Supported workflow:

1. Add or copy full recordings into `data/recordings/`.
2. Annotate time ranges and labels in `metadata.csv`.
3. Validate the metadata.
4. Generate clips.

The sample [`data/metadata.csv`](data/metadata.csv) references placeholder filenames. Validation will report missing files until you replace those entries with real recordings.

## How to fill out metadata.csv

Each row describes one annotated trajectory example.

| Column | Description |
|---|---|
| `example_id` | Unique identifier for the example (used as clip filename) |
| `recording_file` | Source file name under `data/recordings/` |
| `start_time` | Annotation start time in seconds |
| `end_time` | Annotation end time in seconds (must be greater than `start_time`) |
| `label` | One of: `trajectory_1`, `trajectory_2`, `trajectory_3`, `trajectory_4` |
| `performer_id` | Stable identifier for the performer (e.g. slug or catalog ID) |
| `recording_id` | Stable identifier for the source recording/session |
| `tonic_hz` | Tonic frequency in Hz for the recording (optional but recommended) |
| `confidence` | Annotation confidence: `high`, `medium`, or `low` |
| `notes` | Free-text annotation notes, ambiguity notes, or exclusion rationale |

See [`label_definitions.md`](label_definitions.md) for class definitions and annotation guidance.

Valid labels:

- `trajectory_1` — Fixed
- `trajectory_2` — Bend: Simple
- `trajectory_3` — Bend: Sloped Start
- `trajectory_4` — Bend: Sloped End

## How to validate annotations

From the project root (`trajectory-classification/`):

```bash
pip install -r requirements.txt
python scripts/validate_metadata.py
```

The validator checks:

- Required columns are present
- `example_id` values are unique
- Referenced recordings exist
- Times are numeric, nonnegative, and properly ordered
- Labels and confidence values are valid
- `end_time` does not exceed the source recording duration

All errors are printed together. The script exits with status code `1` if any validation error is found.

## How to generate clips

After validation passes (or while iterating on a subset of rows):

```bash
python scripts/create_clips.py
```

Optional flags:

- `--overwrite` — replace existing clips in `data/clips/`
- `--context SECONDS` — include equal padding before `start_time` and after `end_time`, clamped to the recording bounds

Example with half a second of context:

```bash
python scripts/create_clips.py --context 0.5
```

Clips are written to:

```text
data/clips/<example_id>.wav
```

Clips preserve the source sample rate, are saved as mono, and are not normalized or otherwise altered.

## Why recording_id and performer_id must be preserved

These identifiers support proper dataset handling beyond simple clip storage:

- **Train/validation/test splits:** Examples from the same recording or performer should not leak across splits. Grouping by `recording_id` or `performer_id` prevents the model from seeing near-duplicates during evaluation.
- **Performer-aware analysis:** Models may overfit to timbre or performance style. Tracking `performer_id` allows performer-held-out evaluation.
- **Provenance and debugging:** When pitch-track or label errors appear, IDs make it possible to trace examples back to a specific session and annotator notes.

Do not discard or overwrite these fields when adding new rows.

## Why ambiguous examples should initially be excluded

Trajectory shape classification depends on subtle pitch-contour differences. Borderline cases—where two classes are equally plausible—introduce label noise that disproportionately hurts small, specialized datasets.

Recommended practice:

- Annotate uncertain cases with `confidence: low` and detailed `notes`
- Exclude ambiguous examples from the initial training set until definitions are refined
- Revisit deferred examples after reviewing model errors and updating [`label_definitions.md`](label_definitions.md)

Starting with high-confidence, clearly prototypical examples yields cleaner baselines and makes later error analysis more informative.

## Setup

```bash
cd trajectory-classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Next steps (not yet implemented)

- Export IDTAP inventory rows into `data/metadata.csv`
- Generate clips from IDTAP-linked audio
- CQT and audio feature extraction
- Pitch tracking
- Model training and evaluation
