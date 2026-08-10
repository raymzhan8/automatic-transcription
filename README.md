# Automatic Transcription — Trajectory Classification

Pipeline for building pitch-trajectory classification datasets from [IDTAP](https://idtap.org) (Swara Studio) transcriptions, with optional vocal denoising/separation and CNN-based classification.

## Project layout

```text
.
├── build_dataset.py              # Phase 1: export labeled audio clips from IDTAP
├── dataset_utils.py              # Shared dataset-building helpers
├── export_denoised_cnn_dataset.py # Denoise/separate vocals and export CNN spectrogram datasets
├── run_ab_train_eval.py          # A/B train/eval: raw vs denoised vs vocals
├── survey_trajectories.py        # Survey IDTAP trajectory type distribution
├── scripts/
│   └── debug_idtap.py            # IDTAP authentication and API debugging
├── trajectory-classification/    # CNN training, prediction, and dataset tooling
│   ├── scripts/                  # train_cnn.py, predict_cnn.py, create_clips.py, …
│   ├── src/                      # models, spec_dataset, idtap_io
│   └── configs/                  # trajectory label mappings
├── output/                       # Generated datasets (gitignored)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

IDTAP authentication uses Google login via the `idtap` package:

```python
from idtap import login_google, SwaraClient

login_google()
client = SwaraClient()
```

## Workflow

### 1. Build base dataset from IDTAP

```bash
python build_dataset.py
```

Exports labeled trajectory clips and metadata under `output/cnn_dataset/`.

### 2. Export denoised / vocals variants

```bash
python export_denoised_cnn_dataset.py
```

Runs the [vocalprep](https://github.com/jon-myers/denoising-experiments) pipeline (Mel-Roformer denoise + karaoke separation) and writes spectrogram images to `output/cnn_dataset_denoised/` and `output/cnn_dataset_vocals/`.

### 3. Train and compare models

```bash
python run_ab_train_eval.py
```

Trains CNN classifiers on raw, denoised, and vocals datasets and writes comparison metrics to `trajectory-classification/outputs/ab_denoise_runs/`.

See [`trajectory-classification/README.md`](trajectory-classification/README.md) for detailed dataset preparation, metadata validation, and standalone CNN training instructions.

## Classes

Four pitch-trajectory classes mapped from IDTAP annotations:

| Label | IDTAP type |
|-------|------------|
| `trajectory_1` | Fixed |
| `trajectory_2` | Bend: Simple |
| `trajectory_3` | Bend: Sloped Start |
| `trajectory_4` | Bend: Sloped End |

## Notes

- Generated data (`output/`, `.cache/`, audio files) is excluded from git — regenerate locally after cloning.
- Exploratory notebooks (`*.ipynb`) are kept for reference but are not required to run the pipeline.
