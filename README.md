# Artificial Immune System

Handwritten-digit recognition using mutable reference templates. Each training
image updates one closest template of the correct digit, retaining the original
pixel-affinity formula and exponential voting. The default reaction threshold is 0.3, and
inputs with no votes or tied maximum votes return `unrecognized`. Models own
independent networks. Exponential voting uses finite relative scores, and mutation
uses stochastic rounding to preserve the expected 10% rate for small differences.

## Install

Tested with Python 3.13.5. Python 3.9+ is supported.

```sh
uv venv --python 3.13
uv pip install -e '.[dev]'
source .venv/bin/activate
```

Alternatively, create a virtual environment with Python and run
`python -m pip install -e '.[dev]'`. Runtime dependencies are NumPy and Pillow.
`requirements-lock.txt` records the installed versions for Python 3.13.

## Prepare the data

```sh
python -m ImageRecognition prepare
```

This verifies/downloads the four original MNIST archives from the
[CVDF mirror](https://github.com/cvdfoundation/mnist), then exports the original
28×28 grayscale pixels as PNGs under `data/mnist/images/train/0..9` and
`data/mnist/images/test/0..9`. All 70,000 images are retained. Exporting the PNGs
uses more disk space than the compressed archives. `download` fetches only archives.

The original code requires class folders; PNG export does not crop, resize,
normalize, invert, or binarize the dataset. The original image loader performs
its own thresholding at 200.

## Run

A small train → save → load → evaluate check:

```sh
python -m ImageRecognition run --train-per-class 2 --test-per-class 1
```

Separate steps:

```sh
python -m ImageRecognition train --train-per-class 500
python -m ImageRecognition evaluate --test-per-class 10
python -m ImageRecognition predict data/mnist/images/test/7/00000.png --model artifacts/closest-template/model.pkl
```

Omit `--test-per-class` to evaluate all 10,000 test images. The original Python
pixel-list implementation is slow; full runs can take substantially longer than
the earlier modified classifier. Each training call makes one pass through the
first `--train-per-class` files in each digit folder, in the original
`os.listdir` order. Only the closest template of the correct class receives the
mutation. Training ties are resolved randomly using the run seed.

The CLI uses 21 templates per digit, 500 training images per digit, and reaction affinity 0.3.
`--templates` and `--affinity` allow explicit overrides. Prediction and evaluation
also accept `--affinity`; their default of 0.3 overrides an older model's saved
threshold in memory without rewriting the model file. Reports record both values.
The legacy `main.py` initialization also uses 21 templates and affinity 0.3.
The ordinary CLI does not select configurations on validation data; use the
controlled benchmark below for that comparison. There are no epochs or new
normalization options. Without `--model`, prediction loads reference images as the original
recognition script did. Input should already be an aligned 28×28 single digit.

The old script entry points remain available:

```sh
python ImageRecognition/src/main.py
python ImageRecognition/src/recognition.py data/mnist/images/test/7/00000.png
```

## Retained changes

- Packaging, dependencies, package-compatible imports, and CLI commands.
- Portable data paths. CLI commands accept `--data-dir`; legacy scripts honor
  `AIS_DATA_DIR`. Defaults resolve from the repository location.
- MNIST download/checksum verification, IDX reading, and lossless PNG export.
- Python compatibility: convert sets to lists when calling `random.sample`.
- NumPy compatibility: explicitly copy image arrays before mutating their pixels.
- Pillow compatibility: use `Image.Resampling.LANCZOS` in place of removed
  `Image.ANTIALIAS`. Removed an unused third-party `resizeimage` import.
- A run wrapper for saving/loading networks, including older class-state artifacts.
  It uses pickle; load only trusted locally generated model files.
- Simple evaluation reports and tests of run support and original behavior.

## Training behavior and current limitations

Initialization loads the first 21 image files per digit as separate templates.
For each labeled training image, the closest template of its class is mutated
toward that image: 10% of missing white pixels are randomly added, and 10% of extra white
pixels are randomly removed. The black lists are updated in the opposite direction.
Similarity chooses the template to update, with seeded random tie breaking. The
reaction threshold affects inference only; lowering it does not change learning.

The CLI defaults to 500 training examples per digit (5,000 total), but the saved
model evaluated at 11.47% used only two per digit (20 total). Neither run setting
uses all 60,000 available training images. Directory order is not shuffled, there
is one training pass, and there is no validation/checkpoint selection. Templates
receive different subsequences of similar examples, but frequent updates can
still overwrite earlier information. Training does not use prediction errors or examples
from other classes to decide whether an update is beneficial. Stochastic rounding
avoids freezing small differences: one mismatch changes with probability 0.1,
while ten mismatches select exactly one pixel, at power 0.1.

Older files may retain the old mutation bug or the all-template training behavior.
Retrain from source images into a separate output directory to obtain corrected
templates; changing the inference threshold does not repair their stored pixels.

An all-zero vote, including a blank image, returns the literal string
`unrecognized`. This is an output category, not an additional trained digit.
Evaluation includes rejected images as incorrect in overall accuracy and reports
coverage and accuracy on recognized inputs separately. The legacy
`test_recognition_symbol` count list now has 11 entries; index 10 counts unrecognized.
Networks and counters belong to each instance. Finite exponential weights are
scaled by a common factor per image to avoid overflow. Exact matches dominate
finite matches and vote by count. Equal maximum scores return `unrecognized`.
Reported counters are relative vote totals, not probabilities.

## Closest-template training results

The selected update rule reached 72.25% mean accuracy on a separate 5,000-image
confirmation set across seeds 42, 43, 44, compared with 70.17% before training and
64.45% for all-template training. Each method used its validation-selected threshold.
The +2.08 percentage-point gain over the initial templates is modest. These are
confirmation results, not the previous official-test scores.

See `reports/training-analysis/report.md` for controlled comparisons, limitations,
and template images. The experimental loop is retained for reproducibility:

```sh
python -m experiments.analyze_training --workers 3
```

The normal CLI retains directory order. To reproduce the exact experiment data
selection and verify that production training creates identical templates:

```sh
python -m experiments.verify_applied_training
```

This uses the saved experimental manifests and initial networks. It writes fresh
production models under `artifacts/closest-template/seed-42`, `seed-43`, and `seed-44`,
with seed 42 also at the default `artifacts/closest-template/model.pkl` location.
The pixel lists must match all 210 experimental templates per seed exactly.
It does not reevaluate accuracy. The saved research artifacts must already exist.

## Historical all-template benchmark

```sh
python -m ImageRecognition.src.benchmark --workers 3
```

This explicitly opts into `update_selection="all"` to preserve the historical
control even though default training now updates one closest template. It uses:
100, 500, and 1,000 images per digit; 21 templates per digit; seeds 42, 43, 44;
reaction thresholds 0.3, 0.4, 0.5; mutation power 0.1; pixel threshold 200.
A fixed 5,000-image validation split is excluded from both initialization and
training. Mean validation accuracy selects one size/threshold configuration;
the selected models and their untrained controls then run on the full official
10,000-image test set. No test-set configuration selection or retraining occurs.

The benchmark computes the same integer pixel overlaps in batches and verifies
exact agreement with the original coordinate-list affinity and recognition on
100 images per model/split at every evaluated threshold. It calls the production
vote routine for all predictions. This speeds evaluation without altering training.

Results are in `reports/current-approach/report.md`, with the complete grid,
per-seed/per-digit metrics, and confusion matrices in adjacent JSON files.
`artifacts/current-approach/` contains models, split and ordered training indices,
and per-image predictions. These are separate from historical artifacts.

The old cropping/filtering helpers remain; their existing behaviors and bugs are not repaired.
Do not run the legacy bulk preprocessing helpers on the prepared MNIST folders
unless you intend to modify those images in place.

## Outputs and verification

- `data/mnist/`: downloaded archives and original exported images.
- `artifacts/closest-template/initial.pkl`: network before training.
- `artifacts/closest-template/model.pkl`: saved network (existing files are not automatically retrained).
- `artifacts/closest-template/training.json`: run settings and exact input filenames.
- `reports/closest-template/report.md`, `metrics.json`, `predictions.csv`: evaluation.

Use `--output-dir`, `--report-dir`, and `--model` to choose other locations.
The new default locations preserve the historical `artifacts/original/` and
`reports/original/` outputs. Repeat runs overwrite outputs in their chosen locations. Datasets and model files are
ignored by Git. The previous redesigned model and its 91.61% report have been
removed from active project outputs; that score does not describe this version.

```sh
python -m pytest -q
```

Tests check compatibility, data export, persistence across processes, CLI usage,
the original affinity, closest-template training and the explicit all-template control,
stable weighted voting, corrected
pixel mutation, model isolation, split isolation, evaluation equivalence,
and unrecognized outputs and their evaluation denominators. Reports under
`reports/evaluation-current/` describe commit `d13bbca`, before these three changes;
their accuracy numbers do not describe the updated code.
