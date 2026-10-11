# Artificial Immune System

Run support for the handwritten-digit classifier. The original all-cells training
loop, pixel-affinity calculation, and exponential voting remain. Mutation now moves
pixels correctly in both directions, the default reaction threshold is 0.4, and
inputs with no votes return `unrecognized`.

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
python -m ImageRecognition train --train-per-class 100
python -m ImageRecognition evaluate --test-per-class 10
python -m ImageRecognition predict data/mnist/images/test/7/00000.png --model artifacts/original/model.pkl
```

Omit `--test-per-class` to evaluate all 10,000 test images. The original Python
pixel-list implementation is slow; full runs can take substantially longer than
the earlier modified classifier. Each training call makes one pass through the
first `--train-per-class` files in each digit folder, in the original
`os.listdir` order. Every cell of that class receives the corrected mutation.

The CLI uses 21 cells per digit and reaction affinity 0.4, lowered from 0.8.
`--templates` and `--affinity` allow explicit overrides. Prediction and evaluation
also accept `--affinity`; their default of 0.4 overrides an older model's saved
threshold in memory without rewriting the model file. Reports record both values.
The original `main.py` retains its own settings of 20 cells and affinity 0.4.
The new 0.4 threshold has not been tuned or benchmarked on the full test set.
There are no epochs, validation selection, or new normalization
options. Without `--model`, prediction loads reference images as the original
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
- A run wrapper for saving/loading the original network's class-level state.
  It uses pickle; load only trusted locally generated model files.
- Simple evaluation reports and tests of run support and original behavior.

## Training behavior and current limitations

Initialization loads the first 21 image files per digit as separate templates.
For each labeled training image, every template of its class is mutated toward
that image: 10% of missing white pixels are randomly added, and 10% of extra white
pixels are randomly removed. The black lists are updated in the opposite direction.
There is no affinity-based template selection during training. The reaction
threshold affects inference only; lowering it does not change learning.

The CLI defaults to 100 training examples per digit (1,000 total), but the saved
model evaluated at 11.47% used only two per digit (20 total). Neither run setting
uses all 60,000 available training images. Directory order is not shuffled, there
is one training pass, and there is no validation/checkpoint selection. Every cell
sees the same sequence, which can reduce template diversity and make results
sensitive to recent examples. Training does not use prediction errors or examples
from other classes to decide whether an update is beneficial. Integer rounding
means fewer than ten mismatches in a direction produce no change at power 0.1.

Previously trained files still contain the effects of the old mutation bug.
Retrain from source images into a separate output directory to obtain corrected
templates; changing the inference threshold does not repair their stored pixels.

An all-zero vote, including a blank image, returns the literal string
`unrecognized`. This is an output category, not an additional trained digit.
Evaluation includes rejected images as incorrect in overall accuracy and reports
coverage and accuracy on recognized inputs separately. The legacy
`test_recognition_symbol` count list now has 11 entries; index 10 counts unrecognized.
Network state remains class-level, and exact-match votes can still be infinite.

The old cropping/filtering helpers remain; their existing behaviors and bugs are not repaired.
Do not run the legacy bulk preprocessing helpers on the prepared MNIST folders
unless you intend to modify those images in place.

## Outputs and verification

- `data/mnist/`: downloaded archives and original exported images.
- `artifacts/original/initial.pkl`: network before training.
- `artifacts/original/model.pkl`: saved network (existing files are not automatically retrained).
- `artifacts/original/training.json`: run settings and exact input filenames.
- `reports/original/report.md`, `metrics.json`, `predictions.csv`: evaluation.

Use `--output-dir`, `--report-dir`, and `--model` to choose other locations.
Repeat runs overwrite outputs in those locations. Datasets and model files are
ignored by Git. The previous redesigned model and its 91.61% report have been
removed from active project outputs; that score does not describe this version.

```sh
python -m pytest -q
```

Tests check compatibility, data export, persistence across processes, CLI usage,
the original scoring/all-cells training loop, corrected pixel mutation,
and unrecognized outputs and their evaluation denominators. Reports under
`reports/evaluation-current/` describe commit `d13bbca`, before these three changes;
their accuracy numbers do not describe the updated code.
