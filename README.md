# Artificial Immune System

Run support for the original handwritten-digit classifier. The original training,
mutation, pixel-affinity calculation, and exponential voting are preserved.

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
`os.listdir` order. Every cell of that class receives the original mutation.

The CLI uses the original recognition script's settings: 21 cells per digit and
reaction affinity 0.8. `--templates` and `--affinity` allow explicit overrides.
The original `main.py` retains its own settings of 20 cells and affinity 0.4.
There are no tuned defaults, epochs, validation selection, or new normalization
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

## Preserved original behavior

The mutation logic still adds selected pixels to the black list without removing
them from the white list. Network state remains class-level. An all-zero vote
returns the first class in dictionary order. Exact-match votes can be infinite.
Blank images can raise division-by-zero errors. These behaviors were deliberately
left intact; this is not the redesigned classifier from the earlier experiment.

The core algorithm methods were restored from Git. The old cropping/filtering
helpers were restored too; their existing behaviors and bugs are not repaired.
Do not run the legacy bulk preprocessing helpers on the prepared MNIST folders
unless you intend to modify those images in place.

## Outputs and verification

- `data/mnist/`: downloaded archives and original exported images.
- `artifacts/original/initial.pkl`: network before training.
- `artifacts/original/model.pkl`: trained original network.
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
and preservation of the original scoring/mutation behavior. They do not assert
that the original algorithm's bugs have been fixed.
