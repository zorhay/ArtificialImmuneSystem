# General training improvements: fixed decay and earlier stops

The first experiment, updating only on mistakes, reduced mean validation accuracy from 73.21% to 69.69%. This follow-up tests the other planned ideas: progressively smaller updates and selecting how long to train, using the same settings for every digit.

Each run starts from the same reference templates and ordered images within its seed. Every image updates one closest correct-class template. The recognition threshold stays at 0.3, with 21 templates per digit and intensity threshold >200. Seeds are 42, 43, and 44. The mutation rate is either fixed at 10%, or falls linearly from 10% to 3% over the first 500 images of each class. Earlier stops use the prefix of that same rate schedule.

Training is replayed independently for 100, 250, or 500 images per class. The stopping point is selected using mean accuracy across all digits and seeds on the original 5,000-image validation set. This uses no class-specific weighting or manual choice based on digit 5. The older confirmation set and official test set are not evaluated.

## Validation results

| Update rate / images per digit | Seed 42 | Seed 43 | Seed 44 | Mean | SD (pp) |
|---|---:|---:|---:|---:|---:|
| constant-500 | 72.94% | 73.58% | 73.12% | 73.21% | 0.33 |
| constant-100 | 69.38% | 70.26% | 71.28% | 70.31% | 0.95 |
| constant-250 | 70.94% | 71.42% | 70.76% | 71.04% | 0.34 |
| decay-100 | 70.14% | 69.14% | 71.16% | 70.15% | 1.01 |
| decay-250 | 70.44% | 71.14% | 72.12% | 71.23% | 0.84 |
| decay-500 | 72.04% | 72.40% | 72.76% | 72.40% | 0.36 |

Validation-selected configuration: **constant-500**.

**None of the new configurations improved mean validation accuracy.** The fresh holdout was not evaluated. Keep the current production model.

## Interpretation limits

This is a small, fixed comparison, not an exhaustive search. The three seeds share the same images. Replaying different prefix lengths changes later-class random draws, so this is a comparison of complete training runs rather than a single uninterrupted run with intermediate snapshots. No production model or training method is replaced by this script.

Evaluation uses the previously checked exact-overlap cache and production voting; every model/evaluation also checks 100 images against coordinate-list recognition. Full metrics, per-digit errors and confusion matrices are in `validation.json` and, if used, `fresh.json`. Exact model metadata and predictions are under `artifacts/training-schedule/`.

Run: `.venv/bin/python -m experiments.training_schedule --workers 3`.
