# Mistake-driven training experiment

A single rule was tested for every digit: predict the training image first. If the answer is correct, skip the mutation. Otherwise mutate the closest template of the correct class using the existing 10% update. Unrecognized predictions count as mistakes. Only the experimental script changes; production training and its saved models are preserved.

## Protocol

- Three seeds: 42, 43, 44; identical starting templates and ordered training images to the current baseline.
- 21 templates per digit, 500 training images per digit, one pass, mutation power 0.1, reaction threshold 0.3, intensity >200.
- Prediction during training uses the production coordinate-list affinity and voting. No confidence gate, learning-rate schedule, or checkpoint search.
- No settings or rules specific to poorly performing digits. Every class has 500 training images.
- Validation uses the original held-out 5,000 images. The old confirmation set is not evaluated or used to select a configuration.
- A fresh final holdout was reserved before running: 500 new images per digit, disjoint from every previous training, validation, and confirmation row.
- The fresh holdout opens only if mean validation accuracy improves. The official 10,000-image test set is not used.

## Validation

| Seed | Current training | Mistake-driven | Change (points) | Updates / 5,000 |
|---:|---:|---:|---:|---:|
| 42 | 72.94% | 68.46% | -4.48 | 1674 |
| 43 | 73.58% | 69.66% | -3.92 | 1708 |
| 44 | 73.12% | 70.96% | -2.16 | 1632 |

Mean: **73.21% → 69.69%** (-3.52 percentage points).

## Per-digit validation accuracy

| Digit | Current mean | Candidate mean | Change (points) | Mean updates / 500 |
|---:|---:|---:|---:|---:|
| 0 | 90.40% | 87.13% | -3.27 | 75.7 |
| 1 | 97.27% | 97.40% | +0.13 | 15.3 |
| 2 | 62.47% | 52.93% | -9.53 | 262.3 |
| 3 | 71.87% | 72.67% | +0.80 | 159.0 |
| 4 | 61.13% | 55.47% | -5.67 | 317.0 |
| 5 | 33.53% | 31.73% | -1.80 | 345.3 |
| 6 | 87.80% | 88.60% | +0.80 | 57.0 |
| 7 | 75.20% | 69.93% | -5.27 | 182.3 |
| 8 | 75.00% | 67.93% | -7.07 | 156.3 |
| 9 | 77.47% | 73.13% | -4.33 | 101.0 |

**The candidate did not beat current training on validation.** The fresh holdout was not evaluated and remains available for a later selected candidate. Do not promote this candidate on the current evidence.

## Limits and artifacts

One update policy was tested at fixed settings. It does not test whether confidence-gated updates, a decreasing mutation rate, or checkpoint selection would help. Skipping updates also changes the random-number sequence; paired seeds do not mean identical individual mutations. Class-blocked order is held fixed and could affect mistake-driven learning because updates to later classes can change earlier predictions. The three seeds share evaluation images and are not independent extra datasets. Correctness-gating alone does not explicitly push apart confused classes.

Evaluation reuses the exact-overlap calculation and production voting from the previous verified harness; 100 images per model/evaluation are checked against coordinate-list recognition. Training predictions use that coordinate-list recognition directly.

`protocol.json`, `decision.json`, `validation.json`, and (if opened) `fresh.json` contain the full settings, metrics, per-digit counts, and confusion matrices. Models, exact input indices, and per-image predictions are in `artifacts/mistake-driven/`. Production source hashes are checked unchanged at completion.

Run: `.venv/bin/python -m experiments.mistake_driven_training --workers 3`.
