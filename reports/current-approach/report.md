# Current approach: controlled MNIST evaluation

Selected trained models: **62.76% mean test accuracy**, sample SD **1.27 percentage points** across seeds 42, 43, 44.

Untrained reference control at the same threshold: **71.07%** mean test accuracy. Mutation training reduces mean accuracy by **8.31 percentage points**, with drops of **7.20, 7.77, and 9.97 points** for seeds 42, 43, and 44 respectively.

## What ran

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ImageRecognition.src.benchmark --workers 3
PYTHONPATH=. .venv/bin/python reports/current-approach/verify_native.py
```

Fresh independent models: 100, 500, 1,000 training images per digit × seeds 42, 43, 44. Each trained model is evaluated at thresholds 0.3, 0.4, 0.5: 27 validation settings. Three additional untrained initial networks are controls. Configuration selection uses mean validation accuracy across seeds, then evaluates the selected configuration and the untrained control on all 10,000 official test images for each seed.

## Configuration and isolation

- 21 templates per digit (210 total), original affinity: overlap / maximum white-pixel count.
- Every template in the labeled class mutates toward every training image, once; mutation power 0.1.
- Original 28×28 images; white means intensity strictly greater than 200. No preprocessing changes.
- Fixed validation split: 500 per digit (5,000 total), drawn from official training data with split seed 20261012.
- Initial templates and training examples come only from the remaining 55,000 images; no validation/test leakage.
- Seeded within-class shuffled training order; nested subsets across sizes and identical initialization per seed. Reference images may also occur in training, as in the original approach.
- Selected configuration: **500 per digit, threshold 0.3**.
- No full-pool retraining after selection. Test models are exactly the selected validation checkpoints.

## Implementation corrections included

The previous bidirectional pixel-list correction and unrecognized output remain. This run additionally isolates model dictionaries, uses stochastic rounding (e.g. one mismatch changes with probability 0.1, preserving the expected rate), and evaluates the same exponential weights with a common finite scale. Exact matches dominate finite matches and vote by count; tied maxima return unrecognized. Counters are relative scores, not probabilities. These results describe the corrected implementation, not the older 11.47% checkpoint. That historical result used different training data and an inference threshold of 0.8, so the difference cannot be attributed solely to bug fixes. The run used uncommitted corrections on top of commit `184d92f`; exact source hashes are in `configuration.json`.

## Validation results

| Training / digit | Threshold | Mean accuracy | SD (pp) | Seed 42 | Seed 43 | Seed 44 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3 | 70.65% | 0.96 | 70.08% | 70.12% | 71.76% |
| 0 | 0.4 | 68.62% | 0.40 | 68.18% | 68.96% | 68.72% |
| 0 | 0.5 | 57.65% | 0.77 | 57.74% | 58.38% | 56.84% |
| 100 | 0.3 | 58.91% | 2.58 | 56.12% | 61.20% | 59.40% |
| 100 | 0.4 | 43.39% | 1.72 | 41.52% | 44.90% | 43.74% |
| 100 | 0.5 | 12.20% | 1.26 | 13.52% | 11.00% | 12.08% |
| 500 | 0.3 | 60.79% | 1.25 | 60.88% | 62.00% | 59.50% |
| 500 | 0.4 | 45.01% | 2.19 | 46.12% | 46.42% | 42.48% |
| 500 | 0.5 | 13.92% | 2.91 | 16.32% | 14.76% | 10.68% |
| 1000 | 0.3 | 58.63% | 2.39 | 59.64% | 55.90% | 60.34% |
| 1000 | 0.4 | 43.31% | 2.74 | 46.14% | 40.68% | 43.10% |
| 1000 | 0.5 | 12.23% | 2.39 | 14.90% | 11.52% | 10.28% |

## Full test results

| Training / digit | Seed | Correct / 10,000 | Accuracy | Coverage | Recognized accuracy | Unrecognized | Macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 6982 | 69.82% | 98.64% | 70.78% | 136 | 0.6888 |
| 500 | 42 | 6262 | 62.62% | 93.92% | 66.67% | 608 | 0.6229 |
| 0 | 43 | 7186 | 71.86% | 98.87% | 72.68% | 113 | 0.7089 |
| 500 | 43 | 6409 | 64.09% | 94.58% | 67.76% | 542 | 0.6445 |
| 0 | 44 | 7154 | 71.54% | 98.71% | 72.47% | 129 | 0.7128 |
| 500 | 44 | 6157 | 61.57% | 93.40% | 65.92% | 660 | 0.6230 |

Unrecognized predictions count as incorrect in overall accuracy. Coverage is the fraction assigned a digit; recognized accuracy excludes rejected inputs. SD measures seed variation, not a confidence interval. All runs reuse the same test images; they are not 30,000 independent examples.

## Per-digit test accuracy (recall), trained models

| Digit | Test images | Seed 42 | Seed 43 | Seed 44 | Mean | Mean precision |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 81.43% | 81.84% | 83.78% | 82.35% | 70.48% |
| 1 | 1135 | 88.02% | 88.55% | 86.08% | 87.55% | 70.79% |
| 2 | 1032 | 49.71% | 58.82% | 63.66% | 57.40% | 77.51% |
| 3 | 1010 | 71.88% | 71.98% | 41.49% | 61.78% | 62.17% |
| 4 | 982 | 35.74% | 54.58% | 49.19% | 46.50% | 80.52% |
| 5 | 892 | 19.39% | 23.09% | 19.73% | 20.74% | 76.86% |
| 6 | 958 | 77.77% | 65.66% | 59.19% | 67.54% | 82.77% |
| 7 | 1028 | 72.57% | 69.94% | 68.48% | 70.33% | 77.61% |
| 8 | 974 | 46.41% | 63.04% | 70.23% | 59.89% | 52.49% |
| 9 | 1009 | 75.22% | 55.90% | 66.30% | 65.81% | 53.22% |

## Error breakdown

Across the three trained runs, mean coverage is **93.97%** and mean accuracy among recognized inputs is **66.79%**. Rejections are mostly zero votes: 608, 539, 659 per seed; tied maxima account for 0, 3, 1 additional rejections. Thus unresolved ties are not driving the measured failure.

Digit **5** is weakest: mean recall **20.74%**, despite mean precision **76.86%**. It is usually assigned another digit or rejected. The most frequent errors below pool three predictions per original test image; counts are not distinct images.

| True digit | Predicted digit | Errors across three seeds |
|---:|---:|---:|
| 4 | 9 | 904 |
| 5 | 3 | 520 |
| 3 | 8 | 446 |
| 5 | 8 | 409 |
| 8 | 3 | 347 |

All per-seed precision, recall, F1, rejection counts, and confusion matrices are retained in `test.json`.

## Interpretation and limits

Training performs worse than its own initial network in every seed. Among trained configurations, 500 examples per digit performs best on validation; increasing to 1,000 does not improve the mean. Threshold 0.3 is the best of the three tested values, but it is the edge of the grid, not a proven optimum.

Mean affinity between distinct templates of the same class rises from **0.3215 before training to 0.4126 after 500 images per digit**. This is consistent with reduced template diversity, but does not establish the cause of the accuracy loss. Details are in `template_diagnostics.json`. More training is not assumed to help: all cells receive the same image sequence, so repeated updates can erase useful initial shape diversity and emphasize recent examples. There is no discriminative update against other classes. These remain properties of the approach.

Evaluation computes identical integer white-pixel overlaps in batches and calls the production vote routine. For every model/split, 100 images are checked against the original coordinate-list affinity and recognition at every evaluated threshold, with exact equality required. Training always calls the production all-cells loop. This optimization changes evaluation cost, not the similarity formula.

The configuration grid is deliberately narrow. No claim is made that it is globally optimal. Seed variation and all per-class errors are retained. Historical reports remain untouched.

## Artifacts

- `configuration.json`: settings, parent commit, and SHA-256 of every source file at run time.
- `selection.json`, `validation.json`, `test.json`: complete grid, metrics, 10×11 confusion matrices, parity checks.
- `/Users/zori/projects/ArtificialImmuneSystem/artifacts/current-approach/split.json`: exact validation and training-pool indices.
- `/Users/zori/projects/ArtificialImmuneSystem/artifacts/current-approach/seed-*/train-*/`: model.pkl, ordered reference/training indices, and per-image predictions for each evaluated threshold. Index means the original MNIST IDX row.

## Reuse the selected configuration

For the fixed, predeclared seed-42 checkpoint:

```sh
.venv/bin/python -m ImageRecognition evaluate --model artifacts/current-approach/seed-42/train-500/model.pkl --affinity 0.3 --report-dir reports/current-approach/native-cli
```

Pass `--affinity 0.3` explicitly: the general CLI default remains 0.4. The three seeds are reported together; seed 43 is not selected simply because it has the highest test accuracy.

Data checks verified all four MNIST archive checksums and pixel equality between all 24,806 distinct training PNGs used here and the original IDX arrays (`data_verification.json`). The test suite passed **20 tests**. Training time summed over nine trained runs was **363.80 seconds**, with three runs executing concurrently; that sum is not total elapsed benchmark time.

## Full production-path verification

The selected seed-42 model was also evaluated on **all 10,000 test PNGs** through `recognition_image`, including the original coordinate-list affinity implementation. **All 10,000 predictions exactly matched** the batched benchmark; **6,262 were correct (62.62%)**. This independent path took **293.50 seconds**. See `native_verification.json` and `verify_native.py` for the result and reproducible command.
