# Does the current model fit its training examples?

This diagnostic evaluates the unchanged closest-template models on the exact 5,000 images each seed used for training, then compares with the original held-out validation set. No additional training, parameter selection, old confirmation evaluation, or fresh-holdout evaluation occurs.

Mean training accuracy: **75.05%**. Mean validation accuracy: **73.21%**. Training advantage: **+1.84 percentage points**.

| Digit | Training accuracy, mean | Validation accuracy, mean |
|---:|---:|---:|
| 0 | 92.93% | 90.40% |
| 1 | 98.47% | 97.27% |
| 2 | 67.73% | 62.47% |
| 3 | 75.27% | 71.87% |
| 4 | 65.93% | 61.13% |
| 5 | 32.13% | 33.53% |
| 6 | 88.87% | 87.80% |
| 7 | 76.60% | 75.20% |
| 8 | 74.93% | 75.00% |
| 9 | 77.67% | 77.47% |

Training accuracy includes initial reference images that are part of the training sequence. Training subsets differ across seeds; validation images are shared. This comparison is descriptive, not proof that no component overfits. Poor performance on both sets points to limitations of learning or representation rather than memorizing training images successfully and only failing on new images.

Run: `.venv/bin/python -m experiments.diagnose_training_fit`.
