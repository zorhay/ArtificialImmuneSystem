# Digit performance of the applied closest-template model

Digit **5** is the weakest: **98/500 correct (19.6%)** for the saved default model (seed 42). It is also weakest in seeds 43 and 44, with **28.0% mean accuracy** across all three runs.

This breakdown uses the existing 5,000-image confirmation set: 500 examples of each digit. These images were excluded from training and configuration selection. It is not a new evaluation on the official 10,000-image test set. The applied models were checked against the evaluated experimental templates and matched exactly; their recognition threshold is 0.3.

Accuracy here means the percentage of images of a given digit correctly recognized (recall). Rejected images count as incorrect. Overall accuracy is 72.30% for the default model and 72.25% averaged across seeds.

## Default model, weakest first

| Digit | Correct / 500 | Accuracy | Most common wrong predictions (counts) | Three-run mean accuracy |
|---:|---:|---:|---|---:|
| 5 | 98 | 19.6% | 3 (128), 0 (81), 8 (67) | 28.0% |
| 4 | 320 | 64.0% | 9 (120), 6 (16), 1 (12) | 63.8% |
| 2 | 330 | 66.0% | 1 (36), 3 (32), 8 (30) | 65.7% |
| 8 | 331 | 66.2% | 3 (61), 1 (44), 9 (21) | 70.3% |
| 7 | 367 | 73.4% | 9 (70), unrecognized (15), 1 (14) | 74.3% |
| 9 | 401 | 80.2% | 7 (20), 1 (19), 4 (19) | 75.1% |
| 3 | 411 | 82.2% | 8 (23), 1 (18), 0 (13) | 72.3% |
| 6 | 425 | 85.0% | 0 (20), 2 (13), 8 (11) | 85.3% |
| 0 | 451 | 90.2% | 6 (17), 3 (14), 2 (5) | 90.9% |
| 1 | 481 | 96.2% | 8 (7), 3 (4), unrecognized (4) | 96.8% |

For digit 5, the default model rejects only 25 of 500 images; most failures are assignments to other digits. Its precision for predictions of 5 is 87.5%: when it says “5,” it is usually right, but it misses most actual fives. Investigating missed fives is the first priority.

## Across seeds

| Digit | Seed 42 | Seed 43 | Seed 44 | Mean |
|---:|---:|---:|---:|---:|
| 5 | 19.6% | 32.8% | 31.6% | 28.0% |
| 4 | 64.0% | 65.2% | 62.2% | 63.8% |
| 2 | 66.0% | 64.0% | 67.2% | 65.7% |
| 8 | 66.2% | 70.2% | 74.6% | 70.3% |
| 3 | 82.2% | 67.0% | 67.8% | 72.3% |
| 7 | 73.4% | 73.8% | 75.6% | 74.3% |
| 9 | 80.2% | 74.0% | 71.0% | 75.1% |
| 6 | 85.0% | 86.6% | 84.2% | 85.3% |
| 0 | 90.2% | 92.0% | 90.4% | 90.9% |
| 1 | 96.2% | 97.6% | 96.6% | 96.8% |

The same 500 images per digit are reused across seeds; the three runs are not 1,500 independent images per digit. Some rankings vary by seed, but 5 is consistently last.

Reproduce: `.venv/bin/python -m experiments.report_digit_performance`.

Source metrics: `reports/training-analysis/confirmation.json`. Machine-readable ranking: `reports/closest-template/digit-performance.json`.
