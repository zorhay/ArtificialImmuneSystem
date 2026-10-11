**Model-state clarification:** This file is the pre-training snapshot (zero training updates). The wrapper's metadata below describes the planned subsequent 20-example training run, not training already applied to this snapshot. See [the detailed comparison](../report.md).

# Original-algorithm run report

Correct: 1252 / 10000 (12.52%).

Training examples: 20; templates per digit: 21; reaction threshold: 0.8; pixel threshold: 200.

This runs the original training, mutation, affinity, and voting methods. It includes the original mutation bug, zero-score ties, and infinite exact-match scores. Only imports, data paths, and Python/NumPy compatibility were adjusted in the core files.

MNIST source: [CVDF mirror](https://github.com/cvdfoundation/mnist). Reference and training images use the official training set; evaluation uses the official test set. Images are exported without normalization or pixel changes. No validation tuning is performed.

All-zero score cases: 9711. The original classifier resolves these by dictionary order; no nearest-template fallback has been added.

All 10,000 official test images were evaluated.

| Digit | Correct | Evaluated | Accuracy |
|---|---:|---:|---:|
| 0 | 980 | 980 | 100.00% |
| 1 | 250 | 1135 | 22.03% |
| 2 | 0 | 1032 | 0.00% |
| 3 | 0 | 1010 | 0.00% |
| 4 | 2 | 982 | 0.20% |
| 5 | 0 | 892 | 0.00% |
| 6 | 9 | 958 | 0.94% |
| 7 | 8 | 1028 | 0.78% |
| 8 | 1 | 974 | 0.10% |
| 9 | 2 | 1009 | 0.20% |
