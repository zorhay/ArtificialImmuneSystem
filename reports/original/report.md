# Original-algorithm run report

Correct: 1 / 10 (10.00%).

Training examples: 20; templates per digit: 21; reaction threshold: 0.8; pixel threshold: 200.

This runs the original training, mutation, affinity, and voting methods. It includes the original mutation bug, zero-score ties, and infinite exact-match scores. Only imports, data paths, and Python/NumPy compatibility were adjusted in the core files.

MNIST source: [CVDF mirror](https://github.com/cvdfoundation/mnist). Reference and training images use the official training set; evaluation uses the official test set. Images are exported without normalization or pixel changes. No validation tuning is performed.

All-zero score cases: 10. The original classifier resolves these by dictionary order; no nearest-template fallback has been added.

This is a sampled execution check, not a full-test benchmark.

| Digit | Correct | Evaluated | Accuracy |
|---|---:|---:|---:|
| 0 | 1 | 1 | 100.00% |
| 1 | 0 | 1 | 0.00% |
| 2 | 0 | 1 | 0.00% |
| 3 | 0 | 1 | 0.00% |
| 4 | 0 | 1 | 0.00% |
| 5 | 0 | 1 | 0.00% |
| 6 | 0 | 1 | 0.00% |
| 7 | 0 | 1 | 0.00% |
| 8 | 0 | 1 | 0.00% |
| 9 | 0 | 1 | 0.00% |
