# Template classifier run report

Correct: 4 / 10 (40.00%).

Training examples: 20; templates per digit: 21; reaction threshold: 0.4; pixel threshold: 200.

Saved training algorithm: original all-cells training with corrected bidirectional mutation. New training uses corrected bidirectional mutation; loading an older model does not retrain it. Inference uses the original affinity and weighted votes, with an explicit unrecognized output.

MNIST source: [CVDF mirror](https://github.com/cvdfoundation/mnist). Reference and training images use the official training set; evaluation uses the official test set. Images are exported without normalization or pixel changes. No validation tuning is performed.

Unrecognized inputs: 2. Coverage: 80.00%. Unrecognized inputs count as incorrect in overall accuracy. There is no default digit or nearest-template fallback.

Saved model threshold: 0.4; evaluation threshold: 0.4.

This is a sampled execution check, not a full-test benchmark.

| Digit | Correct | Evaluated | Accuracy |
|---|---:|---:|---:|
| 0 | 1 | 1 | 100.00% |
| 1 | 1 | 1 | 100.00% |
| 2 | 0 | 1 | 0.00% |
| 3 | 0 | 1 | 0.00% |
| 4 | 0 | 1 | 0.00% |
| 5 | 0 | 1 | 0.00% |
| 6 | 1 | 1 | 100.00% |
| 7 | 0 | 1 | 0.00% |
| 8 | 0 | 1 | 0.00% |
| 9 | 1 | 1 | 100.00% |
