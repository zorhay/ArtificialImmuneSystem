# Current implementation: full MNIST evaluation

Run date: 2026-10-12T01:10:21.617945+04:00 (Asia/Yerevan).

**Current saved-model accuracy: 11.47% (1,147 / 10,000 correct).**

## 1. What was run

Source commit: `d13bbca72dbda59aa73360d191227dcb9304bfab`. The committed implementation was used without changing its training, mutation, affinity, scoring, thresholds, or prediction behavior.

Two existing model files were evaluated, each on the complete official MNIST test set:

1. `artifacts/original/model.pkl`: the current saved model, initialized from 21 images per digit and then trained on 2 images per digit, one pass (20 training examples, 420 cell-mutation calls).
2. `artifacts/original/initial.pkl`: the same 210 reference templates before any training updates. This is the reference-only comparison corresponding to the original recognition-script workflow.

**No retraining or tuning occurred during this evaluation. The current saved model is the earlier smoke-run model, not a model trained on all 60,000 MNIST training images.** The 20 update images are a subset of the 210 initialization images. Neither uses test images.

Commands executed from the repository root:

```sh
.venv/bin/python -m ImageRecognition evaluate --model artifacts/original/model.pkl --report-dir reports/evaluation-current/trained
.venv/bin/python -m ImageRecognition evaluate --model artifacts/original/initial.pkl --report-dir reports/evaluation-current/initial
```

The `--test-per-class` option was omitted so every test image was evaluated. Both processes call the original `recognition_image` → `recognition_antigen` → `affinity` methods. The second model ran in a separate process because the original classifier uses shared class state.

The downloaded archives passed their expected checksums. All 10,000 exported test images were verified pixel-for-pixel and label-for-label against the original IDX files. All 210 initial templates were also checked against the training IDX images. No test image is blank after thresholding at 200. Prediction logs contain exactly 10,000 distinct test indices per run.

## 2. Detailed accuracy report

| Measurement | Current saved model | Initial templates |
|---|---:|---:|
| Accuracy | 11.47% | 12.52% |
| Correct / evaluated | 1,147 / 10,000 | 1,252 / 10,000 |
| Errors | 8,853 | 8,748 |
| Balanced accuracy (macro recall) | 11.49% | 12.42% |
| Macro precision | 61.00% | 61.01% |
| Macro F1 | 0.0446 | 0.0588 |
| Support-weighted F1 | 0.0474 | 0.0633 |
| All-zero vote cases | 9,815 | 9,711 |
| Images with at least one reacting template | 185 | 289 |
| Accuracy on reacting images only | 100.00% | 100.00% |
| Evaluation elapsed time | 309.41 s | 284.16 s |

Two independent processes ran concurrently; elapsed timings are not isolated performance benchmarks.

The 20-example training pass changed overall accuracy by -1.05 percentage points. It corrected 13 errors and lost 118 previously correct predictions; 9,869 predictions were unchanged.

For context, always predicting 0 scores 9.80% on this test set; always predicting its most frequent class (1) scores 11.35%. These are descriptive constant-output baselines, not tuned models.

### Current saved model: per-digit results

| Digit | Test images | Correct | Predicted as digit | Precision | Recall | F1 | No reaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 980 | 9833 | 9.97% | 100.00% | 0.1813 | 962 |
| 1 | 1135 | 155 | 155 | 100.00% | 13.66% | 0.2403 | 980 |
| 2 | 1032 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 1032 |
| 3 | 1010 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 1010 |
| 4 | 982 | 1 | 1 | 100.00% | 0.10% | 0.0020 | 981 |
| 5 | 892 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 892 |
| 6 | 958 | 6 | 6 | 100.00% | 0.63% | 0.0124 | 952 |
| 7 | 1028 | 2 | 2 | 100.00% | 0.19% | 0.0039 | 1026 |
| 8 | 974 | 1 | 1 | 100.00% | 0.10% | 0.0021 | 973 |
| 9 | 1009 | 2 | 2 | 100.00% | 0.20% | 0.0040 | 1007 |

Precision = correct predictions of a digit / all predictions of that digit. Recall = correct predictions of a digit / all test images of that digit. F1 is their harmonic mean. Undefined precision/F1 for unpredicted classes is reported as zero. No-reaction cases are included in every metric, using the classifier's actual output.

### Current saved model: confusion matrix

Rows are true digits; columns are predicted digits.

| True / predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 980 | 155 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1032 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1010 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 981 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 5 | 892 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 952 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| 7 | 1026 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 8 | 973 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 9 | 1007 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

### Initial templates: per-digit results

| Digit | Test images | Correct | Predicted as digit | Precision | Recall | F1 | No reaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 980 | 9728 | 10.07% | 100.00% | 0.1830 | 963 |
| 1 | 1135 | 250 | 250 | 100.00% | 22.03% | 0.3610 | 885 |
| 2 | 1032 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 1032 |
| 3 | 1010 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 1010 |
| 4 | 982 | 2 | 2 | 100.00% | 0.20% | 0.0041 | 980 |
| 5 | 892 | 0 | 0 | 0.00% | 0.00% | 0.0000 | 892 |
| 6 | 958 | 9 | 9 | 100.00% | 0.94% | 0.0186 | 949 |
| 7 | 1028 | 8 | 8 | 100.00% | 0.78% | 0.0154 | 1020 |
| 8 | 974 | 1 | 1 | 100.00% | 0.10% | 0.0021 | 973 |
| 9 | 1009 | 2 | 2 | 100.00% | 0.20% | 0.0040 | 1007 |

### Initial templates: confusion matrix

Rows are true digits; columns are predicted digits.

| True / predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 980 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 885 | 250 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1032 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1010 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 980 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| 5 | 892 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 949 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |
| 7 | 1020 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| 8 | 973 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 9 | 1007 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

### Observed failure behavior

For the current model, **9,815/10,000 (98.15%) inputs produced zero votes in every class**. In that case, the original `max(counters, key=counters.get)` returns the first dictionary key, which is `0`. These are ordinary predictions in the original code; there is no rejection or nearest-template fallback. The CSV explicitly flags each occurrence.

Of the current model's 1,147 correct answers, 962 came from zero-vote defaults matching a true zero, and 185 came from images with at least one reacting template. High recall for digit 0 therefore should not be interpreted as reliable recognition of zeros.

The saved post-training templates contain 1,674 added white-pixel coordinates and 0 removed white-pixel coordinates relative to initialization. 210/210 cells contain coordinates in both their white and black lists. This confirms the original mutation behavior remains present. These observations do not isolate its causal effect from the small training set or the reaction threshold.

## 3. Configurations

| Setting | Value |
|---|---|
| Dataset | MNIST, official train/test partition |
| Available training set | 60,000 images |
| Initialization | First 21 files per digit in original `os.listdir` order; 210 cells total |
| Actual current-model training | First 2 files per digit in the same order; 20 images, one pass |
| Initial-template comparison training | None |
| Test set | All 10,000 official test images; no sampling |
| Image representation | 28×28 grayscale, coordinates of white and black pixels |
| Binarization | Pixel > 200 becomes 255; otherwise 0 |
| Reaction affinity | ≥ 0.8 |
| Affinity | Shared white pixels / maximum of the two white-pixel counts |
| Vote increment | a × exp(1 / (1 − a)); original infinity handling |
| Mutation | Original all-same-class-cells update, power 0.1 |
| Random seed for saved training | 42 |
| Evaluation seed argument | Default 42; unused when evaluating all files |
| Class insertion/tie order | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Preprocessing beyond thresholding | None; no cropping, normalization, inversion, or augmentation |
| Validation/checkpoint selection | None |
| Python / NumPy / Pillow | 3.13.5 / 2.5.4 / 12.3.0 |
| Platform | macOS-27.0-arm64-arm-64bit-Mach-O |

The legacy `main.py` uses 20 cells and affinity 0.4 only to load a network. That is a different configuration and was not evaluated here. This report uses the saved model's settings, matching the original recognition script's 21-cell/0.8 configuration.

Directory order is filesystem-dependent. The full lists of source filenames used for initialization and training are recorded in the per-run metrics JSON, so seed 42 alone should not be treated as a cross-filesystem reproduction guarantee.

### Model identity

| Model file | SHA-256 |
|---|---|
| `artifacts/original/model.pkl` | `97f394e72f02039c59783372d2351bf3da64fb5edfe93cb670078015e7ba7567` |
| `artifacts/original/initial.pkl` | `27f444cfc956e7b0fcd469e7f6e2b2d02256283767a7c469a61fdd5ef5c8f877` |

## Artifacts and interpretation

- [Summary metrics](summary.json)
- [Current-model predictions, 10,000 rows](trained/predictions.csv)
- [Initial-template predictions, 10,000 rows](initial/predictions.csv)
- [Current-model detailed metrics](trained/detailed_metrics.json)
- [Initial-template detailed metrics](initial/detailed_metrics.json)
- [Environment, checksums, and commands](provenance.json)
- [Template-state diagnostics](template_diagnostics.json)

These measurements describe these two saved model states on MNIST. They do not establish the best accuracy achievable by the original algorithm, accuracy after full-dataset training, or performance with other thresholds/template counts. Only one initialization and one mutation seed are represented. No settings were selected using these test results.

The initial model's raw wrapper metadata records the planned 20-example follow-up training run. That file was saved before `training()` was called; its actual mutation training count is zero, as stated throughout this detailed report.
