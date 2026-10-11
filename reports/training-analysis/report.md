# Improving the current template training approach

The most promising tested change is **closest template, 10%**. On a separate 5,000-image confirmation set, it recognizes **72.25%**, compared with **70.17% before training** and **64.45% with current training**. These are means across seeds 42, 43, 44, with each method using its validation-selected threshold.

These figures use new confirmation images from the official training split. They are **not the earlier official-test scores** and should not be compared directly with 62.76% or 71.07%. The production training implementation was not changed in this investigation.

## What current training does

Think of the 21 templates for each digit as 21 example handwriting styles. Currently every new labeled image changes all 21 examples. They are all repeatedly pulled toward the same sequence of handwriting styles. The training loop never checks which example already resembles the incoming image.

The update is also aggressive: at a 10% mutation rate, the last seven images carry roughly half the expected pixel pattern, and the last 29 carry about 95%. Feeding 500 or 1,000 images therefore does not mean they remain equally represented. Earlier information is repeatedly overwritten. This is an exact statement about expected pixel values under current stochastic rounding, not a claim that every random template has exactly those proportions.

For one pixel, E[x_t] = (1-p) E[x_(t-1)] + p*a_t. The influence of the initial pixel falls as (1-p)^t. The last 95% of the expected pattern spans 29 updates at p=0.1, 99 at p=0.03, and 299 at p=0.01. See `memory_analysis.json`.

## Controlled comparisons

All variants share 21 templates per digit, exactly the same initial images and ordered training images within each seed, 500 training images per digit, one pass, intensity threshold >200, original affinity, and production voting. Each variant is repeated across three seeds. Only the named update behavior changes. Closest-template selection is within the known correct class, with random tie breaking; the existing pixel mutation routine performs the update.

The table holds the recognition threshold at **0.3**, so the difference is training rather than threshold tuning. Validation uses the same held-out 5,000 images as the preceding evaluation.

| Training variant | Mean validation accuracy | SD (percentage points) | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|---:|---:|
| Before training | 70.65% | 0.96 | 70.08% | 70.12% | 71.76% |
| All templates, 10% | 60.79% | 1.25 | 60.88% | 62.00% | 59.50% |
| All templates, 3% | 63.69% | 1.24 | 64.18% | 64.62% | 62.28% |
| All templates, 1% | 65.31% | 0.97 | 66.26% | 65.34% | 64.32% |
| Closest template, 10% | 73.21% | 0.33 | 72.94% | 73.58% | 73.12% |
| Closest template, 3% | 70.48% | 0.70 | 71.16% | 69.76% | 70.52% |
| All templates, 10%, old rounding | 60.93% | 0.55 | 61.00% | 61.44% | 60.34% |

Changing only which template receives the existing 10% update raises mean validation accuracy from 60.79% to 73.21%. Reducing the mutation rate while still updating every template also helps in this experiment, but does not surpass the initial references.

## Did the previous rounding change hurt?

At threshold 0.3, integer-floor rounding scores **60.93%**, compared with **60.79%** for stochastic rounding: old minus current is **+0.13 percentage points**. Both use the corrected bidirectional pixel mutation; this does not rerun the older broken mutation. The same data and seeds are used, but different rounding consumes random numbers differently, so this is a three-seed comparison, not an exact separation of every individual pixel change. Integer rounding also reduces the effective update rate, not just updates involving fewer than ten differences. Its dead zone is a behavioral choice and should not automatically have been called a correctness bug.

## Recognition threshold comparison

Thresholds 0.0, 0.2, 0.3, and 0.4 were evaluated for every model. A value of 0.0 includes any positive affinity; blank inputs and tied maxima can still be unrecognized. There is no nearest-template fallback.

| Variant | Best validation threshold | Mean validation accuracy |
|---|---:|---:|
| Before training | 0.0 | 71.11% |
| All templates, 10% | 0.0 | 64.54% |
| All templates, 3% | 0.0 | 67.86% |
| All templates, 1% | 0.0 | 69.63% |
| Closest template, 10% | 0.3 | 73.21% |
| Closest template, 3% | 0.0 | 70.65% |
| All templates, 10%, old rounding | 0.0 | 64.71% |

## Separate confirmation

After validation selection, the selected candidate and both controls were evaluated once on 5,000 additional images (500 per digit). They were excluded from the prior validation set and from all training examples used by any seed in the preceding 100/500/1,000-per-digit experiments. The confirmation results were not used to change the selected variant or threshold. No official test images were evaluated in this investigation.

| Method | Threshold | Seed 42 | Seed 43 | Seed 44 | Mean | SD (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Before training | 0.0 | 69.28% | 69.98% | 71.24% | 70.17% | 0.99 |
| All templates, 10% | 0.0 | 63.94% | 65.66% | 63.74% | 64.45% | 1.06 |
| Closest template, 10% | 0.3 | 72.30% | 72.32% | 72.12% | 72.25% | 0.11 |

The selected candidate improves confirmation accuracy by **7.80 points** over current training and **2.08 points** over its untrained references. These comparisons use the same confirmation images but method-specific validation-selected thresholds. The fixed-threshold validation comparison above isolates training behavior. The three seeds reuse the same confirmation images and are not 15,000 independent examples.

## Recommendation

Use **closest template, 10%** as the next candidate for the production training loop, with reaction threshold **0.3** for this configuration. Keep the existing affinity, pixel representation, voting, and number of templates. This lets different reference images keep representing different handwriting styles. It is supported by the controlled comparison; it is not a claim that the causal mechanism is fully isolated.

Do not simply add more training images or epochs to the current all-template loop: the earlier benchmark already found that 1,000 images per class did not beat 500. Preserve an untrained control in future evaluations, and select checkpoints on validation accuracy so a harmful training run is not automatically accepted.

The rounding audit is a limited comparison; no other previous fix was ablated. Only the listed variants and one training size were explored. Updating one template also reduces the number of mutation calls from 105,000 to 5,000. A random-single-template control was not run, so the benefit of similarity-based selection has not been separated from the benefit of making fewer updates. Closest-template mutation may still make noisy templates, and updates still do not penalize confusion with other classes. Performance on non-digit inputs has not been measured.

## Reproduce and inspect

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.analyze_training --workers 3
.venv/bin/python -m experiments.report_training_analysis
```

- `protocol.json`: predeclared comparisons, selection procedure, source hashes, and data separation.
- `validation.json`, `selection.json`: every candidate, threshold, seed, and per-digit metric.
- `confirmation.json`, `summary.json`: confirmation scores and confusion matrices.
- `artifacts/training-analysis/`: models, exact training indices, per-template update counts, predictions, and confirmation indices.
- 23 tests passed, including experimental all-template training parity with production, closest-template selection, and restoring rounding after its isolated audit.
- The production source hashes remained unchanged during this investigation.

## Visual comparison

![Validation accuracy across variants](/Users/zori/projects/ArtificialImmuneSystem/reports/training-analysis/validation-comparison.png)

![Reference images before and after training](/Users/zori/projects/ArtificialImmuneSystem/reports/training-analysis/template-comparison.png)

Template comparison uses the first reference template of each digit for seed 42, at the same template index before and after each training variant; no example selection by appearance.
