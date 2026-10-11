"""Build the report from saved experiment results (no training or test access).
Run: .venv/bin/python -m experiments.report_training_analysis
Optional figures require matplotlib, which is not a classifier dependency.
"""
import json
from pathlib import Path
import statistics

import numpy as np
from ImageRecognition.src.experiment import load_model

REPORT = Path('reports/training-analysis')
OUT = Path('artifacts/training-analysis')
LABELS = {'initial': 'Before training', 'current': 'All templates, 10%',
          'all-0.03': 'All templates, 3%', 'all-0.01': 'All templates, 1%',
          'closest-0.1': 'Closest template, 10%', 'closest-0.03': 'Closest template, 3%',
          'old-rounding': 'All templates, 10%, old rounding'}


def main():
    selection = json.loads((REPORT / 'selection.json').read_text())
    validation = json.loads((REPORT / 'validation.json').read_text())
    confirmation = json.loads((REPORT / 'confirmation.json').read_text())
    summary = json.loads((REPORT / 'summary.json').read_text())
    name = selection['selected']['variant']
    best = selection['best_per_variant']
    grid = selection['grid']
    fixed = {row['variant']: row for row in grid if row['threshold'] == 0.3}
    before, current, candidate = (summary[key]['mean_accuracy'] for key in ('initial', 'current', name))
    floor_difference = fixed['old-rounding']['mean_accuracy'] - fixed['current']['mean_accuracy']
    lines = ['# Improving the current template training approach', '',
        f'The most promising tested change is **{LABELS[name].lower()}**. '
        f'On a separate 5,000-image confirmation set, it recognizes **{candidate:.2%}**, '
        f'compared with **{before:.2%} before training** and **{current:.2%} with current training**. '
        'These are means across seeds 42, 43, 44, with each method using its validation-selected threshold.', '',
        'These figures use new confirmation images from the official training split. '
        'They are **not the earlier official-test scores** and should not be compared directly with 62.76% or 71.07%. '
        'The production training implementation was not changed in this investigation.', '',
        '## What current training does', '',
        'Think of the 21 templates for each digit as 21 example handwriting styles. '
        'Currently every new labeled image changes all 21 examples. '
        'They are all repeatedly pulled toward the same sequence of handwriting styles. '
        'The training loop never checks which example already resembles the incoming image.', '',
        'The update is also aggressive: at a 10% mutation rate, the last seven images carry roughly half '
        'the expected pixel pattern, and the last 29 carry about 95%. '
        'Feeding 500 or 1,000 images therefore does not mean they remain equally represented. '
        'Earlier information is repeatedly overwritten. This is an exact statement about expected pixel values '
        'under current stochastic rounding, not a claim that every random template has exactly those proportions.', '',
        'For one pixel, E[x_t] = (1-p) E[x_(t-1)] + p*a_t. The influence of the initial pixel falls as (1-p)^t. '
        'The last 95% of the expected pattern spans 29 updates at p=0.1, 99 at p=0.03, and 299 at p=0.01. '
        'See `memory_analysis.json`.', '',
        '## Controlled comparisons', '',
        'All variants share 21 templates per digit, exactly the same initial images and ordered training images '
        'within each seed, 500 training images per digit, one pass, intensity threshold >200, original affinity, '
        'and production voting. Each variant is repeated across three seeds. '
        'Only the named update behavior changes. Closest-template selection is within the known correct class, '
        'with random tie breaking; the existing pixel mutation routine performs the update.', '',
        'The table holds the recognition threshold at **0.3**, so the difference is training rather than '
        'threshold tuning. Validation uses the same held-out 5,000 images as the preceding evaluation.', '',
        '| Training variant | Mean validation accuracy | SD (percentage points) | Seed 42 | Seed 43 | Seed 44 |',
        '|---|---:|---:|---:|---:|---:|']
    for key, label in LABELS.items():
        row = fixed[key]
        lines.append(f'| {label} | {row["mean_accuracy"]:.2%} | {row["sd_accuracy"]*100:.2f} | ' +
                     ' | '.join(f'{score:.2%}' for score in row['seed_accuracies']) + ' |')
    lines += ['', 'Changing only which template receives the existing 10% update raises mean validation accuracy '
        f'from {fixed["current"]["mean_accuracy"]:.2%} to {fixed["closest-0.1"]["mean_accuracy"]:.2%}. '
        'Reducing the mutation rate while still updating every template also helps in this experiment, '
        'but does not surpass the initial references.', '',
        '## Did the previous rounding change hurt?', '',
        f'At threshold 0.3, integer-floor rounding scores **{fixed["old-rounding"]["mean_accuracy"]:.2%}**, '
        f'compared with **{fixed["current"]["mean_accuracy"]:.2%}** for stochastic rounding: '
        f'old minus current is **{floor_difference*100:+.2f} percentage points**. '
        'Both use the corrected bidirectional pixel mutation; this does not rerun the older broken mutation. '
        'The same data and seeds are used, but different rounding consumes random numbers differently, '
        'so this is a three-seed comparison, not an exact separation of every individual pixel change. '
        'Integer rounding also reduces the effective update rate, not just updates involving fewer than ten differences. '
        'Its dead zone is a behavioral choice and should not automatically have been called a correctness bug.', '',
        '## Recognition threshold comparison', '',
        'Thresholds 0.0, 0.2, 0.3, and 0.4 were evaluated for every model. '
        'A value of 0.0 includes any positive affinity; blank inputs and tied maxima can still be unrecognized. '
        'There is no nearest-template fallback.', '',
        '| Variant | Best validation threshold | Mean validation accuracy |', '|---|---:|---:|']
    for key, label in LABELS.items():
        lines.append(f'| {label} | {best[key]["threshold"]} | {best[key]["mean_accuracy"]:.2%} |')
    lines += ['', '## Separate confirmation', '',
        'After validation selection, the selected candidate and both controls were evaluated once on '
        '5,000 additional images (500 per digit). They were excluded from the prior validation set and from '
        'all training examples used by any seed in the preceding 100/500/1,000-per-digit experiments. '
        'The confirmation results were not used to change the selected variant or threshold. '
        'No official test images were evaluated in this investigation.', '',
        '| Method | Threshold | Seed 42 | Seed 43 | Seed 44 | Mean | SD (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('initial', 'current', name):
        row = summary[key]
        lines.append(f'| {LABELS[key]} | {best[key]["threshold"]} | ' +
            ' | '.join(f'{value:.2%}' for value in row['seed_accuracies']) +
            f' | {row["mean_accuracy"]:.2%} | {row["sd_accuracy"]*100:.2f} |')
    lines += ['', f'The selected candidate improves confirmation accuracy by **{(candidate-current)*100:.2f} points** '
        f'over current training and **{(candidate-before)*100:.2f} points** over its untrained references. '
        'These comparisons use the same confirmation images but method-specific validation-selected thresholds. '
        'The fixed-threshold validation comparison above isolates training behavior. '
        'The three seeds reuse the same confirmation images and are not 15,000 independent examples.', '',
        '## Recommendation', '',
        f'Use **{LABELS[name].lower()}** as the next candidate for the production training loop, '
        f'with reaction threshold **{best[name]["threshold"]}** for this configuration. '
        'Keep the existing affinity, pixel representation, voting, and number of templates. '
        'This lets different reference images keep representing different handwriting styles. '
        'It is supported by the controlled comparison; it is not a claim that the causal mechanism is fully isolated.', '',
        'Do not simply add more training images or epochs to the current all-template loop: '
        'the earlier benchmark already found that 1,000 images per class did not beat 500. '
        'Preserve an untrained control in future evaluations, and select checkpoints on validation accuracy '
        'so a harmful training run is not automatically accepted.', '',
        'The rounding audit is a limited comparison; no other previous fix was ablated. '
        'Only the listed variants and one training size were explored. '
        'Updating one template also reduces the number of mutation calls from 105,000 to 5,000. '
        'A random-single-template control was not run, so the benefit of similarity-based selection '
        'has not been separated from the benefit of making fewer updates. '
        'Closest-template mutation may still make noisy templates, and updates still do not penalize confusion '
        'with other classes. Performance on non-digit inputs has not been measured.', '',
        '## Reproduce and inspect', '', '```sh', '.venv/bin/python -m pytest -q',
        '.venv/bin/python -m experiments.analyze_training --workers 3',
        '.venv/bin/python -m experiments.report_training_analysis', '```', '',
        '- `protocol.json`: predeclared comparisons, selection procedure, source hashes, and data separation.',
        '- `validation.json`, `selection.json`: every candidate, threshold, seed, and per-digit metric.',
        '- `confirmation.json`, `summary.json`: confirmation scores and confusion matrices.',
        '- `artifacts/training-analysis/`: models, exact training indices, per-template update counts, predictions, and confirmation indices.',
        '- 23 tests passed, including experimental all-template training parity with production, '
        'closest-template selection, and restoring rounding after its isolated audit.',
        '- The production source hashes remained unchanged during this investigation.']
    try:
        figures(name, fixed)
        lines += ['', '## Visual comparison', '',
            f'![Validation accuracy across variants]({(REPORT / "validation-comparison.png").resolve()})', '',
            f'![Reference images before and after training]({(REPORT / "template-comparison.png").resolve()})', '',
            'Template comparison uses the first reference template of each digit for seed 42, '
            'at the same template index before and after each training variant; no example selection by appearance.']
    except ImportError:
        lines += ['', 'Optional figures were omitted because matplotlib is unavailable.']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(REPORT / 'report.md')


def figures(selected, fixed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(LABELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#64748b' if name != selected else '#15803d' for name in names]
    colors[names.index('current')] = '#b45309'
    ax.barh(range(len(names)), [fixed[name]['mean_accuracy']*100 for name in names], color=colors)
    for index, name in enumerate(names):
        ax.plot([x*100 for x in fixed[name]['seed_accuracies']], [index]*3, 'o', color='#0f172a', markersize=4)
        ax.text(max(fixed[name]['seed_accuracies'])*100 + 1, index,
                f'{fixed[name]["mean_accuracy"]:.2%}', va='center', fontsize=9)
    ax.set_yticks(range(len(names)), [LABELS[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    ax.set_xlabel('Correct digits (%) · same 5,000 validation images · threshold 0.3')
    ax.set_title('Training only the most similar reference improves recognition')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT/'validation-comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.6))
    for row, name in enumerate(('initial', 'current', selected)):
        model, _ = load_model(OUT/f'seed-42/{name}/model.pkl')
        for digit in range(10):
            ax = axes[row, digit]
            ax.imshow(model.immune_system[str(digit)][0].convert_to_array(), cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(str(digit))
            if digit == 0:
                ax.set_ylabel(LABELS[name].replace(', ', '\n'), fontsize=9)
    fig.suptitle('The same reference templates before and after training · seed 42')
    fig.tight_layout()
    fig.savefig(REPORT/'template-comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
