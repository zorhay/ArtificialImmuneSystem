"""Rank digits using existing confirmation predictions for the applied models.
Run: .venv/bin/python -m experiments.report_digit_performance
"""
import hashlib
import json
from pathlib import Path
import statistics

from ImageRecognition.src.benchmark import write_json
from ImageRecognition.src.experiment import load_model
from ImageRecognition.src.immune_algorithm import DEFAULT_AFFINITY


def main():
    source = Path('reports/training-analysis/confirmation.json')
    runs = [row for row in json.loads(source.read_text()) if row['variant'] == 'closest-0.1']
    assert sorted(row['seed'] for row in runs) == [42, 43, 44]
    # Confirm that these saved metrics describe the deployed templates and threshold.
    for row in runs:
        seed = row['seed']
        applied, _ = load_model(f'artifacts/closest-template/seed-{seed}/model.pkl')
        evaluated, _ = load_model(f'artifacts/training-analysis/seed-{seed}/closest-0.1/model.pkl')
        assert applied.main_affinity_to_react == row['threshold'] == DEFAULT_AFFINITY
        assert applied.immune_system.keys() == evaluated.immune_system.keys()
        for digit, cells in applied.immune_system.items():
            assert len(cells) == len(evaluated.immune_system[digit])
            for actual, previous in zip(cells, evaluated.immune_system[digit]):
                assert actual.white_pixels == previous.white_pixels
                assert actual.black_pixels == previous.black_pixels
    default_model, _ = load_model('artifacts/closest-template/model.pkl')
    seed42, _ = load_model('artifacts/closest-template/seed-42/model.pkl')
    assert default_model.main_affinity_to_react == DEFAULT_AFFINITY
    for digit, cells in default_model.immune_system.items():
        assert [c.white_pixels for c in cells] == [c.white_pixels for c in seed42.immune_system[digit]]
    default = next(row for row in runs if row['seed'] == 42)
    ranking = []
    for digit in map(str, range(10)):
        row = default['per_digit'][digit]
        counts = default['confusion_matrix'][int(digit)]
        errors = sorted([{'prediction': 'unrecognized' if j == 10 else str(j), 'count': count}
                         for j, count in enumerate(counts) if j != int(digit) and count],
                        key=lambda item: -item['count'])
        ranking.append({'digit': digit, 'default_correct': row['correct'], 'examples_per_run': row['total'],
            'default_accuracy': row['recall'], 'default_precision': row['precision'],
            'default_unrecognized': row['unrecognized'], 'default_most_common_errors': errors[:3],
            'mean_accuracy': statistics.mean(run['per_digit'][digit]['recall'] for run in runs),
            'seed_accuracies': {str(run['seed']): run['per_digit'][digit]['recall'] for run in runs}})
    ranking.sort(key=lambda row: row['default_accuracy'])
    output = Path('reports/closest-template')
    result = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'evaluation': 'existing 5000-image confirmation set from official training split, 500 images per digit',
        'new_evaluation': False, 'default_seed': 42, 'threshold': DEFAULT_AFFINITY,
        'production_templates_verified_against_evaluated_models': True,
        'default_overall_accuracy': default['accuracy'], 'ranking': ranking}
    write_json(output / 'digit-performance.json', result)
    lines = ['# Digit performance of the applied closest-template model', '',
        'Digit **5** is the weakest: **98/500 correct (19.6%)** for the saved default model (seed 42). '
        'It is also weakest in seeds 43 and 44, with **28.0% mean accuracy** across all three runs.', '',
        'This breakdown uses the existing 5,000-image confirmation set: 500 examples of each digit. '
        'These images were excluded from training and configuration selection. It is not a new evaluation '
        'on the official 10,000-image test set. The applied models were checked against the evaluated '
        'experimental templates and matched exactly; their recognition threshold is 0.3.', '',
        'Accuracy here means the percentage of images of a given digit correctly recognized '
        '(recall). Rejected images count as incorrect. Overall accuracy is 72.30% for the default model '
        'and 72.25% averaged across seeds.', '',
        '## Default model, weakest first', '',
        '| Digit | Correct / 500 | Accuracy | Most common wrong predictions (counts) | Three-run mean accuracy |',
        '|---:|---:|---:|---|---:|']
    for row in ranking:
        errors = ', '.join(f'{item["prediction"]} ({item["count"]})' for item in row['default_most_common_errors'])
        lines.append(f'| {row["digit"]} | {row["default_correct"]} | {row["default_accuracy"]:.1%} | {errors} | {row["mean_accuracy"]:.1%} |')
    lines += ['', 'For digit 5, the default model rejects only 25 of 500 images; most failures are '
        'assignments to other digits. Its precision for predictions of 5 is 87.5%: when it says “5,” '
        'it is usually right, but it misses most actual fives. Investigating missed fives is the first priority.', '',
        '## Across seeds', '',
        '| Digit | Seed 42 | Seed 43 | Seed 44 | Mean |', '|---:|---:|---:|---:|---:|']
    for row in sorted(ranking, key=lambda row: row['mean_accuracy']):
        lines.append(f'| {row["digit"]} | ' + ' | '.join(f'{row["seed_accuracies"][str(seed)]:.1%}' for seed in (42,43,44)) +
                     f' | {row["mean_accuracy"]:.1%} |')
    lines += ['', 'The same 500 images per digit are reused across seeds; the three runs are not '
        '1,500 independent images per digit. Some rankings vary by seed, but 5 is consistently last.', '',
        'Reproduce: `.venv/bin/python -m experiments.report_digit_performance`.', '',
        'Source metrics: `reports/training-analysis/confirmation.json`. '
        'Machine-readable ranking: `reports/closest-template/digit-performance.json`.']
    (output / 'digit-performance.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(ranking[:5], indent=2))


if __name__ == '__main__':
    main()
