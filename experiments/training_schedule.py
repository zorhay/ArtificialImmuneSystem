"""Compare a fixed decay schedule and earlier training stops, uniformly across digits.
Run after mistake_driven_training: python -m experiments.training_schedule --workers 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from experiments.analyze_training import evaluate, select_cells
from ImageRecognition.src.benchmark import training_order, write_json
from ImageRecognition.src.dataset import DEFAULT_DATA_DIR, load_mnist
from ImageRecognition.src.experiment import folders, load_model, save_model

BASE = Path('artifacts/current-approach')
OUT = Path('artifacts/training-schedule')
REPORT = Path('reports/training-schedule')
SEEDS = (42, 43, 44)
VARIANTS = [('constant', 500), ('constant', 100), ('constant', 250),
            ('decay', 100), ('decay', 250), ('decay', 500)]


def train_scheduled(model, paths, names, seed, count, schedule):
    random.seed(seed)
    for digit, directory in paths.items():
        cells = model.immune_system[digit]
        for step, name in enumerate(names[digit][:count]):
            antigen = model.load_antigen_by_path(directory / name)
            position = select_cells(cells, antigen, 'closest')[0]
            # Restart the same schedule for each class; no digit-specific settings.
            power = 0.1 if schedule == 'constant' else 0.1 - 0.07 * step / 499
            cells[position].mutation(antigen, power)


def validation_seed(seed):
    images, labels = load_mnist(split='train')
    split = json.loads((BASE / 'split.json').read_text())
    order = training_order(split['training_pool'], seed)
    names = {digit: [f'{i:05d}.png' for i in values[:500]] for digit, values in order.items()}
    validation = np.array([i for values in split['validation'].values() for i in values])
    results = []
    for schedule, count in VARIANTS:
        name = f'{schedule}-{count}'
        destination = OUT / f'seed-{seed}' / name
        if name == 'constant-500':
            result = json.loads(Path(f'artifacts/training-analysis/seed-{seed}/closest-0.1/validation/metrics.json').read_text())['0.3']
        else:
            model, metadata = load_model(BASE / f'seed-{seed}/train-0/model.pkl')
            model.set_affinity_to_react(0.3)
            start = time.perf_counter()
            train_scheduled(model, folders(DEFAULT_DATA_DIR, 'train'), names, seed, count, schedule)
            metadata = {**metadata, 'algorithm': 'closest-template mutation', 'schedule': schedule,
                        'training_per_class': count, 'training_examples': count*10,
                        'reaction_affinity': 0.3, 'seed': seed,
                        'training_indices': {digit: values[:count] for digit, values in order.items()},
                        'training_seconds': time.perf_counter()-start,
                        'mutation_schedule': '0.1 constant' if schedule == 'constant' else '0.1 - 0.07 * zero_based_class_step / 499'}
            save_model(model, destination / 'model.pkl', metadata)
            write_json(destination / 'training.json', metadata)
            result = evaluate(model, images, labels, validation, [0.3], destination / 'validation')['0.3']
        results.append({'seed': seed, 'variant': name, **result})
        print(f'Seed {seed}, {name}: validation {result["accuracy"]:.2%}', flush=True)
    return results


def fresh_run(task):
    seed, name, indices = task
    images, labels = load_mnist(split='train')
    path = Path(f'artifacts/closest-template/seed-{seed}/model.pkl') if name == 'constant-500' else OUT / f'seed-{seed}' / name / 'model.pkl'
    model, _ = load_model(path)
    model.set_affinity_to_react(0.3)
    result = evaluate(model, images, labels, np.array(indices), [0.3], OUT / f'seed-{seed}' / name / 'fresh')['0.3']
    print(f'Fresh seed {seed}, {name}: {result["accuracy"]:.2%}', flush=True)
    return {'seed': seed, 'variant': name, **result}


def run(workers):
    previous = json.loads(Path('reports/mistake-driven/decision.json').read_text())
    if previous['fresh_holdout_opened']:
        raise ValueError('The reserved holdout was already opened; choose a new protocol rather than reusing it')
    fresh = json.loads(Path('artifacts/mistake-driven/fresh_indices.json').read_text())
    write_json(REPORT / 'protocol.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': SEEDS,
        'variants': VARIANTS, 'reaction_affinity': 0.3, 'templates_per_digit': 21,
        'pixel_threshold': 200, 'passes': 1,
        'decay': 'linear from 0.1 to 0.03 over 500 images of each class; the same schedule restarts for every class',
        'checkpoints': 'independently replay each prefix from identical initial templates and seed; later-class random draws differ with prefix length',
        'selection': 'best mean validation accuracy across seeds; ties prefer current constant-500 baseline',
        'fresh_holdout': 'reuse reserved but unopened 5000 rows from mistake-driven protocol; evaluate only the selected winner and current baseline',
        'gate': 'only open fresh holdout if validation beats current baseline',
        'post_selection': 'no additional tuning after fresh results; no automatic production changes',
        'official_test_set_used': False,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('ImageRecognition/src').glob('*.py')}})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = [row for group in executor.map(validation_seed, SEEDS) for row in group]
    write_json(REPORT / 'validation.json', results)
    grid = []
    for schedule, count in VARIANTS:
        name = f'{schedule}-{count}'
        scores = [row['accuracy'] for row in results if row['variant'] == name]
        grid.append({'variant': name, 'mean_accuracy': statistics.mean(scores),
                     'sd_accuracy': statistics.stdev(scores), 'seed_accuracies': scores})
    chosen = max(grid, key=lambda row: row['mean_accuracy'])
    open_fresh = chosen['mean_accuracy'] > grid[0]['mean_accuracy']
    write_json(REPORT / 'decision.json', {'grid': grid, 'selected': chosen, 'fresh_holdout_opened': open_fresh})
    print('Selected: '+str(chosen), flush=True)
    fresh_results = []
    if open_fresh:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fresh_results = list(executor.map(fresh_run, [(seed, name, fresh) for seed in SEEDS
                                                        for name in ('constant-500', chosen['variant'])]))
        write_json(REPORT / 'fresh.json', fresh_results)
    write_report(grid, chosen, fresh_results)


def write_report(grid, chosen, fresh):
    lines = ['# General training improvements: fixed decay and earlier stops', '',
        'The first experiment, updating only on mistakes, reduced mean validation accuracy from '
        '73.21% to 69.69%. This follow-up tests the other planned ideas: progressively smaller updates '
        'and selecting how long to train, using the same settings for every digit.', '',
        'Each run starts from the same reference templates and ordered images within its seed. '
        'Every image updates one closest correct-class template. The recognition threshold stays at 0.3, '
        'with 21 templates per digit and intensity threshold >200. Seeds are 42, 43, and 44. '
        'The mutation rate is either fixed at 10%, or falls linearly from 10% to 3% over the first '
        '500 images of each class. Earlier stops use the prefix of that same rate schedule.', '',
        'Training is replayed independently for 100, 250, or 500 images per class. The stopping point '
        'is selected using mean accuracy across all digits and seeds on the original 5,000-image validation set. '
        'This uses no class-specific weighting or manual choice based on digit 5. '
        'The older confirmation set and official test set are not evaluated.', '',
        '## Validation results', '',
        '| Update rate / images per digit | Seed 42 | Seed 43 | Seed 44 | Mean | SD (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in grid:
        lines.append(f'| {row["variant"]} | '+' | '.join(f'{value:.2%}' for value in row['seed_accuracies'])+
                     f' | {row["mean_accuracy"]:.2%} | {row["sd_accuracy"]*100:.2f} |')
    lines += ['', f'Validation-selected configuration: **{chosen["variant"]}**.', '']
    if fresh:
        lines += ['## Fresh final check', '',
            'The 5,000 images reserved before the mistake-driven experiment remained unopened because that '
            'experiment failed. They are disjoint from all prior training, validation, and confirmation rows. '
            'Only this validation-selected candidate and the current baseline were evaluated on them. '
            'No settings were changed afterward.', '',
            '| Seed | Current: constant-500 | Selected candidate | Change (pp) |', '|---:|---:|---:|---:|']
        for seed in SEEDS:
            b = next(row['accuracy'] for row in fresh if row['seed'] == seed and row['variant'] == 'constant-500')
            c = next(row['accuracy'] for row in fresh if row['seed'] == seed and row['variant'] == chosen['variant'])
            lines.append(f'| {seed} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} |')
        b = statistics.mean(row['accuracy'] for row in fresh if row['variant'] == 'constant-500')
        c = statistics.mean(row['accuracy'] for row in fresh if row['variant'] == chosen['variant'])
        lines += ['', f'Mean fresh accuracy: **{b:.2%} → {c:.2%}** ({(c-b)*100:+.2f} percentage points). '
            'Compare these methods on these same images; these are not the earlier confirmation or official-test scores.', '',
            '| Digit | Current mean | Candidate mean | Change (pp) |', '|---:|---:|---:|---:|']
        for digit in map(str, range(10)):
            b = statistics.mean(row['per_digit'][digit]['recall'] for row in fresh if row['variant'] == 'constant-500')
            c = statistics.mean(row['per_digit'][digit]['recall'] for row in fresh if row['variant'] == chosen['variant'])
            lines.append(f'| {digit} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} |')
    else:
        lines += ['**None of the new configurations improved mean validation accuracy.** '
                  'The fresh holdout was not evaluated. Keep the current production model.']
    lines += ['', '## Interpretation limits', '',
        'This is a small, fixed comparison, not an exhaustive search. The three seeds share the same images. '
        'Replaying different prefix lengths changes later-class random draws, so this is a comparison of complete '
        'training runs rather than a single uninterrupted run with intermediate snapshots. '
        'No production model or training method is replaced by this script.', '',
        'Evaluation uses the previously checked exact-overlap cache and production voting; every model/evaluation '
        'also checks 100 images against coordinate-list recognition. Full metrics, per-digit errors and confusion '
        'matrices are in `validation.json` and, if used, `fresh.json`. Exact model metadata and predictions are under '
        '`artifacts/training-schedule/`.', '',
        'Run: `.venv/bin/python -m experiments.training_schedule --workers 3`.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    print(REPORT/'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    run(parser.parse_args().workers)
