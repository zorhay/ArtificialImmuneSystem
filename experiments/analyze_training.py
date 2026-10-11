"""Isolated training experiments; production algorithm and official test set unchanged.
Run: .venv/bin/python -m experiments.analyze_training --workers 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
from PIL import Image

from ImageRecognition.src.benchmark import (cached_affinities, grouped, metrics,
    training_order, verify_cache, write_json)
from ImageRecognition.src.dataset import load_mnist
from ImageRecognition.src.experiment import load_model, save_model
from ImageRecognition.src.limphocit import Limphocit_B

BASE = Path('artifacts/current-approach')
OUT = Path('artifacts/training-analysis')
REPORT = Path('reports/training-analysis')
SEEDS = (42, 43, 44)
THRESHOLDS = (0.0, 0.2, 0.3, 0.4)
VARIANTS = {
    'initial': ('none', 0.0, 'stochastic'),
    'current': ('all', 0.1, 'stochastic'),
    'all-0.03': ('all', 0.03, 'stochastic'),
    'all-0.01': ('all', 0.01, 'stochastic'),
    'closest-0.1': ('closest', 0.1, 'stochastic'),
    'closest-0.03': ('closest', 0.03, 'stochastic'),
    'old-rounding': ('all', 0.1, 'floor'),
}


def confirmation_indices(split, labels):
    """Exclude every training row used in the previous 100/500/1000 experiment."""
    used = set()
    for seed in SEEDS:
        for indices in training_order(split['training_pool'], seed).values():
            used.update(indices[:1000])
    rng = random.Random(20261013)
    result = []
    for digit, indices in split['training_pool'].items():
        remaining = [i for i in indices if i not in used]
        rng.shuffle(remaining)
        chosen = remaining[:500]
        assert len(chosen) == 500 and all(labels[i] == int(digit) for i in chosen)
        result.extend(chosen)
    assert not set(result) & used
    assert not set(result) & {i for values in split['validation'].values() for i in values}
    return result


def evaluate(model, images, labels, indices, thresholds, destination):
    destination.mkdir(parents=True, exist_ok=True)
    affinities = cached_affinities(model, images[indices])
    # Reuse the production affinity and voting equivalence check once per model.
    check = verify_cache(model, images[indices], affinities, [0.3])
    results = {}
    for threshold in thresholds:
        model.set_affinity_to_react(threshold)
        predictions, zero, ties = [], 0, 0
        for row in affinities:
            prediction = model.recognition_affinities(grouped(model, row))
            predictions.append(prediction)
            if prediction == 'unrecognized':
                if any(model.counters.values()):
                    ties += 1
                else:
                    zero += 1
        result = metrics(labels[indices], predictions, zero, ties)
        result['cache_verification'] = check
        results[str(threshold)] = result
        with (destination / f'predictions-{threshold}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['training_idx_index', 'true_digit', 'prediction'])
            writer.writerows(zip(map(int, indices), map(int, labels[indices]), predictions))
    write_json(destination / 'metrics.json', results)
    return results


def select_cells(cells, antigen, mode):
    if mode == 'all':
        return list(range(len(cells)))
    affinities = [cell.affinity(antigen) for cell in cells]
    maximum = max(affinities)
    winners = [index for index, value in enumerate(affinities) if value == maximum]
    return [random.choice(winners)] if len(winners) > 1 else winners


def train_variant(model, images, order, seed, mode, power, rounding):
    random.seed(seed)
    counts = {digit: [0] * len(cells) for digit, cells in model.immune_system.items()}
    original_count = Limphocit_B.mutation_count
    if rounding == 'floor':
        Limphocit_B.mutation_count = staticmethod(lambda differences, rate: int(differences * rate))
    try:
        for digit, indices in order.items():
            cells = model.immune_system[digit]
            for index in indices[:500]:
                antigen = Limphocit_B.get_from_pil_image(Image.fromarray(images[index]))
                for position in select_cells(cells, antigen, mode):
                    cells[position].mutation(antigen, power)
                    counts[digit][position] += 1
    finally:
        Limphocit_B.mutation_count = staticmethod(original_count)
    return counts


def diagnostics(model):
    affinities, whites = [], []
    for cells in model.immune_system.values():
        sets = [set(cell.white_pixels) for cell in cells]
        whites.extend(map(len, sets))
        for i, first in enumerate(sets):
            for second in sets[i + 1:]:
                affinities.append(len(first & second) / max(len(first), len(second)))
    return {'mean_same_class_template_affinity': statistics.mean(affinities),
            'mean_white_pixels': statistics.mean(whites)}


def validation_seed(seed):
    images, labels = load_mnist(split='train')
    split = json.loads((BASE / 'split.json').read_text())
    indices = np.array([i for values in split['validation'].values() for i in values])
    order = training_order(split['training_pool'], seed)
    results = {}
    for name, (mode, power, rounding) in VARIANTS.items():
        destination = OUT / f'seed-{seed}' / name
        initial_path = BASE / f'seed-{seed}/train-0/model.pkl'
        model, metadata = load_model(initial_path)
        start = time.perf_counter()
        counts = None
        if name == 'current':
            model, metadata = load_model(BASE / f'seed-{seed}/train-500/model.pkl')
        elif name != 'initial':
            counts = train_variant(model, images, order, seed, mode, power, rounding)
            metadata.update(training_per_class=500, training_examples=5000,
                training_indices={digit: values[:500] for digit, values in order.items()})
        metadata.update(experiment=name, update_selection=mode, mutation_power=power,
                        mutation_rounding=rounding, experimental_training_seconds=time.perf_counter() - start,
                        update_counts=counts)
        save_model(model, destination / 'model.pkl', metadata)
        write_json(destination / 'training.json', metadata)
        results[name] = {'thresholds': evaluate(model, images, labels, indices, THRESHOLDS,
                                               destination / 'validation'),
                         'diagnostics': diagnostics(model)}
        print(f'Seed {seed}, {name}: ' + ', '.join(f'{t}: {r["accuracy"]:.2%}'
              for t, r in results[name]['thresholds'].items()), flush=True)
    return seed, results


def confirm(task):
    seed, name, threshold, indices = task
    images, labels = load_mnist(split='train')
    destination = OUT / f'seed-{seed}' / name
    model, _ = load_model(destination / 'model.pkl')
    result = evaluate(model, images, labels, np.array(indices), [threshold], destination / 'confirmation')[str(threshold)]
    print(f'Confirmation seed {seed}, {name}, {threshold}: {result["accuracy"]:.2%}', flush=True)
    return {'seed': seed, 'variant': name, 'threshold': threshold, **result}


def run(workers):
    images, labels = load_mnist(split='train')
    split = json.loads((BASE / 'split.json').read_text())
    confirmation = confirmation_indices(split, labels)
    write_json(OUT / 'confirmation_indices.json', confirmation)
    write_json(REPORT / 'protocol.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'variants': VARIANTS,
        'seeds': SEEDS, 'training_per_digit': 500, 'templates_per_digit': 21,
        'pixel_threshold': 200, 'thresholds': THRESHOLDS, 'official_test_set_used': False,
        'validation': 'same fixed 5000 validation rows as current-approach benchmark',
        'confirmation': '5000 previously unused training rows, disjoint from all prior training rows and validation',
        'selection': 'best mean validation accuracy per variant; select best non-control, non-floor experimental variant; tie prefer listed order',
        'confirmation_comparators': 'selected experimental variant, initial templates, current training, each at its validation-selected threshold',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('ImageRecognition/src').glob('*.py')},
        'note': 'All other classifier behavior remains fixed. Old rounding is an audit of the previous integer floor, not the old broken mutation.'})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = dict(executor.map(validation_seed, SEEDS))
    write_json(REPORT / 'validation.json', results)
    grid = []
    for name in VARIANTS:
        for threshold in THRESHOLDS:
            scores = [results[seed][name]['thresholds'][str(threshold)]['accuracy'] for seed in SEEDS]
            grid.append({'variant': name, 'threshold': threshold, 'mean_accuracy': statistics.mean(scores),
                         'sd_accuracy': statistics.stdev(scores), 'seed_accuracies': scores})
    best = {name: max((row for row in grid if row['variant'] == name), key=lambda row: row['mean_accuracy'])
            for name in VARIANTS}
    selected = max((row for name, row in best.items() if name not in ('initial', 'current', 'old-rounding')),
                   key=lambda row: row['mean_accuracy'])
    write_json(REPORT / 'selection.json', {'grid': grid, 'best_per_variant': best, 'selected': selected})
    print('Selected experimental variant: ' + str(selected), flush=True)
    tasks = [(seed, name, best[name]['threshold'], confirmation)
             for seed in SEEDS for name in ('initial', 'current', selected['variant'])]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        confirmed = list(executor.map(confirm, tasks))
    write_json(REPORT / 'confirmation.json', confirmed)
    summaries = {}
    for name in ('initial', 'current', selected['variant']):
        rows = [row for row in confirmed if row['variant'] == name]
        summaries[name] = {'mean_accuracy': statistics.mean(row['accuracy'] for row in rows),
            'sd_accuracy': statistics.stdev(row['accuracy'] for row in rows),
            'seed_accuracies': [row['accuracy'] for row in rows]}
    write_json(REPORT / 'summary.json', summaries)
    print('Confirmation summary: ' + str(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    run(parser.parse_args().workers)
