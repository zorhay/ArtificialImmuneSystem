"""Test mistake-driven closest-template mutation without changing production training.
Run: .venv/bin/python -m experiments.mistake_driven_training --workers 3
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
OUT = Path('artifacts/mistake-driven')
REPORT = Path('reports/mistake-driven')
SEEDS = (42, 43, 44)
AFFINITY = 0.3


def make_fresh_holdout(split, prior_confirmation):
    excluded = set(prior_confirmation)
    excluded.update(i for indices in split['validation'].values() for i in indices)
    for seed in SEEDS:
        for indices in training_order(split['training_pool'], seed).values():
            excluded.update(indices[:1000])
    rng = random.Random(20261014)
    fresh = []
    for indices in split['training_pool'].values():
        candidates = [index for index in indices if index not in excluded]
        rng.shuffle(candidates)
        if len(candidates) < 500:
            raise ValueError('Insufficient unused images for the fresh holdout')
        fresh.extend(candidates[:500])
    assert len(fresh) == len(set(fresh)) == 5000 and not set(fresh) & excluded
    return fresh


def train_on_mistakes(model, paths, names, seed, set_size=500):
    """One fixed rule for all classes; prediction is made before each update."""
    random.seed(seed)
    stats = {}
    for digit, directory in paths.items():
        stats[digit] = {'seen': 0, 'correct_skipped': 0, 'wrong_updated': 0,
                        'unrecognized_updated': 0, 'updates_per_template': [0] * len(model.immune_system[digit])}
        for name in names[digit][:set_size]:
            antigen = model.load_antigen_by_path(directory / name)
            prediction = model.recognition_antigen(antigen)
            stats[digit]['seen'] += 1
            if prediction == digit:
                stats[digit]['correct_skipped'] += 1
                continue
            reason = 'unrecognized_updated' if prediction == 'unrecognized' else 'wrong_updated'
            stats[digit][reason] += 1
            position = select_cells(model.immune_system[digit], antigen, 'closest')[0]
            model.immune_system[digit][position].mutation(antigen, 0.1)
            stats[digit]['updates_per_template'][position] += 1
    return stats


def validation_run(seed):
    images, labels = load_mnist(split='train')
    split = json.loads((BASE / 'split.json').read_text())
    order = training_order(split['training_pool'], seed)
    indices = np.array([i for values in split['validation'].values() for i in values])
    model, metadata = load_model(BASE / f'seed-{seed}/train-0/model.pkl')
    model.set_affinity_to_react(AFFINITY)
    names = {digit: [f'{index:05d}.png' for index in values[:500]] for digit, values in order.items()}
    start = time.perf_counter()
    stats = train_on_mistakes(model, folders(DEFAULT_DATA_DIR, 'train'), names, seed)
    metadata = {**metadata, 'algorithm': 'mistake-driven closest same-class mutation',
        'training_per_class': 500, 'training_examples': 5000, 'reaction_affinity': AFFINITY,
        'training_indices': {digit: values[:500] for digit, values in order.items()},
        'training_seconds': time.perf_counter() - start, 'training_stats': stats,
        'update_rule': 'skip correct predictions; update one closest correct-class template on wrong or unrecognized predictions',
        'mutation_power': 0.1, 'seed': seed}
    destination = OUT / f'seed-{seed}'
    save_model(model, destination / 'model.pkl', metadata)
    write_json(destination / 'training.json', metadata)
    result = evaluate(model, images, labels, indices, [AFFINITY], destination / 'validation')[str(AFFINITY)]
    baseline = json.loads(Path(f'artifacts/training-analysis/seed-{seed}/closest-0.1/validation/metrics.json').read_text())[str(AFFINITY)]
    updates = sum(sum(row['updates_per_template']) for row in stats.values())
    print(f'Seed {seed}: validation {result["accuracy"]:.2%} vs current {baseline["accuracy"]:.2%}; {updates}/5000 updates', flush=True)
    return {'seed': seed, 'candidate': result, 'baseline': baseline, 'training_stats': stats,
            'updates': updates, 'training_seconds': metadata['training_seconds']}


def fresh_run(task):
    seed, variant, indices = task
    images, labels = load_mnist(split='train')
    path = OUT / f'seed-{seed}/model.pkl' if variant == 'mistake-driven' else Path(f'artifacts/closest-template/seed-{seed}/model.pkl')
    model, _ = load_model(path)
    model.set_affinity_to_react(AFFINITY)
    result = evaluate(model, images, labels, np.array(indices), [AFFINITY], OUT / f'seed-{seed}' / variant / 'fresh')[str(AFFINITY)]
    print(f'Fresh holdout seed {seed}, {variant}: {result["accuracy"]:.2%}', flush=True)
    return {'seed': seed, 'variant': variant, **result}


def run(workers):
    split = json.loads((BASE / 'split.json').read_text())
    previous = json.loads(Path('artifacts/training-analysis/confirmation_indices.json').read_text())
    fresh = make_fresh_holdout(split, previous)
    write_json(OUT / 'fresh_indices.json', fresh)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': SEEDS,
        'mutation_power': 0.1, 'reaction_affinity': AFFINITY, 'templates_per_digit': 21,
        'training_per_digit': 500, 'epochs': 1, 'pixel_threshold': 200,
        'training_order': 'same class-blocked, within-class shuffled order and initial templates as previous experiment',
        'candidate': 'update only on wrong or unrecognized training predictions, closest same-class template, uniform rule for all digits',
        'control': 'current closest-template training updates every training image',
        'validation': 'original 5000-image validation set; previous confirmation set not evaluated',
        'fresh_holdout': '5000 new official-training rows, 500 per digit, excludes all prior training, validation, and confirmation rows',
        'gate': 'open fresh holdout only if candidate mean validation accuracy exceeds control; one fixed candidate, no threshold/rate tuning',
        'promotion': 'no automatic production change; report evidence and per-digit tradeoffs',
        'official_test_set_used': False,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in Path('ImageRecognition/src').glob('*.py')}}
    write_json(REPORT / 'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        validation = list(executor.map(validation_run, SEEDS))
    write_json(REPORT / 'validation.json', validation)
    baseline = statistics.mean(row['baseline']['accuracy'] for row in validation)
    candidate = statistics.mean(row['candidate']['accuracy'] for row in validation)
    decision = {'baseline_mean_validation': baseline, 'candidate_mean_validation': candidate,
                'validation_gain': candidate-baseline, 'fresh_holdout_opened': candidate > baseline}
    write_json(REPORT / 'decision.json', decision)
    fresh_results = []
    if candidate > baseline:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fresh_results = list(executor.map(fresh_run, [(seed, variant, fresh) for seed in SEEDS
                                                          for variant in ('current', 'mistake-driven')]))
        write_json(REPORT / 'fresh.json', fresh_results)
    for path, expected in protocol['production_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    write_report(validation, decision, fresh_results)


def write_report(validation, decision, fresh):
    lines = ['# Mistake-driven training experiment', '',
        'A single rule was tested for every digit: predict the training image first. '
        'If the answer is correct, skip the mutation. Otherwise mutate the closest template of the correct class '
        'using the existing 10% update. Unrecognized predictions count as mistakes. '
        'Only the experimental script changes; production training and its saved models are preserved.', '',
        '## Protocol', '',
        '- Three seeds: 42, 43, 44; identical starting templates and ordered training images to the current baseline.',
        '- 21 templates per digit, 500 training images per digit, one pass, mutation power 0.1, reaction threshold 0.3, intensity >200.',
        '- Prediction during training uses the production coordinate-list affinity and voting. No confidence gate, learning-rate schedule, or checkpoint search.',
        '- No settings or rules specific to poorly performing digits. Every class has 500 training images.',
        '- Validation uses the original held-out 5,000 images. The old confirmation set is not evaluated or used to select a configuration.',
        '- A fresh final holdout was reserved before running: 500 new images per digit, disjoint from every previous training, validation, and confirmation row.',
        '- The fresh holdout opens only if mean validation accuracy improves. The official 10,000-image test set is not used.', '',
        '## Validation', '',
        '| Seed | Current training | Mistake-driven | Change (points) | Updates / 5,000 |',
        '|---:|---:|---:|---:|---:|']
    for row in validation:
        b, c = row['baseline']['accuracy'], row['candidate']['accuracy']
        lines.append(f'| {row["seed"]} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} | {row["updates"]} |')
    lines += ['', f'Mean: **{decision["baseline_mean_validation"]:.2%} → {decision["candidate_mean_validation"]:.2%}** '
              f'({decision["validation_gain"]*100:+.2f} percentage points).', '',
              '## Per-digit validation accuracy', '',
              '| Digit | Current mean | Candidate mean | Change (points) | Mean updates / 500 |',
              '|---:|---:|---:|---:|---:|']
    for digit in map(str, range(10)):
        b = statistics.mean(row['baseline']['per_digit'][digit]['recall'] for row in validation)
        c = statistics.mean(row['candidate']['per_digit'][digit]['recall'] for row in validation)
        updates = statistics.mean(sum(row['training_stats'][digit]['updates_per_template']) for row in validation)
        lines.append(f'| {digit} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} | {updates:.1f} |')
    if fresh:
        lines += ['', '## Fresh holdout', '', '| Seed | Current | Mistake-driven | Change (points) |', '|---:|---:|---:|---:|']
        for seed in SEEDS:
            b = next(row['accuracy'] for row in fresh if row['seed'] == seed and row['variant'] == 'current')
            c = next(row['accuracy'] for row in fresh if row['seed'] == seed and row['variant'] == 'mistake-driven')
            lines.append(f'| {seed} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} |')
        b = statistics.mean(row['accuracy'] for row in fresh if row['variant'] == 'current')
        c = statistics.mean(row['accuracy'] for row in fresh if row['variant'] == 'mistake-driven')
        lines += ['', f'Fresh mean: **{b:.2%} → {c:.2%}** ({(c-b)*100:+.2f} points). '
                  'These are new images, so compare the two methods here rather than comparing against earlier test/confirmation scores.', '',
                  '| Digit | Current mean | Candidate mean | Change (points) |', '|---:|---:|---:|---:|']
        for digit in map(str, range(10)):
            b = statistics.mean(row['per_digit'][digit]['recall'] for row in fresh if row['variant'] == 'current')
            c = statistics.mean(row['per_digit'][digit]['recall'] for row in fresh if row['variant'] == 'mistake-driven')
            lines.append(f'| {digit} | {b:.2%} | {c:.2%} | {(c-b)*100:+.2f} |')
    else:
        lines += ['', '**The candidate did not beat current training on validation.** '
                  'The fresh holdout was not evaluated and remains available for a later selected candidate. '
                  'Do not promote this candidate on the current evidence.']
    lines += ['', '## Limits and artifacts', '',
        'One update policy was tested at fixed settings. It does not test whether confidence-gated updates, '
        'a decreasing mutation rate, or checkpoint selection would help. Skipping updates also changes the random-number '
        'sequence; paired seeds do not mean identical individual mutations. Class-blocked order is held fixed and could '
        'affect mistake-driven learning because updates to later classes can change earlier predictions. '
        'The three seeds share evaluation images and are not independent extra datasets. '
        'Correctness-gating alone does not explicitly push apart confused classes.', '',
        'Evaluation reuses the exact-overlap calculation and production voting from the previous verified harness; '
        '100 images per model/evaluation are checked against coordinate-list recognition. '
        'Training predictions use that coordinate-list recognition directly.', '',
        '`protocol.json`, `decision.json`, `validation.json`, and (if opened) `fresh.json` contain the full settings, '
        'metrics, per-digit counts, and confusion matrices. Models, exact input indices, and per-image predictions '
        'are in `artifacts/mistake-driven/`. Production source hashes are checked unchanged at completion.', '',
        'Run: `.venv/bin/python -m experiments.mistake_driven_training --workers 3`.']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Report: {REPORT / "report.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    run(parser.parse_args().workers)
