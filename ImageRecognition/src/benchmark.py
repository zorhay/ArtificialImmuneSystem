"""Controlled evaluation of the existing all-cells mutation classifier.

Run: python -m ImageRecognition.src.benchmark
Training calls the production training loop. Evaluation caches the identical
white-pixel overlap affinity, then calls production voting for each threshold.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

from .dataset import DEFAULT_DATA_DIR, load_mnist
from .experiment import folders, load_model, save_model
from .immune_algorithm import ImmuneAlgorithm, UNRECOGNIZED
from .limphocit import Limphocit_B

SIZES = (100, 500, 1000)
SEEDS = (42, 43, 44)
THRESHOLDS = (0.3, 0.4, 0.5)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_split(labels, validation_per_class=500, split_seed=20261012):
    rng = random.Random(split_seed)
    validation, pool = {}, {}
    for digit in range(10):
        indices = np.flatnonzero(labels == digit).tolist()
        rng.shuffle(indices)
        validation[str(digit)] = indices[:validation_per_class]
        pool[str(digit)] = indices[validation_per_class:]
    return validation, pool


def training_order(pool, seed):
    rng = random.Random(seed)
    result = {digit: list(indices) for digit, indices in pool.items()}
    for indices in result.values():
        rng.shuffle(indices)
    return result


def cached_affinities(model, images):
    """Exact overlap/max(white counts), equivalent to the original two ratios.

    Binary float32 dot products count <=784 pixels exactly; division is float64.
    This is an evaluation optimization only; training and stored cells are unchanged.
    """
    cells = [cell for cells in model.immune_system.values() for cell in cells]
    masks = np.zeros((len(cells), 784), dtype=np.float32)
    for row, cell in enumerate(cells):
        white, black = set(cell.white_pixels), set(cell.black_pixels)
        assert len(white) == len(cell.white_pixels) and not white & black
        assert len(white | black) == 784
        for i, j in white:
            masks[row, i * 28 + j] = 1
    inputs = (images.reshape(-1, 784) > 200).astype(np.float32)
    overlaps = (inputs @ masks.T).astype(np.float64)
    denominator = np.maximum(inputs.sum(axis=1)[:, None], masks.sum(axis=1)[None, :])
    return np.divide(overlaps, denominator, out=np.zeros_like(overlaps), where=denominator != 0)


def grouped(model, row):
    result, offset = {}, 0
    for digit, cells in model.immune_system.items():
        result[digit] = row[offset:offset + len(cells)].tolist()
        offset += len(cells)
    return result


def verify_cache(model, images, affinities, thresholds):
    # Evenly spaced samples cover every class in the class-grouped validation set.
    indices = np.linspace(0, len(images) - 1, min(100, len(images)), dtype=int)
    for index in indices:
        antigen = Limphocit_B.get_from_pil_image(Image.fromarray(images[index]))
        native = [cell.affinity(antigen) for cells in model.immune_system.values() for cell in cells]
        np.testing.assert_array_equal(native, affinities[index])
        for threshold in thresholds:
            model.set_affinity_to_react(threshold)
            expected = model.recognition_antigen(antigen)
            actual = model.recognition_affinities(grouped(model, affinities[index]))
            assert expected == actual
    return {'images': len(indices), 'affinity_pairs': len(indices) * affinities.shape[1],
            'prediction_checks': len(indices) * len(thresholds), 'exact_match': True}


def metrics(truth, predictions, zero, ties):
    confusion = np.zeros((10, 11), dtype=int)
    for label, prediction in zip(truth, predictions):
        confusion[int(label), 10 if prediction == UNRECOGNIZED else int(prediction)] += 1
    correct = int(np.trace(confusion[:, :10]))
    rejected = int(confusion[:, 10].sum())
    recognized = len(truth) - rejected
    per_digit = {}
    for digit in range(10):
        total = int(confusion[digit].sum())
        predicted = int(confusion[:, digit].sum())
        hits = int(confusion[digit, digit])
        precision = hits / predicted if predicted else 0.0
        recall = hits / total if total else 0.0
        per_digit[str(digit)] = {'correct': hits, 'total': total,
            'unrecognized': int(confusion[digit, 10]), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {'examples': len(truth), 'correct': correct, 'accuracy': correct / len(truth),
            'unrecognized': rejected, 'zero_votes': zero, 'tied_maxima': ties,
            'coverage': recognized / len(truth),
            'recognized_accuracy': correct / recognized if recognized else None,
            'macro_f1': statistics.mean(row['f1'] for row in per_digit.values()),
            'per_digit': per_digit, 'confusion_matrix': confusion.tolist(),
            'confusion_columns': [str(i) for i in range(10)] + [UNRECOGNIZED]}


def evaluate_cached(model, images, labels, indices, thresholds, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    affinities = cached_affinities(model, images)
    check = verify_cache(model, images, affinities, thresholds)
    results = {}
    for threshold in thresholds:
        model.set_affinity_to_react(threshold)
        predictions, zero, ties = [], 0, 0
        for row in affinities:
            prediction = model.recognition_affinities(grouped(model, row))
            predictions.append(prediction)
            if prediction == UNRECOGNIZED:
                if any(model.counters.values()):
                    ties += 1
                else:
                    zero += 1
        result = metrics(labels, predictions, zero, ties)
        result['cache_verification'] = check
        results[str(threshold)] = result
        with (destination / f'predictions-{threshold}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx_index', 'true_digit', 'prediction'])
            writer.writerows(zip(map(int, indices), map(int, labels), predictions))
    write_json(destination / 'metrics.json', results)
    return results


def train_seed(task):
    seed, data_dir, artifacts, validation, pool = task
    images, labels = load_mnist(data_dir, 'train')
    order = training_order(pool, seed)
    selected = np.array([i for values in validation.values() for i in values])
    paths = folders(data_dir, 'train')
    names = {digit: [f'{i:05d}.png' for i in indices[:max(SIZES)]] for digit, indices in order.items()}
    reference = {digit: values[:21] for digit, values in order.items()}
    results = {}
    for count in (0, *SIZES):
        random.seed(seed)
        Limphocit_B.init()
        model = ImmuneAlgorithm(28, 28, 0.4, 21)
        for digit, indices in reference.items():
            model.immune_system[digit] = [model.load_antigen_by_path(paths[digit] / f'{i:05d}.png') for i in indices]
        metadata = {'algorithm': 'all-cells bidirectional mutation', 'mutation_version': 3,
            'mutation_power': 0.1, 'mutation_rounding': 'stochastic', 'seed': seed,
            'templates_per_digit': 21, 'pixel_threshold': 200, 'reaction_affinity': 0.4,
            'training_per_class': count, 'training_examples': count * 10,
            'reference_indices': reference,
            'training_indices': {digit: indices[:count] for digit, indices in order.items()},
            'order': 'digit 0 through 9; seeded shuffled images within digit; one pass'}
        start = time.perf_counter()
        if count:
            model.training(paths, count, filenames=names, update_selection='all')
        metadata['training_seconds'] = time.perf_counter() - start
        destination = Path(artifacts) / f'seed-{seed}' / f'train-{count}'
        save_model(model, destination / 'model.pkl', metadata)
        write_json(destination / 'training.json', metadata)
        results[str(count)] = evaluate_cached(model, images[selected], labels[selected], selected,
                                              THRESHOLDS, destination / 'validation')
        print(f'Seed {seed}, {count}/digit: validation ' + ', '.join(
            f'{threshold}: {result["accuracy"]:.2%}' for threshold, result in results[str(count)].items()), flush=True)
    return seed, results


def test_seed(task):
    seed, count, threshold, data_dir, artifacts = task
    images, labels = load_mnist(data_dir, 'test')
    path = Path(artifacts) / f'seed-{seed}' / f'train-{count}'
    model, _ = load_model(path / 'model.pkl')
    result = evaluate_cached(model, images, labels, np.arange(len(labels)), [threshold], path / 'test')
    print(f'Test seed {seed}, {count}/digit, threshold {threshold}: {result[str(threshold)]["accuracy"]:.2%}', flush=True)
    return seed, count, result[str(threshold)]


def run(data_dir, artifacts, reports, workers):
    artifacts, reports = Path(artifacts), Path(reports)
    images, labels = load_mnist(data_dir, 'train')
    validation, pool = make_split(labels)
    assert all(set(validation[digit]).isdisjoint(pool[digit]) for digit in pool)
    write_json(artifacts / 'split.json', {'validation': validation, 'training_pool': pool})
    source = Path(__file__).parent
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'numpy': np.__version__, 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob('*.py'))},
        'seeds': SEEDS, 'train_per_class': SIZES, 'reaction_thresholds': THRESHOLDS,
        'templates_per_digit': 21, 'mutation_power': 0.1, 'pixel_threshold': 200,
        'validation_per_class': 500, 'split_seed': 20261012, 'workers': workers,
        'selection': 'highest mean validation overall accuracy across seeds; ties prefer smaller training size, then lower threshold',
        'test_policy': 'selected configuration and predeclared untrained reference baseline, all three seeds, no test tuning',
        'artifacts': str(artifacts.resolve()), 'data_dir': str(Path(data_dir).resolve())}
    write_json(reports / 'configuration.json', config)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        validation_results = dict(executor.map(train_seed, [(seed, data_dir, artifacts, validation, pool) for seed in SEEDS]))
    write_json(reports / 'validation.json', validation_results)
    grid = []
    for count in (0, *SIZES):
        for threshold in THRESHOLDS:
            scores = [validation_results[seed][str(count)][str(threshold)]['accuracy'] for seed in SEEDS]
            grid.append({'train_per_class': count, 'threshold': threshold,
                         'mean_accuracy': statistics.mean(scores), 'std_accuracy': statistics.stdev(scores),
                         'seed_accuracies': dict(zip(SEEDS, scores))})
    # The reference-only control cannot win the training-configuration search.
    chosen = max((row for row in grid if row['train_per_class']),
                 key=lambda row: (row['mean_accuracy'], -row['train_per_class'], -row['threshold']))
    write_json(reports / 'selection.json', {'selected': chosen, 'grid': grid})
    print(f'Selected on validation only: {chosen}', flush=True)
    tasks = [(seed, count, chosen['threshold'], data_dir, artifacts)
             for seed in SEEDS for count in (0, chosen['train_per_class'])]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        test_results = list(executor.map(test_seed, tasks))
    write_json(reports / 'test.json', [{'seed': seed, 'train_per_class': count, **result} for seed, count, result in test_results])
    write_report(reports, artifacts, grid, chosen, test_results)


def write_report(reports, artifacts, grid, chosen, test_results):
    trained = [result for _, count, result in test_results if count]
    baseline = [result for _, count, result in test_results if not count]
    mean = statistics.mean(result['accuracy'] for result in trained)
    sd = statistics.stdev(result['accuracy'] for result in trained)
    lines = ['# Current approach: controlled MNIST evaluation', '',
        f'Selected trained models: **{mean:.2%} mean test accuracy**, sample SD **{sd * 100:.2f} percentage points** across seeds 42, 43, 44.', '',
        f'Untrained reference control at the same threshold: **{statistics.mean(r["accuracy"] for r in baseline):.2%}** mean test accuracy.', '',
        '## What ran', '',
        '```sh', '.venv/bin/python -m pytest -q',
        '.venv/bin/python -m ImageRecognition.src.benchmark --workers 3', '```', '',
        'Fresh independent models: 100, 500, 1,000 training images per digit × seeds 42, 43, 44. '
        'Each trained model is evaluated at thresholds 0.3, 0.4, 0.5: 27 validation settings. '
        'Three additional untrained initial networks are controls. '
        'Configuration selection uses mean validation accuracy across seeds, then evaluates the selected '
        'configuration and the untrained control on all 10,000 official test images for each seed.', '',
        '## Configuration and isolation', '',
        '- 21 templates per digit (210 total), original affinity: overlap / maximum white-pixel count.',
        '- Every template in the labeled class mutates toward every training image, once; mutation power 0.1.',
        '- Original 28×28 images; white means intensity strictly greater than 200. No preprocessing changes.',
        '- Fixed validation split: 500 per digit (5,000 total), drawn from official training data with split seed 20261012.',
        '- Initial templates and training examples come only from the remaining 55,000 images; no validation/test leakage.',
        '- Seeded within-class shuffled training order; nested subsets across sizes and identical initialization per seed. '
        'Reference images may also occur in training, as in the original approach.',
        f'- Selected configuration: **{chosen["train_per_class"]} per digit, threshold {chosen["threshold"]}**.',
        '- No full-pool retraining after selection. Test models are exactly the selected validation checkpoints.', '',
        '## Implementation corrections included', '',
        'The previous bidirectional pixel-list correction and unrecognized output remain. '
        'This run additionally isolates model dictionaries, uses stochastic rounding '
        '(e.g. one mismatch changes with probability 0.1, preserving the expected rate), '
        'and evaluates the same exponential weights with a common finite scale. '
        'Exact matches dominate finite matches and vote by count; tied maxima return unrecognized. '
        'Counters are relative scores, not probabilities. '
        'These results describe the corrected implementation, not the older 11.47% checkpoint.', '',
        '## Validation results', '',
        '| Training / digit | Threshold | Mean accuracy | SD (pp) | Seed 42 | Seed 43 | Seed 44 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in grid:
        values = row['seed_accuracies']
        lines.append(f'| {row["train_per_class"]} | {row["threshold"]} | {row["mean_accuracy"]:.2%} | {row["std_accuracy"] * 100:.2f} | ' +
                     ' | '.join(f'{values[seed]:.2%}' for seed in SEEDS) + ' |')
    lines += ['', '## Full test results', '',
        '| Training / digit | Seed | Correct / 10,000 | Accuracy | Coverage | Recognized accuracy | Unrecognized | Macro F1 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed, count, result in test_results:
        lines.append(f'| {count} | {seed} | {result["correct"]} | {result["accuracy"]:.2%} | {result["coverage"]:.2%} | '
                     f'{result["recognized_accuracy"]:.2%} | {result["unrecognized"]} | {result["macro_f1"]:.4f} |')
    lines += ['', 'Unrecognized predictions count as incorrect in overall accuracy. Coverage is the fraction assigned '
              'a digit; recognized accuracy excludes rejected inputs. SD measures seed variation, not a confidence interval. '
              'All runs reuse the same test images; they are not 30,000 independent examples.', '',
              '## Per-digit test accuracy (recall), trained models', '',
              '| Digit | Test images | Seed 42 | Seed 43 | Seed 44 | Mean | Mean precision |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for digit in map(str, range(10)):
        rows = [result['per_digit'][digit] for result in trained]
        lines.append(f'| {digit} | {rows[0]["total"]} | ' + ' | '.join(f'{row["recall"]:.2%}' for row in rows) +
                     f' | {statistics.mean(row["recall"] for row in rows):.2%} | {statistics.mean(row["precision"] for row in rows):.2%} |')
    lines += ['', '## Interpretation and limits', '',
        'Compare each trained model with its own initial network to measure what mutation adds. '
        'More training is not assumed to help: all cells receive the same image sequence, '
        'so repeated updates can erase useful initial shape diversity and emphasize recent examples. '
        'There is no discriminative update against other classes. These remain properties of the approach.', '',
        'Evaluation computes identical integer white-pixel overlaps in batches and calls the production vote routine. '
        'For every model/split, 100 images are checked against the original coordinate-list affinity and recognition '
        'at every evaluated threshold, with exact equality required. Training always calls the production all-cells loop. '
        'This optimization changes evaluation cost, not the similarity formula.', '',
        'The configuration grid is deliberately narrow. No claim is made that it is globally optimal. '
        'Seed variation and all per-class errors are retained. Historical reports remain untouched.', '',
        '## Artifacts', '',
        '- `configuration.json`: settings, parent commit, and SHA-256 of every source file at run time.',
        '- `selection.json`, `validation.json`, `test.json`: complete grid, metrics, 10×11 confusion matrices, parity checks.',
        f'- `{artifacts.resolve()}/split.json`: exact validation and training-pool indices.',
        f'- `{artifacts.resolve()}/seed-*/train-*/`: model.pkl, ordered reference/training indices, '
        'and per-image predictions for each evaluated threshold. Index means the original MNIST IDX row.']
    (Path(reports) / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Report: {Path(reports) / "report.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/current-approach'))
    parser.add_argument('--reports', type=Path, default=Path('reports/current-approach'))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run(args.data_dir, args.artifacts, args.reports, args.workers)
