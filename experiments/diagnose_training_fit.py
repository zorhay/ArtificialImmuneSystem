"""Compare current models on their own training images versus held-out validation."""
from pathlib import Path
import json
import statistics

import numpy as np

from experiments.analyze_training import evaluate
from ImageRecognition.src.benchmark import write_json
from ImageRecognition.src.dataset import load_mnist
from ImageRecognition.src.experiment import load_model


def main():
    images, labels = load_mnist(split='train')
    rows = []
    for seed in (42, 43, 44):
        model, metadata = load_model(f'artifacts/closest-template/seed-{seed}/model.pkl')
        indices = np.array([index for values in metadata['training_indices'].values() for index in values])
        assert len(indices) == len(set(indices.tolist())) == 5000
        result = evaluate(model, images, labels, indices, [0.3], Path(f'artifacts/training-schedule/seed-{seed}/current-training-fit'))['0.3']
        validation = json.loads(Path(f'artifacts/training-analysis/seed-{seed}/closest-0.1/validation/metrics.json').read_text())['0.3']
        rows.append({'seed': seed, 'training': result, 'validation': validation})
        print(f'Seed {seed}: training {result["accuracy"]:.2%}, validation {validation["accuracy"]:.2%}', flush=True)
    write_json('reports/training-schedule/training-fit.json', rows)
    train = statistics.mean(row['training']['accuracy'] for row in rows)
    valid = statistics.mean(row['validation']['accuracy'] for row in rows)
    lines = ['# Does the current model fit its training examples?', '',
        'This diagnostic evaluates the unchanged closest-template models on the exact 5,000 images '
        'each seed used for training, then compares with the original held-out validation set. '
        'No additional training, parameter selection, old confirmation evaluation, or fresh-holdout evaluation occurs.', '',
        f'Mean training accuracy: **{train:.2%}**. Mean validation accuracy: **{valid:.2%}**. '
        f'Training advantage: **{(train-valid)*100:+.2f} percentage points**.', '',
        '| Digit | Training accuracy, mean | Validation accuracy, mean |', '|---:|---:|---:|']
    for digit in map(str, range(10)):
        a = statistics.mean(row['training']['per_digit'][digit]['recall'] for row in rows)
        b = statistics.mean(row['validation']['per_digit'][digit]['recall'] for row in rows)
        lines.append(f'| {digit} | {a:.2%} | {b:.2%} |')
    lines += ['', 'Training accuracy includes initial reference images that are part of the training sequence. '
        'Training subsets differ across seeds; validation images are shared. This comparison is descriptive, '
        'not proof that no component overfits. Poor performance on both sets points to limitations of learning '
        'or representation rather than memorizing training images successfully and only failing on new images.', '',
        'Run: `.venv/bin/python -m experiments.diagnose_training_fit`.']
    Path('reports/training-schedule/training-fit.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
