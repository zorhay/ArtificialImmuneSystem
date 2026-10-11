"""Verify every selected seed-42 test prediction through production PNG recognition.

Run from the project root:
PYTHONPATH=. .venv/bin/python reports/current-approach/verify_native.py
"""
import csv
import json
from pathlib import Path
import time

from ImageRecognition.src.experiment import load_model

root = Path(__file__).resolve().parents[2]
report_dir = root / 'reports/current-approach'
configuration = json.loads((report_dir / 'configuration.json').read_text())
selected = json.loads((report_dir / 'selection.json').read_text())['selected']
folder = Path(configuration['artifacts']) / 'seed-42' / f'train-{selected["train_per_class"]}'
model, _ = load_model(folder / 'model.pkl')
model.set_affinity_to_react(selected['threshold'])
with (folder / 'test' / f'predictions-{selected["threshold"]}.csv').open() as stream:
    rows = list(csv.DictReader(stream))
start = time.perf_counter()
correct = 0
for count, row in enumerate(rows, 1):
    path = Path(configuration['data_dir']) / 'images/test' / row['true_digit'] / f'{int(row["idx_index"]):05d}.png'
    prediction = model.recognition_image(path)
    assert prediction == row['prediction'], (row, prediction)
    correct += prediction == row['true_digit']
    if count % 1000 == 0:
        print(f'Native production recognition: {count}/10000 predictions match', flush=True)
result = {'seed': 42, 'training_per_class': selected['train_per_class'],
          'threshold': selected['threshold'], 'test_examples': len(rows),
          'matching_predictions': len(rows), 'correct': correct,
          'accuracy': correct / len(rows), 'seconds': time.perf_counter() - start,
          'path': 'model.recognition_image -> original coordinate-list affinity -> production voting',
          'command': 'PYTHONPATH=. .venv/bin/python reports/current-approach/verify_native.py'}
(report_dir / 'native_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
