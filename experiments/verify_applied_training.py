"""Train through the production method and compare against the selected experiment.
Run: .venv/bin/python -m experiments.verify_applied_training
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

from ImageRecognition.src.benchmark import write_json
from ImageRecognition.src.dataset import DEFAULT_DATA_DIR
from ImageRecognition.src.experiment import folders, load_model, save_model
from ImageRecognition.src.immune_algorithm import DEFAULT_AFFINITY

OUT = Path('artifacts/closest-template')


def verify_seed(seed):
    expected, metadata = load_model(f'artifacts/training-analysis/seed-{seed}/closest-0.1/model.pkl')
    model, initial_metadata = load_model(f'artifacts/current-approach/seed-{seed}/train-0/model.pkl')
    model.set_affinity_to_react(DEFAULT_AFFINITY)
    destination = OUT / f'seed-{seed}'
    initial_metadata = {**initial_metadata, 'reaction_affinity': DEFAULT_AFFINITY}
    save_model(model, destination / 'initial.pkl', initial_metadata)
    if seed == 42:
        save_model(model, OUT / 'initial.pkl', initial_metadata)
    names = {digit: [f'{index:05d}.png' for index in indices]
             for digit, indices in metadata['training_indices'].items()}
    random.seed(seed)
    start = time.perf_counter()
    model.training(folders(DEFAULT_DATA_DIR, 'train'), 500, filenames=names)
    seconds = time.perf_counter() - start
    for digit, cells in model.immune_system.items():
        for actual, previous in zip(cells, expected.immune_system[digit]):
            assert actual.white_pixels == previous.white_pixels, (seed, digit)
            assert actual.black_pixels == previous.black_pixels, (seed, digit)
    metadata = {**metadata, 'algorithm': 'closest same-class template with bidirectional mutation',
                'update_selection': 'closest', 'reaction_affinity': DEFAULT_AFFINITY,
                'training_seconds': seconds, 'production_templates_match_experiment': True,
                'note': 'Replayed the experiment reference/training manifests through default production training.'}
    save_model(model, destination / 'model.pkl', metadata)
    write_json(destination / 'training.json', metadata)
    if seed == 42:
        save_model(model, OUT / 'model.pkl', metadata)
        write_json(OUT / 'training.json', metadata)
    result = {'seed': seed, 'training_examples': 5000, 'matching_templates': 210,
              'exact_pixel_list_match': True, 'training_seconds': seconds,
              'model': str((destination / 'model.pkl').resolve())}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(verify_seed, (42, 43, 44)))
    write_json('reports/closest-template/production_verification.json', {
        'command': '.venv/bin/python -m experiments.verify_applied_training',
        'runs': results, 'new_accuracy_evaluation': False,
        'default_model_seed': 42,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in Path('ImageRecognition/src').glob('*.py')}})
