from pathlib import Path

import numpy as np
from PIL import Image

from experiments.mistake_driven_training import make_fresh_holdout, train_on_mistakes, SEEDS
from ImageRecognition.src.benchmark import training_order
from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


def make_cell(pixels):
    return Limphocit_B.get_from_pil_image(Image.fromarray(np.array(pixels, dtype=np.uint8)*255))


def test_correct_predictions_skip_wrong_and_unrecognized_update_correct_class(tmp_path, monkeypatch):
    Limphocit_B.init()
    model = ImmuneAlgorithm(2, 2, 0.3, 2)
    first = make_cell([[1, 0], [0, 0]])
    second = make_cell([[0, 1], [0, 0]])
    rival = make_cell([[0, 1], [1, 0]])
    model.immune_system = {'0': [first, second], '1': [rival]}
    images = [first.convert_to_image(), rival.convert_to_image(), Image.new('L', (2, 2))]
    names = ['correct.png', 'wrong.png', 'unrecognized.png']
    for name, image in zip(names, images):
        image.save(tmp_path / name)
    calls = []
    monkeypatch.setattr(Limphocit_B, 'mutation', lambda self, antigen, power: calls.append((self.id, power)))
    stats = train_on_mistakes(model, {'0': tmp_path}, {'0': names}, seed=42, set_size=3)['0']
    assert stats['seen'] == 3
    assert stats['correct_skipped'] == stats['wrong_updated'] == stats['unrecognized_updated'] == 1
    assert len(calls) == 2
    assert calls[0] == (second.id, 0.1)
    assert all(identifier in (first.id, second.id) and power == 0.1 for identifier, power in calls)
    assert sum(stats['updates_per_template']) == 2


def test_fresh_holdout_excludes_prior_training_validation_and_confirmation():
    split = {'training_pool': {str(digit): list(range(digit*4000, (digit+1)*4000)) for digit in range(10)},
             'validation': {str(digit): list(range(40000+digit*50, 40050+digit*50)) for digit in range(10)}}
    previous = list(range(0, 40000, 17))
    fresh = make_fresh_holdout(split, previous)
    excluded = set(previous)
    excluded.update(index for values in split['validation'].values() for index in values)
    for seed in SEEDS:
        for values in training_order(split['training_pool'], seed).values():
            excluded.update(values[:1000])
    assert len(fresh) == len(set(fresh)) == 5000
    assert not set(fresh) & excluded
    assert fresh == make_fresh_holdout(split, previous)
    assert all(sum(index//4000 == digit for index in fresh) == 500 for digit in range(10))
