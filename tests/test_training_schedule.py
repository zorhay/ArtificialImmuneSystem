import copy
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

from experiments.training_schedule import train_scheduled
from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


def test_constant_schedule_matches_current_training(tmp_path):
    Limphocit_B.init()
    model = ImmuneAlgorithm(2, 2, 0.3, 2)
    pixels = [np.array([[255, 0], [0, 0]], dtype=np.uint8), np.array([[0, 255], [255, 0]], dtype=np.uint8)]
    model.immune_system['0'] = [Limphocit_B.get_from_pil_image(Image.fromarray(image)) for image in pixels]
    for index, image in enumerate(pixels):
        Image.fromarray(image).save(tmp_path / f'{index}.png')
    reference = copy.deepcopy(model)
    paths, names = {'0': tmp_path}, {'0': ['0.png', '1.png']}
    random.seed(42)
    reference.training(paths, 2, filenames=names)
    train_scheduled(model, paths, names, 42, 2, 'constant')
    for actual, expected in zip(model.immune_system['0'], reference.immune_system['0']):
        assert actual.white_pixels == expected.white_pixels
        assert actual.black_pixels == expected.black_pixels


def test_decay_applies_the_same_positive_decreasing_schedule_to_every_class(monkeypatch):
    Limphocit_B.init()
    first = Limphocit_B.get_from_pil_image(Image.new('L', (2, 2), 255))
    second = Limphocit_B.get_from_pil_image(Image.new('L', (2, 2), 255))
    model = ImmuneAlgorithm(2, 2, 0.3, 1)
    model.immune_system = {'0': [first], '9': [second]}
    monkeypatch.setattr(model, 'load_antigen_by_path', lambda path: first)
    rates = {first.id: [], second.id: []}
    monkeypatch.setattr(Limphocit_B, 'mutation', lambda self, antigen, power: rates[self.id].append(power))
    train_scheduled(model, {'0': Path('.'), '9': Path('.')}, {'0': ['sample']*500, '9': ['sample']*500}, 42, 500, 'decay')
    assert rates[first.id] == rates[second.id]
    assert rates[first.id][0] == pytest.approx(0.1)
    assert rates[first.id][-1] == pytest.approx(0.03)
    assert all(0 < later <= earlier for earlier, later in zip(rates[first.id], rates[first.id][1:]))
