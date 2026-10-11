import copy
import random

import numpy as np
from PIL import Image
import pytest

from experiments.analyze_training import train_variant
from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


def model_with_two_templates():
    Limphocit_B.init()
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[0, 2:20, 3:7] = 255
    images[1, 8:24, 15:20] = 255
    model = ImmuneAlgorithm(28, 28, 0.3, 2)
    model.immune_system['0'] = [Limphocit_B.get_from_pil_image(Image.fromarray(image)) for image in images]
    return model, images


@pytest.mark.parametrize('mode', ['all', 'closest'])
def test_experimental_update_reproduces_production_training(tmp_path, mode):
    first, images = model_with_two_templates()
    second = copy.deepcopy(first)
    for index, image in enumerate(images):
        Image.fromarray(image).save(tmp_path / f'{index}.png')
    random.seed(42)
    first.training({'0': tmp_path}, 2, filenames={'0': ['0.png', '1.png']}, update_selection=mode)
    counts = train_variant(second, images, {'0': [0, 1]}, 42, mode, 0.1, 'stochastic')
    assert sum(counts['0']) == (4 if mode == 'all' else 2)
    for expected, actual in zip(first.immune_system['0'], second.immune_system['0']):
        assert expected.white_pixels == actual.white_pixels
        assert expected.black_pixels == actual.black_pixels


def test_closest_only_updates_the_most_similar_template():
    model, images = model_with_two_templates()
    images[0, 2:20, 7] = 255
    before = copy.deepcopy(model.immune_system['0'][1].white_pixels)
    counts = train_variant(model, images, {'0': [0]}, 42, 'closest', 1.0, 'stochastic')
    assert counts == {'0': [1, 0]}
    assert model.immune_system['0'][1].white_pixels == before
    target = Limphocit_B.get_from_pil_image(Image.fromarray(images[0]))
    assert model.immune_system['0'][0].affinity(target) == 1


def test_old_rounding_experiment_restores_production_rounding():
    model, images = model_with_two_templates()
    original = Limphocit_B.mutation_count
    train_variant(model, images, {'0': [0]}, 42, 'all', 0.1, 'floor')
    assert Limphocit_B.mutation_count is original
