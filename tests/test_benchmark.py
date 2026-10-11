import random

import numpy as np
from PIL import Image

from ImageRecognition.src.benchmark import cached_affinities, make_split, training_order, verify_cache
from ImageRecognition.src.experiment import load_model, save_model
from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


def test_split_is_disjoint_reproducible_and_training_subsets_are_nested():
    labels = np.repeat(np.arange(10), 50)
    validation, pool = make_split(labels, validation_per_class=5)
    assert (validation, pool) == make_split(labels, validation_per_class=5)
    first = training_order(pool, 42)
    assert first == training_order(pool, 42)
    assert first != training_order(pool, 43)
    for digit in pool:
        assert len(validation[digit]) == 5
        assert set(validation[digit]).isdisjoint(first[digit])
        assert set(first[digit][:10]) <= set(first[digit][:20])
        assert all(labels[index] == int(digit) for index in first[digit])


def test_batched_evaluation_matches_original_coordinate_affinity():
    Limphocit_B.init()
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, size=(8, 28, 28), dtype=np.uint8)
    images[0] = 0
    images[1] = 200  # Strict threshold: 200 must remain black.
    images[2] = 201
    model = ImmuneAlgorithm(28, 28, 0.4, 2)
    model.immune_system = {str(digit): [Limphocit_B.get_from_pil_image(Image.fromarray(image))
                                      for image in images[digit * 2:digit * 2 + 2]] for digit in range(4)}
    affinities = cached_affinities(model, images)
    assert verify_cache(model, images, affinities, [0.3, 0.4, 0.5])['exact_match']
    assert np.all(affinities[:2] == 0)


def test_loading_two_models_keeps_the_first_network_intact(tmp_path):
    Limphocit_B.init()
    first, second = ImmuneAlgorithm(28, 28, 0.4, 1), ImmuneAlgorithm(28, 28, 0.4, 1)
    random.seed(42)
    first.immune_system['0'] = [Limphocit_B.generate(20)]
    second.immune_system['1'] = [Limphocit_B.generate(30)]
    save_model(first, tmp_path / 'first.pkl', {})
    save_model(second, tmp_path / 'second.pkl', {})
    restored, _ = load_model(tmp_path / 'first.pkl')
    load_model(tmp_path / 'second.pkl')
    assert list(restored.immune_system) == ['0']
    assert len(restored.immune_system['0'][0].white_pixels) == 20
