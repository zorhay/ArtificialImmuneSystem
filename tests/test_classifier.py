import math

import numpy as np
from PIL import Image
import pytest

from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


@pytest.fixture(autouse=True)
def reset_original_state():
    ImmuneAlgorithm.immune_system.clear()
    ImmuneAlgorithm.counters.clear()
    Limphocit_B.init()
    yield
    ImmuneAlgorithm.immune_system.clear()
    ImmuneAlgorithm.counters.clear()


def cell(pixels):
    return Limphocit_B.get_from_pil_image(Image.fromarray(np.array(pixels, dtype=np.uint8) * 255))


def test_affinity_formula_is_preserved_and_blank_inputs_do_not_react():
    first, second = cell([[1, 1], [1, 0]]), cell([[0, 1], [1, 0]])
    assert first.affinity(second) == pytest.approx(2 / 3)
    assert cell([[0, 0], [0, 0]]).affinity(second) == 0
    assert second.affinity(cell([[0, 0], [0, 0]])) == 0


def test_mutation_moves_pixels_in_both_directions():
    source, target = cell([[1, 1], [0, 0]]), cell([[0, 0], [1, 1]])
    source.mutation(target, 1)
    assert set(source.white_pixels) == set(target.white_pixels)
    assert set(source.black_pixels) == set(target.black_pixels)
    assert not set(source.white_pixels) & set(source.black_pixels)
    assert len(source.white_pixels) + len(source.black_pixels) == 4


def test_partial_mutation_preserves_partition_and_moves_toward_target():
    source = cell([[1] * 10] * 5 + [[0] * 10] * 5)
    target = cell([[0] * 10] * 5 + [[1] * 10] * 5)
    before = set(source.white_pixels)
    source.mutation(target, 0.1)
    after = set(source.white_pixels)
    assert len(after - before) == len(before - after) == 5
    assert not after & set(source.black_pixels)
    assert len(after | set(source.black_pixels)) == 100


def test_original_infinite_exact_match_score_is_preserved():
    model = ImmuneAlgorithm(2, 2, 0.8, 1)
    assert math.isinf(model.get_increment_value(1.0))


def test_no_votes_returns_unrecognized_instead_of_first_digit():
    model = ImmuneAlgorithm(2, 2, 0.9, 1)
    model.immune_system.update({"2": [cell([[1, 1], [0, 0]])],
                               "7": [cell([[0, 0], [1, 1]])]})
    model.counters.update({"2": 0, "7": 0})
    assert model.recognition_antigen(cell([[0, 0], [1, 0]])) == "unrecognized"
    assert model.get_counters() == {"2": 0, "7": 0}
    model.set_affinity_to_react(0.4)
    assert model.recognition_antigen(cell([[0, 0], [1, 0]])) == "7"
    assert model.recognition_antigen(cell([[0, 0], [0, 0]])) == "unrecognized"


def test_original_class_state_is_preserved():
    first, second = ImmuneAlgorithm(2, 2, 0.4, 1), ImmuneAlgorithm(2, 2, 0.4, 1)
    assert first.immune_system is second.immune_system


def test_symbol_evaluation_counts_unrecognized_at_index_ten(tmp_path):
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(tmp_path / "blank.png")
    model = ImmuneAlgorithm(2, 2, 0.4, 1)
    model.immune_system["0"] = [cell([[1, 0], [0, 0]])]
    model.counters["0"] = 0
    assert model.test_recognition_symbol(tmp_path, 1, "0") == [0] * 10 + [1]


def test_original_training_calls_every_same_class_cell(tmp_path, monkeypatch):
    image_dir = tmp_path / "2"
    image_dir.mkdir()
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(image_dir / "one.png")
    first, second = cell([[1, 0], [0, 0]]), cell([[0, 1], [0, 0]])
    model = ImmuneAlgorithm(2, 2, 0.8, 2)
    model.immune_system["2"] = [first, second]
    calls = []
    monkeypatch.setattr(Limphocit_B, "mutation", lambda self, antigen, power: calls.append((self.id, power)))
    model.training({"2": image_dir}, 1)
    assert calls == [(first.id, 0.1), (second.id, 0.1)]
