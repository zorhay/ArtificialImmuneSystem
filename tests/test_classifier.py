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


def test_original_affinity_and_blank_failure_are_preserved():
    first, second = cell([[1, 1], [1, 0]]), cell([[0, 1], [1, 0]])
    assert first.affinity(second) == pytest.approx(2 / 3)
    with pytest.raises(ZeroDivisionError):
        cell([[0, 0], [0, 0]]).affinity(second)


def test_original_mutation_behavior_runs_on_current_python():
    source, target = cell([[1, 1], [0, 0]]), cell([[0, 0], [1, 1]])
    source.mutation(target, 1)
    # Deliberately preserve the original bug, rather than the previous "fix".
    assert set(source.white_pixels) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(source.white_pixels) & set(source.black_pixels)


def test_original_infinite_exact_match_score_is_preserved():
    model = ImmuneAlgorithm(2, 2, 0.8, 1)
    assert math.isinf(model.get_increment_value(1.0))


def test_original_zero_vote_tie_is_preserved():
    model = ImmuneAlgorithm(2, 2, 0.9, 1)
    model.immune_system.update({"2": [cell([[1, 1], [0, 0]])],
                               "7": [cell([[0, 0], [1, 1]])]})
    model.counters.update({"2": 0, "7": 0})
    assert model.recognition_antigen(cell([[0, 0], [1, 0]])) == "2"
    assert model.get_counters() == {"2": 0, "7": 0}


def test_original_class_state_is_preserved():
    first, second = ImmuneAlgorithm(2, 2, 0.4, 1), ImmuneAlgorithm(2, 2, 0.4, 1)
    assert first.immune_system is second.immune_system


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
