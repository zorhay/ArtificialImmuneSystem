import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from ImageRecognition.src import dataset
from ImageRecognition.src.dataset import read_idx
from ImageRecognition.src.experiment import evaluate, load_model, save_model, train
from ImageRecognition.src.immune_algorithm import ImmuneAlgorithm
from ImageRecognition.src.limphocit import Limphocit_B


def test_idx_reader_validates_header_and_payload(tmp_path):
    path = tmp_path / "labels.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 3) + bytes([3, 5, 7]))
    assert read_idx(path).tolist() == [3, 5, 7]
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 4) + bytes([3, 5, 7]))
    with pytest.raises(ValueError, match="payload"):
        read_idx(path)


def test_export_preserves_original_pixels(tmp_path, monkeypatch):
    images = np.arange(10 * 28 * 28, dtype=np.int64).astype(np.uint8).reshape(10, 28, 28)
    labels = np.arange(10)
    monkeypatch.setattr(dataset, "load_mnist", lambda *args: (images, labels))
    dataset.prepare_mnist(tmp_path)
    for split in ("train", "test"):
        with Image.open(tmp_path / "images" / split / "7" / "00007.png") as image:
            np.testing.assert_array_equal(np.asarray(image), images[7])


def test_cli_predict_and_legacy_entrypoint_reload_original_state(tmp_path):
    pixels = np.zeros((28, 28), dtype=np.uint8)
    pixels[3:25, 12:16] = 255
    image_path, model_path = tmp_path / "one.png", tmp_path / "model.pkl"
    Image.fromarray(pixels).save(image_path)
    Limphocit_B.init()
    model = ImmuneAlgorithm(28, 28, 0.8, 1)
    model.immune_system.clear()
    model.counters.clear()
    model.immune_system["1"] = [Limphocit_B.get_from_image(image_path)]
    model.counters["1"] = 0
    save_model(model, model_path, {})
    root = Path(__file__).resolve().parents[1]
    for entrypoint in (["-m", "ImageRecognition", "predict"], [str(root / "ImageRecognition/src/recognition.py")]):
        completed = subprocess.run([sys.executable, *entrypoint, str(image_path), "--model", str(model_path)],
                                   check=True, capture_output=True, text=True, cwd=root)
        result = json.loads(completed.stdout)
        assert result["prediction"] == 1
        assert result["reaction_affinity"] == 0.4
    # The override is for inference; the serialized model itself stays unchanged.
    restored, _ = load_model(model_path)
    assert restored.main_affinity_to_react == 0.8
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(image_path)
    blank = subprocess.run([sys.executable, "-m", "ImageRecognition", "predict", str(image_path),
                            "--model", str(model_path)], check=True, capture_output=True, text=True, cwd=root)
    assert json.loads(blank.stdout)["prediction"] == "unrecognized"


def test_original_train_save_load_evaluate_pipeline(tmp_path):
    for split in ("train", "test"):
        for digit in range(10):
            path = tmp_path / "images" / split / str(digit)
            path.mkdir(parents=True)
            pixels = np.zeros((28, 28), dtype=np.uint8)
            pixels[3:25, 2 * digit + 2] = 255
            Image.fromarray(pixels).save(path / "one.png")
    model_path = train(tmp_path, tmp_path / "output", templates=1, train_per_class=1)
    results = evaluate(tmp_path, model_path, tmp_path / "report", test_per_class=1)
    assert results["correct"] == results["test_examples"] == 10
    assert results["evaluation_affinity"] == results["saved_model_affinity"] == 0.4
    assert results["unrecognized"] == 0
    assert (tmp_path / "report" / "predictions.csv").exists()
    # Rejected images must remain in the accuracy denominator and in the CSV.
    for path in (tmp_path / "images" / "test").rglob("*.png"):
        Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(path)
    rejected = evaluate(tmp_path, model_path, tmp_path / "rejected", test_per_class=1)
    assert rejected["unrecognized"] == rejected["test_examples"] == 10
    assert rejected["accuracy"] == rejected["coverage"] == 0
    assert rejected["recognized_accuracy"] is None


def test_cli_missing_model_reports_error(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "ImageRecognition", "predict", "missing.png",
                             "--model", str(tmp_path / "missing.pkl")], capture_output=True, text=True, cwd=root)
    assert result.returncode == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_download_verifies_and_reuses_cache(tmp_path, monkeypatch):
    payload = b"fixture archive"
    monkeypatch.setattr(dataset, "FILES", {"fixture.gz": hashlib.md5(payload).hexdigest()})
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", open_url)
    dataset.download_mnist(tmp_path)
    dataset.download_mnist(tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "fixture.gz").read_bytes() == payload


def test_bad_download_never_replaces_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "FILES", {"fixture.gz": hashlib.md5(b"expected").hexdigest()})
    monkeypatch.setattr(dataset.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad download"))
    path = tmp_path / "fixture.gz"
    path.write_bytes(b"existing cache")
    with pytest.raises(ValueError, match="Checksum"):
        dataset.download_mnist(tmp_path)
    assert path.read_bytes() == b"existing cache"
    assert not (tmp_path / "fixture.gz.part").exists()
