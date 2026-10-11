"""Run orchestration around the original, unmodified classifier methods."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import random
import time

from .immune_algorithm import ImmuneAlgorithm
from .limphocit import Limphocit_B


def folders(data_dir, split):
    result = {str(digit): Path(data_dir) / "images" / split / str(digit) for digit in range(10)}
    if not all(path.is_dir() for path in result.values()):
        raise FileNotFoundError("MNIST image folders are missing. Run 'python -m ImageRecognition prepare'.")
    return result


def save_model(model, path, metadata):
    """Persist class-level state that the original pickle method omits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump({"format": "original-ais-run-v1", "model": model,
                     "immune_system": model.immune_system, "counters": model.counters,
                     "limfoset": Limphocit_B.limfoset, "id_list": Limphocit_B.id_list,
                     "metadata": metadata}, stream, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    # These are locally generated pickle files; load only trusted model files.
    with Path(path).open("rb") as stream:
        saved = pickle.load(stream)
    if not isinstance(saved, dict) or saved.get("format") != "original-ais-run-v1":
        raise ValueError("Expected a model saved by the original-algorithm run wrapper")
    ImmuneAlgorithm.immune_system.clear()
    ImmuneAlgorithm.immune_system.update(saved["immune_system"])
    ImmuneAlgorithm.counters.clear()
    ImmuneAlgorithm.counters.update(saved["counters"])
    Limphocit_B.limfoset = saved["limfoset"]
    Limphocit_B.id_list = saved["id_list"]
    return saved["model"], saved["metadata"]


def train(data_dir, output_dir, templates=21, affinity=0.8, train_per_class=100, seed=42):
    if templates < 1 or train_per_class < 1 or not 0 <= affinity <= 1:
        raise ValueError("Counts must be positive and affinity must be between 0 and 1")
    training = folders(data_dir, "train")
    files = {digit: os.listdir(path) for digit, path in training.items()}
    if any(len(names) < max(templates, train_per_class) for names in files.values()):
        raise ValueError("Requested more templates/training examples than a digit folder contains")
    random.seed(seed)
    model = ImmuneAlgorithm(28, 28, affinity, templates)
    model.reload_limphocites(training)
    metadata = {
        "algorithm": "original repository implementation",
        "templates_per_digit": templates, "reaction_affinity": affinity,
        "pixel_threshold": 200, "mutation_power": 0.1, "seed": seed,
        "training_per_class": train_per_class, "training_examples": 10 * train_per_class,
        "reference_files": {k: v[:templates] for k, v in files.items()},
        "training_files": {k: v[:train_per_class] for k, v in files.items()},
        "note": "Original os.listdir ordering and all-cells mutation loop are preserved. No tuning or checkpoint selection.",
    }
    output_dir = Path(output_dir)
    save_model(model, output_dir / "initial.pkl", metadata)
    print(f"Original training: {10 * train_per_class} images, {10 * templates} cells, one pass", flush=True)
    start = time.perf_counter()
    model.training(training, train_per_class)
    metadata["training_seconds"] = time.perf_counter() - start
    model_path = output_dir / "model.pkl"
    save_model(model, model_path, metadata)
    (output_dir / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {model_path} after {metadata['training_seconds']:.2f}s", flush=True)
    return model_path


def evaluate(data_dir, model_path, report_dir, test_per_class=None, seed=42):
    if test_per_class is not None and test_per_class < 1:
        raise ValueError("test_per_class must be positive")
    model, metadata = load_model(model_path)
    testing = folders(data_dir, "test")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows, per_digit = [], {}
    start = time.perf_counter()
    for digit, path in testing.items():
        names = os.listdir(path)
        if test_per_class is not None:
            if test_per_class > len(names):
                raise ValueError(f"Digit {digit} has only {len(names)} test images")
            names = rng.sample(names, test_per_class)
        correct = 0
        for name in names:
            prediction = model.recognition_image(path / name)
            correct += prediction == digit
            rows.append({"file": str(path / name), "true_digit": digit, "prediction": prediction,
                         "all_scores_zero": all(score == 0 for score in model.get_counters().values())})
        per_digit[digit] = {"correct": correct, "total": len(names), "accuracy": correct / len(names)}
        print(f"Digit {digit}: {correct}/{len(names)}", flush=True)
    results = {"algorithm": metadata["algorithm"], "training": metadata,
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
               "evaluation_seconds": time.perf_counter() - start,
               "test_examples": len(rows), "correct": sum(r["correct"] for r in per_digit.values()),
               "test_per_class": test_per_class, "evaluation_seed": seed, "per_digit": per_digit,
               "all_scores_zero": sum(row["all_scores_zero"] for row in rows)}
    results["accuracy"] = results["correct"] / results["test_examples"]
    with (report_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Original-algorithm run report", "",
             f"Correct: {results['correct']} / {len(rows)} ({results['accuracy']:.2%}).", "",
             f"Training examples: {metadata['training_examples']}; templates per digit: {model.network_power}; "
             f"reaction threshold: {model.main_affinity_to_react}; pixel threshold: 200.", "",
             "This runs the original training, mutation, affinity, and voting methods. "
             "It includes the original mutation bug, zero-score ties, and infinite exact-match scores. "
             "Only imports, data paths, and Python/NumPy compatibility were adjusted in the core files.", "",
             "MNIST source: [CVDF mirror](https://github.com/cvdfoundation/mnist). "
             "Reference and training images use the official training set; evaluation uses the official test set. "
             "Images are exported without normalization or pixel changes. No validation tuning is performed.", "",
             f"All-zero score cases: {results['all_scores_zero']}. The original classifier resolves these "
             "by dictionary order; no nearest-template fallback has been added.", "",
             "This is a sampled execution check, not a full-test benchmark." if test_per_class is not None
             else "All 10,000 official test images were evaluated.", "",
             "| Digit | Correct | Evaluated | Accuracy |", "|---|---:|---:|---:|"]
    for digit, row in per_digit.items():
        lines.append(f"| {digit} | {row['correct']} | {row['total']} | {row['accuracy']:.2%} |")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Result: {results['accuracy']:.2%} on {len(rows)} images. Report: {report_dir / 'report.md'}", flush=True)
    return results
