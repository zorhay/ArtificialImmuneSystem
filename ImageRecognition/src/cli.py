"""Portable commands around the original folder-based immune algorithm."""

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_DATA_DIR, ROOT, download_mnist, prepare_mnist
from .experiment import evaluate, folders, load_model, train
from .immune_algorithm import DEFAULT_AFFINITY, UNRECOGNIZED, ImmuneAlgorithm

DEFAULT_OUTPUT = ROOT / "artifacts" / "original"
DEFAULT_REPORT = ROOT / "reports" / "original"


def training_options(parser):
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--templates", type=int, default=21, help="Original recognition.py default: 21 per digit")
    parser.add_argument("--affinity", type=float, default=DEFAULT_AFFINITY, help="Reaction threshold (default: 0.4)")
    parser.add_argument("--train-per-class", type=int, default=100, help="Original training() set_size argument")
    parser.add_argument("--seed", type=int, default=42)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    training = commands.add_parser("train", help="Call the original training() method once")
    training_options(training)
    run = commands.add_parser("run", help="Download, export images, train, and evaluate the original algorithm")
    training_options(run)
    run.add_argument("--test-per-class", type=int, default=None, help="Omit to evaluate the full test set")
    run.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluation.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.pkl")
    evaluation.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    evaluation.add_argument("--test-per-class", type=int, default=None)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--affinity", type=float, default=DEFAULT_AFFINITY,
                            help="Evaluation threshold; overrides saved model setting (default: 0.4)")
    prediction = commands.add_parser("predict")
    prediction.add_argument("image", type=Path)
    prediction.add_argument("--model", type=Path, help="Load a trained model; otherwise use original reference-image initialization")
    prediction.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prediction.add_argument("--affinity", type=float, default=DEFAULT_AFFINITY,
                            help="Prediction threshold; overrides saved model setting (default: 0.4)")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "affinity") and not 0 <= args.affinity <= 1:
            raise ValueError("Affinity must be between 0 and 1")
        if args.command in ("download", "prepare", "run"):
            download_mnist(args.data_dir)
        if args.command in ("prepare", "run"):
            prepare_mnist(args.data_dir)
        if args.command in ("train", "run"):
            model_path = train(args.data_dir, args.output_dir, args.templates, args.affinity,
                               args.train_per_class, args.seed)
            if args.command == "run":
                evaluate(args.data_dir, model_path, args.report_dir, args.test_per_class, args.seed, args.affinity)
        elif args.command == "evaluate":
            evaluate(args.data_dir, args.model, args.report_dir, args.test_per_class, args.seed, args.affinity)
        elif args.command == "predict":
            if args.model:
                model, _ = load_model(args.model)
            else:
                model = ImmuneAlgorithm(28, 28, args.affinity, 21)
                model.reload_limphocites(folders(args.data_dir, "train"))
            model.set_affinity_to_react(args.affinity)
            result = model.recognition_image(args.image)
            print(json.dumps({"prediction": result if result == UNRECOGNIZED else int(result),
                              "reaction_affinity": model.main_affinity_to_react}, indent=2))
    except (OSError, ValueError, ZeroDivisionError) as exc:
        parser.exit(1, f"Error: {exc}\n")
