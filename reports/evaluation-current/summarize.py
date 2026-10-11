"""Build the detailed report from the existing evaluator's CSVs; no inference changes.

Run from the repository root after both evaluation commands in provenance.json:
    .venv/bin/python reports/evaluation-current/summarize.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(stage, actual_training_examples):
    path = HERE / stage
    raw = json.loads((path / "metrics.json").read_text())
    with (path / "predictions.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 10000
    assert len({Path(row["file"]).stem for row in records}) == 10000
    matrix = [[0] * 10 for _ in range(10)]
    no_vote = [row for row in records if row["all_scores_zero"] == "True"]
    reacting = [row for row in records if row["all_scores_zero"] != "True"]
    for row in records:
        matrix[int(row["true_digit"])][int(row["prediction"])] += 1
    per_digit = []
    for digit in range(10):
        support = sum(matrix[digit])
        predicted = sum(row[digit] for row in matrix)
        tp = matrix[digit][digit]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_digit.append({"digit": digit, "support": support, "predicted": predicted,
                          "correct": tp, "precision": precision, "recall": recall, "f1": f1,
                          "no_reaction": sum(int(row["true_digit"]) == digit for row in no_vote)})
    correct = sum(matrix[d][d] for d in range(10))
    assert correct == raw["correct"]
    assert len(no_vote) == raw["all_scores_zero"]
    reactive_correct = sum(row["true_digit"] == row["prediction"] for row in reacting)
    zero_correct = sum(row["true_digit"] == row["prediction"] for row in no_vote)
    assert all(row["prediction"] == "0" for row in no_vote)
    summary = {
        "stage": stage, "actual_training_examples": actual_training_examples,
        "test_examples": len(records), "correct": correct, "errors": len(records) - correct,
        "accuracy": correct / len(records),
        "macro_precision": sum(d["precision"] for d in per_digit) / 10,
        "balanced_accuracy": sum(d["recall"] for d in per_digit) / 10,
        "macro_f1": sum(d["f1"] for d in per_digit) / 10,
        "weighted_f1": sum(d["f1"] * d["support"] for d in per_digit) / len(records),
        "no_reaction_count": len(no_vote), "no_reaction_correct": zero_correct,
        "reacting_count": len(reacting), "reacting_correct": reactive_correct,
        "reacting_accuracy": reactive_correct / len(reacting) if reacting else None,
        "evaluation_seconds": raw["evaluation_seconds"],
        "per_digit": per_digit, "confusion_matrix": matrix,
    }
    (path / "detailed_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, records


def percentage(value):
    return "n/a" if value is None else f"{value:.2%}"


def digit_table(result):
    lines = ["| Digit | Test images | Correct | Predicted as digit | Precision | Recall | F1 | No reaction |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in result["per_digit"]:
        lines.append(f"| {d['digit']} | {d['support']} | {d['correct']} | {d['predicted']} | "
                     f"{d['precision']:.2%} | {d['recall']:.2%} | {d['f1']:.4f} | {d['no_reaction']} |")
    return lines


def confusion_table(result):
    lines = ["Rows are true digits; columns are predicted digits.", "",
             "| True / predicted | " + " | ".join(map(str, range(10))) + " |",
             "|---|" + "---:|" * 10]
    for digit, row in enumerate(result["confusion_matrix"]):
        lines.append(f"| {digit} | " + " | ".join(map(str, row)) + " |")
    return lines


def main():
    provenance = json.loads((HERE / "provenance.json").read_text())
    diagnostics = json.loads((HERE / "template_diagnostics.json").read_text())
    trained, trained_records = summarize("trained", 20)
    initial, initial_records = summarize("initial", 0)
    before = {row["file"]: row for row in initial_records}
    improved = sum(before[r["file"]]["prediction"] != r["true_digit"] and r["prediction"] == r["true_digit"]
                   for r in trained_records)
    regressed = sum(before[r["file"]]["prediction"] == r["true_digit"] and r["prediction"] != r["true_digit"]
                    for r in trained_records)
    same = sum(before[r["file"]]["prediction"] == r["prediction"] for r in trained_records)
    metadata = json.loads((HERE / "trained" / "metrics.json").read_text())["training"]
    combined = {"current": trained, "initial": initial,
                "comparison": {"accuracy_delta_percentage_points": 100 * (trained["accuracy"] - initial["accuracy"]),
                               "errors_corrected": improved, "correct_predictions_lost": regressed,
                               "unchanged_predictions": same}, "provenance": provenance}
    (HERE / "summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    added = sum(d["pixels_added"] for d in diagnostics["per_digit_template_state"])
    removed = sum(d["pixels_removed"] for d in diagnostics["per_digit_template_state"])
    overlap = sum(d["templates_with_overlapping_white_black_pixels"] for d in diagnostics["per_digit_template_state"])
    lines = ["# Current implementation: full MNIST evaluation", "",
             f"Run date: {provenance['timestamp']} ({provenance['timezone']}).", "",
             f"**Current saved-model accuracy: {trained['accuracy']:.2%} "
             f"({trained['correct']:,} / 10,000 correct).**", "",
             "## 1. What was run", "",
             f"Source commit: `{provenance['commit']}`. The committed implementation was used without "
             "changing its training, mutation, affinity, scoring, thresholds, or prediction behavior.", "",
             "Two existing model files were evaluated, each on the complete official MNIST test set:", "",
             "1. `artifacts/original/model.pkl`: the current saved model, initialized from 21 images per digit "
             "and then trained on 2 images per digit, one pass (20 training examples, 420 cell-mutation calls).",
             "2. `artifacts/original/initial.pkl`: the same 210 reference templates before any training updates. "
             "This is the reference-only comparison corresponding to the original recognition-script workflow.", "",
             "**No retraining or tuning occurred during this evaluation. The current saved model is the "
             "earlier smoke-run model, not a model trained on all 60,000 MNIST training images.** "
             "The 20 update images are a subset of the 210 initialization images. Neither uses test images.", "",
             "Commands executed from the repository root:", "", "```sh", *provenance["commands"], "```", "",
             "The `--test-per-class` option was omitted so every test image was evaluated. Both processes "
             "call the original `recognition_image` → `recognition_antigen` → `affinity` methods. "
             "The second model ran in a separate process because the original classifier uses shared class state.", "",
             "The downloaded archives passed their expected checksums. All 10,000 exported test images "
             "were verified pixel-for-pixel and label-for-label against the original IDX files. All 210 "
             "initial templates were also checked against the training IDX images. No test image is blank "
             "after thresholding at 200. Prediction logs contain exactly 10,000 distinct test indices per run.", "",
             "## 2. Detailed accuracy report", "",
             "| Measurement | Current saved model | Initial templates |", "|---|---:|---:|",
             f"| Accuracy | {trained['accuracy']:.2%} | {initial['accuracy']:.2%} |",
             f"| Correct / evaluated | {trained['correct']:,} / 10,000 | {initial['correct']:,} / 10,000 |",
             f"| Errors | {trained['errors']:,} | {initial['errors']:,} |",
             f"| Balanced accuracy (macro recall) | {trained['balanced_accuracy']:.2%} | {initial['balanced_accuracy']:.2%} |",
             f"| Macro precision | {trained['macro_precision']:.2%} | {initial['macro_precision']:.2%} |",
             f"| Macro F1 | {trained['macro_f1']:.4f} | {initial['macro_f1']:.4f} |",
             f"| Support-weighted F1 | {trained['weighted_f1']:.4f} | {initial['weighted_f1']:.4f} |",
             f"| All-zero vote cases | {trained['no_reaction_count']:,} | {initial['no_reaction_count']:,} |",
             f"| Images with at least one reacting template | {trained['reacting_count']:,} | {initial['reacting_count']:,} |",
             f"| Accuracy on reacting images only | {percentage(trained['reacting_accuracy'])} | {percentage(initial['reacting_accuracy'])} |",
             f"| Evaluation elapsed time | {trained['evaluation_seconds']:.2f} s | {initial['evaluation_seconds']:.2f} s |", "",
             provenance["evaluation_note"], "",
             f"The 20-example training pass changed overall accuracy by "
             f"{100 * (trained['accuracy'] - initial['accuracy']):+.2f} percentage points. "
             f"It corrected {improved} errors and lost {regressed} previously correct predictions; "
             f"{same:,} predictions were unchanged.", "",
             "For context, always predicting 0 scores 9.80% on this test set; always predicting its most "
             "frequent class (1) scores 11.35%. These are descriptive constant-output baselines, not tuned models.", "",
             "### Current saved model: per-digit results", "", *digit_table(trained), "",
             "Precision = correct predictions of a digit / all predictions of that digit. "
             "Recall = correct predictions of a digit / all test images of that digit. "
             "F1 is their harmonic mean. Undefined precision/F1 for unpredicted classes is reported as zero. "
             "No-reaction cases are included in every metric, using the classifier's actual output.", "",
             "### Current saved model: confusion matrix", "", *confusion_table(trained), "",
             "### Initial templates: per-digit results", "", *digit_table(initial), "",
             "### Initial templates: confusion matrix", "", *confusion_table(initial), "",
             "### Observed failure behavior", "",
             f"For the current model, **{trained['no_reaction_count']:,}/10,000 "
             f"({trained['no_reaction_count']/10000:.2%}) inputs produced zero votes in every class**. "
             "In that case, the original `max(counters, key=counters.get)` returns the first dictionary key, "
             "which is `0`. These are ordinary predictions in the original code; there is no rejection or "
             "nearest-template fallback. The CSV explicitly flags each occurrence.", "",
             f"Of the current model's {trained['correct']:,} correct answers, "
             f"{trained['no_reaction_correct']:,} came from zero-vote defaults matching a true zero, "
             f"and {trained['reacting_correct']:,} came from images with at least one reacting template. "
             "High recall for digit 0 therefore should not be interpreted as reliable recognition of zeros.", "",
             f"The saved post-training templates contain {added:,} added white-pixel coordinates and "
             f"{removed:,} removed white-pixel coordinates relative to initialization. "
             f"{overlap}/210 cells contain coordinates in both their white and black lists. "
             "This confirms the original mutation behavior remains present. These observations do not "
             "isolate its causal effect from the small training set or the reaction threshold.", "",
             "## 3. Configurations", "",
             "| Setting | Value |", "|---|---|",
             "| Dataset | MNIST, official train/test partition |",
             "| Available training set | 60,000 images |",
             "| Initialization | First 21 files per digit in original `os.listdir` order; 210 cells total |",
             "| Actual current-model training | First 2 files per digit in the same order; 20 images, one pass |",
             "| Initial-template comparison training | None |",
             "| Test set | All 10,000 official test images; no sampling |",
             "| Image representation | 28×28 grayscale, coordinates of white and black pixels |",
             "| Binarization | Pixel > 200 becomes 255; otherwise 0 |",
             "| Reaction affinity | ≥ 0.8 |",
             "| Affinity | Shared white pixels / maximum of the two white-pixel counts |",
             "| Vote increment | a × exp(1 / (1 − a)); original infinity handling |",
             "| Mutation | Original all-same-class-cells update, power 0.1 |",
             "| Random seed for saved training | 42 |",
             "| Evaluation seed argument | Default 42; unused when evaluating all files |",
             "| Class insertion/tie order | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |",
             "| Preprocessing beyond thresholding | None; no cropping, normalization, inversion, or augmentation |",
             "| Validation/checkpoint selection | None |",
             f"| Python / NumPy / Pillow | {provenance['python']} / {provenance['numpy']} / {provenance['pillow']} |",
             f"| Platform | {provenance['platform']} |", "",
             "The legacy `main.py` uses 20 cells and affinity 0.4 only to load a network. "
             "That is a different configuration and was not evaluated here. This report uses the saved "
             "model's settings, matching the original recognition script's 21-cell/0.8 configuration.", "",
             "Directory order is filesystem-dependent. The full lists of source filenames used for "
             "initialization and training are recorded in the per-run metrics JSON, so seed 42 alone "
             "should not be treated as a cross-filesystem reproduction guarantee.", "",
             "### Model identity", "", "| Model file | SHA-256 |", "|---|---|"]
    for path, info in provenance["models"].items():
        lines.append(f"| `{path}` | `{info['sha256']}` |")
    lines.extend(["", "## Artifacts and interpretation", "",
                  "- [Summary metrics](summary.json)",
                  "- [Current-model predictions, 10,000 rows](trained/predictions.csv)",
                  "- [Initial-template predictions, 10,000 rows](initial/predictions.csv)",
                  "- [Current-model detailed metrics](trained/detailed_metrics.json)",
                  "- [Initial-template detailed metrics](initial/detailed_metrics.json)",
                  "- [Environment, checksums, and commands](provenance.json)",
                  "- [Template-state diagnostics](template_diagnostics.json)", "",
                  "These measurements describe these two saved model states on MNIST. They do not establish "
                  "the best accuracy achievable by the original algorithm, accuracy after full-dataset "
                  "training, or performance with other thresholds/template counts. Only one initialization "
                  "and one mutation seed are represented. No settings were selected using these test results.", "",
                  "The initial model's raw wrapper metadata records the planned 20-example follow-up "
                  "training run. That file was saved before `training()` was called; its actual mutation "
                  "training count is zero, as stated throughout this detailed report.", ""])
    (HERE / "report.md").write_text("\n".join(lines))
    initial_report = HERE / "initial" / "report.md"
    note = ("**Model-state clarification:** This file is the pre-training snapshot (zero training updates). "
            "The wrapper's metadata below describes the planned subsequent 20-example training run, "
            "not training already applied to this snapshot. See [the detailed comparison](../report.md).\n\n")
    text = initial_report.read_text()
    if not text.startswith("**Model-state clarification:"):
        initial_report.write_text(note + text)
    print(json.dumps({"current_accuracy": trained["accuracy"], "initial_accuracy": initial["accuracy"],
                      "current_correct": trained["correct"], "no_reaction": trained["no_reaction_count"],
                      "per_digit": trained["per_digit"]}, indent=2))


if __name__ == "__main__":
    main()
