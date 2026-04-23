"""Compute Cohen's Kappa between human labels and model predictions.

Loads the locked test set, runs the classifier on test texts,
and compares against ground-truth labels.

Usage: PYTHONPATH=. .venv/bin/python classifier/evaluate.py
"""

import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from classifier.classifier_interface import Classifier
from classifier.train_setfit import (
    LABELS, LABELED_PATH, TEST_IDS_PATH, EVAL_REPORT_DIR,
    load_labeled_data, make_stratified_split,
)


def compute_kappa(model_name: str = "setfit-mpnet") -> float:
    """Load test set, predict, compute Cohen's Kappa vs ground truth."""
    texts, labels = load_labeled_data(LABELED_PATH)
    _, _, test_texts, test_labels = make_stratified_split(texts, labels)

    clf = Classifier(model_name=model_name)
    predictions = clf.predict(test_texts)
    pred_labels = [label for label, _ in predictions]

    kappa = cohen_kappa_score(test_labels, pred_labels, labels=LABELS)
    return kappa


def main():
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    for model_name in ["setfit-mpnet", "setfit-minilm"]:
        kappa = compute_kappa(model_name)
        short = model_name.replace("setfit-", "")
        line = f"{short}: Cohen's Kappa = {kappa:.4f}"
        print(line)
        lines.append(line)

    report_path = EVAL_REPORT_DIR / "kappa_score.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
