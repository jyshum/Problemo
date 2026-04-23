import pytest
import json
import tempfile
from pathlib import Path


def make_labeled_jsonl(tmp_path: Path, n_per_class: int = 30) -> Path:
    """Write a minimal labeled.jsonl with balanced classes."""
    labels = ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"]
    texts = {
        "WORKFLOW_PAIN": "I manually do this repetitive task every week and it takes hours to complete",
        "TOOL_REQUEST": "Is there a tool that can automatically handle this task for me without manual work",
        "PRODUCT_COMPLAINT": "This app is completely broken and missing basic features that should be there",
        "NOISE": "Just a random post about life today nothing to see here at all",
    }
    path = tmp_path / "labeled.jsonl"
    with open(path, "w") as f:
        for i in range(n_per_class):
            for label in labels:
                f.write(json.dumps({"id": f"{label}_{i}", "text": texts[label], "label": label}) + "\n")
    return path


def test_load_labeled_data_returns_texts_and_labels(tmp_path):
    from classifier.train_setfit import load_labeled_data

    path = make_labeled_jsonl(tmp_path, n_per_class=5)
    texts, labels = load_labeled_data(path)
    assert len(texts) == 20
    assert len(labels) == 20
    assert "WORKFLOW_PAIN" in labels


def test_make_stratified_split_returns_four_lists(tmp_path):
    from classifier.train_setfit import make_stratified_split

    texts = ["text number " + str(i) + " with some extra words" for i in range(100)]
    labels = ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"] * 25

    test_ids_path = tmp_path / "test_ids.txt"
    train_t, train_l, test_t, test_l = make_stratified_split(
        texts, labels, test_size=0.2, seed=42, test_ids_path=test_ids_path
    )
    assert len(train_t) == len(train_l)
    assert len(test_t) == len(test_l)
    assert len(train_t) + len(test_t) == 100
    assert test_ids_path.exists()


def test_make_stratified_split_locks_test_ids(tmp_path):
    from classifier.train_setfit import make_stratified_split

    texts = ["text number " + str(i) + " with some extra words" for i in range(100)]
    labels = ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"] * 25
    test_ids_path = tmp_path / "test_ids.txt"

    # First run — writes test_ids.txt
    _, _, test_t1, _ = make_stratified_split(texts, labels, test_size=0.2, seed=42, test_ids_path=test_ids_path)

    # Second run — must return same test set regardless of different seed
    _, _, test_t2, _ = make_stratified_split(texts, labels, test_size=0.2, seed=99, test_ids_path=test_ids_path)

    assert set(test_t1) == set(test_t2)
