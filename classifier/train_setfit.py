"""Train real SetFit classifiers on labeled.jsonl and write eval reports.

Uses SetFit contrastive fine-tuning (not frozen embeddings) to hit macro F1 >= 0.75.
Trains two backbones sequentially on the same locked test set.

Writes model artifacts to classifier/models/ and eval reports
to classifier/eval_report/.

Usage: PYTHONPATH=. python classifier/train_setfit.py
"""

import json
import gc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from config import Config

LABELS = ["WORKFLOW_PAIN", "TOOL_REQUEST", "NOISE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

LABELED_PATH = Path("classifier/data/labeled.jsonl")
TEST_IDS_PATH = Path("classifier/data/test_ids.txt")
MODELS_DIR = Path("classifier/models")
EVAL_REPORT_DIR = Path("classifier/eval_report")


def load_labeled_data(path: Path) -> tuple[list[str], list[str]]:
    """Load labeled.jsonl → (texts, labels), skipping unknown labels."""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["label"] not in LABELS:
                continue
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def make_stratified_split(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    test_ids_path: Path = TEST_IDS_PATH,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified 80/20 split. Locks test indices to test_ids_path on first run."""
    indices = list(range(len(texts)))

    if test_ids_path.exists():
        test_indices = [int(i) for i in test_ids_path.read_text().splitlines() if i.strip()]
        train_indices = [i for i in indices if i not in set(test_indices)]
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_indices, test_indices = next(sss.split(indices, labels))
        train_indices = list(train_indices)
        test_indices = list(test_indices)
        test_ids_path.parent.mkdir(parents=True, exist_ok=True)
        test_ids_path.write_text("\n".join(str(i) for i in test_indices))
        print(f"Locked test set written to {test_ids_path} ({len(test_indices)} posts)")

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    return train_texts, train_labels, test_texts, test_labels


def train_and_evaluate(
    backbone: str,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    model_short_name: str,
) -> dict:
    """Fine-tune a real SetFit model via contrastive learning and evaluate."""
    print(f"\n=== Training {model_short_name} ({backbone}) ===")

    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": [LABEL2ID[l] for l in train_labels],
    })
    eval_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": [LABEL2ID[l] for l in test_labels],
    })

    model = SetFitModel.from_pretrained(backbone, labels=LABELS)

    args = TrainingArguments(
        num_epochs=3,
        num_iterations=20,
        batch_size=8,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
    )

    trainer.train()

    pred_ids = model.predict(test_texts)
    # SetFit returns label strings directly or integer IDs depending on version
    pred_labels = [
        i if isinstance(i, str) else ID2LABEL[int(i)]
        for i in pred_ids
    ]

    model_path = MODELS_DIR / f"setfit-{model_short_name}"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_path))
    print(f"  Model saved to {model_path}")

    # Free GPU memory before next model
    del model, trainer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "backbone": backbone,
        "model_short_name": model_short_name,
        "pred_labels": pred_labels,
        "true_labels": test_labels,
    }


def write_eval_reports(results: list[dict], test_texts: list[str]) -> None:
    """Write classification_report.txt, confusion matrices, and error_analysis.md."""
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = []
    for result in results:
        name = result["model_short_name"]
        pred = result["pred_labels"]
        true = result["true_labels"]

        report_lines.append(f"\n=== {name} (SetFit fine-tuned) ===\n")
        report_lines.append(classification_report(true, pred, labels=LABELS))

        cm = confusion_matrix(true, pred, labels=LABELS)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
        disp.plot(ax=ax, xticks_rotation=30)
        ax.set_title(f"Confusion Matrix - {name} (SetFit fine-tuned)")
        fig.tight_layout()
        fig.savefig(EVAL_REPORT_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix for {name}")

    (EVAL_REPORT_DIR / "classification_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Saved {EVAL_REPORT_DIR / 'classification_report.txt'}")

    # Error analysis on primary model
    primary = results[0]
    lines = [f"# Error Analysis\n\nModel: {primary['model_short_name']} (SetFit fine-tuned)\n\n"]

    for label in LABELS:
        fps = [
            (test_texts[i], primary["true_labels"][i])
            for i, (t, p) in enumerate(zip(primary["true_labels"], primary["pred_labels"]))
            if p == label and t != label
        ][:10]
        fns = [
            (test_texts[i], primary["pred_labels"][i])
            for i, (t, p) in enumerate(zip(primary["true_labels"], primary["pred_labels"]))
            if t == label and p != label
        ][:10]

        lines.append(f"## {label}\n")
        lines.append(f"### False Positives (predicted {label}, actually something else)\n")
        for text, true in fps:
            lines.append(f"- **True: {true}** | {text[:200]}\n")
        lines.append(f"\n### False Negatives (actually {label}, predicted something else)\n")
        for text, pred in fns:
            lines.append(f"- **Predicted: {pred}** | {text[:200]}\n")
        lines.append("\n")

    (EVAL_REPORT_DIR / "error_analysis.md").write_text("".join(lines), encoding="utf-8")
    print(f"Saved {EVAL_REPORT_DIR / 'error_analysis.md'}")

    kappa_path = EVAL_REPORT_DIR / "kappa_score.txt"
    if not kappa_path.exists():
        kappa_path.write_text(
            "Cohen's Kappa: not yet computed.\n"
            "Run: PYTHONPATH=. python classifier/evaluate.py --kappa\n",
            encoding="utf-8",
        )


def main():
    config = Config()

    texts, labels = load_labeled_data(LABELED_PATH)
    print(f"Loaded {len(texts)} labeled posts")

    train_texts, train_labels, test_texts, test_labels = make_stratified_split(texts, labels)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    backbones = [
        (config.setfit_backbone, "mpnet"),
        ("sentence-transformers/all-mpnet-base-v2", "mpnet-v2"),
    ]

    results = []
    for backbone, short_name in backbones:
        model_path = MODELS_DIR / f"setfit-{short_name}"
        if model_path.exists():
            print(f"\n=== Skipping {short_name} (already trained, loading saved model) ===")
            from setfit import SetFitModel
            model = SetFitModel.from_pretrained(str(model_path))
            pred_ids = model.predict(test_texts)
            pred_labels = [
                i if isinstance(i, str) else ID2LABEL[int(i)]
                for i in pred_ids
            ]
            results.append({
                "backbone": backbone,
                "model_short_name": short_name,
                "pred_labels": pred_labels,
                "true_labels": test_labels,
            })
            del model
        else:
            result = train_and_evaluate(
                backbone=backbone,
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                model_short_name=short_name,
            )
            results.append(result)

    write_eval_reports(results, test_texts)
    print("\nTraining complete. Eval reports written to classifier/eval_report/")


if __name__ == "__main__":
    main()
