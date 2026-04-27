"""Quick eval of saved SetFit mpnet model on locked test set."""
import json
from pathlib import Path
from sklearn.metrics import classification_report, cohen_kappa_score
from setfit import SetFitModel

LABELS = ["WORKFLOW_PAIN", "TOOL_REQUEST", "NOISE"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

texts, labels = [], []
with open("classifier/data/labeled.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row["label"] in LABELS:
            texts.append(row["text"])
            labels.append(row["label"])

test_ids = [int(i) for i in Path("classifier/data/test_ids.txt").read_text().splitlines() if i.strip()]
test_texts = [texts[i] for i in test_ids]
test_labels = [labels[i] for i in test_ids]

print("Loading saved mpnet model...")
model = SetFitModel.from_pretrained("classifier/models/setfit-mpnet")
preds = model.predict(test_texts)
pred_labels = [i if isinstance(i, str) else ID2LABEL[int(i)] for i in preds]

print("\n=== SetFit mpnet (contrastive fine-tuned) ===\n")
print(classification_report(test_labels, pred_labels, labels=LABELS))
