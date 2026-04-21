# Phase 2: Labeling + SetFit Classifier — Design Spec

**Status:** Approved
**Date:** 2026-04-21
**Builds on:** Phase 1 (11,441 posts in `data/posts.db`)

---

## Goal

Train a SetFit classifier that predicts one of four classes for every post in the DB. Write confident predictions (≥0.70) back to `predicted_class` + `confidence` columns. Posts below threshold stay NULL. Produce a full eval report as a portfolio artifact.

---

## Classes

| Class | Definition | Signal phrases |
|-------|-----------|----------------|
| `WORKFLOW_PAIN` | Person describes a repetitive task done manually/inefficiently with no satisfying solution | "every week I", "I manually", "takes hours", "still using spreadsheets", "no tool for", "wish there was a way to" |
| `TOOL_REQUEST` | Person explicitly asks if a tool exists or requests one be built | "is there an app that", "someone should build", "does a tool exist for" |
| `PRODUCT_COMPLAINT` | Person unhappy with a specific named existing product | App/product name + "broken", "missing", "wish it had" |
| `NOISE` | General venting, off-topic, career advice, not actionable | — |

Only `WORKFLOW_PAIN` and `TOOL_REQUEST` pass to Phase 3 clustering.

---

## Architecture

```
DB (11,441 posts)
      │
      ▼
scripts/sample_posts.py          ← stratified sample → sampled.jsonl
      │
      ▼
scripts/label_posts.py           ← CLI tool, you label 180 posts
      │
      ▼
scripts/label_with_claude.py     ← Claude Sonnet labels 300 more
      │
      ▼
classifier/data/labeled.jsonl    ← 480 labeled posts merged
      │
      ▼
classifier/train_setfit.py       ← trains mpnet + modernbert, writes eval report
      │
      ├── classifier/models/setfit-mpnet/
      ├── classifier/models/setfit-modernbert/
      └── classifier/eval_report/
      │
      ▼
scripts/run_inference.py         ← predict all 11,441 posts
      │
      ▼
DB: predicted_class + confidence written back
```

---

## File Structure

**New files created:**
```
classifier/
├── label_guide.md                     # Generated draft, reviewed by user before labeling
├── classifier_interface.py            # Classifier class — backend-agnostic predict()
├── train_setfit.py                    # Train both backbones, evaluate, write reports
├── evaluate.py                        # Re-evaluate saved model on locked test set
├── data/
│   ├── sampled.jsonl                  # 180 posts for manual labeling
│   ├── labeled.jsonl                  # 480 posts (180 human + 300 Claude)
│   └── test_ids.txt                   # Locked test set IDs (written once, never overwritten)
└── eval_report/
    ├── classification_report.txt      # Per-class precision, recall, F1, macro avg (both models)
    ├── confusion_matrix.png           # Visual matrix for each model
    ├── kappa_score.txt                # Cohen's Kappa from re-labeling 50 posts
    └── error_analysis.md             # 10 FP + 10 FN per class, qualitative notes

scripts/
├── generate_label_guide.py            # One-time: writes classifier/label_guide.md
├── sample_posts.py                    # One-time: stratified sample from DB
├── label_posts.py                     # CLI labeling tool (resume-safe)
├── label_with_claude.py               # Claude Sonnet batch labeling
└── run_inference.py                   # Predict all posts, write back to DB
```

**Modified files:**
```
pipeline/db.py                         # +update_predictions(posts: list[Post])
requirements.txt                       # +setfit, +sentence-transformers, +matplotlib, +scikit-learn
```

---

## Labeling Strategy

Stratified sampling — not random — to ensure balanced classes with minimal wasted effort:

| Class | Manual labels | Claude labels | Keyword filter used |
|-------|--------------|---------------|---------------------|
| WORKFLOW_PAIN | 50 | 75 | "manually", "hours", "spreadsheet", "every week", "no tool" |
| TOOL_REQUEST | 50 | 75 | "is there a", "does anyone know", "someone should build" |
| PRODUCT_COMPLAINT | 50 | 75 | product/app names + "broken", "missing", "wish it had" |
| NOISE | 30 | 75 | random sample, no filter |
| **Total** | **180** | **300** | |

`label_with_claude.py` sends batches of 10 posts to Claude Sonnet with:
- The full `label_guide.md` as system context
- 20 human-labeled examples as few-shot demonstrations
- Returns structured JSON labels

---

## Classifier Interface

```python
# classifier/classifier_interface.py
class Classifier:
    def __init__(self, config: Config)
        # loads model specified by config.classifier_model_type + backbone
        # raises FileNotFoundError if model not trained yet

    def predict(self, posts: list[Post]) -> list[Post]
        # sets post.predicted_class and post.confidence on each post
        # leaves both as None if confidence < config.confidence_threshold
        # returns the same list (mutates in place)
```

Phase 4 imports this and calls `predict()` without knowing the backend is SetFit.

---

## Training

`classifier/train_setfit.py` trains both backbones sequentially:

1. **`sentence-transformers/paraphrase-mpnet-base-v2`** (primary)
2. **`nomic-ai/modernbert-embed-base`** (comparison)

Process for each:
- Load `labeled.jsonl`
- Stratified 80/20 train/test split
- Lock test IDs to `data/test_ids.txt` on first run (never overwritten on subsequent runs)
- Train SetFit model (num_epochs=1, num_iterations=20 — SetFit default)
- Evaluate on locked test set
- Save model to `classifier/models/setfit-{backbone-short-name}/`
- Write eval report files

Both models use the same locked test set for fair comparison.

---

## Evaluation

**Automated (run during training):**
- Per-class precision, recall, F1
- Macro-average F1 (target: ≥ 0.75)
- Confusion matrix image

**Manual (run ~2 weeks after initial labeling):**
- Re-label 50 posts without referring to original labels
- Run `python classifier/evaluate.py --kappa` to compute Cohen's Kappa (target: ≥ 0.70)

**Coverage report (run after inference):**
```
Total posts:        11,441
Confident (≥0.70):  ~8,000  (target: >60%)
Left NULL:          ~3,400
Passing to Phase 3: WORKFLOW_PAIN + TOOL_REQUEST count
```

If NULL rate is too high, lower `config.confidence_threshold` (e.g., 0.65) and re-run inference — no retraining needed.

---

## DB Change

```python
# pipeline/db.py — new method
def update_predictions(self, posts: list[Post]) -> None:
    # UPDATE posts SET predicted_class=?, confidence=? WHERE id=?
    # Only writes posts where post.predicted_class is not None
    # Batch commits for performance
```

---

## API Keys Required

| Key | Used by | When |
|-----|---------|------|
| `ANTHROPIC_API_KEY` | `label_with_claude.py` | Step 4 of labeling only |

Set in `.env` before running `label_with_claude.py`. Estimated cost: **~$0.70** total for 300 posts via Claude Sonnet 4.6.

All training and inference runs fully local on M2 (MPS). `config.embedding_use_fp16 = False` (fp16 unstable on MPS).

---

## Success Criteria

- [ ] `classifier/data/labeled.jsonl` contains 480 labeled posts
- [ ] `classifier/label_guide.md` reviewed and edited before any labeling
- [ ] Macro F1 ≥ 0.75 on locked test set
- [ ] Cohen's Kappa ≥ 0.70 on 50 re-labeled posts
- [ ] All 11,441 posts have had inference run (confident ones have `predicted_class` set)
- [ ] `classifier/eval_report/` contains all four files
- [ ] `pipeline/db.py` passes all existing tests after `update_predictions` addition

---

## What Phase 3 Receives

Posts in DB where `predicted_class IN ('WORKFLOW_PAIN', 'TOOL_REQUEST')` — approximately 5,000–7,000 posts depending on classifier coverage. These are the input to RoBERTa comparison (Phase 3) and BERTopic clustering (Phase 4).
