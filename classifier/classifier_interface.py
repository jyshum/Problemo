"""Classifier interface: load saved model and predict on texts.

Loads the sentence-transformer encoder + logistic regression head
saved by train_setfit.py. Returns predicted class and confidence
for each input text.

Usage:
    from classifier.classifier_interface import Classifier
    clf = Classifier()
    clf.predict_posts(posts)  # mutates posts in-place
"""

import joblib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config

LABELS = ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODELS_DIR = Path("classifier/models")


class Classifier:
    """Wraps a sentence-transformer encoder + logistic regression head."""

    def __init__(self, model_name: str = "setfit-mpnet", models_dir: Path = MODELS_DIR):
        model_path = models_dir / model_name
        backbone = (model_path / "backbone.txt").read_text().strip()
        self.encoder = SentenceTransformer(backbone, device="cpu")
        self.head = joblib.load(model_path / "head.joblib")

    def predict(self, texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
        """Predict class and confidence for a list of texts.

        Returns list of (predicted_class, confidence) tuples.
        """
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=batch_size)
        label_ids = self.head.predict(embeddings)
        probas = self.head.predict_proba(embeddings)
        confidences = [float(probas[i, label_ids[i]]) for i in range(len(label_ids))]
        labels = [ID2LABEL[int(lid)] for lid in label_ids]
        return list(zip(labels, confidences))

    def predict_posts(self, posts: list, config: Config | None = None) -> list:
        """Predict on Post objects, setting predicted_class and confidence in-place.

        Only overwrites predictions where confidence exceeds threshold.
        Returns the list of posts (mutated).
        """
        if config is None:
            config = Config()

        texts = [p.text for p in posts]
        predictions = self.predict(texts)
        for post, (label, conf) in zip(posts, predictions):
            if conf >= config.confidence_threshold:
                post.predicted_class = label
                post.confidence = round(conf, 4)
            else:
                post.predicted_class = "UNCERTAIN"
                post.confidence = round(conf, 4)
        return posts
