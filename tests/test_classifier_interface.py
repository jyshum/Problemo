import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path

from pipeline.ingestion.schema import Post


def make_post(id_, text):
    return Post(
        id=id_,
        text=text,
        source="reddit",
        created_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        score=10,
        author_karma=100,
        account_age_days=60,
        url=f"https://example.com/{id_}",
    )


def test_predict_returns_labels_and_confidences():
    from classifier.classifier_interface import Classifier

    clf = Classifier()
    results = clf.predict(["I manually track everything in spreadsheets every week"])
    assert len(results) == 1
    label, conf = results[0]
    assert label in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"]
    assert 0.0 <= conf <= 1.0


def test_predict_posts_sets_fields():
    from classifier.classifier_interface import Classifier
    from config import Config

    config = Config()
    config.confidence_threshold = 0.0  # accept all predictions

    clf = Classifier()
    posts = [
        make_post("1", "I manually do this repetitive task every single week and it takes hours"),
        make_post("2", "Just a random post about nothing important today"),
    ]
    clf.predict_posts(posts, config=config)

    for p in posts:
        assert p.predicted_class is not None
        assert p.confidence is not None
        assert p.confidence >= 0.0


def test_predict_posts_marks_uncertain_below_threshold():
    from classifier.classifier_interface import Classifier
    from config import Config

    config = Config()
    config.confidence_threshold = 0.99  # almost nothing will pass

    clf = Classifier()
    posts = [make_post("1", "some ambiguous text that could be anything really")]
    clf.predict_posts(posts, config=config)

    assert posts[0].predicted_class == "UNCERTAIN"
