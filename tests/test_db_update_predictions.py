import pytest
from pathlib import Path
from datetime import datetime, timezone

from pipeline.db import PostDB
from pipeline.ingestion.schema import Post


def make_post(id_, predicted_class=None, confidence=None):
    return Post(
        id=id_,
        text="some text here",
        source="reddit",
        created_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        score=10,
        author_karma=100,
        account_age_days=60,
        url="https://example.com",
        predicted_class=predicted_class,
        confidence=confidence,
    )


@pytest.fixture
def db(tmp_path):
    db = PostDB(tmp_path / "test.db")
    db.insert_posts([make_post("a"), make_post("b"), make_post("c")])
    return db


def test_update_predictions_writes_class_and_confidence(db):
    posts = [make_post("a", "WORKFLOW_PAIN", 0.92)]
    db.update_predictions(posts)
    updated = [p for p in db.get_all_posts() if p.id == "a"][0]
    assert updated.predicted_class == "WORKFLOW_PAIN"
    assert abs(updated.confidence - 0.92) < 1e-6


def test_update_predictions_skips_none_class(db):
    posts = [make_post("b", None, None)]
    db.update_predictions(posts)
    unchanged = [p for p in db.get_all_posts() if p.id == "b"][0]
    assert unchanged.predicted_class is None


def test_update_predictions_batch_multiple(db):
    posts = [
        make_post("a", "WORKFLOW_PAIN", 0.85),
        make_post("b", "NOISE", 0.91),
    ]
    db.update_predictions(posts)
    result = {p.id: p for p in db.get_all_posts()}
    assert result["a"].predicted_class == "WORKFLOW_PAIN"
    assert result["b"].predicted_class == "NOISE"
    assert result["c"].predicted_class is None  # untouched
