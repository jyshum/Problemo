import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.db import PostDB
from pipeline.ingestion.schema import Post


def _make_post(id: str = "test1") -> Post:
    return Post(
        id=id,
        text="I manually track invoices every week",
        source="reddit",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=42,
        author_karma=500,
        account_age_days=365,
        url=f"https://reddit.com/r/freelance/{id}",
        subreddit="freelance",
        freshness_score=0.85,
    )


def test_create_table(tmp_path):
    db = PostDB(tmp_path / "test.db")
    # Table should exist after init
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posts'")
    assert cursor.fetchone() is not None
    conn.close()


def test_insert_and_retrieve(tmp_path):
    db = PostDB(tmp_path / "test.db")
    post = _make_post()
    db.insert_posts([post])
    results = db.get_all_posts()
    assert len(results) == 1
    assert results[0].id == "test1"
    assert results[0].subreddit == "freelance"
    assert results[0].freshness_score == 0.85


def test_insert_duplicate_skipped(tmp_path):
    db = PostDB(tmp_path / "test.db")
    post = _make_post()
    db.insert_posts([post])
    db.insert_posts([post])  # duplicate
    results = db.get_all_posts()
    assert len(results) == 1


def test_count_posts(tmp_path):
    db = PostDB(tmp_path / "test.db")
    posts = [_make_post(f"post_{i}") for i in range(5)]
    db.insert_posts(posts)
    assert db.count() == 5


def test_count_by_source(tmp_path):
    db = PostDB(tmp_path / "test.db")
    posts = [
        _make_post("r1"),
        _make_post("r2"),
    ]
    posts[1].source = "hn"
    db.insert_posts(posts)
    counts = db.count_by_source()
    assert counts["reddit"] == 1
    assert counts["hn"] == 1
