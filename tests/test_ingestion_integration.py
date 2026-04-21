"""Integration test for the full ingestion pipeline.

Tests the flow: fetch -> filter -> compute freshness -> store.
Uses mocked sources but real filters and DB.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from pipeline.ingestion.run import ingest_all
from pipeline.ingestion.schema import Post
from pipeline.db import PostDB


def _fake_posts():
    """Mix of posts that should and shouldn't pass filters."""
    now = datetime.now(timezone.utc)
    return [
        # Should pass — authentic, recent
        Post(
            id="good1", text="I manually track everything in spreadsheets every single week",
            source="reddit", created_utc=now - timedelta(days=5),
            score=20, author_karma=200, account_age_days=90,
            url="http://example.com/1", subreddit="freelance",
        ),
        # Should fail — low karma
        Post(
            id="bad_karma", text="This is a valid length post but from a new account",
            source="reddit", created_utc=now - timedelta(days=2),
            score=5, author_karma=10, account_age_days=90,
            url="http://example.com/2", subreddit="freelance",
        ),
        # Should fail — too short
        Post(
            id="bad_short", text="lol",
            source="hn", created_utc=now - timedelta(days=1),
            score=50, author_karma=500, account_age_days=365,
            url="http://example.com/3", subreddit=None,
        ),
        # Should pass — HN post
        Post(
            id="good2", text="Is there a tool that automates contract generation for freelancers?",
            source="hn", created_utc=now - timedelta(days=10),
            score=45, author_karma=1000, account_age_days=730,
            url="http://example.com/4", subreddit=None,
        ),
    ]


@patch("pipeline.ingestion.run._fetch_from_all_sources")
def test_ingest_all_filters_and_stores(mock_fetch, tmp_path):
    mock_fetch.return_value = _fake_posts()

    db = PostDB(tmp_path / "test.db")
    stats = ingest_all(db=db)

    assert db.count() == 2  # Only good1 and good2 should pass
    assert stats["fetched"] == 4
    assert stats["stored"] == 2
    assert stats["filtered_out"] == 2

    # Verify freshness was computed
    posts = db.get_all_posts()
    for post in posts:
        assert post.freshness_score is not None
        assert 0.0 < post.freshness_score <= 1.0
