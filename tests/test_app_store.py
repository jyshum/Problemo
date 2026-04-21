from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.app_store import fetch_app_reviews, _normalize_review


def _make_mock_review(**overrides):
    review = MagicMock()
    review.id = overrides.get("id", "rev_001")
    review.title = overrides.get("title", "Terrible experience")
    review.body = overrides.get("body", "The app crashes every time I try to export invoices.")
    review.created_at = overrides.get("created_at", datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc))
    review.rating = overrides.get("rating", 1)
    review.author_name = overrides.get("author_name", "frustrated_user")
    review.app_id = overrides.get("app_id", "com.example.app")
    return review


def test_normalize_review():
    review = _make_mock_review()
    post = _normalize_review(review, app_name="InvoiceApp")
    assert post.id == "appstore_rev_001"
    assert "Terrible experience" in post.text
    assert "crashes every time" in post.text
    assert post.source == "appstore"
    assert post.score == 1
    assert post.subreddit is None


def test_normalize_review_no_body():
    review = _make_mock_review(body="")
    post = _normalize_review(review, app_name="SomeApp")
    assert post.text == "Terrible experience"


@patch("pipeline.ingestion.app_store.AppStoreReviews")
def test_fetch_app_reviews_returns_posts(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    review = _make_mock_review(
        id="r1",
        title="Wish it had feature X",
        body="I need this for my workflow",
        rating=3,
    )
    mock_result = MagicMock()
    mock_result.reviews = [review]
    mock_client.fetch.return_value = mock_result

    posts = fetch_app_reviews(app_id="com.example.app", app_name="ExampleApp")
    assert len(posts) == 1
    assert posts[0].id == "appstore_r1"
    assert posts[0].source == "appstore"
