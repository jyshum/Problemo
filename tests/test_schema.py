from datetime import datetime, timezone
from pipeline.ingestion.schema import Post


def test_post_creation():
    post = Post(
        id="abc123",
        text="I manually track invoices every week in a spreadsheet",
        source="reddit",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=42,
        author_karma=500,
        account_age_days=365,
        url="https://reddit.com/r/freelance/abc123",
        subreddit="freelance",
    )
    assert post.id == "abc123"
    assert post.source == "reddit"
    assert post.subreddit == "freelance"
    assert post.predicted_class is None
    assert post.confidence is None
    assert post.freshness_score is None


def test_post_to_dict():
    post = Post(
        id="abc123",
        text="Test post",
        source="hn",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=10,
        author_karma=100,
        account_age_days=60,
        url="https://news.ycombinator.com/item?id=abc123",
        subreddit=None,
    )
    d = post.to_dict()
    assert d["id"] == "abc123"
    assert d["source"] == "hn"
    assert d["subreddit"] is None
    assert d["created_utc"] == "2026-03-15T00:00:00+00:00"


def test_post_from_dict():
    d = {
        "id": "xyz789",
        "text": "Some text",
        "source": "appstore",
        "created_utc": "2026-03-15T00:00:00+00:00",
        "score": 5,
        "author_karma": 0,
        "account_age_days": 0,
        "url": "https://apps.apple.com/xyz",
        "subreddit": None,
        "freshness_score": 0.85,
        "predicted_class": None,
        "confidence": None,
    }
    post = Post.from_dict(d)
    assert post.id == "xyz789"
    assert post.source == "appstore"
    assert post.freshness_score == 0.85
    assert post.created_utc == datetime(2026, 3, 15, tzinfo=timezone.utc)


def test_post_text_combines_title_and_body():
    """Posts from Reddit/HN have title+body. The text field stores the combined version."""
    post = Post(
        id="t1",
        text="Title here. Body content follows.",
        source="reddit",
        created_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        score=1,
        author_karma=100,
        account_age_days=90,
        url="http://example.com",
        subreddit="webdev",
    )
    assert "Title here" in post.text
    assert "Body content follows" in post.text
