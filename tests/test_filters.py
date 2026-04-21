import math
from datetime import datetime, timezone, timedelta
from pipeline.ingestion.schema import Post
from pipeline.ingestion.filters import passes_authenticity_filter, compute_freshness_score


def _make_post(**overrides) -> Post:
    defaults = {
        "id": "test1",
        "text": "This is a long enough test post for the filter to accept",
        "source": "reddit",
        "created_utc": datetime(2026, 4, 1, tzinfo=timezone.utc),
        "score": 10,
        "author_karma": 100,
        "account_age_days": 90,
        "url": "http://example.com",
        "subreddit": "webdev",
    }
    defaults.update(overrides)
    return Post(**defaults)


def test_passes_all_filters():
    post = _make_post()
    assert passes_authenticity_filter(post) is True


def test_fails_account_age():
    post = _make_post(account_age_days=10)
    assert passes_authenticity_filter(post) is False


def test_fails_karma():
    post = _make_post(author_karma=20)
    assert passes_authenticity_filter(post) is False


def test_fails_text_length():
    post = _make_post(text="too short")
    assert passes_authenticity_filter(post) is False


def test_fails_empty_text():
    post = _make_post(text="")
    assert passes_authenticity_filter(post) is False


def test_hn_post_skips_karma_and_age_checks():
    """HN and App Store don't provide karma/age — only text length applies."""
    post = _make_post(source="hn", author_karma=0, account_age_days=0)
    assert passes_authenticity_filter(post) is True


def test_appstore_post_skips_karma_and_age_checks():
    post = _make_post(source="appstore", author_karma=0, account_age_days=0)
    assert passes_authenticity_filter(post) is True


def test_hn_post_still_fails_text_length():
    post = _make_post(source="hn", author_karma=0, account_age_days=0, text="short")
    assert passes_authenticity_filter(post) is False


def test_freshness_score_recent_post():
    """A post from today should have freshness close to 1.0."""
    now = datetime.now(timezone.utc)
    score = compute_freshness_score(now, half_life_days=45)
    assert score > 0.99


def test_freshness_score_at_half_life():
    """A post exactly one half-life old should score 0.5."""
    now = datetime.now(timezone.utc)
    post_date = now - timedelta(days=45)
    score = compute_freshness_score(post_date, half_life_days=45)
    assert abs(score - 0.5) < 0.01


def test_freshness_score_very_old():
    """A post 180 days old (4 half-lives) should score ~0.0625."""
    now = datetime.now(timezone.utc)
    post_date = now - timedelta(days=180)
    score = compute_freshness_score(post_date, half_life_days=45)
    expected = 0.5 ** (180 / 45)  # 0.0625
    assert abs(score - expected) < 0.01
