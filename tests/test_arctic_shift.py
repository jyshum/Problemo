"""Tests for Arctic Shift ingestion.

Uses mocked HTTP responses — actual API calls are tested manually.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.arctic_shift import fetch_subreddit_posts, _normalize_post


def test_normalize_post():
    """Arctic Shift returns Reddit JSON format — verify normalization to Post."""
    raw = {
        "id": "abc123",
        "title": "Anyone use a tool for invoice tracking?",
        "selftext": "I spend hours every week doing this manually.",
        "subreddit": "freelance",
        "created_utc": 1710460800,  # 2024-03-15 00:00:00 UTC
        "score": 25,
        "author_fullname": "t2_user1",
        "author": "testuser",
    }
    author_meta = {"link_karma": 500, "created_utc": 1672531200}  # account created 2023-01-01

    post = _normalize_post(raw, author_meta)
    assert post.id == "abc123"
    assert "Anyone use a tool" in post.text
    assert "hours every week" in post.text
    assert post.source == "reddit"
    assert post.subreddit == "freelance"
    assert post.score == 25
    assert post.author_karma == 500


def test_normalize_post_no_selftext():
    """Posts with empty selftext should still work (title-only)."""
    raw = {
        "id": "def456",
        "title": "Is there a tool for X?",
        "selftext": "",
        "subreddit": "webdev",
        "created_utc": 1710460800,
        "score": 5,
        "author_fullname": "t2_user2",
        "author": "user2",
    }
    author_meta = {"link_karma": 200, "created_utc": 1672531200}

    post = _normalize_post(raw, author_meta)
    assert post.text == "Is there a tool for X?"
    assert post.source == "reddit"


@patch("pipeline.ingestion.arctic_shift.requests.get")
def test_fetch_subreddit_posts_parses_response(mock_get):
    """Verify fetch_subreddit_posts returns Post objects from API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "id": "post1",
                "title": "Test title",
                "selftext": "Test body",
                "subreddit": "freelance",
                "created_utc": 1710460800,
                "score": 10,
                "author_fullname": "t2_u1",
                "author": "u1",
            }
        ]
    }
    mock_get.return_value = mock_response

    posts = fetch_subreddit_posts(
        subreddit="freelance",
        author_meta_lookup=lambda author: {"link_karma": 100, "created_utc": 1672531200},
    )
    assert len(posts) == 1
    assert posts[0].id == "post1"
    assert posts[0].subreddit == "freelance"
