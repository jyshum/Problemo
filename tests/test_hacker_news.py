from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.hacker_news import fetch_hn_posts, _normalize_hit


def test_normalize_hit():
    raw = {
        "objectID": "12345",
        "title": "Ask HN: How do you track invoices?",
        "story_text": "I'm tired of doing this manually every week.",
        "created_at_i": 1710460800,
        "points": 35,
        "author": "hn_user",
        "num_comments": 12,
    }
    post = _normalize_hit(raw)
    assert post.id == "hn_12345"
    assert "How do you track invoices" in post.text
    assert "tired of doing this manually" in post.text
    assert post.source == "hn"
    assert post.score == 35
    assert post.subreddit is None


def test_normalize_hit_no_story_text():
    raw = {
        "objectID": "67890",
        "title": "Show HN: My invoice tool",
        "story_text": None,
        "created_at_i": 1710460800,
        "points": 10,
        "author": "another_user",
        "num_comments": 3,
    }
    post = _normalize_hit(raw)
    assert post.text == "Show HN: My invoice tool"


@patch("pipeline.ingestion.hacker_news.requests.get")
def test_fetch_hn_posts_paginates(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "hits": [
            {
                "objectID": "1",
                "title": "I manually do X",
                "story_text": "Details here",
                "created_at_i": 1710460800,
                "points": 5,
                "author": "user1",
                "num_comments": 2,
            }
        ],
        "nbPages": 1,
    }
    mock_get.return_value = mock_response

    posts = fetch_hn_posts(query="I manually", days_back=7)
    assert len(posts) == 1
    assert posts[0].id == "hn_1"
    assert posts[0].source == "hn"
