"""Fetch Reddit posts from Arctic Shift's pre-filtered search API.

Uses https://arctic-shift.photon-reddit.com/api/posts/search
to download subreddit-specific data without full monthly dumps.
"""

from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from config import Config
from pipeline.ingestion.schema import Post


def _normalize_post(raw: dict, author_meta: dict) -> Post:
    """Convert Arctic Shift JSON to Post schema."""
    title = raw.get("title", "")
    selftext = raw.get("selftext", "")
    text = f"{title}. {selftext}".strip() if selftext else title

    created_utc = datetime.fromtimestamp(raw["created_utc"], tz=timezone.utc)

    # Calculate account age from author creation date
    author_created = datetime.fromtimestamp(
        author_meta.get("created_utc", raw["created_utc"]), tz=timezone.utc
    )
    account_age_days = (created_utc - author_created).days

    return Post(
        id=raw["id"],
        text=text,
        source="reddit",
        created_utc=created_utc,
        score=raw.get("score", 0),
        author_karma=author_meta.get("link_karma", 0),
        account_age_days=max(account_age_days, 0),
        url=f"https://reddit.com/r/{raw.get('subreddit', '')}/comments/{raw['id']}",
        subreddit=raw.get("subreddit"),
    )


def fetch_subreddit_posts(
    subreddit: str,
    author_meta_lookup: Callable[[str], dict],
    config: Optional[Config] = None,
    after: Optional[datetime] = None,
    limit: int = 1000,
) -> list[Post]:
    """Fetch posts from Arctic Shift API for a single subreddit.

    Args:
        subreddit: Target subreddit name (without r/ prefix).
        author_meta_lookup: Function that takes an author name and returns
            {"link_karma": int, "created_utc": int}.
        config: Pipeline config. Defaults to Config().
        after: Only fetch posts after this datetime. Defaults to lookback_window_days ago.
        limit: Max posts per request (API limit is 1000).

    Returns:
        List of Post objects (not yet filtered).
    """
    if config is None:
        config = Config()

    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    params = {
        "subreddit": subreddit,
        "limit": min(limit, 1000),
        "sort": "desc",
        "sort_type": "created_utc",
    }

    if after:
        params["after"] = int(after.timestamp())

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json().get("data", [])

    posts = []
    for raw in data:
        author = raw.get("author", "[deleted]")
        if author in ("[deleted]", "[removed]", "AutoModerator"):
            continue
        author_meta = author_meta_lookup(author)
        post = _normalize_post(raw, author_meta)
        posts.append(post)

    return posts
