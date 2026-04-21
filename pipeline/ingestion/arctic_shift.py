"""Fetch Reddit posts from Arctic Shift's pre-filtered search API.

Uses https://arctic-shift.photon-reddit.com/api/posts/search
to download subreddit-specific data without full monthly dumps.

API constraints: max limit=100 per request, paginate via created_utc cursor.
"""

from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import time

import requests

from config import Config
from pipeline.ingestion.schema import Post

API_PAGE_SIZE = 100  # Arctic Shift rejects limit > 100


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


def _request_with_retry(url: str, params: dict, retries: int = 3) -> requests.Response:
    """GET with retry on 429/5xx."""
    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response  # unreachable but satisfies type checker


def fetch_subreddit_posts(
    subreddit: str,
    author_meta_lookup: Callable[[str], dict],
    config: Optional[Config] = None,
    after: Optional[datetime] = None,
    limit: int = 1000,
) -> list[Post]:
    """Fetch posts from Arctic Shift API for a single subreddit.

    Paginates using created_utc cursor since API caps at 100 per request.

    Args:
        subreddit: Target subreddit name (without r/ prefix).
        author_meta_lookup: Function that takes an author name and returns
            {"link_karma": int, "created_utc": int}.
        config: Pipeline config. Defaults to Config().
        after: Only fetch posts after this datetime. Defaults to lookback_window_days ago.
        limit: Total max posts to fetch across all pages.

    Returns:
        List of Post objects (not yet filtered).
    """
    if config is None:
        config = Config()

    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    # Default: look back from config
    if after is None:
        after = datetime.now(timezone.utc) - timedelta(days=config.lookback_window_days)

    all_posts = []
    cursor = None  # created_utc of oldest post in last batch

    while len(all_posts) < limit:
        params = {
            "subreddit": subreddit,
            "limit": API_PAGE_SIZE,
            "sort": "desc",
            "sort_type": "created_utc",
            "after": int(after.timestamp()),
        }
        if cursor is not None:
            params["before"] = cursor

        response = _request_with_retry(base_url, params)
        data = response.json().get("data", [])

        if not data:
            break

        for raw in data:
            author = raw.get("author", "[deleted]")
            if author in ("[deleted]", "[removed]", "AutoModerator"):
                continue
            author_meta = author_meta_lookup(author)
            post = _normalize_post(raw, author_meta)
            all_posts.append(post)

        # Use the oldest post's timestamp as cursor for next page
        cursor = data[-1]["created_utc"]

        # If we got fewer than page size, no more data
        if len(data) < API_PAGE_SIZE:
            break

        time.sleep(0.5)  # Be polite to the API

    return all_posts[:limit]
