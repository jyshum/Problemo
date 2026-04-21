"""Fetch posts from Hacker News via Algolia API.

API endpoint: https://hn.algolia.com/api/v1/search_by_date
- No auth required
- 1,000 hit cap per query — paginate by 7-day date windows
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config import Config
from pipeline.ingestion.schema import Post


def _normalize_hit(hit: dict) -> Post:
    """Convert Algolia HN hit to Post schema."""
    title = hit.get("title") or ""
    story_text = hit.get("story_text") or ""
    text = f"{title}. {story_text}".strip() if story_text else title

    created_utc = datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc)

    return Post(
        id=f"hn_{hit['objectID']}",
        text=text,
        source="hn",
        created_utc=created_utc,
        score=hit.get("points", 0) or 0,
        author_karma=0,  # HN API doesn't expose karma in search results
        account_age_days=0,  # Not available from Algolia
        url=f"https://news.ycombinator.com/item?id={hit['objectID']}",
        subreddit=None,
    )


def fetch_hn_posts(
    query: str,
    days_back: int = 7,
    config: Optional[Config] = None,
) -> list[Post]:
    """Fetch HN posts matching a query within a date window.

    Args:
        query: Search term (e.g., "is there a tool").
        days_back: How many days back to search.
        config: Pipeline config.

    Returns:
        List of Post objects.
    """
    base_url = "https://hn.algolia.com/api/v1/search_by_date"

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)

    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>{int(start.timestamp())}",
        "hitsPerPage": 200,
        "page": 0,
    }

    all_posts = []

    while True:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hits = data.get("hits", [])
        for hit in hits:
            post = _normalize_hit(hit)
            all_posts.append(post)

        # Paginate if more pages exist (up to 1000 hits cap)
        current_page = params["page"]
        total_pages = data.get("nbPages", 1)
        if current_page + 1 >= total_pages or len(all_posts) >= 1000:
            break
        params["page"] = current_page + 1

    return all_posts
