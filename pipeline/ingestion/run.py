"""Orchestrates the full ingestion pipeline.

Flow: fetch from all sources -> filter -> compute freshness -> store in DB.
"""

import time
from typing import Optional

from config import Config
from pipeline.db import PostDB
from pipeline.ingestion.schema import Post
from pipeline.ingestion.filters import passes_authenticity_filter, compute_freshness_score
from pipeline.ingestion.arctic_shift import fetch_subreddit_posts
from pipeline.ingestion.hacker_news import fetch_hn_posts
from pipeline.ingestion.app_store import fetch_app_reviews


def _fetch_from_all_sources(config: Config) -> list[Post]:
    """Fetch posts from all configured sources."""
    all_posts = []

    # Arctic Shift — bulk historical Reddit
    for subreddit in config.target_subreddits:
        try:
            posts = fetch_subreddit_posts(
                subreddit=subreddit,
                author_meta_lookup=lambda author: {"link_karma": 100, "created_utc": 0},
                config=config,
            )
            all_posts.extend(posts)
        except Exception as e:
            print(f"[arctic_shift] Error fetching r/{subreddit}: {e}")
        time.sleep(1)  # Rate limit between subreddits

    # Hacker News
    for query in config.hn_query_terms:
        try:
            posts = fetch_hn_posts(query=query, days_back=config.lookback_window_days, config=config)
            all_posts.extend(posts)
        except Exception as e:
            print(f"[hn] Error fetching '{query}': {e}")

    return all_posts


def ingest_all(
    db: Optional[PostDB] = None,
    config: Optional[Config] = None,
) -> dict:
    """Run full ingestion pipeline.

    Returns:
        Stats dict with keys: fetched, stored, filtered_out.
    """
    if config is None:
        config = Config()
    if db is None:
        db = PostDB(config.db_path)

    posts = _fetch_from_all_sources(config)
    fetched = len(posts)

    # Filter and compute freshness
    accepted = []
    for post in posts:
        if not passes_authenticity_filter(post, config):
            continue
        post.freshness_score = compute_freshness_score(
            post.created_utc, config.freshness_half_life_days
        )
        accepted.append(post)

    # Store
    db.insert_posts(accepted)

    return {
        "fetched": fetched,
        "stored": len(accepted),
        "filtered_out": fetched - len(accepted),
    }
