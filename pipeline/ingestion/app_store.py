"""Fetch app reviews via the app-reviews package.

Handles App Store reviews. No API keys required.
"""

from datetime import datetime, timezone
from typing import Optional

from app_reviews import AppStoreReviews

from pipeline.ingestion.schema import Post


def _normalize_review(review, app_name: str) -> Post:
    """Convert an app-reviews Review object to Post schema."""
    title = review.title or ""
    body = review.body or ""
    text = f"{title}. {body}".strip() if body else title

    created_utc = review.created_at
    if created_utc is None:
        created_utc = datetime.now(timezone.utc)
    elif created_utc.tzinfo is None:
        created_utc = created_utc.replace(tzinfo=timezone.utc)

    return Post(
        id=f"appstore_{review.id}",
        text=text,
        source="appstore",
        created_utc=created_utc,
        score=review.rating or 0,
        author_karma=0,
        account_age_days=0,
        url=f"https://apps.apple.com/app/{app_name}",
        subreddit=None,
    )


def fetch_app_reviews(
    app_id: str,
    app_name: str,
    country: str = "us",
    max_reviews: int = 500,
) -> list[Post]:
    """Fetch reviews for a single app.

    Args:
        app_id: App bundle ID or store ID.
        app_name: Human-readable app name (for URL construction).
        country: App Store country code.
        max_reviews: Maximum reviews to fetch.

    Returns:
        List of Post objects.
    """
    client = AppStoreReviews()
    result = client.fetch(app_id, countries=[country], limit=max_reviews)

    posts = []
    for review in result.reviews:
        post = _normalize_review(review, app_name=app_name)
        posts.append(post)

    return posts
