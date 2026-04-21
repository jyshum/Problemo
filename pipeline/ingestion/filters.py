"""Authenticity and freshness filters applied at ingestion time.

Posts failing any authenticity filter are discarded and never stored.
Freshness is computed at ingestion and stored with the post.
"""

import math
from datetime import datetime, timezone

from config import Config
from pipeline.ingestion.schema import Post


def passes_authenticity_filter(post: Post, config: Config | None = None) -> bool:
    """Return True if post passes all authenticity checks.

    Karma and account age checks only apply to Reddit posts —
    HN and App Store APIs don't expose this metadata.
    Text length check applies to all sources.
    """
    if config is None:
        config = Config()

    # Text length applies to all sources
    if len(post.text) < config.min_text_length:
        return False

    # Karma and account age only apply to Reddit
    if post.source == "reddit":
        if post.account_age_days < config.min_account_age_days:
            return False
        if post.author_karma < config.min_karma:
            return False

    return True


def compute_freshness_score(
    created_utc: datetime, half_life_days: int = 45
) -> float:
    """Exponential decay with configurable half-life.

    Returns a score between 0.0 and 1.0.
    A post created now scores 1.0.
    A post one half-life old scores 0.5.
    """
    now = datetime.now(timezone.utc)
    age_days = (now - created_utc).total_seconds() / 86400
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)
