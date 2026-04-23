"""Stratified sample from DB → classifier/data/sampled.jsonl.

Samples posts by keyword match per class, then random for NOISE.
Run once before manual labeling.

Usage: PYTHONPATH=. .venv/bin/python scripts/sample_posts.py
"""

import json
import random
from pathlib import Path

from config import Config
from pipeline.db import PostDB
from pipeline.ingestion.schema import Post

REMOVED_MARKERS = {"[removed]", "[deleted]"}

NOISE_TITLE_PREFIXES = (
    "show hn:",
    "show hn -",
    "ask hn: anyone",
    "ask hn: what do you",
    "ask hn: how do you feel",
    "ask hn: who is",
    "ask hn: poll",
)


def _is_usable(post: Post) -> bool:
    if post.text.strip() in REMOVED_MARKERS:
        return False
    if len(post.text.strip()) < 30:
        return False
    title_lower = post.text.lower()
    if any(title_lower.startswith(prefix) for prefix in NOISE_TITLE_PREFIXES):
        return False
    return True


KEYWORD_FILTERS: dict[str, list[str]] = {
    "WORKFLOW_PAIN": ["manually", "spreadsheet", "every week", "no tool", "takes hours", "tedious", "time-consuming"],
    "TOOL_REQUEST": ["is there a", "is there an app", "does anyone know of", "someone should build", "looking for a tool", "looking for software", "recommend a tool", "recommend an app", "any tools for", "any software for"],
    "PRODUCT_COMPLAINT": ["broken", "missing feature", "crashes", "doesn't work", "terrible", "stopped working", "wish it had"],
}

# Subreddits most likely to contain actionable tech/business pain points
SIGNAL_SUBREDDITS = {"smallbusiness", "Entrepreneur", "freelance", "webdev", "programming"}

SAMPLE_COUNTS: dict[str, int] = {
    "WORKFLOW_PAIN": 50,
    "TOOL_REQUEST": 50,
    "PRODUCT_COMPLAINT": 50,
    "NOISE": 30,
}

OUTPUT_PATH = Path("classifier/data/sampled.jsonl")


def sample_by_keywords(
    posts: list[Post],
    keywords: list[str],
    n: int,
    exclude_ids: set[str] | None = None,
    seed: int = 42,
) -> list[Post]:
    """Return up to n posts whose text contains any keyword (case-insensitive)."""
    if exclude_ids is None:
        exclude_ids = set()
    matched = [
        p for p in posts
        if p.id not in exclude_ids
        and _is_usable(p)
        and any(kw.lower() in p.text.lower() for kw in keywords)
    ]
    random.seed(seed)
    random.shuffle(matched)
    return matched[:n]


def sample_random(
    posts: list[Post],
    n: int,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[Post]:
    """Return n random posts, excluding any IDs in exclude_ids."""
    if exclude_ids is None:
        exclude_ids = set()
    pool = [p for p in posts if p.id not in exclude_ids and _is_usable(p)]
    random.seed(seed)
    random.shuffle(pool)
    return pool[:n]


def write_sampled_jsonl(posts: list[Post], output_path: Path) -> None:
    """Write posts to JSONL with id, text, url, source, label fields."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for post in posts:
            f.write(json.dumps({
                "id": post.id,
                "text": post.text,
                "url": post.url,
                "source": post.source,
                "suggested_class": None,
                "label": None,
            }) + "\n")


def main():
    config = Config()
    db = PostDB(config.db_path)
    all_posts = db.get_all_posts()
    db.close()

    print(f"Loaded {len(all_posts)} posts from DB")

    sampled: list[Post] = []
    used_ids: set[str] = set()

    # For signal classes, prefer posts from high-signal subreddits
    signal_posts = [p for p in all_posts if p.subreddit in SIGNAL_SUBREDDITS]
    hn_posts = [p for p in all_posts if p.source == "hn"]
    signal_pool = signal_posts + hn_posts
    print(f"  Signal pool (tech/business subreddits + HN): {len(signal_pool)} posts")

    for class_name in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT"]:
        keywords = KEYWORD_FILTERS[class_name]
        n = SAMPLE_COUNTS[class_name]
        # Try signal pool first, fall back to all posts if not enough matches
        batch = sample_by_keywords(signal_pool, keywords, n=n, exclude_ids=used_ids)
        if len(batch) < n:
            extra = sample_by_keywords(all_posts, keywords, n=n - len(batch), exclude_ids=used_ids | {p.id for p in batch})
            batch += extra
        print(f"  {class_name}: {len(batch)} posts sampled (keyword-filtered from signal pool)")
        sampled.extend(batch)
        used_ids.update(p.id for p in batch)

    noise_batch = sample_random(all_posts, n=SAMPLE_COUNTS["NOISE"], exclude_ids=used_ids)
    print(f"  NOISE: {len(noise_batch)} posts sampled (random)")
    sampled.extend(noise_batch)

    write_sampled_jsonl(sampled, OUTPUT_PATH)
    print(f"\nWrote {len(sampled)} posts to {OUTPUT_PATH}")
    print("Next: PYTHONPATH=. .venv/bin/python scripts/label_posts.py")


if __name__ == "__main__":
    main()
