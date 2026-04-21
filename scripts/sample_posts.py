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

KEYWORD_FILTERS: dict[str, list[str]] = {
    "WORKFLOW_PAIN": ["manually", "hours", "spreadsheet", "every week", "no tool", "have to", "tedious"],
    "TOOL_REQUEST": ["is there a", "is there an", "does anyone know", "someone should build", "looking for"],
    "PRODUCT_COMPLAINT": ["broken", "missing", "crashes", "doesn't work", "terrible", "stopped working"],
}

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
    pool = [p for p in posts if p.id not in exclude_ids]
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

    for class_name in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT"]:
        keywords = KEYWORD_FILTERS[class_name]
        n = SAMPLE_COUNTS[class_name]
        batch = sample_by_keywords(all_posts, keywords, n=n, exclude_ids=used_ids)
        print(f"  {class_name}: {len(batch)} posts sampled (keyword-filtered)")
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
