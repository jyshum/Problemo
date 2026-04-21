"""Manual verification: run ingestion and report row counts.

Run this after setting up .env with real credentials.
Usage: python scripts/verify_ingestion.py
"""

from config import Config
from pipeline.db import PostDB
from pipeline.ingestion.run import ingest_all


def main():
    config = Config()
    db = PostDB(config.db_path)

    print("Starting ingestion...")
    print(f"  Target subreddits: {config.target_subreddits}")
    print(f"  HN query terms: {len(config.hn_query_terms)}")
    print()

    stats = ingest_all(db=db, config=config)

    print(f"\nIngestion complete:")
    print(f"  Fetched: {stats['fetched']}")
    print(f"  Stored:  {stats['stored']}")
    print(f"  Filtered out: {stats['filtered_out']}")
    print()

    counts = db.count_by_source()
    print("Posts by source:")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print(f"  TOTAL: {db.count()}")

    if db.count() >= 10_000:
        print("\n  Phase 1 milestone reached: >=10,000 posts stored.")
    else:
        print(f"\n  Need >=10,000 posts. Currently at {db.count()}. Run again or expand sources.")

    db.close()


if __name__ == "__main__":
    main()
