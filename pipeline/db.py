"""SQLite storage for posts.

Handles table creation, insertion (with dedup), and basic queries.
"""

import sqlite3
from pathlib import Path

from pipeline.ingestion.schema import Post


class PostDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                score INTEGER NOT NULL,
                author_karma INTEGER NOT NULL,
                account_age_days INTEGER NOT NULL,
                url TEXT NOT NULL,
                subreddit TEXT,
                freshness_score REAL,
                predicted_class TEXT,
                confidence REAL
            )
        """)
        self._conn.commit()

    def insert_posts(self, posts: list[Post]):
        for post in posts:
            d = post.to_dict()
            self._conn.execute("""
                INSERT OR IGNORE INTO posts
                (id, text, source, created_utc, score, author_karma,
                 account_age_days, url, subreddit, freshness_score,
                 predicted_class, confidence)
                VALUES (:id, :text, :source, :created_utc, :score,
                        :author_karma, :account_age_days, :url, :subreddit,
                        :freshness_score, :predicted_class, :confidence)
            """, d)
        self._conn.commit()

    def get_all_posts(self) -> list[Post]:
        cursor = self._conn.execute("SELECT * FROM posts")
        return [Post.from_dict(dict(row)) for row in cursor.fetchall()]

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM posts")
        return cursor.fetchone()[0]

    def count_by_source(self) -> dict[str, int]:
        cursor = self._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM posts GROUP BY source"
        )
        return {row["source"]: row["cnt"] for row in cursor.fetchall()}

    def close(self):
        self._conn.close()
