"""Post schema — single source of truth for all pipeline data.

All ingestion sources normalize to this schema before storage.
Downstream code never imports source-specific types.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Post:
    id: str
    text: str  # title + body combined
    source: str  # "reddit", "hn", "appstore"
    created_utc: datetime
    score: int
    author_karma: int
    account_age_days: int
    url: str
    subreddit: Optional[str] = None
    freshness_score: Optional[float] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "created_utc": self.created_utc.isoformat(),
            "score": self.score,
            "author_karma": self.author_karma,
            "account_age_days": self.account_age_days,
            "url": self.url,
            "subreddit": self.subreddit,
            "freshness_score": self.freshness_score,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Post":
        created = d["created_utc"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        return cls(
            id=d["id"],
            text=d["text"],
            source=d["source"],
            created_utc=created,
            score=d["score"],
            author_karma=d["author_karma"],
            account_age_days=d["account_age_days"],
            url=d["url"],
            subreddit=d.get("subreddit"),
            freshness_score=d.get("freshness_score"),
            predicted_class=d.get("predicted_class"),
            confidence=d.get("confidence"),
        )
