# Phase 1: Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the project and build a working multi-source ingestion pipeline that stores ≥10,000 authentic, freshness-scored posts in SQLite.

**Architecture:** Four ingestion modules (Arctic Shift, PRAW, Hacker News, App Store) each normalize raw data into a shared `Post` schema. Posts pass through authenticity and freshness filters at ingestion time before being stored in SQLite. All configuration lives in `config.py`.

**Tech Stack:** Python 3.11+, SQLite, PRAW, requests, app-reviews, zstandard (for any .zst files)

---

## File Structure

```
pain-point-engine/
├── .env.example              # Template showing required/optional keys
├── .gitignore
├── config.py                 # All configuration values
├── requirements.txt          # Pinned production dependencies
├── requirements-dev.txt      # Pinned dev/test dependencies
├── pipeline/
│   ├── __init__.py
│   ├── db.py                 # SQLite connection and table setup
│   └── ingestion/
│       ├── __init__.py
│       ├── schema.py         # Post dataclass + to_dict/from_dict
│       ├── filters.py        # Authenticity + freshness scoring
│       ├── arctic_shift.py   # Arctic Shift pre-filtered API
│       ├── reddit_incremental.py  # PRAW weekly pulls
│       ├── hacker_news.py    # Algolia HN API
│       └── app_store.py      # app-reviews package
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures
    ├── test_schema.py
    ├── test_filters.py
    ├── test_db.py
    ├── test_arctic_shift.py
    ├── test_hacker_news.py
    ├── test_app_store.py
    └── test_reddit_incremental.py
```

---

## Task 1: Project Bootstrap

**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Create project directory and virtual environment**

```bash
cd ~/Desktop/pain_point
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Install core dependencies and pin versions**

```bash
pip install praw requests zstandard python-dotenv anthropic openai app-reviews
pip freeze > requirements.txt
```

Review `requirements.txt` — it will contain exact pinned versions of everything installed. Remove any packages that are clearly unrelated (pip sometimes pulls in extras).

- [ ] **Step 3: Install dev dependencies and pin separately**

```bash
pip install pytest pytest-cov
pip freeze | grep -E "^(pytest|pytest-cov|iniconfig|pluggy|coverage)" > requirements-dev.txt
```

- [ ] **Step 4: Create `.env.example`**

```bash
# .env.example
# Required
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=pain-point-engine/1.0 by /u/YOUR_USERNAME
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional (pipeline degrades gracefully without these)
OPENAI_API_KEY=
PRODUCTHUNT_API_TOKEN=
```

- [ ] **Step 5: Create `.gitignore`**

```
# Virtual environment
.venv/

# Environment variables
.env

# Data directories
data/raw/
data/processed/

# Database
dashboard/feedback.db
*.db

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
```

- [ ] **Step 6: Create directory skeleton**

```bash
mkdir -p pipeline/ingestion
mkdir -p pipeline/clustering
mkdir -p pipeline/scoring
mkdir -p classifier/data
mkdir -p classifier/eval_report
mkdir -p dashboard
mkdir -p data/raw
mkdir -p data/processed
mkdir -p tests
touch pipeline/__init__.py
touch pipeline/ingestion/__init__.py
touch tests/__init__.py
```

- [ ] **Step 7: Commit — project bootstrap**

```bash
git init
git add .gitignore .env.example requirements.txt requirements-dev.txt
git add pipeline/__init__.py pipeline/ingestion/__init__.py tests/__init__.py
git commit -m "chore: project bootstrap with dependencies and directory structure"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_config.py
from config import Config


def test_config_has_required_fields():
    cfg = Config()
    assert cfg.target_subreddits == [
        "freelance", "smallbusiness", "Entrepreneur", "nursing",
        "medicine", "legaladvice", "webdev", "programming",
        "Teachers", "GradSchool",
    ]
    assert cfg.min_account_age_days == 30
    assert cfg.min_karma == 50
    assert cfg.min_text_length == 30
    assert cfg.freshness_half_life_days == 45
    assert cfg.staleness_threshold_days == 60
    assert cfg.embedding_model == "BAAI/bge-m3"
    assert cfg.embedding_use_fp16 is False
    assert cfg.classifier_model_type == "setfit"
    assert cfg.bertopic_min_topic_size == 10
    assert cfg.umap_random_state == 42
    assert cfg.scoring_composite_weights == {"frequency": 0.25, "intensity": 0.25, "opportunity": 0.50}


def test_config_hn_query_terms():
    cfg = Config()
    assert "is there a tool" in cfg.hn_query_terms
    assert "I manually" in cfg.hn_query_terms
    assert len(cfg.hn_query_terms) == 7
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement config.py**

```python
# config.py
"""All configuration for the pain-point-engine pipeline.

Change model choices, tunable values, and API settings here.
No magic numbers or model names hardcoded in source files.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # === Ingestion ===
    target_subreddits: list[str] = field(default_factory=lambda: [
        "freelance", "smallbusiness", "Entrepreneur", "nursing",
        "medicine", "legaladvice", "webdev", "programming",
        "Teachers", "GradSchool",
    ])

    hn_query_terms: list[str] = field(default_factory=lambda: [
        "is there a tool",
        "wish there was",
        "I manually",
        "takes hours",
        "no good solution",
        "Ask HN: How do you",
        "frustrated with",
    ])

    lookback_window_days: int = 90
    max_posts_per_run_per_source: int = 5000

    # === Filtering ===
    min_account_age_days: int = 30
    min_karma: int = 50
    min_text_length: int = 30

    # === Freshness ===
    freshness_half_life_days: int = 45
    staleness_threshold_days: int = 60

    # === Models: Embedding ===
    embedding_model: str = "BAAI/bge-m3"
    embedding_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_use_fp16: bool = False  # fp16 not stable on MPS

    # === Models: Classification ===
    classifier_model_type: str = "setfit"  # "setfit" or "hf"
    setfit_backbone: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    setfit_comparison_backbone: str = "nomic-ai/modernbert-embed-base"
    hf_backbone: str = "roberta-base"
    confidence_threshold: float = 0.7

    # === Models: LLM ===
    cluster_labeling_model: str = "claude-haiku-4-5-20251001"
    cluster_labeling_fallback: str = "gpt-5-nano"
    dataset_labeling_model: str = "claude-sonnet-4-6-20250514"

    # === Clustering ===
    bertopic_min_topic_size: int = 10
    umap_random_state: int = 42
    nr_topics: str = "auto"
    outlier_reduction_strategy: str = "embeddings"

    # === Scoring ===
    scoring_frequency_cap: int = 10
    scoring_composite_weights: dict = field(default_factory=lambda: {
        "frequency": 0.25,
        "intensity": 0.25,
        "opportunity": 0.50,
    })
    producthunt_api_token: str = field(default_factory=lambda: os.getenv("PRODUCTHUNT_API_TOKEN", ""))

    # === Scheduling ===
    cron_interval_days: int = 7

    # === Paths ===
    data_raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    data_processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    db_path: Path = field(default_factory=lambda: Path("data/posts.db"))

    # === API Keys (loaded from .env) ===
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    reddit_user_agent: str = field(default_factory=lambda: os.getenv("REDDIT_USER_AGENT", "pain-point-engine/1.0"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add config module with all pipeline settings"
```

---

## Task 3: Post Schema

**Files:**
- Create: `pipeline/ingestion/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_schema.py
from datetime import datetime, timezone
from pipeline.ingestion.schema import Post


def test_post_creation():
    post = Post(
        id="abc123",
        text="I manually track invoices every week in a spreadsheet",
        source="reddit",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=42,
        author_karma=500,
        account_age_days=365,
        url="https://reddit.com/r/freelance/abc123",
        subreddit="freelance",
    )
    assert post.id == "abc123"
    assert post.source == "reddit"
    assert post.subreddit == "freelance"
    assert post.predicted_class is None
    assert post.confidence is None
    assert post.freshness_score is None


def test_post_to_dict():
    post = Post(
        id="abc123",
        text="Test post",
        source="hn",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=10,
        author_karma=100,
        account_age_days=60,
        url="https://news.ycombinator.com/item?id=abc123",
        subreddit=None,
    )
    d = post.to_dict()
    assert d["id"] == "abc123"
    assert d["source"] == "hn"
    assert d["subreddit"] is None
    assert d["created_utc"] == "2026-03-15T00:00:00+00:00"


def test_post_from_dict():
    d = {
        "id": "xyz789",
        "text": "Some text",
        "source": "appstore",
        "created_utc": "2026-03-15T00:00:00+00:00",
        "score": 5,
        "author_karma": 0,
        "account_age_days": 0,
        "url": "https://apps.apple.com/xyz",
        "subreddit": None,
        "freshness_score": 0.85,
        "predicted_class": None,
        "confidence": None,
    }
    post = Post.from_dict(d)
    assert post.id == "xyz789"
    assert post.source == "appstore"
    assert post.freshness_score == 0.85
    assert post.created_utc == datetime(2026, 3, 15, tzinfo=timezone.utc)


def test_post_text_combines_title_and_body():
    """Posts from Reddit/HN have title+body. The text field stores the combined version."""
    post = Post(
        id="t1",
        text="Title here. Body content follows.",
        source="reddit",
        created_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        score=1,
        author_karma=100,
        account_age_days=90,
        url="http://example.com",
        subreddit="webdev",
    )
    assert "Title here" in post.text
    assert "Body content follows" in post.text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.schema'`

- [ ] **Step 3: Implement schema.py**

```python
# pipeline/ingestion/schema.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_schema.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/schema.py tests/test_schema.py
git commit -m "feat: add Post schema with serialization"
```

---

## Task 4: Authenticity Filters

**Files:**
- Create: `pipeline/ingestion/filters.py`
- Create: `tests/test_filters.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_filters.py
import math
from datetime import datetime, timezone, timedelta
from pipeline.ingestion.schema import Post
from pipeline.ingestion.filters import passes_authenticity_filter, compute_freshness_score


def _make_post(**overrides) -> Post:
    defaults = {
        "id": "test1",
        "text": "This is a long enough test post for the filter to accept",
        "source": "reddit",
        "created_utc": datetime(2026, 4, 1, tzinfo=timezone.utc),
        "score": 10,
        "author_karma": 100,
        "account_age_days": 90,
        "url": "http://example.com",
        "subreddit": "webdev",
    }
    defaults.update(overrides)
    return Post(**defaults)


def test_passes_all_filters():
    post = _make_post()
    assert passes_authenticity_filter(post) is True


def test_fails_account_age():
    post = _make_post(account_age_days=10)
    assert passes_authenticity_filter(post) is False


def test_fails_karma():
    post = _make_post(author_karma=20)
    assert passes_authenticity_filter(post) is False


def test_fails_text_length():
    post = _make_post(text="too short")
    assert passes_authenticity_filter(post) is False


def test_fails_empty_text():
    post = _make_post(text="")
    assert passes_authenticity_filter(post) is False


def test_hn_post_skips_karma_and_age_checks():
    """HN and App Store don't provide karma/age — only text length applies."""
    post = _make_post(source="hn", author_karma=0, account_age_days=0)
    assert passes_authenticity_filter(post) is True


def test_appstore_post_skips_karma_and_age_checks():
    post = _make_post(source="appstore", author_karma=0, account_age_days=0)
    assert passes_authenticity_filter(post) is True


def test_hn_post_still_fails_text_length():
    post = _make_post(source="hn", author_karma=0, account_age_days=0, text="short")
    assert passes_authenticity_filter(post) is False


def test_freshness_score_recent_post():
    """A post from today should have freshness close to 1.0."""
    now = datetime.now(timezone.utc)
    score = compute_freshness_score(now, half_life_days=45)
    assert score > 0.99


def test_freshness_score_at_half_life():
    """A post exactly one half-life old should score 0.5."""
    now = datetime.now(timezone.utc)
    post_date = now - timedelta(days=45)
    score = compute_freshness_score(post_date, half_life_days=45)
    assert abs(score - 0.5) < 0.01


def test_freshness_score_very_old():
    """A post 180 days old (4 half-lives) should score ~0.0625."""
    now = datetime.now(timezone.utc)
    post_date = now - timedelta(days=180)
    score = compute_freshness_score(post_date, half_life_days=45)
    expected = 0.5 ** (180 / 45)  # 0.0625
    assert abs(score - expected) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_filters.py -v
```

Expected: FAIL — `cannot import name 'passes_authenticity_filter'`

- [ ] **Step 3: Implement filters.py**

```python
# pipeline/ingestion/filters.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_filters.py -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/filters.py tests/test_filters.py
git commit -m "feat: add authenticity filters and freshness scoring"
```

---

## Task 5: SQLite Storage Layer

**Files:**
- Create: `pipeline/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_db.py
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.db import PostDB
from pipeline.ingestion.schema import Post


def _make_post(id: str = "test1") -> Post:
    return Post(
        id=id,
        text="I manually track invoices every week",
        source="reddit",
        created_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
        score=42,
        author_karma=500,
        account_age_days=365,
        url=f"https://reddit.com/r/freelance/{id}",
        subreddit="freelance",
        freshness_score=0.85,
    )


def test_create_table(tmp_path):
    db = PostDB(tmp_path / "test.db")
    # Table should exist after init
    conn = sqlite3.connect(tmp_path / "test.db")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posts'")
    assert cursor.fetchone() is not None
    conn.close()


def test_insert_and_retrieve(tmp_path):
    db = PostDB(tmp_path / "test.db")
    post = _make_post()
    db.insert_posts([post])
    results = db.get_all_posts()
    assert len(results) == 1
    assert results[0].id == "test1"
    assert results[0].subreddit == "freelance"
    assert results[0].freshness_score == 0.85


def test_insert_duplicate_skipped(tmp_path):
    db = PostDB(tmp_path / "test.db")
    post = _make_post()
    db.insert_posts([post])
    db.insert_posts([post])  # duplicate
    results = db.get_all_posts()
    assert len(results) == 1


def test_count_posts(tmp_path):
    db = PostDB(tmp_path / "test.db")
    posts = [_make_post(f"post_{i}") for i in range(5)]
    db.insert_posts(posts)
    assert db.count() == 5


def test_count_by_source(tmp_path):
    db = PostDB(tmp_path / "test.db")
    posts = [
        _make_post("r1"),
        _make_post("r2"),
    ]
    posts[1].source = "hn"
    db.insert_posts(posts)
    counts = db.count_by_source()
    assert counts["reddit"] == 1
    assert counts["hn"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.db'`

- [ ] **Step 3: Implement db.py**

```python
# pipeline/db.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_db.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/db.py tests/test_db.py
git commit -m "feat: add SQLite storage layer with dedup"
```

---

## Task 6: Arctic Shift Ingestion

**Files:**
- Create: `pipeline/ingestion/arctic_shift.py`
- Create: `tests/test_arctic_shift.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_arctic_shift.py
"""Tests for Arctic Shift ingestion.

Uses mocked HTTP responses — actual API calls are tested manually.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.arctic_shift import fetch_subreddit_posts, _normalize_post


def test_normalize_post():
    """Arctic Shift returns Reddit JSON format — verify normalization to Post."""
    raw = {
        "id": "abc123",
        "title": "Anyone use a tool for invoice tracking?",
        "selftext": "I spend hours every week doing this manually.",
        "subreddit": "freelance",
        "created_utc": 1710460800,  # 2024-03-15 00:00:00 UTC
        "score": 25,
        "author_fullname": "t2_user1",
        "author": "testuser",
    }
    author_meta = {"link_karma": 500, "created_utc": 1672531200}  # account created 2023-01-01

    post = _normalize_post(raw, author_meta)
    assert post.id == "abc123"
    assert "Anyone use a tool" in post.text
    assert "hours every week" in post.text
    assert post.source == "reddit"
    assert post.subreddit == "freelance"
    assert post.score == 25
    assert post.author_karma == 500


def test_normalize_post_no_selftext():
    """Posts with empty selftext should still work (title-only)."""
    raw = {
        "id": "def456",
        "title": "Is there a tool for X?",
        "selftext": "",
        "subreddit": "webdev",
        "created_utc": 1710460800,
        "score": 5,
        "author_fullname": "t2_user2",
        "author": "user2",
    }
    author_meta = {"link_karma": 200, "created_utc": 1672531200}

    post = _normalize_post(raw, author_meta)
    assert post.text == "Is there a tool for X?"
    assert post.source == "reddit"


@patch("pipeline.ingestion.arctic_shift.requests.get")
def test_fetch_subreddit_posts_parses_response(mock_get):
    """Verify fetch_subreddit_posts returns Post objects from API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "id": "post1",
                "title": "Test title",
                "selftext": "Test body",
                "subreddit": "freelance",
                "created_utc": 1710460800,
                "score": 10,
                "author_fullname": "t2_u1",
                "author": "u1",
            }
        ]
    }
    mock_get.return_value = mock_response

    posts = fetch_subreddit_posts(
        subreddit="freelance",
        author_meta_lookup=lambda author: {"link_karma": 100, "created_utc": 1672531200},
    )
    assert len(posts) == 1
    assert posts[0].id == "post1"
    assert posts[0].subreddit == "freelance"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_arctic_shift.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement arctic_shift.py**

```python
# pipeline/ingestion/arctic_shift.py
"""Fetch Reddit posts from Arctic Shift's pre-filtered search API.

Uses https://arctic-shift.photon-reddit.com/api/posts/search
to download subreddit-specific data without full monthly dumps.
"""

from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from config import Config
from pipeline.ingestion.schema import Post


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


def fetch_subreddit_posts(
    subreddit: str,
    author_meta_lookup: Callable[[str], dict],
    config: Optional[Config] = None,
    after: Optional[datetime] = None,
    limit: int = 1000,
) -> list[Post]:
    """Fetch posts from Arctic Shift API for a single subreddit.

    Args:
        subreddit: Target subreddit name (without r/ prefix).
        author_meta_lookup: Function that takes an author name and returns
            {"link_karma": int, "created_utc": int}.
        config: Pipeline config. Defaults to Config().
        after: Only fetch posts after this datetime. Defaults to lookback_window_days ago.
        limit: Max posts per request (API limit is 1000).

    Returns:
        List of Post objects (not yet filtered).
    """
    if config is None:
        config = Config()

    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    params = {
        "subreddit": subreddit,
        "limit": min(limit, 1000),
        "sort": "desc",
        "sort_type": "created_utc",
    }

    if after:
        params["after"] = int(after.timestamp())

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json().get("data", [])

    posts = []
    for raw in data:
        author = raw.get("author", "[deleted]")
        if author in ("[deleted]", "[removed]", "AutoModerator"):
            continue
        author_meta = author_meta_lookup(author)
        post = _normalize_post(raw, author_meta)
        posts.append(post)

    return posts
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_arctic_shift.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/arctic_shift.py tests/test_arctic_shift.py
git commit -m "feat: add Arctic Shift ingestion module"
```

---

## Task 7: Hacker News Ingestion

**Files:**
- Create: `pipeline/ingestion/hacker_news.py`
- Create: `tests/test_hacker_news.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_hacker_news.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.hacker_news import fetch_hn_posts, _normalize_hit


def test_normalize_hit():
    raw = {
        "objectID": "12345",
        "title": "Ask HN: How do you track invoices?",
        "story_text": "I'm tired of doing this manually every week.",
        "created_at_i": 1710460800,
        "points": 35,
        "author": "hn_user",
        "num_comments": 12,
    }
    post = _normalize_hit(raw)
    assert post.id == "hn_12345"
    assert "How do you track invoices" in post.text
    assert "tired of doing this manually" in post.text
    assert post.source == "hn"
    assert post.score == 35
    assert post.subreddit is None


def test_normalize_hit_no_story_text():
    raw = {
        "objectID": "67890",
        "title": "Show HN: My invoice tool",
        "story_text": None,
        "created_at_i": 1710460800,
        "points": 10,
        "author": "another_user",
        "num_comments": 3,
    }
    post = _normalize_hit(raw)
    assert post.text == "Show HN: My invoice tool"


@patch("pipeline.ingestion.hacker_news.requests.get")
def test_fetch_hn_posts_paginates(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "hits": [
            {
                "objectID": "1",
                "title": "I manually do X",
                "story_text": "Details here",
                "created_at_i": 1710460800,
                "points": 5,
                "author": "user1",
                "num_comments": 2,
            }
        ],
        "nbPages": 1,
    }
    mock_get.return_value = mock_response

    posts = fetch_hn_posts(query="I manually", days_back=7)
    assert len(posts) == 1
    assert posts[0].id == "hn_1"
    assert posts[0].source == "hn"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_hacker_news.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement hacker_news.py**

```python
# pipeline/ingestion/hacker_news.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_hacker_news.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/hacker_news.py tests/test_hacker_news.py
git commit -m "feat: add Hacker News ingestion via Algolia API"
```

---

## Task 8: App Store Ingestion

**Files:**
- Create: `pipeline/ingestion/app_store.py`
- Create: `tests/test_app_store.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_app_store.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.app_store import fetch_app_reviews, _normalize_review


def test_normalize_review():
    raw = {
        "id": "rev_001",
        "title": "Terrible experience",
        "content": "The app crashes every time I try to export invoices.",
        "date": "2026-03-15T10:00:00Z",
        "rating": 1,
        "userName": "frustrated_user",
    }
    post = _normalize_review(raw, app_name="InvoiceApp")
    assert post.id == "appstore_rev_001"
    assert "Terrible experience" in post.text
    assert "crashes every time" in post.text
    assert post.source == "appstore"
    assert post.score == 1
    assert post.subreddit is None


def test_normalize_review_no_content():
    raw = {
        "id": "rev_002",
        "title": "Broken",
        "content": "",
        "date": "2026-03-15T10:00:00Z",
        "rating": 2,
        "userName": "user2",
    }
    post = _normalize_review(raw, app_name="SomeApp")
    assert post.text == "Broken"


@patch("pipeline.ingestion.app_store.app_reviews")
def test_fetch_app_reviews_returns_posts(mock_app_reviews):
    mock_app_reviews.AppStore.return_value.reviews.return_value = [
        {
            "id": "r1",
            "title": "Wish it had feature X",
            "content": "I need this for my workflow",
            "date": "2026-03-15T10:00:00Z",
            "rating": 3,
            "userName": "reviewer1",
        }
    ]

    posts = fetch_app_reviews(app_id="com.example.app", app_name="ExampleApp")
    assert len(posts) == 1
    assert posts[0].id == "appstore_r1"
    assert posts[0].source == "appstore"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_app_store.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement app_store.py**

```python
# pipeline/ingestion/app_store.py
"""Fetch app reviews via the app-reviews package.

Handles both App Store and Google Play reviews.
No API keys required.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import app_reviews

from pipeline.ingestion.schema import Post


def _normalize_review(raw: dict, app_name: str) -> Post:
    """Convert app review to Post schema."""
    title = raw.get("title", "")
    content = raw.get("content", "")
    text = f"{title}. {content}".strip() if content else title

    # Parse date
    date_str = raw.get("date", "")
    if date_str:
        created_utc = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    else:
        created_utc = datetime.now(timezone.utc)

    return Post(
        id=f"appstore_{raw['id']}",
        text=text,
        source="appstore",
        created_utc=created_utc,
        score=raw.get("rating", 0),
        author_karma=0,  # Not available for app reviews
        account_age_days=0,  # Not available for app reviews
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
    store = app_reviews.AppStore(app_id=app_id, country=country)
    reviews = store.reviews(count=max_reviews)

    posts = []
    for review in reviews:
        post = _normalize_review(review, app_name=app_name)
        posts.append(post)
        time.sleep(0.1)  # Rate limiting

    return posts
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_app_store.py -v
```

Expected: 3 passed

Note: The `app_reviews` package API may differ from what's mocked here. After installing the package, run `python -c "import app_reviews; help(app_reviews)"` to verify the actual interface and adjust the implementation/mocks accordingly.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/app_store.py tests/test_app_store.py
git commit -m "feat: add App Store ingestion via app-reviews"
```

---

## Task 9: Reddit Incremental (PRAW)

**Files:**
- Create: `pipeline/ingestion/reddit_incremental.py`
- Create: `tests/test_reddit_incremental.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_reddit_incremental.py
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pipeline.ingestion.reddit_incremental import fetch_recent_posts, _normalize_submission


def _mock_submission():
    sub = MagicMock()
    sub.id = "praw_001"
    sub.title = "Anyone know a tool for time tracking?"
    sub.selftext = "I waste so much time switching between apps."
    sub.subreddit.display_name = "freelance"
    sub.created_utc = 1710460800.0
    sub.score = 15
    sub.author.link_karma = 300
    sub.author.created_utc = 1672531200.0
    sub.permalink = "/r/freelance/comments/praw_001/test/"
    sub.is_self = True
    sub.removed_by_category = None
    return sub


def test_normalize_submission():
    sub = _mock_submission()
    post = _normalize_submission(sub)
    assert post.id == "praw_001"
    assert "tool for time tracking" in post.text
    assert "waste so much time" in post.text
    assert post.source == "reddit"
    assert post.subreddit == "freelance"
    assert post.author_karma == 300


def test_normalize_submission_deleted_author():
    sub = _mock_submission()
    sub.author = None
    post = _normalize_submission(sub)
    assert post.author_karma == 0
    assert post.account_age_days == 0


@patch("pipeline.ingestion.reddit_incremental.praw.Reddit")
def test_fetch_recent_posts(mock_reddit_cls):
    mock_reddit = MagicMock()
    mock_reddit_cls.return_value = mock_reddit

    sub = _mock_submission()
    mock_subreddit = MagicMock()
    mock_subreddit.new.return_value = [sub]
    mock_reddit.subreddit.return_value = mock_subreddit

    posts = fetch_recent_posts(subreddit="freelance")
    assert len(posts) == 1
    assert posts[0].id == "praw_001"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_reddit_incremental.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement reddit_incremental.py**

```python
# pipeline/ingestion/reddit_incremental.py
"""Fetch recent Reddit posts via PRAW for weekly incremental updates.

PRAW free tier: 60 requests/minute.
This module fetches the last 7 days of posts from target subreddits.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import praw

from config import Config
from pipeline.ingestion.schema import Post


def _normalize_submission(submission) -> Post:
    """Convert a PRAW Submission to Post schema."""
    title = submission.title or ""
    selftext = submission.selftext or ""
    text = f"{title}. {selftext}".strip() if selftext else title

    created_utc = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)

    # Handle deleted/suspended authors
    if submission.author:
        author_karma = getattr(submission.author, "link_karma", 0)
        author_created = getattr(submission.author, "created_utc", submission.created_utc)
        account_age_days = (created_utc - datetime.fromtimestamp(author_created, tz=timezone.utc)).days
    else:
        author_karma = 0
        account_age_days = 0

    return Post(
        id=submission.id,
        text=text,
        source="reddit",
        created_utc=created_utc,
        score=submission.score,
        author_karma=author_karma,
        account_age_days=max(account_age_days, 0),
        url=f"https://reddit.com{submission.permalink}",
        subreddit=submission.subreddit.display_name,
    )


def fetch_recent_posts(
    subreddit: str,
    config: Optional[Config] = None,
    days_back: int = 7,
    limit: int = 500,
) -> list[Post]:
    """Fetch recent posts from a subreddit via PRAW.

    Args:
        subreddit: Subreddit name (without r/ prefix).
        config: Pipeline config with Reddit credentials.
        days_back: How many days back to fetch.
        limit: Max posts to fetch per subreddit.

    Returns:
        List of Post objects (not yet filtered).
    """
    if config is None:
        config = Config()

    reddit = praw.Reddit(
        client_id=config.reddit_client_id,
        client_secret=config.reddit_client_secret,
        user_agent=config.reddit_user_agent,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    sub = reddit.subreddit(subreddit)

    posts = []
    for submission in sub.new(limit=limit):
        created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created < cutoff:
            break
        if submission.removed_by_category:
            continue
        post = _normalize_submission(submission)
        posts.append(post)

    return posts
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_reddit_incremental.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingestion/reddit_incremental.py tests/test_reddit_incremental.py
git commit -m "feat: add PRAW incremental Reddit ingestion"
```

---

## Task 10: Integration — Full Ingestion Pipeline

**Files:**
- Create: `pipeline/ingestion/run.py`
- Create: `tests/test_ingestion_integration.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_ingestion_integration.py
"""Integration test for the full ingestion pipeline.

Tests the flow: fetch → filter → compute freshness → store.
Uses mocked sources but real filters and DB.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from pipeline.ingestion.run import ingest_all
from pipeline.ingestion.schema import Post
from pipeline.db import PostDB


def _fake_posts():
    """Mix of posts that should and shouldn't pass filters."""
    now = datetime.now(timezone.utc)
    return [
        # Should pass — authentic, recent
        Post(
            id="good1", text="I manually track everything in spreadsheets every single week",
            source="reddit", created_utc=now - timedelta(days=5),
            score=20, author_karma=200, account_age_days=90,
            url="http://example.com/1", subreddit="freelance",
        ),
        # Should fail — low karma
        Post(
            id="bad_karma", text="This is a valid length post but from a new account",
            source="reddit", created_utc=now - timedelta(days=2),
            score=5, author_karma=10, account_age_days=90,
            url="http://example.com/2", subreddit="freelance",
        ),
        # Should fail — too short
        Post(
            id="bad_short", text="lol",
            source="hn", created_utc=now - timedelta(days=1),
            score=50, author_karma=500, account_age_days=365,
            url="http://example.com/3", subreddit=None,
        ),
        # Should pass — HN post
        Post(
            id="good2", text="Is there a tool that automates contract generation for freelancers?",
            source="hn", created_utc=now - timedelta(days=10),
            score=45, author_karma=1000, account_age_days=730,
            url="http://example.com/4", subreddit=None,
        ),
    ]


@patch("pipeline.ingestion.run._fetch_from_all_sources")
def test_ingest_all_filters_and_stores(mock_fetch, tmp_path):
    mock_fetch.return_value = _fake_posts()

    db = PostDB(tmp_path / "test.db")
    stats = ingest_all(db=db)

    assert db.count() == 2  # Only good1 and good2 should pass
    assert stats["fetched"] == 4
    assert stats["stored"] == 2
    assert stats["filtered_out"] == 2

    # Verify freshness was computed
    posts = db.get_all_posts()
    for post in posts:
        assert post.freshness_score is not None
        assert 0.0 < post.freshness_score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingestion_integration.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.ingestion.run'`

- [ ] **Step 3: Implement run.py**

```python
# pipeline/ingestion/run.py
"""Orchestrates the full ingestion pipeline.

Flow: fetch from all sources → filter → compute freshness → store in DB.
"""

from typing import Optional

from config import Config
from pipeline.db import PostDB
from pipeline.ingestion.schema import Post
from pipeline.ingestion.filters import passes_authenticity_filter, compute_freshness_score
from pipeline.ingestion.arctic_shift import fetch_subreddit_posts
from pipeline.ingestion.reddit_incremental import fetch_recent_posts
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

    # PRAW — incremental Reddit (last 7 days)
    if config.reddit_client_id:
        for subreddit in config.target_subreddits:
            try:
                posts = fetch_recent_posts(subreddit=subreddit, config=config)
                all_posts.extend(posts)
            except Exception as e:
                print(f"[praw] Error fetching r/{subreddit}: {e}")

    # Hacker News
    for query in config.hn_query_terms:
        try:
            posts = fetch_hn_posts(query=query, config=config)
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_ingestion_integration.py -v
```

Expected: 1 passed

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass (schema, filters, db, arctic_shift, hacker_news, app_store, reddit_incremental, integration).

- [ ] **Step 6: Commit**

```bash
git add pipeline/ingestion/run.py tests/test_ingestion_integration.py
git commit -m "feat: add ingestion orchestrator with filter + freshness + storage"
```

---

## Task 11: Manual Verification Run

**Files:**
- Create: `scripts/verify_ingestion.py`

- [ ] **Step 1: Create verification script**

```python
# scripts/verify_ingestion.py
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
        print("\n✓ Phase 1 milestone reached: ≥10,000 posts stored.")
    else:
        print(f"\n✗ Need ≥10,000 posts. Currently at {db.count()}. Run again or expand sources.")

    db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Set up .env with real credentials**

Create `.env` (copy from `.env.example`) and fill in:
- `REDDIT_CLIENT_ID` — from https://www.reddit.com/prefs/apps (create a "script" type app)
- `REDDIT_CLIENT_SECRET` — same page
- `REDDIT_USER_AGENT` — e.g., `pain-point-engine/1.0 by /u/your_username`
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com/settings/keys

- [ ] **Step 3: Run verification**

```bash
python scripts/verify_ingestion.py
```

Expected output: row counts by source. If total < 10,000, run multiple times or increase `lookback_window_days` in config.

- [ ] **Step 4: Commit**

```bash
mkdir -p scripts
git add scripts/verify_ingestion.py
git commit -m "feat: add ingestion verification script"
```

---

## Phase 1 Completion Criteria

- [ ] All tests pass: `pytest tests/ -v`
- [ ] `data/posts.db` contains ≥10,000 posts
- [ ] `db.count_by_source()` shows posts from at least Reddit + HN
- [ ] Each stored post has a non-null `freshness_score`
- [ ] No posts with `account_age_days < 30` or `author_karma < 50` in DB
