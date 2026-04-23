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
    dataset_labeling_model: str = "claude-sonnet-4-6"

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
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
