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
