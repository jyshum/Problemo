import pytest
from datetime import datetime, timezone

from pipeline.ingestion.schema import Post


def make_post(id_, text):
    return Post(
        id=id_,
        text=text,
        source="reddit",
        created_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        score=10,
        author_karma=100,
        account_age_days=60,
        url=f"https://example.com/{id_}",
    )


def test_sample_by_keywords_returns_matching_posts():
    from scripts.sample_posts import sample_by_keywords

    posts = [
        make_post("1", "I manually track everything in spreadsheets every week"),
        make_post("2", "Great day today, loving life"),
        make_post("3", "takes hours to reconcile manually"),
    ]
    result = sample_by_keywords(posts, ["manually", "spreadsheet"], n=10)
    ids = {p.id for p in result}
    assert "1" in ids
    assert "3" in ids
    assert "2" not in ids


def test_sample_by_keywords_respects_n():
    from scripts.sample_posts import sample_by_keywords

    posts = [make_post(str(i), "manually do this every week") for i in range(20)]
    result = sample_by_keywords(posts, ["manually"], n=5)
    assert len(result) == 5


def test_sample_by_keywords_excludes_ids():
    from scripts.sample_posts import sample_by_keywords

    posts = [make_post(str(i), "manually every week") for i in range(10)]
    result = sample_by_keywords(posts, ["manually"], n=10, exclude_ids={"0", "1", "2"})
    ids = {p.id for p in result}
    assert "0" not in ids
    assert "1" not in ids


def test_sample_random_returns_n():
    from scripts.sample_posts import sample_random

    posts = [make_post(str(i), "some text here") for i in range(50)]
    result = sample_random(posts, n=10)
    assert len(result) == 10


def test_sample_random_excludes_ids():
    from scripts.sample_posts import sample_random

    posts = [make_post(str(i), "some text here") for i in range(20)]
    result = sample_random(posts, n=15, exclude_ids={"0", "1", "2", "3", "4"})
    ids = {p.id for p in result}
    for excluded in ["0", "1", "2", "3", "4"]:
        assert excluded not in ids
