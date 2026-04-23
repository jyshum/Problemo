import json
import pytest


def test_parse_labels_returns_dict_of_id_to_label():
    from scripts.label_with_claude import parse_labels

    response = json.dumps({
        "labels": [
            {"id": "abc123", "label": "WORKFLOW_PAIN"},
            {"id": "def456", "label": "NOISE"},
        ]
    })
    result = parse_labels(response)
    assert result == {"abc123": "WORKFLOW_PAIN", "def456": "NOISE"}


def test_parse_labels_handles_malformed_json():
    from scripts.label_with_claude import parse_labels

    result = parse_labels("not valid json at all")
    assert result == {}


def test_parse_labels_handles_missing_keys():
    from scripts.label_with_claude import parse_labels

    result = parse_labels(json.dumps({"something": "else"}))
    assert result == {}


def test_parse_labels_filters_invalid_labels():
    from scripts.label_with_claude import parse_labels

    response = json.dumps({
        "labels": [
            {"id": "abc", "label": "WORKFLOW_PAIN"},
            {"id": "def", "label": "INVALID_CLASS"},
        ]
    })
    result = parse_labels(response)
    assert "abc" in result
    assert "def" not in result


def test_build_prompt_includes_post_text():
    from scripts.label_with_claude import build_prompt

    posts = [{"id": "x1", "text": "I manually copy data every week"}]
    examples = []
    prompt = build_prompt(posts, examples)
    assert "I manually copy data every week" in prompt
    assert "x1" in prompt


def test_build_prompt_includes_examples():
    from scripts.label_with_claude import build_prompt

    posts = [{"id": "x1", "text": "some text"}]
    examples = [{"id": "e1", "text": "example text", "label": "WORKFLOW_PAIN"}]
    prompt = build_prompt(posts, examples)
    assert "WORKFLOW_PAIN" in prompt
    assert "example text" in prompt
