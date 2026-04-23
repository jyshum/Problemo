"""Claude Sonnet batch labeling for classifier/data/labeled.jsonl.

Sends batches of 10 unlabeled posts to Claude Sonnet with the label guide
as system context and up to 20 few-shot examples from human labels.
Merges results with human labels into labeled.jsonl (~480 posts total).

Requires ANTHROPIC_API_KEY in .env.

Usage: PYTHONPATH=. .venv/bin/python scripts/label_with_claude.py
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from config import Config

load_dotenv()

LABEL_GUIDE_PATH = Path("classifier/label_guide.md")
PROGRESS_PATH = Path("classifier/data/label_progress.jsonl")
OUTPUT_PATH = Path("classifier/data/labeled.jsonl")

VALID_LABELS = {"WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"}

KEYWORD_FILTERS: dict[str, list[str]] = {
    "WORKFLOW_PAIN": ["manually", "spreadsheet", "every week", "no tool", "takes hours", "tedious", "time-consuming"],
    "TOOL_REQUEST": ["is there a", "is there an app", "does anyone know of", "someone should build", "looking for a tool", "looking for software", "recommend a tool", "any tools for"],
    "PRODUCT_COMPLAINT": ["broken", "missing feature", "crashes", "doesn't work", "terrible", "stopped working", "wish it had"],
}

SIGNAL_SUBREDDITS = {"smallbusiness", "Entrepreneur", "freelance", "webdev", "programming"}

CLAUDE_SAMPLE_COUNTS: dict[str, int] = {
    "WORKFLOW_PAIN": 75,
    "TOOL_REQUEST": 75,
    "PRODUCT_COMPLAINT": 75,
    "NOISE": 75,
}

REMOVED_MARKERS = {"[removed]", "[deleted]"}


def _is_usable(text: str) -> bool:
    return text.strip() not in REMOVED_MARKERS and len(text.strip()) >= 30


def build_prompt(posts: list[dict], examples: list[dict]) -> str:
    """Build the user message for Claude: few-shot examples + posts to label."""
    lines = []

    if examples:
        lines.append("## Few-shot examples\n")
        for ex in examples:
            lines.append(f"ID: {ex['id']}")
            lines.append(f"Text: {ex['text'][:300]}")
            lines.append(f"Label: {ex['label']}\n")

    lines.append("## Posts to label\n")
    for post in posts:
        lines.append(f"ID: {post['id']}")
        lines.append(f"Text: {post['text'][:400]}\n")

    lines.append(
        "\nReturn a JSON object with this exact structure:\n"
        '{"labels": [{"id": "<post_id>", "label": "<LABEL>"}, ...]}\n'
        "Valid labels: WORKFLOW_PAIN, TOOL_REQUEST, PRODUCT_COMPLAINT, NOISE\n"
        "Return JSON only, no other text."
    )
    return "\n".join(lines)


def parse_labels(response_text: str) -> dict[str, str]:
    """Parse Claude's JSON response → {id: label}. Returns {} on any error.

    Handles responses wrapped in markdown code blocks or with surrounding text.
    """
    import re
    text = response_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If still not starting with {, try to extract first {...} block
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
        return {
            item["id"]: item["label"]
            for item in data.get("labels", [])
            if item.get("label") in VALID_LABELS
        }
    except Exception:
        return {}


def load_human_labels() -> list[dict]:
    """Load human-labeled posts from label_progress.jsonl."""
    if not PROGRESS_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRESS_PATH} not found. Run scripts/label_posts.py first."
        )
    rows = []
    with open(PROGRESS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("label") in VALID_LABELS:
                rows.append({"id": row["id"], "text": row["text"], "label": row["label"]})
    return rows


def sample_for_claude(human_ids: set[str], config: Config) -> list[dict]:
    """Sample posts for Claude labeling (stratified, excluding human-labeled posts)."""
    import random
    from pipeline.db import PostDB

    db = PostDB(config.db_path)
    all_posts = db.get_all_posts()
    db.close()

    signal_posts = [p for p in all_posts if p.subreddit in SIGNAL_SUBREDDITS]
    hn_posts = [p for p in all_posts if p.source == "hn"]
    signal_pool = signal_posts + hn_posts

    result = []
    used_ids: set[str] = set(human_ids)

    for class_name in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT"]:
        keywords = KEYWORD_FILTERS[class_name]
        n = CLAUDE_SAMPLE_COUNTS[class_name]
        matched = [
            p for p in signal_pool
            if p.id not in used_ids
            and _is_usable(p.text)
            and any(kw.lower() in p.text.lower() for kw in keywords)
        ]
        random.seed(42 + list(CLAUDE_SAMPLE_COUNTS.keys()).index(class_name))
        random.shuffle(matched)
        batch = matched[:n]
        for p in batch:
            result.append({"id": p.id, "text": p.text, "suggested_class": class_name})
        used_ids.update(p.id for p in batch)

    # NOISE: random sample from all usable posts
    noise_pool = [
        p for p in all_posts
        if p.id not in used_ids and _is_usable(p.text)
    ]
    random.seed(99)
    random.shuffle(noise_pool)
    for p in noise_pool[:CLAUDE_SAMPLE_COUNTS["NOISE"]]:
        result.append({"id": p.id, "text": p.text, "suggested_class": "NOISE"})

    return result


def merge_labeled(human_rows: list[dict], claude_rows: list[dict], output_path: Path) -> None:
    """Write all labeled posts to output_path as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = human_rows + claude_rows
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps({"id": row["id"], "text": row["text"], "label": row["label"]}) + "\n")
    print(f"Wrote {len(all_rows)} labeled posts to {output_path}")


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        return

    config = Config()
    client = anthropic.Anthropic(api_key=api_key)
    label_guide = LABEL_GUIDE_PATH.read_text()
    model = config.dataset_labeling_model

    # Load human labels
    human_rows = load_human_labels()
    print(f"Loaded {len(human_rows)} human-labeled posts")

    # Build few-shot examples: up to 5 per class
    by_class: dict[str, list] = defaultdict(list)
    for row in human_rows:
        by_class[row["label"]].append(row)
    examples = []
    for label in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"]:
        examples.extend(by_class[label][:5])
    print(f"Using {len(examples)} few-shot examples ({len(by_class['WORKFLOW_PAIN'][:5])} WP, "
          f"{len(by_class['TOOL_REQUEST'][:5])} TR, {len(by_class['PRODUCT_COMPLAINT'][:5])} PC, "
          f"{len(by_class['NOISE'][:5])} N)")

    # Sample posts for Claude
    human_ids = {r["id"] for r in human_rows}
    to_label = sample_for_claude(human_ids, config)
    print(f"Sampling {len(to_label)} posts for Claude to label...")

    # Batch into groups of 10
    claude_rows = []
    batch_size = 10
    total_batches = (len(to_label) + batch_size - 1) // batch_size

    for i in range(0, len(to_label), batch_size):
        batch = to_label[i: i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}...", end=" ", flush=True)

        try:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                system=f"You are a precise text classifier. Use this label guide:\n\n{label_guide}",
                messages=[{"role": "user", "content": build_prompt(batch, examples)}],
            )
            labels = parse_labels(message.content[0].text)
        except Exception as e:
            print(f"ERROR: {e}")
            labels = {}

        for post in batch:
            if post["id"] in labels:
                post["label"] = labels[post["id"]]
                claude_rows.append(post)

        print(f"{len(labels)} labeled")
        time.sleep(0.5)

    print(f"\nClaude labeled {len(claude_rows)} posts")
    merge_labeled(human_rows, claude_rows, OUTPUT_PATH)

    # Final distribution
    from collections import Counter
    all_labels = [r["label"] for r in human_rows + claude_rows]
    print("\nFinal label distribution:")
    for label in ["WORKFLOW_PAIN", "TOOL_REQUEST", "PRODUCT_COMPLAINT", "NOISE"]:
        count = Counter(all_labels)[label]
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
