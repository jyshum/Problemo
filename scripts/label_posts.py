"""Resume-safe CLI labeling tool for classifier/data/sampled.jsonl.

Press 1-4 to label, s to skip, q to quit. Progress saves after each label.
Run from project root.

Usage: PYTHONPATH=. .venv/bin/python scripts/label_posts.py
"""

import json
import sys
from pathlib import Path

SAMPLED_PATH = Path("classifier/data/sampled.jsonl")
PROGRESS_PATH = Path("classifier/data/label_progress.jsonl")

LABELS = {
    "1": "WORKFLOW_PAIN",
    "2": "TOOL_REQUEST",
    "3": "PRODUCT_COMPLAINT",
    "4": "NOISE",
}


def load_sampled(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_progress(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def count_labeled(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("label") is not None)


def main():
    if not SAMPLED_PATH.exists():
        print(f"ERROR: {SAMPLED_PATH} not found. Run scripts/sample_posts.py first.")
        sys.exit(1)

    rows = load_sampled(SAMPLED_PATH)

    # Merge any existing progress
    if PROGRESS_PATH.exists():
        progress = {r["id"]: r for r in load_sampled(PROGRESS_PATH)}
        for row in rows:
            if row["id"] in progress:
                row["label"] = progress[row["id"]].get("label")

    total = len(rows)
    already_labeled = count_labeled(rows)
    print(f"\n=== Pain Point Labeling Tool ===")
    print(f"Total: {total} | Already labeled: {already_labeled} | Remaining: {total - already_labeled}")
    print(f"\nKeys: 1=WORKFLOW_PAIN  2=TOOL_REQUEST  3=PRODUCT_COMPLAINT  4=NOISE  s=skip  q=quit\n")

    for i, row in enumerate(rows):
        if row.get("label") is not None:
            continue

        print(f"--- [{i+1}/{total}] ({row['source']}) ---")
        print(row["text"][:600])
        print(f"\nURL: {row['url']}")
        print()

        while True:
            key = input("Label (1/2/3/4/s/q): ").strip().lower()
            if key == "q":
                save_progress(rows, PROGRESS_PATH)
                print(f"\nSaved. {count_labeled(rows)}/{total} labeled. Resume anytime.")
                sys.exit(0)
            elif key == "s":
                break
            elif key in LABELS:
                row["label"] = LABELS[key]
                save_progress(rows, PROGRESS_PATH)
                print(f"  → {LABELS[key]}\n")
                break
            else:
                print("  Invalid key. Use 1, 2, 3, 4, s, or q.")

    save_progress(rows, PROGRESS_PATH)
    labeled = count_labeled(rows)
    print(f"\nDone! {labeled}/{total} labeled.")
    if labeled < total:
        print(f"  {total - labeled} posts skipped.")
    print(f"\nProgress saved to {PROGRESS_PATH}")
    print("Next step: review progress, then run scripts/label_with_claude.py")


if __name__ == "__main__":
    main()
