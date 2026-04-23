"""Run classifier on all posts in the DB and write predictions back.

Usage: PYTHONPATH=. .venv/bin/python scripts/run_inference.py
"""

from collections import Counter

from classifier.classifier_interface import Classifier
from config import Config
from pipeline.db import PostDB


def main():
    config = Config()
    db = PostDB(config.db_path)

    posts = db.get_all_posts()
    print(f"Loaded {len(posts)} posts from DB")

    clf = Classifier()
    print("Running predictions...")
    clf.predict_posts(posts, config=config)

    db.update_predictions(posts)
    print("Predictions written to DB")

    # Summary
    counts = Counter(p.predicted_class for p in posts)
    print(f"\nPrediction distribution:")
    for label in sorted(counts.keys()):
        print(f"  {label}: {counts[label]}")

    uncertain = sum(1 for p in posts if p.predicted_class == "UNCERTAIN")
    print(f"\nTotal: {len(posts)} | Uncertain: {uncertain} ({uncertain/len(posts)*100:.1f}%)")

    db.close()


if __name__ == "__main__":
    main()
