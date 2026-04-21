# Pain Point Discovery Engine — Design Document

**Date:** 2026-04-20
**Status:** Approved
**Source spec:** `pain-point-pipeline-v4.md` (corrected)

---

## Summary

Personal research tool that surfaces validated, unsolved market problems from Reddit, Hacker News, and App Store reviews. ML portfolio artifact with published evaluation methodology. Not a commercial product.

---

## Environment Constraints

- **Machine:** M2 Mac, Apple Silicon
- **Acceleration:** MPS (Metal Performance Shaders) via PyTorch. No CUDA.
- **BGE-M3:** `use_fp16=False` (fp16 unstable on MPS)
- **SetFit:** MPS supported natively via HuggingFace Transformers
- **UMAP/HDBSCAN:** CPU-bound, performant on M2
- **Package management:** venv + pip, pinned exact versions in `requirements.txt`
- **Dev dependencies:** Separate `requirements-dev.txt`

---

## Architecture

```
INGESTION → FILTERING → CLASSIFICATION → CLUSTERING → SCORING → OUTPUT
```

Five sequential layers, each independent. Classifier wrapped behind a `Classifier` interface for backend swapping. All configuration in `config.py`.

---

## Key Design Decisions

### Data Sources
| Source | Method | Notes |
|--------|--------|-------|
| Reddit (historical) | Arctic Shift pre-filtered search API | Subreddit-filtered downloads, not full dumps |
| Reddit (incremental) | PRAW | 60 req/min, weekly pulls, last 7 days |
| Hacker News | Algolia API | Paginate by 7-day windows |
| App Store | `app-reviews` package | App Store + Google Play |

### Models
| Role | Primary | Fallback |
|------|---------|----------|
| Embedding | BGE-M3 (FlagEmbedding, fp16=False) | all-MiniLM-L6-v2 |
| Clustering | BERTopic | Raw HDBSCAN + UMAP |
| Classification | SetFit (paraphrase-mpnet-base-v2) | SetFit (modernbert-embed-base) |
| Cluster labeling | Claude Haiku 4.5 | GPT-5 nano |
| Dataset labeling | Claude Sonnet 4.6 | Claude Haiku 4.5 |

### Scoring — Graceful Degradation
- Opportunity score requires ProductHunt API token (OAuth)
- Without token: returns neutral 5.0 baseline, WTP-only scoring
- Pipeline never crashes on missing token

### Classification — 4 Classes
`WORKFLOW_PAIN`, `TOOL_REQUEST`, `PRODUCT_COMPLAINT`, `NOISE`
Only first two pass to clustering.

---

## Execution Order

### Step 0 — Project Bootstrap
1. Create venv, pin all dependencies
2. Set up `.env`: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `ANTHROPIC_API_KEY`
3. Optional `.env`: `OPENAI_API_KEY`, `PRODUCTHUNT_API_TOKEN`
4. Project skeleton: directories, `config.py`, `schema.py`
5. `git init`, `.gitignore`

### Step 1 — Phase 1: Data Foundation
- Arctic Shift pre-filtered API for target subreddits
- Ingestion modules (all sources → `Post` schema)
- Authenticity + freshness filters
- PRAW incremental setup
- Gate: ≥10,000 authentic posts stored

### Step 2 — Phase 2: Labeling + SetFit
- `label_guide.md` first
- 200 manual labels → 300 Claude Sonnet labels → 60 spot-checked
- `classifier_interface.py` with `Classifier` wrapper
- Train both SetFit backbones, evaluate
- Cohen's Kappa on 50 re-labeled posts
- Gate: macro F1 ≥ 0.75

### Step 3 — Phase 3: RoBERTa Comparison
- `train_hf.py` with `roberta-base`
- Three-way comparison documented
- Gate: `eval_report/comparison.md` complete

### Step 4 — Phase 4: Clustering + Scoring
- BGE-M3 embedding (fp16=False on MPS)
- BERTopic with UMAP(random_state=42) + HDBSCAN
- Claude Haiku cluster labeling
- All scoring modules + composite
- ProductHunt integration (graceful degradation)
- Gate: top 50 ranked clusters with full metadata

### Step 5 — Phase 5: Dashboard + Feedback
- Streamlit dashboard
- Feedback to SQLite
- APScheduler weekly cron
- Gate: dashboard running, cron running

---

## Success Criteria

- Macro F1 ≥ 0.75
- Cohen's Kappa ≥ 0.70
- Three-way model comparison documented
- Top 20 BERTopic clusters coherent on manual inspection
- At least one output cluster leads to a discovery interview

---

## What This Is Not

- Not a commercial product
- Not real-time (weekly batch)
- Not a replacement for user interviews
- Not optimized for external users in v1
