# Pain Point Discovery Engine
**Version:** 4.0
**Type:** Personal research tool + ML portfolio project
**Goal:** Surface validated, unsolved market problems from online communities to inform what to build next.

---

## Context and Positioning

This is not a commercial product. It is a personal research tool with a rigorously documented ML core. The commercial space (GummySearch, PainHunt, PainOnSocial) is saturated — all closed-source with undisclosed scoring methods. The differentiation here is methodological rigor made public: labeled dataset, inter-annotator agreement scores, evaluation reports, and error analysis. None of the competitors publish this. That documentation is the portfolio artifact.

---

## Model Decisions — Primary and Fallback

This section is the authoritative reference for every model used in the pipeline. All modules must use the primary model unless explicitly overridden in `config.py`.

### Embedding Model
**PRIMARY: `BAAI/bge-m3`**
Chosen because: MTEB score ~68+ vs. all-MiniLM-L6-v2's 56.3. Handles full Reddit post lengths up to 8,192 tokens (posts vary widely in length). Supports dense, sparse, and multi-vector retrieval simultaneously. Free, self-hostable, MIT licensed. Runs via `FlagEmbedding` library.

**FALLBACK: `sentence-transformers/all-MiniLM-L6-v2`**
Use when: CPU-only machine, rapid prototyping, or BGE-M3 memory overhead is prohibitive. Acceptable for development runs. Not acceptable for final evaluation or published results.

---

### Clustering Framework
**PRIMARY: BERTopic**
Chosen because: BERTopic wraps UMAP + HDBSCAN internally but adds class-based TF-IDF (c-TF-IDF) on top, which automatically extracts representative keywords per cluster and produces human-readable topic labels without requiring an LLM call for every cluster. It also provides built-in LLM representation support, topic merging, outlier reduction, and reproducibility via `random_state`. Achieves 15–25% higher coherence scores than LDA on short-text datasets. Fully modular — embedding model, clustering algorithm, and representation model are all swappable.

**FALLBACK: Raw HDBSCAN + UMAP (manual pipeline)**
Use when: BERTopic installation fails or the pipeline requires more granular control over intermediate steps. Requires manual Claude Haiku labeling for every cluster — more expensive and slower.

---

### Classification Model — Phase 1
**PRIMARY: SetFit with `sentence-transformers/paraphrase-mpnet-base-v2` backbone**
Chosen because: Achieves strong performance with as few as 8–20 labeled examples per class. Appropriate for early phase when labeled dataset is small. Well-documented, stable HuggingFace library.

**COMPARISON (run alongside primary, document results): SetFit with `nomic-ai/modernbert-embed-base` backbone**
Research shows ModernBERT achieves 20–50% improvement in few-shot scenarios. Run this as a direct comparison against mpnet-base-v2 on the same labeled dataset. Results go in `eval_report/comparison.md`.

---

### Classification Model — Phase 2
**PRIMARY: `roberta-base` via HuggingFace `Trainer`**
Chosen because: Standard fine-tuning on a well-understood backbone. Educational value — comparing SetFit (few-shot contrastive) against standard fine-tuning (full gradient descent) on the same dataset is the portfolio artifact. Not expected to outperform SetFit at small data scale. Documents tradeoffs honestly.

**NOT a replacement for SetFit.** Both are used. SetFit is the production classifier. RoBERTa fine-tune is the comparison subject.

---

### Cluster Labeling LLM
**PRIMARY: Claude Haiku 4.5 via Anthropic API**
Chosen because: Short structured generation tasks (one-sentence problem statement from 5 posts) require reliable instruction-following more than frontier reasoning capability. Haiku 4.5 at $0.25/M input, $1.25/M output is cost-effective for this use case. API stability and consistent output format are more important than marginal cost savings. GPT-5 nano is cheaper ($0.05/M) but less reliable on structured outputs. Gemini free tier was cut to 20 RPD in December 2025 and is explicitly not used due to demonstrated fragility.

**FALLBACK: GPT-5 nano via OpenAI API**
Use when: Anthropic API is unavailable or cost reduction is critical at scale. Requires additional output validation since instruction-following on short structured prompts is less consistent.

**EXPLICITLY NOT USED: Gemini 2.5 Flash (free tier)**
Reason: Free tier cut from 250 to 20 RPD in December 2025 without notice. Building a personal research tool on a free tier with demonstrated quota instability creates unnecessary fragility. Excluded from the pipeline entirely.

---

### LLM-Assisted Dataset Labeling
**PRIMARY: Claude claude-sonnet-4-6 via Anthropic API**
Chosen because: Few-shot labeling of 300 posts requires higher reasoning quality than cluster naming. Sonnet 4.6 at $3/M input produces more reliable 4-class classifications than Haiku when given ambiguous examples. This is a one-time cost during dataset construction, not a recurring runtime cost.

**FALLBACK: Claude Haiku 4.5**
Use when: Budget is constrained. Accept lower accuracy on ambiguous cases — compensate by increasing manual spot-check percentage from 20% to 35%.

---

## Architecture Overview

Five sequential layers. Each layer is independent. The classifier is wrapped behind an interface so the model backend (SetFit or standard fine-tune) can be swapped without touching downstream code.

```
INGESTION → FILTERING → CLASSIFICATION → CLUSTERING → SCORING → OUTPUT
```

**Ingestion:** Multi-source. Reddit via Arctic Shift dumps (bulk historical) and PRAW (weekly incremental). Hacker News via Algolia API. App Store via `app-reviews`. All sources normalize to a single `Post` schema.

**Filtering:** Authenticity and freshness applied at ingestion time before storage. Removes bots, throwaway accounts, and stale content. Nothing downstream sees unfiltered data.

**Classification:** SetFit wrapped in a `Classifier` interface. Four classes: `WORKFLOW_PAIN`, `TOOL_REQUEST`, `PRODUCT_COMPLAINT`, `NOISE`. Only the first two pass to clustering. Interface is backend-agnostic — swapping to RoBERTa is a one-line config change.

**Clustering:** BGE-M3 embeddings → BERTopic (which internally runs UMAP + HDBSCAN + c-TF-IDF). Produces coherent problem statements with keyword representation. Claude Haiku labels the top clusters with a single precise sentence.

**Scoring:** Three sub-scores combined into a composite. Frequency (unique user count, log-scaled). Intensity (upvote signals, comment depth, emotional language density). Opportunity (ProductHunt cross-reference when API token configured, willingness-to-pay language detection). Defaults to WTP-only scoring when ProductHunt token is absent. Freshness decay applied as a multiplier.

**Output:** Local Streamlit dashboard. Ranked problem cards. One-click feedback writing to SQLite. Weekly cron job re-ingests fresh data and updates cluster scores.

---

## Data Sources

| Source | Method | Cost | Notes |
|--------|--------|------|-------|
| Reddit (historical) | Arctic Shift monthly dumps | Free | Offline `.zst` files — no API, no rate limits, no licensing risk |
| Reddit (incremental) | PRAW (Reddit API) | Free | 60 req/min free tier — sufficient for weekly incremental pulls |
| Hacker News | Algolia API at `hn.algolia.com/api` | Free | No auth required. 1,000 hit cap per query — paginate by 7-day date windows |
| App Store | `app-reviews` Python lib | Free | Handles App Store + Google Play, no API keys required |

**Target subreddits (v1):** r/freelance, r/smallbusiness, r/Entrepreneur, r/nursing, r/medicine, r/legaladvice, r/webdev, r/programming, r/Teachers, r/GradSchool

**HN query terms:** "is there a tool", "wish there was", "I manually", "takes hours", "no good solution", "Ask HN: How do you", "frustrated with"

---

## Post Schema

All sources normalize to one schema before storage. Downstream code never imports source-specific types.

**Fields:** `id`, `text` (title + body combined), `source` (reddit/hn/appstore), `created_utc`, `score`, `author_karma`, `account_age_days`, `url`, `subreddit` (nullable), `freshness_score` (computed at ingestion), `predicted_class` (set after classification), `confidence` (set after classification)

---

## Authenticity Filters

Applied at ingestion. Posts failing any filter are discarded and never stored.

- Account age ≥ 30 days
- Author karma ≥ 50
- Post not deleted or removed
- Text length ≥ 30 characters

For training data labeling only: verify author's last 10 posts span at least 2 subreddits. Prevents single-topic bot accounts from contaminating the labeled dataset.

---

## Freshness Scoring

Exponential decay with 45-day half-life applied to every post at ingestion. Freshness is a composite score multiplier — old posts still appear in clusters but contribute less weight. Any cluster with zero posts from the last 60 days is flagged `POTENTIALLY_STALE` in the dashboard.

---

## Classification System

### The Four Classes

**WORKFLOW_PAIN** — Person describes a repetitive task performed manually or inefficiently with no satisfying solution. Frustration is with a process, not a product. Signal phrases: "every week I", "I manually", "takes hours", "still using spreadsheets", "no tool for", "wish there was a way to".

**TOOL_REQUEST** — Person explicitly asks if a tool exists or requests one be built. Direct opportunity signal. Signal phrases: "is there an app that", "someone should build", "does a tool exist for".

**PRODUCT_COMPLAINT** — Person is unhappy with a specific named existing product. Low opportunity signal unless the complaint reveals a workflow gap.

**NOISE** — General venting, off-topic, not actionable.

### Classifier Interface — Critical Design Decision

A `Classifier` class wraps all model backends behind a single `predict(texts: list[str]) -> list[str]` method. Backend is specified at instantiation via `model_type` parameter (`"setfit"` or `"hf"`). All downstream pipeline code calls only `classifier.predict()`. Swapping backends is a one-line config change. No other file changes required.

### Phase 1 — SetFit

Uses contrastive learning on sentence pairs to fine-tune the backbone, then trains a logistic regression head. Strong performance with 8–20 labeled examples per class. Run both `paraphrase-mpnet-base-v2` and `modernbert-embed-base` backbones and compare — results documented in `eval_report/comparison.md`.

### Phase 2 — Standard Fine-Tuning

After Phase 1 is complete and labeled dataset reaches 400+ examples, add `roberta-base` standard fine-tuning as a comparison backend. One config change, no downstream refactoring. The three-way comparison (SetFit mpnet / SetFit ModernBERT / RoBERTa fine-tuned) is the portfolio artifact.

### Labeling Methodology

Write `classifier/label_guide.md` before labeling a single post. Define each class with: precise definition, 5 positive examples, 3 negative examples, 3 explicit edge cases. Label 200 posts manually. Use Claude Sonnet 4.6 few-shot to label 300 more. Spot-check 60 manually.

Measure inter-annotator agreement: re-label 50 posts two weeks after initial labeling without referring to first labels. Compute Cohen's Kappa. Target ≥ 0.70. Publish the score.

### Evaluation

Stratified 80/20 train/test split. Test set locked immediately after creation. Publish: per-class precision, recall, F1, macro-average F1, confusion matrix (image), error analysis (qualitative). Target macro F1 ≥ 0.75.

---

## Clustering Layer

Only `WORKFLOW_PAIN` and `TOOL_REQUEST` posts enter clustering.

**Primary pipeline:** BGE-M3 embeddings → BERTopic (UMAP + HDBSCAN + c-TF-IDF internally) → Claude Haiku 4.5 for final one-sentence problem statement per cluster.

**BERTopic configuration guidance:**
- Set `random_state=42` on the UMAP model inside BERTopic for reproducibility — UMAP is stochastic by default
- Start `min_topic_size` (BERTopic's wrapper around HDBSCAN's `min_cluster_size`) at 10
- Use `nr_topics="auto"` to merge near-duplicate topics automatically
- Use `reduce_outliers()` after fitting to reassign unclustered posts via semantic similarity rather than discarding them
- Validate by manually reading top 20 clusters — each should describe a coherent, distinct problem

**Why BERTopic over raw HDBSCAN:** BERTopic adds c-TF-IDF keyword extraction on top of HDBSCAN, producing automatic topic representations that are already human-readable. This reduces reliance on LLM labeling calls (only needed for the final polished one-sentence label, not initial cluster description). It also provides built-in topic merging, outlier handling, and topic hierarchy — features that would require custom code with raw HDBSCAN.

---

## Scoring Layer

**Frequency (0–10):** Log-scaled unique user count per cluster. Caps at 10.

**Intensity (0–10):** Composite of average post upvote score, average comment depth, and emotional language density (fraction of posts containing words from a predefined high-frustration lexicon: hate, nightmare, ridiculous, unbearable, manually, broken, waste, etc.).

**Opportunity (0–10):** When ProductHunt API token is configured: API check by cluster keyword — well-reviewed solutions lower the score, absent or poorly-reviewed ones raise it. Solution gap weighted 70%, WTP signal 30%. When token is absent: score based solely on willingness-to-pay language detection ("I'd pay", "worth paying", "would pay for"), returning neutral 5.0 as baseline with WTP signals boosting above. Graceful degradation — pipeline never crashes on missing token.

**Composite:** Frequency 25% + Intensity 25% + Opportunity 50%. Opportunity weighted double because frequency and intensity without commercial signal produces interesting but unbuildable output.

**Freshness multiplier:** Average freshness score of cluster's constituent posts applied to composite. Fully stale clusters (all posts > 6 months old) have composite score halved.

---

## Output Layer

Local Streamlit dashboard. Not public-facing in v1.

**Per cluster card:** Problem statement (Claude Haiku generated), composite score with sub-score breakdown, source distribution, 3 example posts with links, BERTopic keyword representation, freshness indicator, staleness flag, ProductHunt check summary, WTP language examples.

**Filters:** Minimum composite score, source, freshness, feedback status.

**Feedback:** Thumbs up / thumbs down per card writes to local SQLite `feedback.db`. After 4 weeks, compute real-world precision: fraction of high-scoring clusters rated useful. Publish in README.

**Scheduling:** APScheduler weekly cron. Ingests fresh posts, filters, classifies, re-clusters, updates scores.

---

## Repository Structure

```
pain-point-engine/
├── classifier/                    # Primary — lead with this in all communications
│   ├── README.md
│   ├── label_guide.md
│   ├── train_setfit.py            # Runs both mpnet-base-v2 and modernbert-embed-base
│   ├── train_hf.py                # Phase 2: roberta-base standard fine-tuning
│   ├── evaluate.py
│   ├── classifier_interface.py    # Classifier wrapper — single predict() method
│   ├── data/
│   │   ├── labeled.jsonl
│   │   └── split.py
│   └── eval_report/
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── kappa_score.txt
│       ├── comparison.md          # SetFit mpnet / SetFit ModernBERT / RoBERTa
│       └── error_analysis.md
├── pipeline/
│   ├── ingestion/
│   │   ├── arctic_shift.py        # Bulk historical Reddit data
│   │   ├── reddit_incremental.py  # PRAW — weekly fresh pulls (last 7 days)
│   │   ├── hacker_news.py
│   │   ├── app_store.py           # Uses app-reviews package
│   │   ├── filters.py
│   │   └── schema.py              # Post dataclass — single source of truth
│   ├── clustering/
│   │   ├── embed.py               # BGE-M3 via FlagEmbedding
│   │   ├── bertopic_model.py      # BERTopic configuration and fitting
│   │   └── label_clusters.py      # Claude Haiku 4.5 final labeling
│   └── scoring/
│       ├── frequency.py
│       ├── intensity.py
│       ├── opportunity.py
│       └── composite.py
├── dashboard/
│   ├── app.py
│   └── feedback.db               # gitignored
├── data/
│   ├── raw/                      # gitignored
│   └── processed/                # gitignored
├── scheduler.py
├── config.py                     # All model choices and tunable values here
├── requirements.txt
└── README.md
└── .env                          # API Keys
```

---

## Configuration

All model choices and tunable values live in `config.py`. No magic numbers or model names hardcoded in source files. Changing the embedding model, clustering parameters, or LLM backend requires only a change to `config.py`.

Key configuration groups:

- **Models:** embedding model name, embedding fallback name, setfit backbone, hf backbone, labeling llm model, dataset labeling llm model
- **Ingestion:** target subreddits, HN query terms, app IDs, lookback window days
- **Filtering:** min account age, min karma, min text length
- **Freshness:** half-life days, staleness threshold days
- **Classification:** active model type (`"setfit"` or `"hf"`), model path, confidence threshold
- **Clustering:** BERTopic min topic size, UMAP random state, nr topics strategy, outlier reduction strategy
- **Scoring:** frequency cap, intensity lexicon path, opportunity weights, composite weights, producthunt_api_token (empty string = degraded mode)
- **Scheduling:** cron interval, max posts per run per source

---

## Build Phases

### Phase 1 — Data Foundation (Weeks 1–2)
Objective: Clean stored data from all three sources before any ML work begins.

- Download Arctic Shift data for target subreddits via pre-filtered search API (arctic-shift.photon-reddit.com) — not full monthly dumps
- Implement `Post` schema in `pipeline/ingestion/schema.py`
- Build ingestion modules for all three sources — all output `Post` objects
- Implement authenticity and freshness filters in `filters.py`
- Set up PRAW for incremental weekly Reddit pulls (last 7 days per target subreddit)
- Confirm ≥ 10,000 authentic posts stored before proceeding

Milestone: `data/raw/` populated. Row counts by source documented.

---

### Phase 2 — Labeling and SetFit Classifier (Weeks 3–5)
Objective: Working classifier with published, reproducible evaluation.

- Write `classifier/label_guide.md` before any labeling begins
- Manually label 200 posts (target 50 per class, minimum 30)
- Use Claude Sonnet 4.6 few-shot to label 300 more — spot-check 60 manually
- Implement `classifier_interface.py` with `Classifier` wrapper
- Implement `train_setfit.py` — runs both `paraphrase-mpnet-base-v2` and `modernbert-embed-base`
- Stratified 80/20 train/test split — lock test set immediately
- Train both SetFit backbones — document which performs better
- Evaluate on held-out test set — classification report and confusion matrix
- Re-label 50 posts two weeks later — compute Cohen's Kappa
- Write `eval_report/error_analysis.md`
- Plug `Classifier` into pipeline — verify end-to-end flow

Milestone: `classifier/eval_report/` complete and public. Pipeline runs end-to-end.

---

### Phase 3 — Standard Fine-Tuning Comparison (Week 5, after Phase 2 complete)
Objective: Add RoBERTa backend. Complete three-way comparison.

- Implement `train_hf.py` using HuggingFace `Trainer` with `roberta-base`
- Train on same labeled dataset and split as SetFit
- Populate `eval_report/comparison.md` — all three models, F1, training time, inference cost, error overlap
- Set default backend in `config.py` (expected: best SetFit backbone)

Milestone: Three-way comparison documented. Default backend confirmed.

---

### Phase 4 — Clustering and Scoring (Weeks 6–7)
Objective: Coherent, scored problem clusters worth investigating.

- Implement BGE-M3 embedding in `pipeline/clustering/embed.py` using `FlagEmbedding`
- Implement BERTopic pipeline in `bertopic_model.py` with custom UMAP (random_state=42) and HDBSCAN
- Tune `min_topic_size` — validate by reading top 20 clusters manually
- Apply `reduce_outliers()` after fitting
- Implement Claude Haiku 4.5 final labeling in `label_clusters.py`
- Implement all three scoring modules and composite
- Integrate ProductHunt API cross-reference into opportunity scoring
- Implement staleness flag logic

Milestone: `data/processed/ranked_clusters.json` — top 50 clusters with full metadata.

---

### Phase 5 — Dashboard and Feedback Loop (Week 8)
Objective: A tool used weekly for the next month.

- Build Streamlit dashboard with all card fields and filters
- Display BERTopic keyword representation alongside Claude Haiku label per card
- Implement feedback mechanism writing to `feedback.db`
- Implement `scheduler.py` weekly cron
- Write technical blog post: model choices, evaluation findings, what the output revealed
- Begin using dashboard

Milestone: Dashboard running. Cron running. Feedback initialized. Blog post published.

---

## Tech Stack

| Component | Primary | Fallback | Notes |
|-----------|---------|----------|-------|
| Reddit bulk | Arctic Shift dumps | — | No API, no rate limits |
| Reddit incremental | PRAW | — | 60 req/min free tier; weekly pulls only |
| Hacker News | Algolia API | Firebase HN API | Algolia has better search |
| App Store | app-reviews | app-store-web-scraper | Handles App Store + Google Play |
| Storage | SQLite | — | Zero config |
| Embedding | BGE-M3 (FlagEmbedding) | all-MiniLM-L6-v2 | BGE-M3 for production; MiniLM for CPU-only dev runs |
| Clustering | BERTopic | Raw HDBSCAN + UMAP | BERTopic for production; raw HDBSCAN if install fails |
| Classification P1 | SetFit (paraphrase-mpnet-base-v2) | SetFit (modernbert-embed-base) | mpnet-base-v2 is primary; ModernBERT is comparison |
| Classification P2 | roberta-base (HuggingFace Trainer) | — | Phase 2 comparison only |
| Cluster labeling | Claude Haiku 4.5 | GPT-5 nano | Haiku for reliability; nano as budget fallback |
| Dataset labeling | Claude Sonnet 4.6 | Claude Haiku 4.5 | Sonnet for quality; Haiku if budget constrained |
| Dashboard | Streamlit | — | Fastest local UI |
| Scheduling | APScheduler | — | Simple in-process cron |
| Opportunity check | ProductHunt API v2 | WTP-only scoring | Requires OAuth token; degrades gracefully without it |

---

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| Arctic Shift dumps | $0 |
| HN Algolia API | $0 |
| App Store scraper | $0 |
| BGE-M3 (self-hosted) | $0 |
| BERTopic | $0 |
| SetFit training | $0 |
| Claude Haiku 4.5 (cluster labeling, ~50 clusters × 100 output tokens/week) | < $0.01 |
| Claude Sonnet 4.6 (dataset labeling, one-time 300 posts) | ~$0.50 total |
| Hosting (local only in v1) | $0 |
| **Total ongoing** | **< $1/month** |

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Reddit locks down Arctic Shift | PRAW incremental + HN + App Store sustain pipeline in degraded mode |
| ProductHunt API token unavailable | Opportunity score degrades to WTP-only with neutral 5.0 baseline |
| BGE-M3 too slow on CPU | Switch to `all-MiniLM-L6-v2` in `config.py` — one line change |
| BERTopic install conflicts | Fall back to raw HDBSCAN + UMAP pipeline |
| SetFit macro F1 < 0.70 | Switch classifier backend to Claude few-shot. Document failure — still an artifact. |
| Fine-tuning underperforms SetFit | Expected at small data scale. Document it. This is the educational finding. |
| Haiku API unavailable | Switch to GPT-5 nano in `config.py` |
| Scope creep past 8 weeks | Hard rule: ship dashboard at end of Week 8 regardless. Ship then iterate. |

---

## Success Criteria

**Technical**
- Macro F1 ≥ 0.75 on held-out test set
- Cohen's Kappa ≥ 0.70 on re-labeled overlap set
- Three-way model comparison documented with honest analysis
- Multi-source ingestion operational across all three sources
- Top 20 BERTopic clusters coherent on manual inspection

**Portfolio**
- `classifier/` is self-contained, readable, and leads the README
- Full evaluation report published — classification report, confusion matrix, Kappa, error analysis, model comparison
- One technical blog post covering model selection rationale and evaluation findings
- Fluent on all design decisions and tradeoffs without notes

**Real-world**
- At least one output cluster leads to a discovery interview
- At least five discovery interviews conducted from pipeline output
- Pipeline output influences what gets built next

---

## Interview Narrative

> "I built a multi-source pain point discovery engine that classifies Reddit, Hacker News, and App Store content into four opportunity-detection classes using a custom-labeled dataset. I chose BGE-M3 for embeddings after benchmarking against all-MiniLM-L6-v2 on MTEB, and BERTopic over raw HDBSCAN because of its built-in c-TF-IDF topic representation. I wrote a formal labeling guide, measured inter-annotator agreement with Cohen's Kappa, and published a full evaluation report. I compared SetFit with two backbones against standard RoBERTa fine-tuning — the comparison showed the data-efficiency tradeoffs clearly. I built it to find my next project."

---

## What This Is Not

- Not a commercial product competing with PainHunt or PainOnSocial
- Not a real-time system — weekly batch is sufficient
- Not a replacement for talking to real users — it surfaces where to look, not what to build
- Not optimized for external users in v1

---

*The real goal: build a machine that finds real problems. Use it to find one. Build that one. Get hired doing it.*
