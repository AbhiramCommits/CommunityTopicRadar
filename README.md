# CommunityTopicRadar

CommunityTopicRadar is a community-engagement analytics pipeline for program owners who
need to understand a public developer Q&A community without reading thousands of threads.
It scrapes real questions and answers from the Stack Exchange API, stores them in
PostgreSQL or SQLite, and produces trending-topic rankings, a content-gap backlog, member
segments, engagement anomalies, and plain-language topic briefs. A five-tab Streamlit
dashboard and a Power BI-ready star schema make the results usable by non-technical
stakeholders.

## Architecture

```text
Stack Exchange API / cached HTML fallback
        |
        v
scrape.py  --->  data/raw + data/cache
        |
        v
ingest.py  --->  PostgreSQL  (SQLite fallback)
        |             |
        |             +--> engagement_daily
        v
analysis/  --->  TF-IDF + NMF topics, tag gaps, K-Means/DBSCAN segments,
        |         robust-z + IsolationForest anomalies
        v
brief.py   --->  OpenAI briefs or deterministic offline template
        |
        v
export.py  --->  star-schema CSVs + DAX + SQL views
        |
        v
dashboard.py  --->  Streamlit executive dashboard (5 tabs)
```

| Module | Responsibility |
|---|---|
| `src/ctr/config.py` | Reads `DATABASE_URL`, `OPENAI_API_KEY`, site and page settings into a `Settings` dataclass. |
| `src/ctr/models.py` | SQLAlchemy models for posts, answers, tags, authors and daily engagement. |
| `src/ctr/db.py` | Engine creation, schema initialization, sessions and portable SQLite/PostgreSQL upserts. |
| `src/ctr/scrape.py` | Polite cached Stack Exchange client, raw-payload replay, timestamp/HTML normalization and HTML fallback parser. |
| `src/ctr/ingest.py` | Normalizes posts/answers, upserts entities, rebuilds daily engagement and writes ingest statistics. |
| `src/ctr/cli.py` | `ingest`, `analyze`, `brief`, `export` and `dashboard` commands. |
| `src/ctr/analysis/topics.py` | TF-IDF, week-over-week trending terms, NMF topic selection by reconstruction-error elbow. |
| `src/ctr/analysis/content_gap.py` | Tag-level demand, answer supply and ranked content-gap backlog. |
| `src/ctr/analysis/segments.py` | Author feature matrix, silhouette-selected K-Means, DBSCAN comparison and human segment labels. |
| `src/ctr/analysis/anomaly.py` | 28-day rolling median/MAD flags, IsolationForest comparison and precision/recall agreement. |
| `src/ctr/analysis/pipeline.py` | Loads the warehouse, runs all four analyses and writes report artifacts. |
| `src/ctr/brief.py` | Prompt construction, OpenAI briefs with token cap/retry, deterministic fallback and response cache. |
| `src/ctr/export.py` | Writes the six-table star schema, Power BI measures and PostgreSQL views. |
| `src/ctr/dashboard.py` | Five-tab Streamlit dashboard over database/export artifacts. |

## Data

The measured build used the public **Cross Validated** community
(`stats.stackexchange.com`, Stack Exchange site slug `stats`):

- Source: Stack Exchange API 2.3 questions with `filter=withbody`, plus answers in the same creation window.
- Date range: **2024-09-23 to 2026-11-05** (774 calendar days, 773 days with engagement).
- Posts: **6,500**.
- Answers: **7,828**.
- Authors: **5,025**.
- Tags: **1,204**.
- Post-tag links: **21,153**.
- Daily engagement rows: **773**.
- Raw API pages: **65 question pages** and **79 answer pages**.

The full raw dump stays untracked. `data/sample/posts_sample.json` (200 posts) and
`data/sample/answers_sample.json` (184 answers) are committed, and
`python -m ctr.cli ingest --offline` replays `data/raw` when it exists or falls back to the
committed sample, so CI and a fresh checkout run without network access. The sample is
drawn from the most recent 200 posts and is not a substitute for the full measured dataset.

## How to run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `DATABASE_URL` to a PostgreSQL URL when you want the Postgres path. The default is
`sqlite:///data/community.db`. Never commit `.env`; `OPENAI_API_KEY` is read from the
environment only.

```bash
python -m ctr.cli ingest             # live API, cached and saved under data/raw
python -m ctr.cli ingest --offline   # raw replay, or committed sample if raw is absent
python -m ctr.cli analyze
python -m ctr.cli brief              # OpenAI if configured, template fallback otherwise
python -m ctr.cli export
streamlit run src/ctr/dashboard.py
```

PostgreSQL:

```bash
createdb community
psql "$DATABASE_URL" -f sql/schema.sql
DATABASE_URL=postgresql+psycopg2://user:pass@localhost/community python -m ctr.cli ingest
```

Power BI: import `exports/dim_topic.csv`, `dim_author.csv`, `dim_date.csv`,
`fact_post.csv`, `fact_engagement_daily.csv` and `fact_anomaly.csv`, then relate
`topic_id`, `author_id` and `date_key`. The exact DAX measures are in
`exports/powerbi_measures.md`; equivalent PostgreSQL views are in `sql/views.sql`.

### Make targets

```bash
make setup       # create .venv and install pinned dependencies
make all         # ingest --offline, analyze, brief, export
make test        # pytest with the 85% coverage gate
make lint        # ruff
make dashboard   # streamlit run src/ctr/dashboard.py
```

`make all` works from a clean checkout: with no `data/raw` present it replays the
committed sample, and with raw pages present it replays the full measured dataset. The
scheduled-run design, cron example and weekly GitHub Actions workflow are in
`scripts/schedule.md`, `scripts/cron.example` and `.github/workflows/scheduled-scrape.yml`.


## Results (measured)

All values below come from the committed artifacts in `reports/` and `exports/`, generated
from the full 6,500-post dataset.

| Result | Measured value |
|---|---|
| Posts ingested | 6,500 |
| Answers ingested | 7,828 |
| Authors | 5,025 |
| Tags | 1,204 |
| Days of engagement history | 773 |
| Chosen NMF topic count | 9 |
| Top trending term | effects, +600.0% week over week |
| Trending terms 2–5 | random effects +100.0%, statistical +500.0%, glmmtmb +100.0%, age +100.0% |
| Top content gap | lasso: 47.2% answer rate, gap score 0.394 |
| Content gaps 2–5 | difference-in-difference (30.0%, 0.385), classification (46.4%, 0.378), distributions (56.4%, 0.364), mgcv (55.7%, 0.364) |
| K-Means clusters / silhouette | 8 / 0.394 |
| DBSCAN silhouette | 0.838 |
| Segment sizes | 4,765 lurker, 246 answerer, 9 asker, 5 champion |
| Anomalies | 82 metric-day rows across 63 days |
| IsolationForest vs robust-z | precision 0.333, recall 0.170 |
| Tests | 45 passed |
| Coverage on `src/ctr` excluding `dashboard.py` | 86.18% |
| Full offline pipeline wall-clock (`ingest` → `analyze` → `brief` → `export`) | 33.92 seconds |
| Clean-checkout `make all` wall-clock (sample fallback) | 18.18 seconds |

Trending terms are ranked by the week-over-week two-proportion z-score. A `+100.0%` change
means the term had no occurrences in the prior complete week. `+600.0%` means it moved from
one prior-week post to seven current-week posts.

![Dashboard overview](reports/screenshots/dashboard_overview.png)

![Content gaps](reports/screenshots/dashboard_gaps.png)

One generated brief, verbatim:

> distribution / probability / random is a recurring community topic with 1289 posts and 58% of them answered. Question volume fell week over week, and the typical post receives 155 views. Program owners should commission a short plain-language explainer and ask two experienced volunteers to watch new questions on it.

The full set is in `reports/topic_briefs.md`. It was generated with the deterministic
offline template because no `OPENAI_API_KEY` was present in this build. With a key set, the
same pipeline uses `gpt-4o-mini`, caps output at 180 tokens, retries once, and caches results
in `data/cache/briefs.json`.

## Limitations and next steps

- **Scraping rate limits.** The anonymous Stack Exchange API caps page depth at 25 and the
  daily request quota is limited. The client handles this by restating with `todate` windows,
  caching every request and enforcing a one-second delay. Larger collections need an API key,
  a queue and incremental date-window scheduling.
- **Single-site scope.** Results describe Cross Validated only. Tag vocabularies, answer
  culture and moderation rules differ across communities, so the same dashboard should not be
  read as a cross-platform benchmark without running the pipeline per site.
- **Observed participants only.** Authors are people who asked or answered in the scrape
  window. Silent lurkers cannot be counted from this API; the “lurker” segment means
  low-activity one-question participants, not zero-activity visitors.
- **Topic-model stability.** NMF is seeded and the k is chosen by a reconstruction-error
  elbow, but small weekly volumes make short-window keyword rankings noisy and k can shift
  when the scrape window moves. Topic labels should be reviewed by a domain owner before
  program decisions.
- **What changes at 3M users.** Ingestion becomes an incremental stream with deduplication
  and backfill jobs; analytics move from a single process to partitioned/columnar storage;
  clustering and topic fitting need sampled or distributed implementations; caching,
  quota management and moderation/bot filtering become first-class concerns; dashboard
  queries need pre-aggregated marts and row-level access controls.
