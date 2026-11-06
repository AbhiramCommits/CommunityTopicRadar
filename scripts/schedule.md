# Scheduled scrape design

The pipeline is designed to run as a weekly batch job rather than a request-time service.
Each run is reproducible from the raw payloads it saves.

## Job shape

1. Read `DATABASE_URL`, `STACKEXCHANGE_SITE`, `SCRAPE_PAGES` and an optional
   `OPENAI_API_KEY` from the environment.
2. Scrape the newest question pages with the polite, cached `StackExchangeClient`, then
   scrape answers created in the same window.
3. Upsert posts, answers, authors and tags. Rebuild `engagement_daily` in one SQL pass.
4. Re-run trending, content-gap, segment and anomaly analytics.
5. Generate topic briefs. If the model call fails, use the deterministic template so the
   job still completes.
6. Rebuild the star-schema exports and upload `reports/` and `exports/` as job artifacts.
7. Never push report changes from an unattended job. A human reviews the artifacts and
   commits the reports that should become the published snapshot.

## Cron example

`scripts/cron.example` runs every Monday at 03:17 local server time, writes a timestamped
log, and keeps the previous working tree:

```cron
17 3 * * 1 cd /opt/community-topic-radar && /usr/bin/flock -n /tmp/ctr.lock make all >> /var/log/community-topic-radar.log 2>&1
```

Install it with:

```bash
crontab -e
# paste the line from scripts/cron.example
```

Use a dedicated service account, restrict the environment file to that account, and keep
the API key in the environment or a secret manager. The cache in `data/cache/` makes reruns
and retries cheap; `data/raw/` is the durable replay record.

## GitHub Actions schedule

`.github/workflows/scheduled-scrape.yml` runs weekly and on manual dispatch. It runs the
live scrape, plus analyze, brief and export, then uploads the generated reports and exports
as workflow artifacts. It does not commit or push. Add `OPENAI_API_KEY` as a repository
secret only if model-written briefs are wanted; otherwise the template fallback is used.

## Failure handling

- HTTP 429/5xx responses retry with exponential backoff up to four attempts.
- Missing or malformed cache entries are ignored and refetched.
- A failed OpenAI call retries once and then falls back to the template.
- The job fails loudly if ingestion returns zero posts; it will not overwrite a good report
  with an empty run.
- Keep the previous artifact run in the CI retention policy for at least one quarter.
