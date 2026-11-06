-- Daily engagement rollup, portable across PostgreSQL and SQLite.
-- Answers are counted by their own creation date; the answer scrape covers the
-- same creation window as the questions.
WITH post_daily AS (
    SELECT
        DATE(created_at) AS day,
        source_site,
        COUNT(*) AS posts_created,
        COUNT(DISTINCT author_id) AS unique_authors,
        COALESCE(SUM(CASE WHEN is_answered THEN 1 ELSE 0 END), 0) AS answered_count,
        COALESCE(SUM(view_count), 0) AS total_views
    FROM posts
    GROUP BY DATE(created_at), source_site
),
answer_daily AS (
    SELECT
        DATE(created_at) AS day,
        source_site,
        COUNT(*) AS answers_created
    FROM answers
    GROUP BY DATE(created_at), source_site
)
INSERT INTO engagement_daily (
    day,
    source_site,
    posts_created,
    answers_created,
    unique_authors,
    answered_count,
    total_views
)
SELECT
    p.day,
    p.source_site,
    p.posts_created,
    COALESCE(a.answers_created, 0) AS answers_created,
    p.unique_authors,
    p.answered_count,
    p.total_views
FROM post_daily AS p
LEFT JOIN answer_daily AS a
    ON p.day = a.day AND p.source_site = a.source_site
ON CONFLICT (day, source_site) DO UPDATE SET
    posts_created = excluded.posts_created,
    answers_created = excluded.answers_created,
    unique_authors = excluded.unique_authors,
    answered_count = excluded.answered_count,
    total_views = excluded.total_views;
