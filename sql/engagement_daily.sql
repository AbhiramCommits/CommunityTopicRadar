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
),
date_spine AS (
    SELECT day, source_site FROM post_daily
    UNION
    SELECT day, source_site FROM answer_daily
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
    d.day,
    d.source_site,
    COALESCE(p.posts_created, 0) AS posts_created,
    COALESCE(a.answers_created, 0) AS answers_created,
    COALESCE(p.unique_authors, 0) AS unique_authors,
    COALESCE(p.answered_count, 0) AS answered_count,
    COALESCE(p.total_views, 0) AS total_views
FROM date_spine AS d
LEFT JOIN post_daily AS p
    ON p.day = d.day AND p.source_site = d.source_site
LEFT JOIN answer_daily AS a
    ON a.day = d.day AND a.source_site = d.source_site
ON CONFLICT (day, source_site) DO UPDATE SET
    posts_created = excluded.posts_created,
    answers_created = excluded.answers_created,
    unique_authors = excluded.unique_authors,
    answered_count = excluded.answered_count,
    total_views = excluded.total_views;
