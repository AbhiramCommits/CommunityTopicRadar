-- Daily engagement rollup, portable across PostgreSQL and SQLite.
-- answers_created is a question-level proxy: posts created that day which had
-- received at least one answer by scrape time. See README limitations.
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
    DATE(created_at) AS day,
    source_site,
    COUNT(*) AS posts_created,
    COALESCE(SUM(CASE WHEN answer_count > 0 THEN 1 ELSE 0 END), 0) AS answers_created,
    COUNT(DISTINCT author_id) AS unique_authors,
    COALESCE(SUM(CASE WHEN is_answered THEN 1 ELSE 0 END), 0) AS answered_count,
    COALESCE(SUM(view_count), 0) AS total_views
FROM posts
GROUP BY DATE(created_at), source_site
ON CONFLICT (day, source_site) DO UPDATE SET
    posts_created = excluded.posts_created,
    answers_created = excluded.answers_created,
    unique_authors = excluded.unique_authors,
    answered_count = excluded.answered_count,
    total_views = excluded.total_views;
