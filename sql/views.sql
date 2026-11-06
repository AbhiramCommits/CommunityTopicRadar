-- PostgreSQL views equivalent to the Power BI measures.
CREATE OR REPLACE VIEW v_monthly_active_authors AS
SELECT
    DATE_TRUNC('month', created_at)::date AS month,
    COUNT(DISTINCT author_id) AS monthly_active_authors
FROM posts
GROUP BY 1;

CREATE OR REPLACE VIEW v_answer_rate AS
SELECT
    DATE_TRUNC('month', created_at)::date AS month,
    AVG(CASE WHEN is_answered THEN 1.0 ELSE 0.0 END) AS answer_rate
FROM posts
GROUP BY 1;

CREATE OR REPLACE VIEW v_topic_wow_growth AS
WITH weekly AS (
    SELECT
        t.name AS topic,
        DATE_TRUNC('week', p.created_at)::date AS week,
        COUNT(*) AS posts
    FROM posts AS p
    JOIN post_tags AS pt ON pt.post_id = p.id
    JOIN tags AS t ON t.id = pt.tag_id
    GROUP BY 1, 2
),
comparison AS (
    SELECT
        topic,
        week,
        posts,
        LAG(posts) OVER (PARTITION BY topic ORDER BY week) AS prior_posts
    FROM weekly
)
SELECT
    topic,
    week,
    posts,
    prior_posts,
    CASE
        WHEN prior_posts > 0 THEN (posts - prior_posts)::numeric / prior_posts
        ELSE NULL
    END AS wow_growth
FROM comparison;

CREATE OR REPLACE VIEW v_content_gap_backlog AS
WITH tag_stats AS (
    SELECT
        t.name AS topic,
        COUNT(*) AS posts,
        AVG(CASE WHEN p.is_answered THEN 1.0 ELSE 0.0 END) AS answer_rate,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.view_count) AS median_views
    FROM posts AS p
    JOIN post_tags AS pt ON pt.post_id = p.id
    JOIN tags AS t ON t.id = pt.tag_id
    GROUP BY 1
),
scored AS (
    SELECT
        *,
        (
            PERCENT_RANK() OVER (ORDER BY posts)
            + PERCENT_RANK() OVER (ORDER BY median_views)
        ) / 2.0 AS demand_percentile
    FROM tag_stats
)
SELECT
    topic,
    posts,
    answer_rate,
    median_views,
    demand_percentile * (1 - answer_rate) AS gap_score
FROM scored;
