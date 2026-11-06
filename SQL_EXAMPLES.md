# SQL analyst examples

Five queries run against `data/community.db` on 2026-11-05 after the full ingest
(6,500 posts, 7,828 answers). The SQL is SQLite syntax; on PostgreSQL replace
`strftime`/`date(..., '+N month')` with `date_trunc`/interval arithmetic.

## 1. Top tags by unanswered rate

```sql
SELECT t.name AS tag,
       COUNT(*) AS posts,
       ROUND(AVG(CASE WHEN p.is_answered THEN 0.0 ELSE 1.0 END), 3) AS unanswered_rate,
       ROUND(AVG(p.view_count), 0) AS avg_views
FROM posts p
JOIN post_tags pt ON pt.post_id = p.id
JOIN tags t ON t.id = pt.tag_id
GROUP BY t.name
HAVING COUNT(*) >= 30
ORDER BY unanswered_rate DESC, posts DESC
LIMIT 10;
```

```text
          tag             posts  unanswered_rate  avg_views
------------------------  -----  ---------------  ---------
stationarity                 37             0.73      125.0
spatial                      38            0.711      130.0
difference-in-difference     40              0.7      159.0
gaussian-process             31            0.677      171.0
ecology                      52            0.615      182.0
fixed-effects-model          49            0.592      194.0
random-forest                31            0.581      139.0
clustering                   66            0.576      179.0
autocorrelation              68            0.574      168.0
stochastic-processes         49            0.571      180.0
```

## 2. Month-over-month active authors

```sql
SELECT strftime('%Y-%m', created_at) AS month,
       COUNT(DISTINCT author_id) AS active_authors,
       COUNT(*) AS posts
FROM posts
GROUP BY 1
ORDER BY 1 DESC
LIMIT 12;
```

```text
 month   active_authors  posts
-------  --------------  -----
2026-11              15     15
2026-10              72     86
2026-09              96    110
2026-08              75     96
2026-07              90    101
2026-06              89    104
2026-05              92    103
2026-04             122    141
2026-03             133    147
2026-02             139    167
2026-01             156    181
2025-12             162    181
```

## 3. Cohort retention of askers

Cohorts are defined by the month of an author's first question. Retention is the
share of that cohort that asks again the following month.

```sql
WITH first_seen AS (
    SELECT author_id, MIN(strftime('%Y-%m', created_at)) AS cohort
    FROM posts
    WHERE author_id IS NOT NULL
    GROUP BY author_id
),
activity AS (
    SELECT DISTINCT p.author_id,
           f.cohort,
           strftime('%Y-%m', p.created_at) AS activity_month
    FROM posts p
    JOIN first_seen f ON f.author_id = p.author_id
),
retained AS (
    SELECT f.cohort,
           COUNT(DISTINCT f.author_id) AS cohort_size,
           COUNT(DISTINCT CASE
               WHEN a.activity_month = strftime('%Y-%m', date(f.cohort || '-01', '+1 month'))
               THEN a.author_id END) AS retained_next_month
    FROM first_seen f
    JOIN activity a ON a.author_id = f.author_id
    GROUP BY f.cohort
)
SELECT cohort,
       cohort_size,
       retained_next_month,
       ROUND(1.0 * retained_next_month / cohort_size, 3) AS next_month_retention
FROM retained
ORDER BY cohort DESC
LIMIT 12;
```

```text
cohort   cohort_size  retained_next_month  next_month_retention
-------  -----------  -------------------  --------------------
2026-11           10                    0                   0.0
2026-10           47                    0                   0.0
2026-09           66                    2                  0.03
2026-08           47                    3                 0.064
2026-07           57                    3                 0.053
2026-06           65                    6                 0.092
2026-05           60                    1                 0.017
2026-04           68                    1                 0.015
2026-03           86                    6                  0.07
2026-02           87                    3                 0.034
2026-01          100                    5                  0.05
2025-12          107                    3                 0.028
```

## 4. Fastest-growing tags

Compares the 90 days before the latest post with the preceding 90 days, requiring at
least three recent posts.

```sql
WITH bounds AS (
    SELECT MAX(date(created_at)) AS max_day FROM posts
),
tag_windows AS (
    SELECT t.name AS tag,
           SUM(CASE WHEN date(p.created_at) > date(b.max_day, '-90 days') THEN 1 ELSE 0 END) AS recent_90d,
           SUM(CASE WHEN date(p.created_at) BETWEEN date(b.max_day, '-180 days')
                                         AND date(b.max_day, '-90 days') THEN 1 ELSE 0 END) AS prior_90d
    FROM posts p
    JOIN post_tags pt ON pt.post_id = p.id
    JOIN tags t ON t.id = pt.tag_id
    CROSS JOIN bounds b
    GROUP BY t.name
)
SELECT tag, recent_90d, prior_90d,
       recent_90d - prior_90d AS delta,
       ROUND(100.0 * (recent_90d - prior_90d) / MAX(prior_90d, 1), 1) AS pct_change
FROM tag_windows
WHERE recent_90d >= 3
ORDER BY delta DESC, recent_90d DESC
LIMIT 10;
```

```text
          tag            recent_90d  prior_90d  delta  pct_change
-----------------------  ----------  ---------  -----  ----------
multiple-comparisons             13          4      9       225.0
sampling                         10          1      9       900.0
mgcv                             11          4      7       175.0
probability                      16         10      6        60.0
mathematical-statistics          12          6      6       100.0
maximum-likelihood               12          6      6       100.0
likelihood-ratio                  6          0      6       600.0
meta-analysis                     6          1      5       500.0
mixed-model                      29         25      4        16.0
lme4-nlme                         9          5      4        80.0
```

## 5. Contributor leaderboard

```sql
SELECT a.author_id,
       a.author_name,
       COUNT(*) AS answers,
       SUM(CASE WHEN a.is_accepted THEN 1 ELSE 0 END) AS accepted_answers,
       ROUND(AVG(a.score), 2) AS avg_answer_score
FROM answers a
WHERE a.author_id IS NOT NULL
GROUP BY a.author_id, a.author_name
ORDER BY answers DESC, accepted_answers DESC
LIMIT 10;
```

```text
author_id    author_name     answers  accepted_answers  avg_answer_score
---------  ----------------  -------  ----------------  ----------------
    28500  EdM                   445               159              3.35
      686  Peter Flom            419                72              4.09
   380096  jginestet             394               105              3.28
   249135  Thomas Lumley         329               104              4.33
    97844  Rick Hass             223                54               3.8
   164061  Sextus Empiricus      220                42               3.4
   247165  Christian Hennig     211                55              4.83
     2126  Ben Bolker            165                76               5.7
   173082  Ben                   160                39              5.32
     4253  Frank Harrell         144                22              5.95
```
