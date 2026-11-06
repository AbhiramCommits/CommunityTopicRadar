# Power BI measures

Import the six CSVs in this folder as a star schema. Relate
`fact_post[topic_id]` to `dim_topic[topic_id]`, `fact_post[author_id]` to
`dim_author[author_id]`, and every `date_key` to `dim_date[date_key]`.

```DAX
Monthly Active Authors =
DISTINCTCOUNT ( fact_post[author_id] )

Answer Rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_post ), fact_post[is_answered] = TRUE() ),
    COUNTROWS ( fact_post )
)

WoW Topic Growth =
VAR CurrentWeek = MAX ( dim_date[date] )
VAR CurrentPosts =
    CALCULATE (
        COUNTROWS ( fact_post ),
        FILTER ( dim_date, dim_date[date] > CurrentWeek - 7 && dim_date[date] <= CurrentWeek )
    )
VAR PriorPosts =
    CALCULATE (
        COUNTROWS ( fact_post ),
        FILTER (
            dim_date,
            dim_date[date] > CurrentWeek - 14 && dim_date[date] <= CurrentWeek - 7
        )
    )
RETURN DIVIDE ( CurrentPosts - PriorPosts, PriorPosts )

Content Gap Backlog Size =
COUNTROWS ( FILTER ( dim_topic, dim_topic[gap_score] >= 0.30 ) )
```

`fact_engagement_daily[is_anomaly]` and `fact_anomaly` support the anomaly
page. `dim_author[segment]` supports the community-segment page.
