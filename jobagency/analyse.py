import pandas as pd
import numpy as np
from datetime import datetime, timezone


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    schema = {
        "source": "string",
        "job_id": "string",
        "title": "string",
        "company": "string",
        "category": "string",
        "job_type": "string",
        "location": "string",
        "remote": "boolean",
        "salary_low": "float64",
        "salary_high": "float64",
        "salary_period": "string",
        "description_html": "string",
        "url": "string",
    }

    for column, dtype in schema.items():
        if column not in df.columns:
            df[column] = pd.Series(index=df.index, dtype=dtype)
        else:
            df[column] = df[column].astype(dtype)

    if "tags" not in df.columns:
        df["tags"] = pd.Series(index=df.index, dtype="object")

    df["posted_at"] = pd.to_datetime(
        df["posted_at"], utc=True, errors="coerce"
    ) if "posted_at" in df.columns else pd.Series(
        index=df.index, dtype="datetime64[ns, UTC]"
    )

    return df


def overview(df: pd.DataFrame) -> dict:
    total_jobs = len(df)
    by_source = df.groupby("source").size().to_dict()
    remote_job_count = len(df.loc[df["remote"]])
    first_posted = str(df["posted_at"].dt.strftime("%Y-%m-%d").min())
    last_posted = str(df["posted_at"].dt.strftime("%Y-%m-%d").max())

    return {
        "total_jobs": total_jobs,
        "by_source": by_source,
        "remote_count": remote_job_count,
        "first_posted": first_posted,
        "last_posted": last_posted
    }


def count_by(df, col: str, top_n: int = 10):
    COUNTABLE = {"source", "company", "category", "job_type", "location", "salary_period"}
    if col not in COUNTABLE:
        raise ValueError(f"Column {col} not found in DataFrame")
    res = []

    by_col_dict = df.groupby(col).size().to_dict()
    for value, count in by_col_dict.items():
        res.append({"value": value, "count": count})

    return sorted(res, key=lambda x: x["count"], reverse=True)[:top_n]


def salary_stats(df, period: str = "year") -> dict:
    period_df = df.loc[df["salary_period"] == period].copy()

    total_jobs = len(df)

    valid = (
            period_df["salary_low"].notna()
            & period_df["salary_high"].notna()
    )

    salaries = (
                       period_df.loc[valid, "salary_low"]
                       + period_df.loc[valid, "salary_high"]
               ) / 2

    n = len(salaries)

    coverage_pct = (
        (n / total_jobs) * 100
        if total_jobs > 0
        else 0.0
    )

    return {
        "period": period,
        "n": n,
        "total_jobs": total_jobs,
        "coverage_pct": coverage_pct,
        "min": period_df["salary_low"].min() if n else None,
        "max": period_df["salary_high"].max() if n else None,
        "median": float(np.median(salaries)) if n else None,
        "p25": float(np.percentile(salaries, 25)) if n else None,
        "p75": float(np.percentile(salaries, 75)) if n else None,
    }


def top_tags(df, top_n: int = 20) -> list[dict]:
    # return df.explode("tags").groupby("tags").size().reset_index(name="count").rename(columns={"tags": "tag"}).sort_values(by="count",ascending=False).to_dict("records")[:top_n]
    return df.explode("tags")["tags"].value_counts().head(top_n).reset_index(name="count").rename(
        columns={"tags": "tag"}).to_dict(orient="records")


def posting_trend(df, freq: str = "D") -> list[dict]:
    df_copy = df.copy()
    return (df_copy.set_index("posted_at").resample(freq)
            .size().reset_index(name="count").rename(columns={"posted_at": "date"}).assign(
        date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"))


def tag_cooccurrence(
        df,
        top_n: int = 30,
        max_pairs: int = 20,
        min_count: int = 2
) -> list[dict]:
    # Remove missing and duplicate tags within each job
    job_tags = [
        set(tags) if isinstance(tags, list) else set()
        for tags in df["tags"]
    ]

    # Count individual tag frequency
    tag_counts = pd.Series(
        [tag for tags in job_tags for tag in tags]
    ).value_counts()

    # Restrict computation to the most common tags
    tags = tag_counts.head(top_n).index.tolist()

    if len(tags) < 2:
        return []

    tag_to_idx = {tag: i for i, tag in enumerate(tags)}

    # Build binary matrix: rows = jobs, columns = tags
    M = np.zeros((len(job_tags), len(tags)), dtype=np.int64)

    for row, job_tag_set in enumerate(job_tags):
        for tag in job_tag_set:
            if tag in tag_to_idx:
                M[row, tag_to_idx[tag]] = 1

    # Pairwise co-occurrence counts
    cooccurrence = M.T @ M

    # Select unique pairs, excluding the diagonal
    rows, cols = np.triu_indices(len(tags), k=1)

    results = [
        {
            "tag_a": tags[i],
            "tag_b": tags[j],
            "count": int(cooccurrence[i, j])
        }
        for i, j in zip(rows, cols)
        if cooccurrence[i, j] >= min_count
    ]

    results.sort(key=lambda item: item["count"], reverse=True)

    return results[:max_pairs]


def main():
    records = [
        {
            "source": "remotive",
            "job_id": "remotive:1001",
            "title": "Java Backend Developer",
            "company": "Google",
            "category": "Software Development",
            "tags": ["Java", "Spring Boot", "AWS"],
            "job_type": "full_time",
            "posted_at": datetime(2026, 10, 10, tzinfo=timezone.utc),
            "location": "India",
            "remote": True,
            "salary_low": 80000.0,
            "salary_high": 120000.0,
            "salary_period": "year",
            "description_html": "<p>Java backend role</p>",
            "url": "https://example.com/jobs/1001",
        },
        {
            "source": "arbeitnow",
            "job_id": "arbeitnow:1002",
            "title": "Python Developer",
            "company": "Amazon",
            "category": None,
            "tags": ["Python", "Django"],
            "job_type": "full_time",
            "posted_at": datetime(2026, 10, 10, tzinfo=timezone.utc),
            "location": "Berlin",
            "remote": False,
            "salary_low": None,
            "salary_high": None,
            "salary_period": None,
            "description_html": "<p>Python developer role</p>",
            "url": "https://example.com/jobs/1002",
        },
        {
            "source": "remotive",
            "job_id": "remotive:1003",
            "title": "Software Engineer",
            "company": "Microsoft",
            "category": "Software Development",
            "tags": ["Python", "Azure"],
            "job_type": "full_time",
            "posted_at": datetime(2026, 10, 8, tzinfo=timezone.utc),
            "location": "Worldwide",
            "remote": True,
            "salary_low": 100000.0,
            "salary_high": 140000.0,
            "salary_period": "year",
            "description_html": "<p>Software engineering role</p>",
            "url": "https://example.com/jobs/1003",
        },
        {
            "source": "arbeitnow",
            "job_id": "arbeitnow:1004",
            "title": "Data Analyst",
            "company": "Google",
            "category": None,
            "tags": ["SQL", "Python"],
            "job_type": None,
            "posted_at": datetime(2026, 10, 7, tzinfo=timezone.utc),
            "location": "Munich",
            "remote": False,
            "salary_low": None,
            "salary_high": None,
            "salary_period": None,
            "description_html": "<p>Data analyst role</p>",
            "url": "https://example.com/jobs/1004",
        },
        {
            "source": "remotive",
            "job_id": "remotive:1005",
            "title": "Spring Boot Developer",
            "company": "Netflix",
            "category": "Software Development",
            "tags": ["Java", "Spring Boot", "Microservices"],
            "job_type": "full_time",
            "posted_at": datetime(2026, 10, 6, tzinfo=timezone.utc),
            "location": "India",
            "remote": True,
            "salary_low": 90000.0,
            "salary_high": 130000.0,
            "salary_period": "yearly",
            "description_html": "<p>Spring Boot role</p>",
            "url": "https://example.com/jobs/1005",
        },
        {
            "source": "arbeitnow",
            "job_id": "arbeitnow:1006",
            "title": "Full Stack Developer",
            "company": "Amazon",
            "category": None,
            "tags": ["React", "Python"],
            "job_type": "contract",
            "posted_at": datetime(2026, 10, 5, tzinfo=timezone.utc),
            "location": "Remote",
            "remote": True,
            "salary_low": None,
            "salary_high": None,
            "salary_period": None,
            "description_html": "<p>Full stack role</p>",
            "url": "https://example.com/jobs/1006",
        },
    ]
    df = pd.DataFrame(records)
    # print(df)

    print(overview(df))


if __name__ == "__main__":
    main()
