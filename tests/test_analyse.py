"""Tests for jobagency.analyse

The sample frame is built by hand with known answers, not loaded from the API
fixtures. Analysis tests need *predictable* numbers - "the median of these six
salaries is X" - which a live payload cannot give you.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from jobagency.analyse import (
    count_by,
    overview,
    posting_trend,
    salary_stats,
    tag_cooccurrence,
    top_tags,
    to_dataframe,
)


def _record(**overrides) -> dict:
    """A complete record with sensible defaults; override only what matters.

    Keeps each test readable - a test about salary should not have to spell
    out fifteen irrelevant fields.
    """
    base = {
        "source": "remotive", "job_id": "r:0", "title": "Dev",
        "company": "Acme", "category": "Software Development",
        "tags": ["python"], "job_type": "full_time",
        "posted_at": datetime(2026, 10, 10, tzinfo=timezone.utc),
        "location": "Worldwide", "remote": True,
        "salary_low": None, "salary_high": None, "salary_period": None,
        "description_html": "<p>x</p>", "url": "https://e.com/0",
    }
    return {**base, **overrides}


@pytest.fixture
def df() -> pd.DataFrame:
    """Six jobs. Salaries: two annual, one hourly, three none.

    Annual midpoints are 100_000 and 120_000 -> median 110_000.
    Tags: python x4, java x2, aws x2, sql x1.
    Dates span 10-05 to 10-10 with 10-09 deliberately empty.
    """
    return to_dataframe([
        _record(job_id="r:1", company="Google", tags=["python", "aws"],
                salary_low=80_000.0, salary_high=120_000.0, salary_period="year",
                posted_at=datetime(2026, 10, 10, tzinfo=timezone.utc)),
        _record(job_id="r:2", company="Google", tags=["python", "java"],
                salary_low=100_000.0, salary_high=140_000.0, salary_period="year",
                posted_at=datetime(2026, 10, 10, tzinfo=timezone.utc)),
        _record(job_id="r:3", company="Netflix", tags=["java", "aws"],
                salary_low=50.0, salary_high=90.0, salary_period="hour",
                posted_at=datetime(2026, 10, 8, tzinfo=timezone.utc)),
        _record(job_id="a:4", source="arbeitnow", company="Amazon",
                category=None, tags=["python", "sql"], remote=False,
                posted_at=datetime(2026, 10, 7, tzinfo=timezone.utc)),
        _record(job_id="a:5", source="arbeitnow", company="Amazon",
                category=None, tags=["python"], remote=False,
                posted_at=datetime(2026, 10, 6, tzinfo=timezone.utc)),
        _record(job_id="a:6", source="arbeitnow", company="Zalando",
                category=None, tags=[], remote=True,
                posted_at=datetime(2026, 10, 5, tzinfo=timezone.utc)),
    ])


# --- to_dataframe --------------------------------------------------------

def test_to_dataframe_pins_dtypes(df):
    """Inference is data-dependent; pinning keeps behaviour stable per batch."""
    assert str(df["posted_at"].dtype).startswith("datetime64")
    assert df["posted_at"].dt.tz is not None
    assert df["salary_low"].dtype == "float64"
    assert df["remote"].dtype == "boolean"


def test_to_dataframe_adds_absent_columns():
    """A payload missing a field must not produce a KeyError downstream."""
    out = to_dataframe([{"source": "x", "job_id": "x:1", "tags": []}])
    for column in ("company", "salary_low", "posted_at", "remote"):
        assert column in out.columns


def test_to_dataframe_handles_empty_input():
    assert len(to_dataframe([])) == 0


# --- overview ------------------------------------------------------------

def test_overview_counts(df):
    result = overview(df)
    assert result["total_jobs"] == 6
    assert result["by_source"] == {"remotive": 3, "arbeitnow": 3}
    assert result["remote_count"] == 4


def test_overview_dates_are_iso_dates(df):
    result = overview(df)
    assert result["first_posted"] == "2026-10-05"
    assert result["last_posted"] == "2026-10-10"


# --- count_by ------------------------------------------------------------

def test_count_by_is_sorted_by_count_descending(df):
    counts = [row["count"] for row in count_by(df, "company")]
    assert counts == sorted(counts, reverse=True)


def test_count_by_values(df):
    result = {row["value"]: row["count"] for row in count_by(df, "company")}
    assert result == {"Google": 2, "Amazon": 2, "Netflix": 1, "Zalando": 1}


def test_count_by_respects_top_n(df):
    assert len(count_by(df, "company", top_n=2)) == 2


@pytest.mark.parametrize("column", ["description_html", "url", "title", "nope"])
def test_count_by_rejects_columns_outside_the_allowlist(df, column):
    """Grouping by HTML is nonsense; in phase 5 a model picks this argument."""
    with pytest.raises(ValueError):
        count_by(df, column)


# --- salary_stats --------------------------------------------------------

def test_salary_stats_always_reports_n_and_coverage(df):
    """The honesty contract: you cannot get the median without the sample size."""
    for period in ("year", "hour", "decade"):
        result = salary_stats(df, period=period)
        assert "n" in result and "coverage_pct" in result
        assert result["total_jobs"] == 6


def test_salary_stats_annual_only(df):
    """Mixing hourly and annual in one median is meaningless."""
    result = salary_stats(df, period="year")
    assert result["n"] == 2
    assert result["median"] == 110_000.0          # midpoints 100k and 120k


def test_salary_stats_hourly_only(df):
    result = salary_stats(df, period="hour")
    assert result["n"] == 1
    assert result["median"] == 70.0               # midpoint of 50 and 90


def test_salary_stats_coverage_reflects_the_whole_frame(df):
    result = salary_stats(df, period="year")
    assert result["coverage_pct"] == pytest.approx(2 / 6 * 100)


def test_salary_stats_empty_gives_none_not_zero(df):
    """Absence is not zero - the same rule as parse_salary."""
    result = salary_stats(df, period="decade")
    assert result["n"] == 0
    for key in ("min", "max", "median", "p25", "p75"):
        assert result[key] is None


def test_salary_stats_range_uses_low_and_high(df):
    """min should be the lowest offered, max the highest - not midpoints."""
    result = salary_stats(df, period="year")
    assert result["min"] == 80_000.0
    assert result["max"] == 140_000.0


# --- top_tags ------------------------------------------------------------

def test_top_tags_sorted_by_count_descending(df):
    counts = [row["count"] for row in top_tags(df)]
    assert counts == sorted(counts, reverse=True)


def test_top_tags_counts_occurrences_not_rows(df):
    result = {row["tag"]: row["count"] for row in top_tags(df)}
    assert result["python"] == 4      # 4 jobs mention it
    assert result["java"] == 2
    assert result["sql"] == 1


def test_top_tags_uses_the_tag_key(df):
    assert set(top_tags(df)[0]) == {"tag", "count"}


def test_top_tags_respects_top_n(df):
    assert len(top_tags(df, top_n=2)) == 2


# --- posting_trend -------------------------------------------------------

def test_posting_trend_fills_empty_days(df):
    """10-09 has no postings. A line chart with a missing day lies."""
    dates = {row["date"]: row["count"] for row in posting_trend(df)}
    assert dates["2026-10-09"] == 0
    assert dates["2026-10-10"] == 2


def test_posting_trend_is_chronological(df):
    dates = [row["date"] for row in posting_trend(df)]
    assert dates == sorted(dates)


def test_posting_trend_does_not_mutate_the_input(df):
    before = df.copy(deep=True)
    posting_trend(df)
    pd.testing.assert_frame_equal(df, before)


# --- tag_cooccurrence ----------------------------------------------------

def test_cooccurrence_finds_known_pairs(df):
    """python+aws (r:1), python+java (r:2), java+aws (r:3) -> each once.
    python appears with sql once too. With min_count=1 all should show."""
    pairs = {
        frozenset((row["tag_a"], row["tag_b"])): row["count"]
        for row in tag_cooccurrence(df, min_count=1)
    }
    assert pairs[frozenset(("python", "aws"))] == 1
    assert pairs[frozenset(("java", "aws"))] == 1


def test_cooccurrence_has_no_self_pairs(df):
    for row in tag_cooccurrence(df, min_count=1):
        assert row["tag_a"] != row["tag_b"]


def test_cooccurrence_reports_each_pair_once(df):
    """The matrix is symmetric; the upper triangle must not double-report."""
    seen = [frozenset((r["tag_a"], r["tag_b"])) for r in tag_cooccurrence(df, min_count=1)]
    assert len(seen) == len(set(seen))


def test_cooccurrence_respects_min_count(df):
    assert tag_cooccurrence(df, min_count=99) == []


def test_cooccurrence_max_pairs_is_independent_of_top_n(df):
    """The whole point of splitting the parameter: many tags, few pairs."""
    result = tag_cooccurrence(df, top_n=10, max_pairs=1, min_count=1)
    assert len(result) == 1


def test_cooccurrence_handles_jobs_with_no_tags(df):
    """One job has tags=[] - it must not crash or be counted."""
    assert isinstance(tag_cooccurrence(df, min_count=1), list)


# --- the boundary --------------------------------------------------------

def test_every_result_is_json_serialisable(df):
    """analyse.py is the last place a DataFrame exists. Everything it returns
    has to survive FastAPI and json.dumps for a tool result."""
    for result in (
        overview(df),
        count_by(df, "company"),
        salary_stats(df, "year"),
        top_tags(df),
        posting_trend(df),
        tag_cooccurrence(df, min_count=1),
    ):
        json.dumps(result)
