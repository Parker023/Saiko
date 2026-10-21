"""Tests for jobagency.api

Every test runs against a six-row frame injected through
`dependency_overrides` - so no network, no lake, no live data, and known
answers. `TestClient` runs the app in-process: no uvicorn, no port, no waiting.

One override does all the work: `get_dataframe` depends on `get_loaded_data`,
so replacing the latter controls both the happy path and the 503 path.
"""

import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from jobagency import api
from jobagency.analyse import to_dataframe
from jobagency.api import app, get_loaded_data


def _record(**overrides) -> dict:
    """A complete posting with defaults; override only what the test is about."""
    base = {
        "source": "remotive", "job_id": "r:0", "title": "Backend Developer",
        "company": "Acme", "category": "Software Development",
        "tags": ["python"], "job_type": "full_time",
        "posted_at": datetime(2026, 10, 10, tzinfo=timezone.utc),
        "location": "Worldwide", "remote": True,
        "salary_low": None, "salary_high": None, "salary_period": None,
        "description_html": "<p>a role</p>", "url": "https://e.test/0",
    }
    return {**base, **overrides}


# Six postings with deliberately known properties:
#   sources   - remotive 3, arbeitnow 3
#   remote    - 4 true, 2 false
#   salaries  - 2 annual (both bounds), 1 hourly, 3 none
#   tags      - python 3, java 2, aws 2, sql 1
#   companies - Google 2, Amazon 2, Netflix 1, Zalando 1
#   dates     - 10-05 .. 10-10, newest is r:1 and r:2
RECORDS = [
    _record(job_id="r:1", company="Google", tags=["python", "aws"],
            salary_low=80_000.0, salary_high=120_000.0, salary_period="year",
            posted_at=datetime(2026, 10, 10, tzinfo=timezone.utc)),
    _record(job_id="r:2", company="Google", tags=["python", "java"],
            salary_low=100_000.0, salary_high=140_000.0, salary_period="year",
            posted_at=datetime(2026, 10, 10, tzinfo=timezone.utc)),
    _record(job_id="r:3", company="Netflix", tags=["java", "aws"],
            title="Senior Engineer",
            salary_low=50.0, salary_high=90.0, salary_period="hour",
            posted_at=datetime(2026, 10, 8, tzinfo=timezone.utc)),
    _record(job_id="a:4", source="arbeitnow", company="Amazon", category=None,
            tags=["python", "sql"], remote=False, job_type=None,
            posted_at=datetime(2026, 10, 7, tzinfo=timezone.utc)),
    _record(job_id="a:5", source="arbeitnow", company="Amazon", category=None,
            tags=[], remote=False, job_type="contract",
            posted_at=datetime(2026, 10, 6, tzinfo=timezone.utc)),
    _record(job_id="a:6", source="arbeitnow", company="Zalando", category=None,
            tags=["java"], remote=True, job_type=None,
            posted_at=datetime(2026, 10, 5, tzinfo=timezone.utc)),
]


def _loaded(source="live", error=None):
    """The provenance dict the app keeps on app.state."""
    return {
        "data": to_dataframe(RECORDS),
        "source": source,
        "loaded_at": pd.Timestamp("2026-10-10T09:00:00Z"),
        "error": error,
    }


@pytest.fixture(autouse=True)
def no_startup_fetch(monkeypatch):
    """Stop lifespan from doing a real fetch.

    dependency_overrides replaces what HANDLERS receive - it does NOT stop
    `lifespan` from running. Without this, every TestClient(app) fetches both
    live APIs at startup: measured at ~4s per test, 120s for the file.
    """
    monkeypatch.setattr(api, "load_jobs_with_fallback", lambda **kw: _loaded())


def _client_with(provenance):
    """A TestClient whose endpoints see `provenance` instead of real data."""
    app.dependency_overrides[get_loaded_data] = lambda: provenance
    return TestClient(app)


@pytest.fixture
def client():
    """A client whose data is a known six-row frame, loaded 'live'."""
    with _client_with(_loaded()) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def degraded_client():
    """As if the live fetch failed and the cache was used instead."""
    with _client_with(_loaded(source="cache", error=RuntimeError("network down"))) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def empty_client():
    """As if neither source could be loaded."""
    with _client_with({
        "data": None, "source": None,
        "loaded_at": pd.Timestamp("2026-10-10T09:00:00Z"),
        "error": RuntimeError("cache unusable too"),
    }) as c:
        yield c
    app.dependency_overrides.clear()


# --- /health -------------------------------------------------------------

def test_health_reports_a_healthy_live_load(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["source"] == "live"
    assert body["degraded"] is False
    assert body["total_jobs"] == 6
    assert body["reason"] is None


def test_health_reports_data_freshness_not_just_load_time(client):
    """A file read today can still hold old postings - both dates matter."""
    body = client.get("/health").json()
    assert body["loaded_at"].startswith("2026-10-10")
    assert body["newest_posting"] == "2026-10-10"


def test_health_flags_a_cache_fallback_as_degraded(degraded_client):
    body = degraded_client.get("/health").json()
    assert body["source"] == "cache"
    assert body["degraded"] is True
    assert "network down" in body["reason"]


def test_health_is_503_with_no_data(empty_client):
    """A load balancer reads the status code, not the body."""
    response = empty_client.get("/health")
    assert response.status_code == 503
    assert "cache unusable" in response.json()["reason"]


# --- the 503 guard, enforced by the dependency ---------------------------

@pytest.mark.parametrize(
    "path", ["/stats", "/jobs", "/jobs/r:1", "/chart/tags"]
)
def test_data_endpoints_are_503_with_no_data(empty_client, path):
    """get_dataframe raises, so no handler can receive None."""
    assert empty_client.get(path).status_code == 503


# --- /stats --------------------------------------------------------------

def test_stats_returns_every_section(client):
    body = client.get("/stats").json()
    assert set(body) == {
        "overview", "by_category", "by_company", "by_job_type",
        "salary_year", "salary_hour", "top_tags", "trend", "tag_pairs",
    }


def test_stats_salary_always_carries_n_and_coverage(client):
    """The honesty contract, published in the schema."""
    year = client.get("/stats").json()["salary_year"]
    assert year["n"] == 2
    assert year["total_jobs"] == 6
    assert year["median"] == 110_000.0


def test_stats_separates_hourly_from_annual(client):
    body = client.get("/stats").json()
    assert body["salary_year"]["n"] == 2
    assert body["salary_hour"]["n"] == 1
    assert body["salary_hour"]["median"] == 70.0


# --- /jobs: filtering ----------------------------------------------------

def test_jobs_unfiltered_returns_everything(client):
    body = client.get("/jobs").json()
    assert body["total"] == 6
    assert len(body["items"]) == 6


@pytest.mark.parametrize(
    "query, expected",
    [
        ("source=remotive", 3),
        ("source=REMOTIVE", 3),          # case-insensitive
        ("source=arbeitnow", 3),
        ("company=google", 2),           # case-insensitive
        ("company=goog", 2),             # substring
        ("category=software development", 3),
        ("job_type=full_time", 3),
        ("tag=python", 3),               # a list column
        ("tag=PYTHON", 3),
        ("tag=java", 3),
        ("q=engineer", 1),               # title substring
        ("q=developer", 5),
    ],
)
def test_jobs_filters(client, query, expected):
    assert client.get(f"/jobs?{query}").json()["total"] == expected


@pytest.mark.parametrize(
    "query, expected",
    [("", 6), ("remote=true", 4), ("remote=false", 2)],
)
def test_jobs_remote_is_three_state(client, query, expected):
    """bool|None, not bool: without the None state you cannot ask for 'all'."""
    assert client.get(f"/jobs?{query}").json()["total"] == expected


@pytest.mark.parametrize(
    "query, expected",
    [("", 6), ("has_salary=true", 3), ("has_salary=false", 3)],
)
def test_jobs_has_salary_is_three_state(client, query, expected):
    assert client.get(f"/jobs?{query}").json()["total"] == expected


def test_jobs_filters_combine_with_and(client):
    assert client.get("/jobs?source=remotive&tag=python").json()["total"] == 2
    assert client.get("/jobs?source=arbeitnow&tag=python").json()["total"] == 1


def test_jobs_unmatched_filter_gives_an_empty_page_not_an_error(client):
    body = client.get("/jobs?company=nosuchcompany").json()
    assert body["total"] == 0
    assert body["items"] == []


# --- /jobs: validation ---------------------------------------------------

@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=101", "limit=-1", "limit=abc", "offset=-1", "q=a"],
)
def test_jobs_rejects_bad_parameters_with_422(client, query):
    """Pydantic validates before the handler runs. 422, never a 500."""
    assert client.get(f"/jobs?{query}").status_code == 422


def test_jobs_limit_cap_protects_the_payload(client):
    """le=100 is the reason a client cannot ask for every row at once."""
    assert client.get("/jobs?limit=100").status_code == 200
    assert client.get("/jobs?limit=500").status_code == 422


# --- /jobs: pagination ---------------------------------------------------

def test_jobs_total_is_counted_after_filtering_before_slicing(client):
    body = client.get("/jobs?source=remotive&limit=1").json()
    assert body["total"] == 3          # all matches
    assert len(body["items"]) == 1     # this page


def test_jobs_echoes_limit_and_offset(client):
    body = client.get("/jobs?limit=2&offset=4").json()
    assert body["limit"] == 2
    assert body["offset"] == 4


def test_jobs_pages_do_not_overlap(client):
    first = {i["job_id"] for i in client.get("/jobs?limit=3&offset=0").json()["items"]}
    second = {i["job_id"] for i in client.get("/jobs?limit=3&offset=3").json()["items"]}
    assert len(first) == len(second) == 3
    assert first & second == set()      # the explicit sort is what guarantees this


def test_jobs_offset_past_the_end_is_an_empty_page(client):
    body = client.get("/jobs?offset=9999").json()
    assert body["total"] == 6
    assert body["items"] == []


def test_jobs_are_sorted_newest_first(client):
    dates = [i["posted_at"] for i in client.get("/jobs").json()["items"]]
    assert dates == sorted(dates, reverse=True)


# --- /jobs: serialisation ------------------------------------------------

def test_jobs_summary_excludes_the_description(client):
    """2.70 MB with descriptions vs 0.15 MB without - that is why."""
    item = client.get("/jobs?limit=1").json()["items"][0]
    assert "description_html" not in item


def test_jobs_response_contains_no_bare_nan(client):
    """json.dumps writes NaN for a missing float, which is not valid JSON -
    Python's parser accepts it but a browser's JSON.parse throws."""
    raw = client.get("/jobs?limit=100").text
    assert "NaN" not in raw
    json.loads(raw)     # strict parse


def test_jobs_missing_salaries_are_null_not_zero(client):
    items = client.get("/jobs?has_salary=false").json()["items"]
    assert all(i["salary_low"] is None for i in items)


def test_jobs_tags_come_back_as_a_list(client):
    item = client.get("/jobs?tag=python&limit=1").json()["items"][0]
    assert isinstance(item["tags"], list)


# --- /jobs/{job_id} ------------------------------------------------------

def test_job_detail_includes_the_description(client):
    body = client.get("/jobs/r:1").json()
    assert body["job_id"] == "r:1"
    assert body["description_html"] == "<p>a role</p>"


def test_job_detail_404_names_the_id(client):
    response = client.get("/jobs/nope:123")
    assert response.status_code == 404
    assert "nope:123" in response.json()["detail"]


def test_job_detail_handles_ids_containing_a_colon(client):
    """job_ids are prefixed with the source, so they all contain one."""
    assert client.get("/jobs/a:4").status_code == 200


# --- /chart/{name} -------------------------------------------------------

@pytest.mark.parametrize("name", ["trend", "tags", "salary_coverage"])
def test_chart_returns_a_png(client, name):
    response = client.get(f"/chart/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"


def test_chart_404_lists_the_allowed_names(client):
    response = client.get("/chart/nope")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "nope" in detail
    for name in ("trend", "tags", "salary_coverage"):
        assert name in detail


def test_chart_requests_do_not_leak_figures(client):
    """pyplot holds every figure until it is closed; a chart endpoint without
    plt.close leaks until the process dies."""
    before = len(plt.get_fignums())
    for _ in range(5):
        client.get("/chart/tags")
    assert len(plt.get_fignums()) == before


# --- POST /refresh -------------------------------------------------------

def test_refresh_is_a_post_not_a_get(client, monkeypatch):
    """A GET that mutates can be fired by a browser prefetch or a crawler."""
    monkeypatch.setattr(api, "load_jobs_with_fallback", lambda **kw: _loaded())
    assert client.get("/refresh").status_code == 405
    assert client.post("/refresh").status_code == 200


def test_refresh_reloads_and_reports(client, monkeypatch):
    monkeypatch.setattr(api, "load_jobs_with_fallback", lambda **kw: _loaded())
    body = client.post("/refresh").json()
    assert body["status"] == "ok"
    assert body["total_jobs"] == 6


def test_refresh_reports_503_when_nothing_loads(client, monkeypatch):
    monkeypatch.setattr(api, "load_jobs_with_fallback", lambda **kw: {
        "data": None, "source": None,
        "loaded_at": pd.Timestamp("2026-10-10T09:00:00Z"),
        "error": RuntimeError("both failed"),
    })
    response = client.post("/refresh")
    assert response.status_code == 503
    assert "both failed" in response.json()["reason"]


# --- the published contract ----------------------------------------------

def test_openapi_documents_the_real_response_shapes(client):
    """The reason for response_model= : /docs shows the schema, not {}."""
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/stats"]["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]["$ref"].endswith("JobStatsResponse")
    assert schema["paths"]["/jobs"]["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]["$ref"].endswith("JobListResponse")
    assert "image/png" in schema["paths"]["/chart/{name}"]["get"][
        "responses"]["200"]["content"]


def test_openapi_lists_every_model(client):
    names = set(client.get("/openapi.json").json()["components"]["schemas"])
    assert {"JobSummary", "JobDetail", "JobListResponse",
            "JobStatsResponse", "SalaryStats", "Overview"} <= names
