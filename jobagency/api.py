"""HTTP API over the job analysis.

The DataFrame is loaded once in `lifespan` - at app startup, not at import -
so importing this module is free and tests can start the app with whatever
data they like via `dependency_overrides`.
"""

import logging
from contextlib import asynccontextmanager
from typing import Callable

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status, HTTPException, Query
from jobagency.chart import trend_figure, tags_figure, salary_coverage_figure, png_bytes
from jobagency.analyse import posting_trend, salary_stats, top_tags
from jobagency.main import load_jobs_with_fallback, build_report
from jobagency.models import (
    JobDetail,
    JobListResponse,
    JobStatsResponse,
    JobSummary,
)

logger = logging.getLogger(__name__)

CHARTS: dict[str, tuple[Callable, Callable]] = {
    "trend": (posting_trend, trend_figure),
    "tags": (top_tags, tags_figure),
    "salary_coverage": (salary_stats, salary_coverage_figure),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_jobs_with_fallback already returns exactly this shape, so store it
    # as-is: copying it field by field just creates a second place to update.
    app.state.jobs = load_jobs_with_fallback()
    logger.info(
        "loaded %s jobs from %s",
        0 if app.state.jobs["data"] is None else len(app.state.jobs["data"]),
        app.state.jobs["source"],
    )
    yield


app = FastAPI(lifespan=lifespan, title="Job Match Agency")


def get_loaded_data(request: Request) -> dict:
    """The loaded frame plus its provenance.

    A dependency rather than a direct `app.state` read, so a test can do
    `app.dependency_overrides[get_loaded_data] = lambda: {...}`.
    """
    return request.app.state.jobs


def get_dataframe(loaded: dict = Depends(get_loaded_data)) -> pd.DataFrame:
    """The loaded frame."""
    if loaded["data"] is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="no data loaded")
    return loaded["data"]


def _summary(loaded: dict) -> dict:
    """The shape /health and /refresh both report."""
    df: pd.DataFrame = loaded["data"]
    return {
        "status": "ok",
        "source": loaded["source"],
        # degraded is derived, not stored - it cannot drift out of sync
        "degraded": loaded["source"] == "cache",
        "reason": None if loaded["error"] is None else str(loaded["error"]),
        "total_jobs": len(df),
        "loaded_at": loaded["loaded_at"].isoformat(),
        # the data's own freshness, which is different from when we read it:
        # a file written today can still hold month-old postings.
        "newest_posting": (
            None if df.empty else df["posted_at"].max().strftime("%Y-%m-%d")
        ),
    }


@app.get("/health")
def health(response: Response, loaded: dict = Depends(get_loaded_data)) -> dict:
    """Liveness plus how trustworthy the data is.

    503 when there is nothing to serve: a load balancer reads the status
    code, not the body.
    """
    if loaded["data"] is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unavailable",
            "reason": str(loaded["error"]) if loaded["error"] else "no data loaded",
        }
    return _summary(loaded)


@app.post("/refresh")
def refresh(response: Response) -> dict:
    """Re-load the data and report what happened.

    POST, not GET: it changes the server state, and a GET that mutates can be
    fired by a browser prefetch or a crawler.
    """
    app.state.jobs = load_jobs_with_fallback()
    if app.state.jobs["data"] is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "reason": str(app.state.jobs["error"])}
    return _summary(app.state.jobs)


@app.get("/stats", response_model=JobStatsResponse)
def stats(df: pd.DataFrame = Depends(get_dataframe)):
    return build_report(df)


COUNTABLE_TEXT_FILTERS = ("source", "company", "category", "job_type")


def _records(frame: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN turned into None.

    Pydantic will NOT do this for you: NaN is a valid float, so `float | None`
    happily passes it through - and `json.dumps` then emits a bare `NaN`,
    which is not valid JSON. Python's own parser accepts it, so this only
    breaks once a real client (a browser, Streamlit) tries to read it.
    """
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


@app.get("/jobs", response_model=JobListResponse)
def jobs(
        *,
        source: str | None = None,
        company: str | None = Query(default=None, max_length=100),
        category: str | None = Query(default=None, max_length=100),
        job_type: str | None = Query(default=None, max_length=50),
        tag: str | None = Query(default=None, max_length=50),
        q: str | None = Query(default=None, max_length=100, min_length=2),
        remote: bool | None = Query(default=None),
        has_salary: bool | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        df: pd.DataFrame = Depends(get_dataframe),
):
    """Search job postings. All filters are combined with AND."""
    frame = df

    # Case-insensitive exact match: lower BOTH sides. Lowering only the input
    # silently matches nothing because the stored values keep their case.
    for column, value in (
            ("source", source),
            ("category", category),
            ("job_type", job_type),
    ):
        if value is not None:
            frame = frame[frame[column].str.lower() == value.lower()]

    # Substring rather than equality: a caller searching "google" should find
    # "Google Cloud". Na= False, so missing values become False, not NaN.
    if company is not None:
        frame = frame[frame["company"].str.contains(company, case=False, na=False)]

    if q is not None:
        frame = frame[frame["title"].str.contains(q, case=False, na=False)]

    # `tags` holds lists, so .str.contains does not apply - it treats every
    # non-string as missing and silently matches nothing. Test membership
    # per row instead.
    if tag is not None:
        wanted = tag.lower()
        frame = frame[frame["tags"].apply(
            lambda tags: any(wanted in str(t).lower() for t in (tags or []))
        )]

    # Three-state booleans: comparing the mask TO the flag handles True and
    # False in one expression, and None means "do not filter".
    if remote is not None:
        frame = frame[frame["remote"].fillna(False) == remote]

    if has_salary is not None:
        # Both bounds, to match salary_stats' definition of "has a salary".
        present = frame["salary_low"].notna() & frame["salary_high"].notna()
        frame = frame[present == has_salary]

    total = len(frame)

    # Without an explicit sort, pagination is undefined - page 2 could repeat
    # rows from page 1. Newest first is also what a reader wants.
    frame = frame.sort_values("posted_at", ascending=False, kind="stable")
    page = frame.iloc[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [JobSummary.model_validate(row) for row in _records(page)],
    }


@app.get("/jobs/{job_id}", response_model=JobDetail)
def job(job_id: str, df: pd.DataFrame = Depends(get_dataframe)):
    """One job, including its description.

    HTTPException rather than setting response.status_code: it short-circuits,
    so there is no success path to fall through into, and it produces the same
    {"detail": ...} shape as every other error in this API.
    """
    matches = df.loc[df["job_id"] == job_id]
    if matches.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"no job with id {job_id!r}"
        )
    return JobDetail.model_validate(_records(matches.head(1))[0])


@app.get("/chart/{name}", response_class=Response, responses={
    200: {"content": {"image/png": {}}},

})
def chart(name: str, df: pd.DataFrame = Depends(get_dataframe)):
    """A chart, as a PNG."""

    if name not in CHARTS:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"no such chart {name!r} allowed: {', '.join(CHARTS.keys())}", )
    analyse_fn, figure_fn = CHARTS[name]

    fig = figure_fn(analyse_fn(df))
    return Response(content=png_bytes(fig), media_type="image/png")
