"""HTTP API over the job analysis.

The DataFrame is loaded once in `lifespan` - at app startup, not at import -
so importing this module is free and tests can start the app with whatever
data they like via `dependency_overrides`.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status, HTTPException, Query

from jobagency.main import load_jobs_with_fallback, build_report
from jobagency.models import JobStatsResponse, JobSummary, JobDetail

logger = logging.getLogger(__name__)


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


@app.get("/jobs")
def jobs(*,
         source: str | None = None,
         company: str | None = Query(default=None, max_length=100),
         category: str | None = Query(default=None, max_length=100),
         job_type: str | None = Query(default=None, max_length=50),
         tag: str | None = Query(default=None, max_length=50),
         q: str | None = Query(default=None, max_length=100, min_length=2),
         remote: bool | None = Query(default=None),
         has_salary: bool | None = Query(default=None),
         limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0),
         df: pd.DataFrame = Depends(get_dataframe)):
    """Search for jobs."""
    df_new = df.copy()

    # starts filtering
    df_new = df_new[df_new["source"] == source.lower()] if source else df_new
    df_new = df_new[df_new["company"].str.contains(company, case=False, na=False)] if company else df_new
    df_new = df_new[df_new["category"] == category.lower()] if category else df_new
    df_new = df_new[df_new["job_type"] == job_type.lower()] if job_type else df_new
    df_new = df_new[df_new["tags"].str.contains(tag, case=False, na=False)] if tag else df_new
    df_new = df_new[df_new["title"].str.contains(q, case=False, na=False)] if q else df_new
    df_new = df_new[df_new["remote"] == remote] if remote is not None else df_new
    df_new = df_new[df_new["salary_low"].notna()] if has_salary is not None else df_new
    df_new = df.dropna(subset=["salary_low"]) if has_salary is None else df_new
    df_new = df.dropna(subset=["salary_high"]) if has_salary is None else df_new

    total_jobs = len(df_new)
    df_new = df_new.iloc[offset:offset + limit]

    # ends filtering

    items = [JobSummary.model_validate(row) for row in df_new.to_dict(orient="records")]

    return {
        "total": total_jobs,
        "limit": limit,
        "offset": offset,
        "items": items
    }


@app.get("/jobs/{job_id}")
def job(job_id: str, resp: Response, df: pd.DataFrame = Depends(get_dataframe)):
    """Get a single job by its ID."""
    matching_rows = df.loc[df["job_id"] == job_id]

    if not matching_rows.empty:
        row = matching_rows.iloc[0]
        resp.status_code = status.HTTP_200_OK
        return JobDetail.model_validate(row.to_dict())
    else:
        resp.status_code = status.HTTP_404_NOT_FOUND
        return {"error": f"no job with ID {job_id}"}
