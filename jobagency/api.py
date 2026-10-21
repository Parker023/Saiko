"""HTTP API over the job analysis.

The DataFrame is loaded once in `lifespan` - at app startup, not at import -
so importing this module is free and tests can start the app with whatever
data they like via `dependency_overrides`.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Request, Response, status, HTTPException

from jobagency.main import load_jobs_with_fallback, build_report
from jobagency.models import JobStatsResponse

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
def stats(df: pd.DataFrame = Depends(get_dataframe)) :
    return build_report(df)
