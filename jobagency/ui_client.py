"""HTTP client for the Job Match Agency API.

The UI's ONLY link to the backend. Nothing here imports jobagency.analyse,
jobagency.fetch or jobagency.normalise - if it did, FastAPI would become
decorative and you could delete the API without the UI noticing.

Every function returns parsed JSON or raises ApiError. Streamlit code handles
ApiError once, at the top, rather than wrapping every call.
"""

from __future__ import annotations

import os

import requests

# Overridable so the same UI can point at a deployed backend later.
API_BASE = os.environ.get("JOB_API_BASE", "http://localhost:8000")

TIMEOUT = 15.0


class ApiError(RuntimeError):
    """The API could not be reached, or answered with an error."""


def _get(path: str, params: dict | None = None) -> dict:
    """GET one endpoint. The single place network failure is translated.

    Strips None values from params: Streamlit widgets naturally produce None
    for "no filter", and requests would otherwise send `source=None` as a
    literal string.
    """
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        response = requests.get(f"{API_BASE}{path}", params=clean, timeout=TIMEOUT)
    except requests.RequestException as error:
        raise ApiError(f"cannot reach the API at {API_BASE}: {error}") from error

    if response.status_code >= 400:
        # The API uses {"detail": ...} for errors; fall back to raw text.
        try:
            detail = response.json().get("detail") or response.json().get("reason")
        except ValueError:
            detail = response.text[:200]
        raise ApiError(f"{response.status_code} from {path}: {detail}")

    return response.json()


def health() -> dict:
    """Backend liveness, data source and freshness. Never raises on 503 -
    'unhealthy' is information the UI wants to display, not an error."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        return response.json() | {"http_status": response.status_code}
    except requests.RequestException as error:
        raise ApiError(f"cannot reach the API at {API_BASE}: {error}") from error


def stats() -> dict:
    return _get("/stats")


def jobs(**filters) -> dict:
    """A page of job summaries. Accepts the API's query parameters directly."""
    return _get("/jobs", filters)


def job(job_id: str) -> dict:
    return _get(f"/jobs/{job_id}")


def chart_png(name: str) -> bytes:
    """A chart as raw PNG bytes, for st.image."""
    try:
        response = requests.get(f"{API_BASE}/chart/{name}", timeout=TIMEOUT)
    except requests.RequestException as error:
        raise ApiError(f"cannot reach the API at {API_BASE}: {error}") from error
    if response.status_code >= 400:
        raise ApiError(f"{response.status_code} from /chart/{name}")
    return response.content


def refresh() -> dict:
    """Ask the backend to re-load its data. POST, because it mutates."""
    try:
        response = requests.post(f"{API_BASE}/refresh", timeout=60.0)
    except requests.RequestException as error:
        raise ApiError(f"cannot reach the API at {API_BASE}: {error}") from error
    return response.json() | {"http_status": response.status_code}
