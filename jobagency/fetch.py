from __future__ import annotations

import requests as rq
import json
from pathlib import Path
import time
from datetime import datetime, timezone

ARBEIT_NOW_URL = "https://www.arbeitnow.com/api/job-board-api"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"

LAKE_DIR = Path(__file__).parent / "lake"

USER_AGENT = "job-match-agency/0.1 (learning project)"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """A request failed after exhausting retries.

    Callers catch this rather than requests' own exceptions, so swapping
    an HTTP library later does not break them.
    """


def _get_json(url, *, timeout=20.0, attempts=3) -> dict:
    """GET `url` and return the parsed JSON body.

        The only function here that touches the network. Retries timeouts,
        connection errors and transient status codes with exponential backoff;
        does NOT retry 4xx, because a bad request repeated is still bad.
        """

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = rq.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if response.status_code in RETRYABLE_STATUS:
                raise rq.HTTPError(
                    f"{response.status_code} from {url}", response=response
                )
            response.raise_for_status()
            return response.json()
        except (rq.RequestException, ValueError) as error:
            last_error = error
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None and status not in RETRYABLE_STATUS:
                raise FetchError(f"{url} failed: {error}")
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s

    raise FetchError(f"{url} failed after {attempts} attempts: {last_error}")


def fetch_remotive(*, timeout: float = 20.0) -> dict:
    """The Remotive payload, unchanged: {"jobs": [...], "job-count": N, ...}.

        No query parameters - the ones it documents do not work.
        """
    return _get_json(REMOTIVE_URL, timeout=timeout)


def fetch_arbeitnow(
        *, max_pages: int = 1, delay: float = 1.0, timeout: float = 20.0
) -> dict:
    """Arbeitnow, paged, merged into ONE payload shaped {"data": [...]}.

    Follows `links.next` rather than building `?page=N`, because the
    parameter is undocumented on their site and the API hands us the full
    URL anyway.

    Note: the feed is ordered by `created_at` and refreshed hourly, so a
    multipage fetch is a slightly inconsistent snapshot - the same job can
    appear on two pages. `normalise_all` dedupes by job_id for this reason.
    """

    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    jobs: list[dict] = []
    url: str | None = ARBEIT_NOW_URL
    for page in range(max_pages):
        if url is None:
            break
        if page > 0:
            time.sleep(delay)  # the API asks not to be hammered
        payload = _get_json(url, timeout=timeout)
        jobs.extend(payload.get("data", []))
        url = (payload.get("links") or {}).get("next")

    return {"data": jobs}


def fetch_all(
        *, max_pages: int = 1, delay: float = 1.0, timeout: float = 20.0
) -> tuple[dict, dict]:
    """Both sources, in the order `normalise_all` expects: (remotive, arbeitnow)."""
    return (
        fetch_remotive(timeout=timeout),
        fetch_arbeitnow(max_pages=max_pages, delay=delay, timeout=timeout),
    )


def save_raw(payload: dict, name: str, directory: Path = LAKE_DIR) -> Path:
    """Write one payload to <directory>/<name>_<utc timestamp>.json.

    One JSON document per file. Appending several to one file would produce
    something no parser can read back.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{name}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_raw(path: Path) -> dict:
    """Read a saved payload back.

    Returns the same shape a fetch would, so everything downstream works
    identically online or offline.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    remotive, arbeitnow = fetch_all()
    save_raw(remotive, "remotive")
    save_raw(arbeitnow, "arbeitnow")


if __name__ == "__main__":
    main()
