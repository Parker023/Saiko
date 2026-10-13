from __future__ import annotations
import numpy as np
from datetime import datetime, timezone
import re

NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*([kK])?")


def _to_amount(digits: str, k_suffix: str | None):
    """'35,3' + 'k' -> 35300.0; '35,000' -> 35000.0

        A comma means two different things in this feed. Three digits after it are
a thousand separators ("35,000"); one or two is a European decimal comma
        ("35,3"). Guessing wrong is a factor-of-1000 error, so decide explicitly.
        """

    if "," in digits:
        whole, _, fraction = digits.partition(",")
        digits = whole + fraction if len(fraction) >= 3 else f"{whole}.{fraction}"
    value = float(digits)
    return value * 1000 if k_suffix else value


def parse_salary(raw: str | None) -> tuple[float, float, str] | None:
    """Parse a salary string into (low, high, period).

       Returns None when nothing is parseable - which is the common case, since
       most postings publish no salary at all. The period is kept rather than
       converted: turning hourly into annual needs an assumed hour-per-year,
       and that assumption does not belong hidden inside a parser.
       """
    if not raw or not raw.strip():
        return None

    text = raw.strip().lower()
    period = "hour" if "hour" in text or "/hr" in text else "year"
    amounts = [_to_amount(d, k) for d, k in NUMBER.findall(text)]
    if not amounts:
        return None
    arr = np.array(amounts)
    low, high = arr.min(), arr.max()
    return low, high, period


def normalise_remotive(job: dict) -> dict:
    """One Remotive job -> the common record shape."""
    salary = parse_salary(job.get("salary"))
    return {
        "source": "remotive",
        "job_id": f"remotive:{job['id']}",
        "title": job["title"],
        "company": job["company_name"],
        "category": job.get("category"),
        "tags": job.get("tags") or [],
        "job_type": job.get("job_type"),
        # Remotive sends a naive ISO string. Assume UTC and say so, rather
        # than mixing naive and aware datetimes - pandas rises on that.
        "posted_at": datetime.fromisoformat(job["publication_date"]).replace(
            tzinfo=timezone.utc
        ),
        "location": job.get("candidate_required_location"),
        "remote": True,  # it is a remote-only board
        "salary_low": salary[0] if salary else None,
        "salary_high": salary[1] if salary else None,
        "salary_period": salary[2] if salary else None,
        "description_html": job.get("description") or "",
        "url": job.get("url"),
    }


def normalise_arbeitnow(job: dict) -> dict:
    """One Arbeitnow job -> the common record shape."""
    job_types = job.get("job_types") or []
    return {
        "source": "arbeitnow",
        "job_id": f"arbeitnow:{job['slug']}",
        "title": job["title"],
        "company": job["company_name"],
        "category": None,  # this source has no category - do not invent one
        "tags": job.get("tags") or [],
        "job_type": job_types[0] if job_types else None,
        "posted_at": datetime.fromtimestamp(job["created_at"], tz=timezone.utc),
        "location": job.get("location"),
        "remote": bool(job.get("remote")),
        "salary_low": None,  # not published by this source at all
        "salary_high": None,
        "salary_period": None,
        "description_html": job.get("description") or "",
        "url": job.get("url"),
    }


def normalise_all(remotive_payload: dict, arbeitnow_payload: dict) -> list[dict]:
    """Both payloads -> one list of records, duplicates removed by job_id."""
    records = [normalise_remotive(j) for j in remotive_payload.get("jobs", [])]
    records += [normalise_arbeitnow(j) for j in arbeitnow_payload.get("data", [])]

    seen, unique = set(), []
    for record in records:
        if record["job_id"] not in seen:
            seen.add(record["job_id"])
            unique.append(record)
    return unique
