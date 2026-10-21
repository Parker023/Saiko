"""Where fetched payloads live: locally for a few days, then only in miniS3.

Two tiers:

- **Local** - `~/data/jobagency/lake` by default, outside the repo so a `git
  clean` or a reclone cannot destroy it. Holds the last RETENTION_DAYS days.
- **Remote** - a miniS3 bucket on `ani-srv`, reached over Tailscale. Holds
  everything, forever.

The safety rule that makes this work:

    a local file is NEVER deleted unless miniS3 confirms it has a copy.

That one rule handles every failure mode on its own. Tailscale down? The
upload fails, the file stays, and the next run retries it. Server full? Same.
There is no path where a payload exists nowhere - which matters because these
payloads are not reproducible: a posting that drops off the feed is gone, and
the lake is the only record it ever existed.

Configuration, all optional except the API key:

    JOBAGENCY_LAKE    local directory         (default ~/data/jobagency/lake)
    MINIS3_ENDPOINT   base URL incl. /storage/minis3
    MINIS3_API_KEY    ms3_...                 (never hard-code it)
    MINIS3_BUCKET     bucket name             (default "jobagency")

With no endpoint or key configured, everything still works - it just stays
local, logs once, and never prunes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

# miniS3 is an optional extra: the client is a private package, and the whole
# point of this module is that the pipeline survives it being unavailable.
# Importing it unconditionally would make "optional" untrue - a fresh clone
# could not even import jobagency.storage.
try:
    from minis3 import MiniS3Client, MiniS3Error
except ImportError:                                   # pragma: no cover
    MiniS3Client = None

    class MiniS3Error(Exception):
        """Stand-in so the except clauses below still parse and catch nothing."""

from jobagency.fetch import load_raw, save_raw

logger = logging.getLogger(__name__)

LAKE_DIR = Path(os.environ.get("JOBAGENCY_LAKE", "~/data/jobagency/lake")).expanduser()

BUCKET = os.environ.get("MINIS3_BUCKET", "jobagency")

#: Days of payloads kept on disk. Older ones are deleted once uploaded.
RETENTION_DAYS = 3


@lru_cache(maxsize=1)
def _client() -> MiniS3Client | None:
    """The miniS3 client, or None if it is not configured.

    Cached: the client is thread-safe and meant to be created once. Returning
    None rather than raising keeps miniS3 optional - the pipeline must not
    stop working because a home server is off.
    """
    if MiniS3Client is None:
        logger.info(
            "the minis3 client is not installed; payloads stay local and are "
            "not pruned (see README: optional dependencies)"
        )
        return None

    endpoint = os.environ.get("MINIS3_ENDPOINT")
    api_key = os.environ.get("MINIS3_API_KEY")
    if not endpoint or not api_key:
        logger.info(
            "miniS3 not configured (MINIS3_ENDPOINT / MINIS3_API_KEY); "
            "payloads stay local and are not pruned"
        )
        return None
    return MiniS3Client(endpoint, api_key=api_key)


def remote_key(path: Path) -> str:
    """Local filename -> the object key in the bucket.

    `remotive_20261021T134225Z.json` -> `remotive/2026/10/21/remotive_...json`

    Date-partitioned so `list_objects(prefix="remotive/2026/10/")` returns one
    month, and the layout mirrors how you would query it later. The source
    prefix comes first so one bucket could hold other projects' data.
    """
    stamp = _stamp_from(path)
    source = path.stem.rsplit("_", 1)[0]
    if stamp is None:                     # unparseable name: keep it flat
        return f"{source}/unknown/{path.name}"
    return f"{source}/{stamp:%Y/%m/%d}/{path.name}"


def _stamp_from(path: Path) -> datetime | None:
    """The UTC timestamp encoded in a save_raw filename, or None.

    Reading the date from the NAME, not the file's mtime: copying or touching
    a file would change the mtime and silently move it between retention
    buckets. The name is what save_raw promised.
    """
    try:
        _, stamp = path.stem.rsplit("_", 1)
    except ValueError:
        return None
    # Two formats: files written before save_raw gained microsecond precision
    # use the shorter one, and they must keep parsing.
    for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def upload(path: Path) -> bool:
    """Put one local payload into the bucket. True if it is there afterwards.

    Idempotent: miniS3 replaces on an existing key, so re-uploading the same
    file is harmless - which is what makes the catch-up in prune() safe.
    """
    client = _client()
    if client is None:
        return False

    key = remote_key(path)
    try:
        info = client.put_object(BUCKET, key, path, content_type="application/json")
    except MiniS3Error as error:
        # status_code 0 means unreachable - Tailscale off, or the server down.
        logger.warning("miniS3 upload failed for %s (%s)", key, error)
        return False

    logger.info("uploaded %s -> %s/%s (%s bytes)", path.name, BUCKET, key, info.size)
    return True


def prune(directory: Path = LAKE_DIR, *, days: int = RETENTION_DAYS) -> list[Path]:
    """Delete local payloads older than `days` that miniS3 already holds.

    Also retries anything old that was never uploaded, so a run that happened
    while the server was down is caught up on the next one.

    Returns the paths actually deleted.
    """
    client = _client()
    if client is None or not directory.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted: list[Path] = []

    for path in sorted(directory.glob("*.json")):
        stamp = _stamp_from(path)
        if stamp is None or stamp >= cutoff:
            continue                       # unparseable, or still within the window

        key = remote_key(path)
        try:
            present = client.object_exists(BUCKET, key)
        except MiniS3Error as error:
            logger.warning("cannot check %s on miniS3 (%s); keeping it", key, error)
            continue

        # Catch-up: an old file that is not there yet was probably written
        # while the server was unreachable.
        if not present:
            logger.info("%s is older than %s days but not in miniS3; uploading", path.name, days)
            present = upload(path)

        if present:
            path.unlink()
            deleted.append(path)
            logger.info("pruned %s (safe: miniS3 has %s)", path.name, key)

    return deleted


def _digest(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def newest_for(name: str, directory: Path = LAKE_DIR) -> Path | None:
    """The most recent payload already stored for one source, if any.

    Timestamped filenames sort lexically by time, so the last one is newest.
    """
    matches = sorted(directory.glob(f"{name}_*.json")) if directory.exists() else []
    return matches[-1] if matches else None


def archive(
    payload: dict, name: str, *, directory: Path = LAKE_DIR, dedupe: bool = True
) -> Path:
    """Save a payload locally, push it to miniS3, then prune what is expired.

    The order matters: write first so the data exists on disk before anything
    can fail, upload second, prune last - and prune only ever removes files
    whose remote copy is confirmed.

    `dedupe` skips the writing entirely when the payload is byte-identical to
    the newest one already stored for this source. Arbeitnow refreshes hourly,
    so fetching more often than that produces exact repeats: measured at 318
    files holding 12 distinct payloads, 94% waste. What you lose is the record
    that a fetch happened and returned nothing new - which the logs cover.
    """
    if dedupe:
        previous = newest_for(name, directory)
        if previous is not None:
            incoming = json.dumps(payload, indent=2).encode("utf-8")
            if _digest(incoming) == _digest(previous.read_bytes()):
                logger.info(
                    "%s is unchanged since %s; not storing a duplicate",
                    name, previous.name,
                )
                upload(previous)       # in case an earlier upload failed
                prune(directory)
                return previous

    path = save_raw(payload, name, directory)
    upload(path)
    prune(directory)
    return path


def local_payloads(directory: Path = LAKE_DIR) -> list[Path]:
    """Every payload currently on disk, oldest first."""
    return sorted(directory.glob("*.json")) if directory.exists() else []


__all__ = [
    "LAKE_DIR", "BUCKET", "RETENTION_DAYS",
    "archive", "upload", "prune", "remote_key", "local_payloads", "load_raw",
]
