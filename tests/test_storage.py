"""Tests for jobagency.storage

The rule under test, which everything else depends on:

    a local file is NEVER deleted unless miniS3 confirms it has a copy.

These payloads are not reproducible - a posting that drops off the feed is
gone - so a bug here loses data permanently. That is why the failure cases
get more tests than the happy path.

No network: a fake client stands in for miniS3.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from minis3 import MiniS3Error

from jobagency import storage


class FakeMiniS3:
    """An in-memory stand-in. `fail_on` makes one operation raise."""

    def __init__(self, existing=(), fail_on=()):
        self.objects = {key: b"{}" for key in existing}
        self.fail_on = set(fail_on)
        self.puts: list[str] = []
        self.exists_calls: list[str] = []

    def put_object(self, bucket, key, data, content_type=None):
        if "put" in self.fail_on:
            raise MiniS3Error(0, "server unreachable")
        self.puts.append(key)
        self.objects[key] = b"{}"
        return type("Info", (), {"size": 2, "key": key})()

    def object_exists(self, bucket, key):
        self.exists_calls.append(key)
        if "exists" in self.fail_on:
            raise MiniS3Error(0, "server unreachable")
        return key in self.objects


@pytest.fixture
def fake(monkeypatch):
    """Install a fake client, bypassing the lru_cache on _client()."""
    client = FakeMiniS3()
    storage._client.cache_clear()          # drop any real client built earlier
    monkeypatch.setattr(storage, "_client", lambda: client)
    return client                          # monkeypatch undoes the patch itself


@pytest.fixture
def unconfigured(monkeypatch):
    """As if MINIS3_ENDPOINT / MINIS3_API_KEY are not set."""
    storage._client.cache_clear()
    monkeypatch.setattr(storage, "_client", lambda: None)


def _write(directory: Path, name: str, age_days: float) -> Path:
    """A payload file whose NAME encodes an age, as save_raw would write it."""
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    path = directory / f"{name}_{stamp:%Y%m%dT%H%M%SZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": []}))
    return path


# --- remote_key ----------------------------------------------------------

def test_remote_key_is_date_partitioned():
    key = storage.remote_key(Path("remotive_20261021T134225Z.json"))
    assert key == "remotive/2026/10/21/remotive_20261021T134225Z.json"


def test_remote_key_keeps_the_source_prefix():
    """So one bucket can hold other projects without collisions."""
    assert storage.remote_key(
        Path("arbeitnow_20260903T010203Z.json")
    ).startswith("arbeitnow/")


def test_remote_key_survives_an_unparseable_name():
    key = storage.remote_key(Path("weird-name.json"))
    assert "unknown" in key and key.endswith("weird-name.json")


def test_age_is_read_from_the_name_not_the_mtime(tmp_path):
    """Copying or touching a file must not move it between retention buckets."""
    path = _write(tmp_path, "remotive", age_days=10)
    path.touch()                                    # mtime is now
    stamp = storage._stamp_from(path)
    assert (datetime.now(timezone.utc) - stamp).days == 10


# --- upload --------------------------------------------------------------

def test_upload_sends_the_file(tmp_path, fake):
    path = _write(tmp_path, "remotive", age_days=0)
    assert storage.upload(path) is True
    assert fake.puts == [storage.remote_key(path)]


def test_upload_returns_false_when_unconfigured(tmp_path, unconfigured):
    """miniS3 is optional - a missing key must not break the pipeline."""
    assert storage.upload(_write(tmp_path, "remotive", age_days=0)) is False


def test_upload_returns_false_when_the_server_is_unreachable(tmp_path, monkeypatch):
    client = FakeMiniS3(fail_on=["put"])
    monkeypatch.setattr(storage, "_client", lambda: client)
    assert storage.upload(_write(tmp_path, "remotive", age_days=0)) is False


# --- prune: the safety rule ----------------------------------------------

def test_prune_deletes_old_files_that_minis3_has(tmp_path, fake):
    old = _write(tmp_path, "remotive", age_days=5)
    fake.objects[storage.remote_key(old)] = b"{}"

    deleted = storage.prune(tmp_path, days=3)

    assert deleted == [old]
    assert not old.exists()


def test_prune_keeps_files_inside_the_window(tmp_path, fake):
    """Even when miniS3 already has them - the window is the point."""
    recent = _write(tmp_path, "remotive", age_days=1)
    fake.objects[storage.remote_key(recent)] = b"{}"

    assert storage.prune(tmp_path, days=3) == []
    assert recent.exists()


def test_prune_never_deletes_a_file_minis3_does_not_have(tmp_path, monkeypatch):
    """THE rule. An upload that fails must not cost us the only copy."""
    client = FakeMiniS3(fail_on=["put"])       # uploads fail, nothing is stored
    storage._client.cache_clear()
    monkeypatch.setattr(storage, "_client", lambda: client)

    old = _write(tmp_path, "remotive", age_days=99)

    assert storage.prune(tmp_path, days=3) == []
    assert old.exists(), "deleted a payload that exists nowhere else"


def test_prune_retries_an_old_file_that_was_never_uploaded(tmp_path, fake):
    """Catch-up: a run that happened while the server was down is recovered."""
    old = _write(tmp_path, "remotive", age_days=5)
    assert storage.remote_key(old) not in fake.objects

    deleted = storage.prune(tmp_path, days=3)

    assert fake.puts == [storage.remote_key(old)]   # uploaded first
    assert deleted == [old]                         # then safe to delete


def test_prune_keeps_the_file_if_the_existence_check_errors(tmp_path, monkeypatch):
    """Unknown is not the same as confirmed. Never delete on uncertainty."""
    client = FakeMiniS3(fail_on=["exists"])
    monkeypatch.setattr(storage, "_client", lambda: client)

    old = _write(tmp_path, "remotive", age_days=5)
    assert storage.prune(tmp_path, days=3) == []
    assert old.exists()


def test_prune_does_nothing_when_unconfigured(tmp_path, unconfigured):
    """With no miniS3 there is no second copy, so nothing may be removed."""
    old = _write(tmp_path, "remotive", age_days=99)
    assert storage.prune(tmp_path, days=3) == []
    assert old.exists()


def test_prune_ignores_unparseable_filenames(tmp_path, fake):
    """A file we cannot date is a file we cannot judge - leave it alone."""
    stray = tmp_path / "notes.json"
    stray.write_text("{}")
    assert storage.prune(tmp_path, days=3) == []
    assert stray.exists()


def test_prune_handles_a_missing_directory(tmp_path, fake):
    assert storage.prune(tmp_path / "nope", days=3) == []


# --- archive -------------------------------------------------------------

def test_archive_writes_uploads_and_prunes(tmp_path, fake):
    old = _write(tmp_path, "remotive", age_days=5)
    fake.objects[storage.remote_key(old)] = b"{}"

    path = storage.archive({"jobs": [1]}, "remotive", directory=tmp_path)

    assert path.exists()                                   # written
    assert storage.remote_key(path) in fake.objects        # uploaded
    assert not old.exists()                                # pruned


def test_archive_still_writes_locally_when_minis3_is_down(tmp_path, monkeypatch):
    """The whole pipeline must survive the home server being off."""
    client = FakeMiniS3(fail_on=["put"])
    monkeypatch.setattr(storage, "_client", lambda: client)

    path = storage.archive({"jobs": []}, "remotive", directory=tmp_path)

    assert path.exists()
    assert json.loads(path.read_text()) == {"jobs": []}


def test_archive_round_trips_through_load_raw(tmp_path, fake):
    payload = {"jobs": [{"id": 1}], "job-count": 1}
    path = storage.archive(payload, "remotive", directory=tmp_path)
    assert storage.load_raw(path) == payload


# --- deduplication -------------------------------------------------------

def test_archive_skips_a_byte_identical_payload(tmp_path, fake):
    """Arbeitnow refreshes hourly; fetching more often returns exact repeats.
    Measured on real runs: 318 files held 12 distinct payloads."""
    payload = {"jobs": [{"id": 1}]}

    first = storage.archive(payload, "remotive", directory=tmp_path)
    second = storage.archive(payload, "remotive", directory=tmp_path)

    assert second == first
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_archive_stores_a_changed_payload(tmp_path, fake):
    storage.archive({"jobs": [{"id": 1}]}, "remotive", directory=tmp_path)
    storage.archive({"jobs": [{"id": 2}]}, "remotive", directory=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_archive_dedupe_can_be_turned_off(tmp_path, fake):
    payload = {"jobs": []}
    storage.archive(payload, "remotive", directory=tmp_path, dedupe=False)
    storage.archive(payload, "remotive", directory=tmp_path, dedupe=False)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_dedupe_is_per_source(tmp_path, fake):
    """Two sources with the same bytes are still two different things."""
    payload = {"data": []}
    storage.archive(payload, "remotive", directory=tmp_path, dedupe=True)
    storage.archive(payload, "arbeitnow", directory=tmp_path, dedupe=True)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_a_skipped_duplicate_still_retries_its_upload(tmp_path, fake):
    """If the first upload failed, the next identical fetch catches it up."""
    payload = {"jobs": []}
    first = storage.archive(payload, "remotive", directory=tmp_path)
    fake.objects.clear()                       # as if the upload had failed
    fake.puts.clear()

    storage.archive(payload, "remotive", directory=tmp_path)

    assert fake.puts == [storage.remote_key(first)]


def test_storage_imports_without_the_minis3_package(monkeypatch):
    """miniS3 is optional, so the import must be optional too.

    Importing it unconditionally would mean a fresh clone could not even
    import jobagency.storage - "optional" would be a claim the code does not
    honour.
    """
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "minis3", None)
    for name in [m for m in sys.modules if m.startswith("jobagency.storage")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("jobagency.storage")
    importlib.reload(module)

    assert module._client() is None
    assert module.upload(Path("/tmp/does-not-matter.json")) is False
