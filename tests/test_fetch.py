"""Tests for jobagency.fetch

No test here touches the network except the one marked `integration`.
Everything else replaces the single function that does I/O - which is the
payoff for concentrating all of it in `_get_json`.
"""

import json

import pytest
import requests

from jobagency import fetch
from jobagency.fetch import (
    ARBEIT_NOW_URL,
    REMOTIVE_URL,
    FetchError,
    fetch_all,
    fetch_arbeitnow,
    fetch_remotive,
    load_raw,
    save_raw,
)


# --- helpers -------------------------------------------------------------

def _page(slugs, next_url=None):
    """One fake Arbeitnow page."""
    return {
        "data": [{"slug": s} for s in slugs],
        "links": {"next": next_url},
        "meta": {"current_page": 1},
    }


class FakeResponse:
    """Enough of requests.Response for _get_json to work against."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture
def no_sleep(monkeypatch):
    """Backoff sleeps 1s then 2s. Tests should not wait for that."""
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)


# =========================================================================
# the wrappers - _get_json replaced, so paging logic is tested in isolation
# =========================================================================

def test_arbeitnow_merges_pages_into_one_payload(monkeypatch):
    pages = [
        _page(["a", "b"], next_url=f"{ARBEIT_NOW_URL}?page=2"),
        _page(["c", "d"], next_url=None),
    ]
    monkeypatch.setattr(fetch, "_get_json", lambda url, **kw: pages.pop(0))

    result = fetch_arbeitnow(max_pages=2, delay=0)

    assert list(result) == ["data"]          # one payload, not a list of pages
    assert [j["slug"] for j in result["data"]] == ["a", "b", "c", "d"]


def test_arbeitnow_stops_when_next_is_none(monkeypatch):
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return _page(["only"], next_url=None)

    monkeypatch.setattr(fetch, "_get_json", fake)

    fetch_arbeitnow(max_pages=5, delay=0)
    assert len(calls) == 1                   # did not ask for a page that is not there


def test_arbeitnow_respects_max_pages_when_next_never_ends(monkeypatch):
    """A feed that always offers another page must not loop forever."""
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return _page(["x"], next_url=f"{ARBEIT_NOW_URL}?page=99")

    monkeypatch.setattr(fetch, "_get_json", fake)

    result = fetch_arbeitnow(max_pages=3, delay=0)
    assert len(calls) == 3
    assert len(result["data"]) == 3


def test_arbeitnow_follows_the_next_url_it_was_given(monkeypatch):
    """Not ?page=N built by hand - whatever URL the server hands back."""
    calls = []
    pages = [
        _page(["a"], next_url="https://example.test/surprise?cursor=abc"),
        _page(["b"], next_url=None),
    ]

    def fake(url, **kw):
        calls.append(url)
        return pages.pop(0)

    monkeypatch.setattr(fetch, "_get_json", fake)

    fetch_arbeitnow(max_pages=2, delay=0)
    assert calls == [ARBEIT_NOW_URL, "https://example.test/surprise?cursor=abc"]


def test_arbeitnow_handles_a_page_with_no_data_key(monkeypatch):
    monkeypatch.setattr(fetch, "_get_json", lambda url, **kw: {})
    assert fetch_arbeitnow(max_pages=1, delay=0) == {"data": []}


@pytest.mark.parametrize("bad", [0, -1])
def test_arbeitnow_rejects_max_pages_below_one(bad):
    with pytest.raises(ValueError):
        fetch_arbeitnow(max_pages=bad, delay=0)


def test_arbeitnow_propagates_fetch_error(monkeypatch):
    """A failed fetch must not come back as an empty result set."""
    def boom(url, **kw):
        raise FetchError("down")

    monkeypatch.setattr(fetch, "_get_json", boom)
    with pytest.raises(FetchError):
        fetch_arbeitnow(max_pages=1, delay=0)


def test_remotive_calls_the_bare_url_with_no_params(monkeypatch):
    """Its documented params do nothing, so we must not pretend to send any."""
    calls = []
    monkeypatch.setattr(
        fetch, "_get_json",
        lambda url, **kw: calls.append(url) or {"jobs": []},
    )
    fetch_remotive()
    assert calls == [REMOTIVE_URL]


def test_fetch_all_returns_remotive_first(monkeypatch):
    """normalise_all(remotive, arbeitnow) - the order is the whole point."""
    monkeypatch.setattr(
        fetch, "_get_json",
        lambda url, **kw: {"jobs": ["R"]} if url == REMOTIVE_URL
        else _page(["A"], next_url=None),
    )
    first, second = fetch_all(max_pages=1, delay=0)
    assert "jobs" in first
    assert "data" in second


# =========================================================================
# _get_json itself - requests.get replaced, so retry logic is tested
# =========================================================================

def test_get_json_returns_parsed_body(monkeypatch):
    monkeypatch.setattr(fetch.rq, "get",
                        lambda url, **kw: FakeResponse(200, {"hello": "world"}))
    assert fetch._get_json("https://example.test") == {"hello": "world"}


def test_get_json_retries_a_500_then_succeeds(monkeypatch, no_sleep):
    responses = [FakeResponse(500), FakeResponse(200, {"ok": 1})]
    monkeypatch.setattr(fetch.rq, "get", lambda url, **kw: responses.pop(0))

    assert fetch._get_json("https://example.test", attempts=3) == {"ok": 1}
    assert responses == []                   # both were consumed


def test_get_json_does_not_retry_a_404(monkeypatch, no_sleep):
    """A 4xx is our bug. Repeating it three times is three wrong requests."""
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return FakeResponse(404)

    monkeypatch.setattr(fetch.rq, "get", fake)

    with pytest.raises(FetchError):
        fetch._get_json("https://example.test", attempts=3)
    assert len(calls) == 1                   # tried once, gave up


def test_get_json_retries_429(monkeypatch, no_sleep):
    """Rate limiting IS worth retrying, unlike other 4xx."""
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return FakeResponse(429)

    monkeypatch.setattr(fetch.rq, "get", fake)

    with pytest.raises(FetchError):
        fetch._get_json("https://example.test", attempts=3)
    assert len(calls) == 3


def test_get_json_raises_after_exhausting_attempts(monkeypatch, no_sleep):
    monkeypatch.setattr(
        fetch.rq, "get",
        lambda url, **kw: (_ for _ in ()).throw(rq_timeout()),
    )
    with pytest.raises(FetchError):
        fetch._get_json("https://example.test", attempts=2)


def rq_timeout():
    return requests.Timeout("too slow")


@pytest.mark.parametrize("bad", [0, -1])
def test_get_json_rejects_attempts_below_one(bad):
    with pytest.raises(ValueError):
        fetch._get_json("https://example.test", attempts=bad)


def test_get_json_sends_a_user_agent_and_timeout(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(kw)
        return FakeResponse(200)

    monkeypatch.setattr(fetch.rq, "get", fake)
    fetch._get_json("https://example.test", timeout=7.5)

    assert seen["timeout"] == 7.5
    assert "User-Agent" in seen["headers"]


# =========================================================================
# the disk cache
# =========================================================================

def test_save_and_load_round_trip(tmp_path):
    payload = {"jobs": [{"id": 1, "title": "Dev"}], "job-count": 1}
    path = save_raw(payload, "remotive", tmp_path)
    assert load_raw(path) == payload


def test_save_raw_creates_the_directory(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    path = save_raw({"a": 1}, "x", target)
    assert path.exists()


def test_save_raw_filename_has_name_and_timestamp(tmp_path):
    path = save_raw({"a": 1}, "arbeitnow", tmp_path)
    assert path.name.startswith("arbeitnow_")
    assert path.suffix == ".json"


def test_save_raw_writes_one_valid_json_document(tmp_path):
    """Appending several documents to one file would be unreadable."""
    path = save_raw({"a": 1}, "x", tmp_path)
    json.loads(path.read_text())             # raises if it is not valid JSON


def test_two_saves_never_overwrite_each_other(tmp_path):
    """Second-precision names collided when two payloads were written in the
    same second - the first was silently lost. Microseconds fixed it."""
    a = save_raw({"n": 1}, "x", tmp_path)
    b = save_raw({"n": 2}, "x", tmp_path)
    assert a != b
    assert load_raw(a) == {"n": 1}
    assert load_raw(b) == {"n": 2}


# =========================================================================
# the one test that really talks to the internet
# =========================================================================

@pytest.mark.integration
def test_remotive_really_responds():
    """Run deliberately: pytest -m integration

    Kept because both APIs silently ignore documented parameters - this is
    how you find out they changed something.
    """
    payload = fetch_remotive()
    assert "jobs" in payload and len(payload["jobs"]) > 0
