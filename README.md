# Job Match Agency

A local job-search assistant. Fetches postings from two public job boards,
normalises them into one shape, analyses them, serves the results over HTTP,
and (later) answers questions about them with a local LLM.

Built as a learning project covering the full stack: `requests` → `pandas` /
`numpy` / `matplotlib` → `FastAPI` → `Streamlit` → RAG → tool calling.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Optionally, to archive payloads to miniS3 (see
[Storage](#storage-local-lake--minis3)) — `requirements-minis3.txt` explains why
this is not a normal requirement:

```bash
pip install -r requirements-minis3.txt
export MINIS3_ENDPOINT="https://ani-srv.tail1678db.ts.net/storage/minis3"
export MINIS3_API_KEY="ms3_..."          # never commit this
```

**Run the pipeline as a script** — fetch, analyse, write charts, print a summary:

```bash
python -m jobagency.main                # fetch live, 1 page of Arbeitnow
python -m jobagency.main --pages 3      # more pages
python -m jobagency.main --offline      # reuse the newest saved payloads
```

**Run the API** (terminal 1):

```bash
uvicorn jobagency.api:app --reload          # port 8000
```

<http://localhost:8000/docs> gives interactive documentation;
<http://localhost:8000/chart/tags> gives a chart.

**Run the UI** (terminal 2, with the API already running):

```bash
PYTHONPATH=. streamlit run jobagency/ui.py  # port 8501
```

> `PYTHONPATH=.` is needed because `streamlit run jobagency/ui.py` puts
> `jobagency/` on `sys.path`, not the project root — so `from jobagency import
> ui_client` cannot resolve. The proper fix is a `pyproject.toml` and
> `pip install -e .`, which would also settle pytest's rootdir.

**Run the tests:**

```bash
pytest                                  # fast, offline, no network
pytest -m integration -o addopts=""     # the one test that hits a real API
```

---

## How it fits together

```
fetch.py      HTTP only. Every network call goes through _get_json.
    |
normalise.py  Two API shapes -> one record shape. Pure functions.
    |
analyse.py    pandas / numpy. DataFrame in, plain JSON-ready Python out.
    |         <- the last place a DataFrame exists
    +-- chart.py   analysis results -> PNG bytes
    |
main.py       Wires the pipeline; also the CLI.
    |
api.py        HTTP interface. Loads data once at startup, serves it.
    |
    |  ~~~~~~ process boundary: everything below speaks only HTTP ~~~~~~
    |
ui_client.py  One function per endpoint. The UI's only link to the backend.
    |
ui.py         Streamlit. Imports ui_client and nothing else from the package.
```

### The rules that shape it

**All I/O is concentrated in one function.** `fetch._get_json` is the only
thing that touches the network, so timeouts, retries and headers live in one
place — and tests mock one function to cover every source.

**`analyse.py` returns plain Python, never a DataFrame.** That is why
`/stats` is `return build_report(df)` with no adapter, and why `chart.py`
works equally well from a live frame or from a JSON response over HTTP.

**Data is loaded in FastAPI's `lifespan`, not at import.** Importing
`jobagency.api` is free, so tests can start the app with any data via
`dependency_overrides`.

**Preconditions live in dependencies.** `get_dataframe` raises 503 if no data
loaded, so no endpoint can receive `None`.

**Anything a caller names is an allowlist** — `COUNTABLE` columns in
`count_by`, `CHARTS` names in `/chart/{name}`. Never `getattr` on user input.

**The UI is an HTTP client, not an import.** `ui.py` may import `ui_client`
and nothing else from this package. If it imported `analyse` directly the API
would be decorative — so `tests/test_ui_boundary.py` enforces it by parsing
the imports.

---

## Endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | liveness, data source, freshness, `degraded` flag |
| `POST` | `/refresh` | re-loads the data, reports what happened |
| `GET` | `/stats` | the full analysis report |
| `GET` | `/jobs` | a filtered, paginated page of job summaries |
| `GET` | `/jobs/{job_id}` | one job, including its description |
| `GET` | `/chart/{name}` | a PNG: `trend`, `tags` or `salary_coverage` |

### `/jobs` parameters

| Name | Type | Default | Matches |
|---|---|---|---|
| `source` | str | — | exact, case-insensitive |
| `company` | str | — | **substring**, case-insensitive |
| `category` | str | — | exact, case-insensitive |
| `job_type` | str | — | exact, case-insensitive |
| `tag` | str | — | membership in the `tags` list |
| `q` | str | — | substring in `title` (min 2 chars) |
| `remote` | bool | unset | three-state: true / false / don't filter |
| `has_salary` | bool | unset | three-state; requires **both** bounds |
| `limit` | int | 20 | 1–100 |
| `offset` | int | 0 | ≥ 0 |

Filters combine with AND. Results are sorted newest first, then sliced — so
pagination is stable. `total` is the count after filtering, before slicing.

---

## Storage: local lake + miniS3

Fetched payloads live in two tiers.

| | Where | Holds |
|---|---|---|
| **Local** | `~/data/jobagency/lake` (override with `JOBAGENCY_LAKE`) | the last 3 days |
| **Remote** | miniS3 bucket `jobagency` on `ani-srv`, over Tailscale | everything |

Object keys are date-partitioned, source first:

```
remotive/2026/10/21/remotive_20261021T135500123456Z.json
```

So `list_objects(prefix="remotive/2026/10/")` returns one month, and a single
bucket could hold other projects without collisions.

### The rule everything rests on

> **A local file is never deleted unless miniS3 confirms it has a copy.**

That one rule covers every failure mode without special cases:

| Failure | Result |
|---|---|
| Tailscale off | upload fails → file stays → retried next run |
| Server errors during prune | `object_exists` raises → file kept (unknown is not confirmed) |
| miniS3 not installed or not configured | nothing uploads, **nothing prunes** |
| An old file was never uploaded | prune uploads it *first*, then deletes |

It matters because **these payloads are not reproducible**: a posting that
drops off the feed is gone, and the lake is the only record it existed. Eight
of the storage tests are failure cases for that reason.

### Deduplication

`archive()` skips a payload that is byte-identical to the newest one already
stored for that source. Arbeitnow refreshes hourly, so fetching more often
returns exact repeats — measured on real runs at **318 files holding 12
distinct payloads, 94% waste**. What is lost is the record that a fetch
happened and found nothing new; the logs cover that. Pass `dedupe=False` to
keep every fetch.

### miniS3 is optional, including at import

`storage.py` catches the `ImportError`. Without the client the project still
imports, fetches, analyses and serves — it simply never uploads and never
prunes. An unconditional import would make "optional" a claim the code did not
honour.

## The UI

A Streamlit app on port 8501 that talks to the API on 8000 over HTTP. Two
processes, one repo.

- **Sidebar** — backend health, data source, freshness, a refresh button, and
  the filters.
- **Table** — a page of summaries. Click a row to fetch that job's full
  description from `/jobs/{id}`; the list itself never carries descriptions.
- **Insights** — the three charts as PNGs straight from `/chart/{name}`, plus
  headline numbers from `/stats`.

Three details worth knowing:

**Streamlit re-runs the entire script on every interaction.** There are no
event handlers — the script *is* the handler. That is why every API call goes
through `@st.cache_data`, and why the page number lives in `st.session_state`
rather than an ordinary variable.

**The three-state filters are why the API uses `bool | None`.** "Any / Remote
/ On-site" cannot be expressed by a checkbox, and an unset widget sends no
parameter at all rather than an empty one.

**Charts arrive as bytes.** `st.image` takes them directly, so nothing is
written to disk — which is what `chart.png_bytes` exists for, alongside
`save_png` for the CLI.

> The job detail panel renders the board's own HTML with
> `unsafe_allow_html=True`. Acceptable for a local single-user tool; it would
> need sanitising before this was exposed to anyone else.

## The data, and what is wrong with it

Both sources are free and need no API key. Everything below was measured, not
read in a doc.

| | Remotive | Arbeitnow |
|---|---|---|
| Endpoint | `remotive.com/api/remote-jobs` | `arbeitnow.com/api/job-board-api` |
| Volume | ~15 per request | 300+ per page, paginated |
| Salary | a free-text string | none at all |
| Category | yes | none |

**Query parameters mostly do not work.** Remotive ignores `limit` *and*
`category`. Arbeitnow ignores `remote`, `search` and `per_page`. The only one
that works anywhere is Arbeitnow's `page`, and that is followed via
`links.next` rather than constructed. **All filtering happens client-side.**

**Arbeitnow pagination drifts.** The feed is ordered by `created_at` and
refreshed hourly, so the same job can appear on two pages — measured at 6
duplicates across pages 1 and 2. This is why `normalise_all` dedupes by
`job_id`; it is not defensive paranoia.

**Salary coverage is about 2%.** Six of 310 postings publish one, and those
six mix hourly with annual rates. `salary_stats` therefore filters by period
first and **always** returns `n` and `coverage_pct` alongside any figure, so a
caller cannot read a median without seeing how little it rests on. The
`salary_coverage` chart deliberately charts the gap rather than the
distribution.

**Salary strings come in at least seven formats**, including `$35,3k` where
the comma is a decimal separator and `$35,000` where it is a thousands
separator — a factor-of-1000 apart. See `parse_salary` and its tests.

**The sample is dominated by Arbeitnow** (roughly 300 of 310), which is a
German-language board. Any unfiltered statistic describes that market, and
the top "skills" are its business categories rather than developer skills.

---

## Testing

166 fast tests, offline and deterministic, plus one integration test that is
deselected by default.

| File | Covers |
|---|---|
| `test_normalise.py` | salary parsing, the common record shape, dedup |
| `test_analyse.py` | every statistic, the honesty contract, JSON-serialisability |
| `test_fetch.py` | paging, retries, the disk cache — all with the network mocked |
| `test_api.py` | every endpoint, validation, pagination, 404s and 503s |
| `test_storage.py` | the never-delete-unconfirmed rule, dedup, optional import |
| `test_ui_boundary.py` | the UI does not import the backend |

- **Pure functions** (`normalise`, `analyse`) are tested directly against
  hand-built fixtures with known answers.
- **`fetch`** is tested by replacing `_get_json` — so paging, merging, page
  caps and error propagation are covered without a network.
- **Retry logic** is tested by replacing `requests.get`, with `time.sleep`
  patched out so the tests do not wait for backoff.
- **The API** is tested with `TestClient` and `dependency_overrides`, against
  a six-row frame with known answers.
- **`lifespan` is stubbed separately.** `dependency_overrides` replaces what
  *handlers* receive; it does not stop startup. Without stubbing the loader
  too, every test did a live fetch — measured at 120s for the file, 1.7s
  after.
- The **one integration test** exists because both APIs silently ignore
  documented parameters; it is how you find out they changed something.

```bash
pytest -q                               # 166 tests, ~2.0s
pytest -m integration -o addopts=""     # 1 test, hits remotive.com
```

---

## Status

| Phase | | State |
|---|---|---|
| 1 | Fetch, normalise, analyse, chart | ✅ done |
| 2 | FastAPI | ✅ done |
| 3 | Streamlit UI (an HTTP client, not an import) | ✅ done |
| 4 | RAG over the resume and job descriptions | todo |
| 5 | Tool calling — the agent routes numbers vs text | todo |
| 6 | `/ask` endpoint and a chat UI | todo |

### Known gaps

- No `pyproject.toml`, so `PYTHONPATH=.` is needed to run the UI and
  `pytest.ini` has to be force-added past the `*.ini` ignore rule.
- No maximum staleness: the cache is accepted at any age, so a month-old
  cache would be served as if current.
- `q` searches titles only. Most skill mentions are in the descriptions,
  which the list endpoint deliberately excludes — `tag=python` returns 3 of
  310 as a result.
- `/refresh` is unauthenticated, so anyone who can reach it can make the
  server hammer two public APIs.
