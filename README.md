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

**Run the pipeline as a script** — fetch, analyse, write charts, print a summary:

```bash
python -m jobagency.main                # fetch live, 1 page of Arbeitnow
python -m jobagency.main --pages 3      # more pages
python -m jobagency.main --offline      # reuse the newest saved payloads
```

**Run the API:**

```bash
uvicorn jobagency.api:app --reload
```

Then open <http://localhost:8000/docs> for interactive documentation, or
<http://localhost:8000/chart/tags> for a chart.

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

77 fast tests, offline and deterministic, plus one integration test that is
deselected by default.

- **Pure functions** (`normalise`, `analyse`) are tested directly against
  hand-built fixtures with known answers.
- **`fetch`** is tested by replacing `_get_json` — so paging, merging, page
  caps and error propagation are covered without a network.
- **Retry logic** is tested by replacing `requests.get`, with `time.sleep`
  patched out so the tests do not wait for backoff.
- The **one integration test** exists because both APIs silently ignore
  documented parameters; it is how you find out they changed something.

```bash
pytest -q                               # 77 tests, ~0.5s
pytest -m integration -o addopts=""     # 1 test, hits remotive.com
```

---

## Status

| Phase | | State |
|---|---|---|
| 1 | Fetch, normalise, analyse, chart | ✅ done |
| 2 | FastAPI | ✅ endpoints done, tests pending |
| 3 | Streamlit UI (an HTTP client, not an import) | todo |
| 4 | RAG over the resume and job descriptions | todo |
| 5 | Tool calling — the agent routes numbers vs text | todo |
| 6 | `/ask` endpoint and a chat UI | todo |

### Known gaps

- No `test_api.py` yet.
- No maximum staleness: the cache is accepted at any age, so a month-old
  cache would be served as if current.
- `q` searches titles only. Most skill mentions are in the descriptions,
  which the list endpoint deliberately excludes — `tag=python` returns 3 of
  310 as a result.
- `/refresh` is unauthenticated, so anyone who can reach it can make the
  server hammer two public APIs.
