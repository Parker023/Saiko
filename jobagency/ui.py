"""Streamlit UI for the Job Match Agency.

Run it (the API must be running too):

    Uvicorn jobagency.api:app --reload # terminal 1, port 8000
    streamlit run jobagency/ui.py # terminal 2, port 8501

This file speaks ONLY HTTP, through jobagency.ui_client. It deliberately does
not import, analyze, fetch or normalise - if it did, the API would be
decorative, and none of its designs would be exercised.

Streamlit re-runs this whole script on every interaction, which is why every
API call goes through a @st.cache_data wrapper. Without them, each keystroke
in a filter box would re-fetch.
"""

import streamlit as st

from jobagency import ui_client
from jobagency.ui_client import ApiError

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Job Match Agency", page_icon="💼", layout="wide")


# --- cached API calls ----------------------------------------------------
# ttl keeps data from going stale for a long-lived session; show_spinner
# gives feedback on the first (uncached) call only.

@st.cache_data(ttl=60, show_spinner=False)
def _health() -> dict:
    return ui_client.health()


@st.cache_data(ttl=300, show_spinner="Loading statistics...")
def _stats() -> dict:
    return ui_client.stats()


@st.cache_data(ttl=300, show_spinner="Searching...")
def _jobs(**filters) -> dict:
    """Cached per unique filter combination - Streamlit keys the cache on the
    arguments, so changing one filter is one request, and going back to a
    previous combination is none."""
    return ui_client.jobs(**filters)


@st.cache_data(ttl=300, show_spinner=False)
def _job(job_id: str) -> dict:
    return ui_client.job(job_id)


@st.cache_data(ttl=300, show_spinner=False)
def _chart(name: str) -> bytes:
    return ui_client.chart_png(name)


def _render_sidebar_health() -> dict | None:
    """Backend status. Returns the health payload, or None if unreachable.

    Returning None rather than raising lets the caller decide to stop - a UI
    that tracebacks because a server is down is worse than one that says so.
    """
    with st.sidebar:
        st.subheader("Backend")
        try:
            info = _health()
        except ApiError as error:
            st.error("API unreachable")
            st.caption(str(error))
            st.caption(f"Start it with:  `uvicorn jobagency.api:app --reload`")
            return None

        if info.get("http_status") == 503:
            st.error("No data loaded")
            st.caption(info.get("reason", ""))
            return info

        if info.get("degraded"):
            st.warning("Serving cached data")
            st.caption(info.get("reason", ""))
        else:
            st.success("Live data")

        st.metric("Postings", f"{info['total_jobs']:,}")
        st.caption(f"newest posting · {info['newest_posting']}")
        st.caption(f"loaded · {info['loaded_at'][:16].replace('T', ' ')} UTC")

        if st.button("Refresh data", use_container_width=True):
            _do_refresh()

        return info


def _do_refresh() -> None:
    """Re-load on the backend, then clear every cache.

    Both halves matter: without clearing the cache, the backend would have new
    data, and the UI would keep showing the old response for up to 5 minutes.
    """
    try:
        result = ui_client.refresh()
    except ApiError as error:
        st.sidebar.error(str(error))
        return

    st.cache_data.clear()
    if result.get("http_status") == 503:
        st.sidebar.error(f"Refresh failed: {result.get('reason')}")
    else:
        st.sidebar.success(f"Reloaded {result['total_jobs']:,} postings")
    st.rerun()


st.title("💼 Job Match Agency")

health_info = _render_sidebar_health()

if health_info is None or health_info.get("http_status") == 503:
    st.info("Nothing to show until the backend has data.")
    st.stop()          # ends this run cleanly - no half-rendered page

st.caption(
    f"{health_info['total_jobs']:,} postings · "
    f"source: {health_info['source']} · "
    f"newest: {health_info['newest_posting']}"
)


# --- filters -------------------------------------------------------------

def _blank_to_none(value: str) -> str | None:
    """Empty text input means 'no filter', not 'match the empty string'."""
    value = value.strip()
    return value or None


def _read_filters(options: dict) -> dict:
    """Sidebar widgets -> the API's query parameters.

    Every control maps to exactly one parameter. The three-state radios are
    why `remote` and `has_salary` are bool|None on the API: a checkbox has
    only two states and could never express "don't filter on this".
    """
    with st.sidebar:
        st.divider()
        st.subheader("Filters")

        query = _blank_to_none(st.text_input("Title contains", placeholder="engineer"))
        # The API requires min_length=2 and would answer 422; catching it here
        # is kinder than showing the user a validation error they did not ask for.
        if query is not None and len(query) < 2:
            query = None

        company = _blank_to_none(st.text_input("Company", placeholder="google"))

        source = st.selectbox("Source", ["Any", *options["sources"]])
        category = st.selectbox("Category", ["Any", *options["categories"]])
        tag = st.selectbox("Skill tag", ["Any", *options["tags"]])

        remote_choice = st.radio(
            "Location", ["Any", "Remote only", "On-site only"], horizontal=False
        )
        salary_choice = st.radio(
            "Salary", ["Any", "Published", "Not published"], horizontal=False
        )

        page_size = st.select_slider("Results per page", [10, 20, 50, 100], value=20)

    return {
        "q": query,
        "company": company,
        "source": None if source == "Any" else source,
        "category": None if category == "Any" else category,
        "tag": None if tag == "Any" else tag,
        "remote": {"Any": None, "Remote only": True, "On-site only": False}[remote_choice],
        "has_salary": {"Any": None, "Published": True, "Not published": False}[salary_choice],
        "limit": page_size,
    }


def _filter_options(report: dict) -> dict:
    """Dropdown choices, taken from /stats rather than hardcoded.

    The UI never invents a category that does not exist in the data - the
    options come from the same analysis the numbers do.
    """
    return {
        "sources": sorted(report["overview"]["by_source"]),
        "categories": [row["value"] for row in report["by_category"]],
        "tags": [row["tag"] for row in report["top_tags"]],
    }


def _format_salary(row: dict) -> str:
    """A readable range, or an honest dash. 97% of postings have none."""
    low, high, period = row["salary_low"], row["salary_high"], row["salary_period"]
    if low is None or high is None:
        return "—"
    unit = "/hr" if period == "hour" else "/yr"
    if low == high:
        return f"{low:,.0f}{unit}"
    return f"{low:,.0f}–{high:,.0f}{unit}"


try:
    report = _stats()
except ApiError as error:
    st.error(str(error))
    st.stop()

filters = _filter_options(report)
active = _read_filters(filters)

# The page number must survive the rerun that changing it triggers, so it
# lives in session_state rather than an ordinary variable.
st.session_state.setdefault("page", 0)

# Any change to the filters should reset to page 1 - otherwise you can end up
# on page 7 of a 2-page result and see nothing.
signature = tuple(sorted((k, str(v)) for k, v in active.items()))
if st.session_state.get("filter_signature") != signature:
    st.session_state["filter_signature"] = signature
    st.session_state["page"] = 0

limit = active["limit"]
offset = st.session_state["page"] * limit

try:
    page = _jobs(**active, offset=offset)
except ApiError as error:
    st.error(str(error))
    st.stop()

total = page["total"]
st.subheader(f"{total:,} matching postings")

if total == 0:
    st.info("No postings match these filters.")
    st.stop()

selection = st.dataframe(
    [
        {
            "Posted": row["posted_at"][:10],
            "Title": row["title"],
            "Company": row["company"],
            "Location": row["location"] or "—",
            "Remote": "Yes" if row["remote"] else "No",
            "Salary": _format_salary(row),
            "Tags": ", ".join(row["tags"][:4]) or "—",
            "Link": row["url"],
            "id": row["job_id"],
        }
        for row in page["items"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Link": st.column_config.LinkColumn("Link", display_text="open"),
        "id": None,          # carried for selection, not shown
    },
    # on_select="rerun" turns the table into an input: clicking a row triggers
    # a script rerun with the choice available in `selection`.
    on_select="rerun",
    selection_mode="single-row",
)

# --- job detail, fetched only when a row is chosen -----------------------
# The list carries summaries (0.15 MB for all 310). The full description is
# ~9 KB per posting, so it is fetched one at a time, on demand.

chosen_rows = selection.selection.rows if selection and selection.selection else []
if chosen_rows:
    chosen_id = page["items"][chosen_rows[0]]["job_id"]
    try:
        detail = _job(chosen_id)
    except ApiError as error:
        st.error(str(error))
    else:
        with st.expander(f"**{detail['title']}** — {detail['company']}", expanded=True):
            left, right = st.columns([3, 1])
            with right:
                st.caption("Posted");   st.write(detail["posted_at"][:10])
                st.caption("Location"); st.write(detail["location"] or "—")
                st.caption("Salary");   st.write(_format_salary(detail))
                st.caption("Type");     st.write(detail["job_type"] or "—")
                st.link_button("View posting", detail["url"], use_container_width=True)
            with left:
                if detail["tags"]:
                    st.write(" ".join(f"`{tag}`" for tag in detail["tags"]))
                # The API returns the board's own HTML. Rendering someone
                # else's markup is an XSS risk in a real product - acceptable
                # here because it is local and single-user, but it is the
                # reason you would sanitise before shipping this.
                st.markdown(detail["description_html"], unsafe_allow_html=True)

# --- pagination ----------------------------------------------------------

last_page = max(0, (total - 1) // limit)
previous, indicator, following = st.columns([1, 2, 1])

with previous:
    if st.button("← Previous", disabled=st.session_state["page"] == 0,
                 use_container_width=True):
        st.session_state["page"] -= 1
        st.rerun()

with indicator:
    st.markdown(
        f"<div style='text-align:center;padding-top:6px'>"
        f"Page {st.session_state['page'] + 1} of {last_page + 1}</div>",
        unsafe_allow_html=True,
    )

with following:
    if st.button("Next →", disabled=st.session_state["page"] >= last_page,
                 use_container_width=True):
        st.session_state["page"] += 1
        st.rerun()



# --- insights ------------------------------------------------------------
# Charts come from the API as PNG bytes - st.image takes bytes directly, so
# nothing touches disk. That is what png_bytes() exists for.

st.divider()
st.header("Insights")

overview = report["overview"]
salary = report["salary_year"]

a, b, c, d = st.columns(4)
a.metric("Postings", f"{overview['total_jobs']:,}")
b.metric("Remote", f"{overview['remote_count']:,}")
c.metric("Companies", f"{len(report['by_company']):,}+")
# The honesty contract, surfaced: never a median without its sample size.
d.metric(
    "Salary published",
    f"{salary['coverage_pct']:.1f}%",
    help=f"only {salary['n']} of {salary['total_jobs']} postings state a figure",
)

if salary["n"]:
    st.caption(
        f"Annual salary across **{salary['n']}** postings that publish one: "
        f"median {salary['median']:,.0f}, range {salary['min']:,.0f}–{salary['max']:,.0f}. "
        f"That is {salary['coverage_pct']:.1f}% of the data — treat it as anecdote, "
        f"not a market rate."
    )

trend_tab, tags_tab, salary_tab = st.tabs(["Postings over time", "Skills", "Salary coverage"])
for tab, name in ((trend_tab, "trend"), (tags_tab, "tags"), (salary_tab, "salary_coverage")):
    with tab:
        try:
            st.image(_chart(name), use_container_width=True)
        except ApiError as error:
            st.warning(f"chart unavailable: {error}")

with st.expander("Skills requested together"):
    pairs = report["tag_pairs"]
    if not pairs:
        st.caption("no pairs above the threshold")
    for pair in pairs[:10]:
        st.write(f"`{pair['tag_a']}` + `{pair['tag_b']}` — {pair['count']} postings")
