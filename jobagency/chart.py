"""Render analysis results as PNG charts.

Takes the plain lists and dicts that `analyse` returns - never a DataFrame.
`analyse` is the last place a DataFrame exists, so charts work equally well
from a live frame or from a JSON response over HTTP (which is what phase 3
will do).

The figure builders return a Figure and write nothing. `save_png` and
`png_bytes` are the two ways to get it out, and both close the figure.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

# MUST come before importing pyplot. The default backend here is "macosx",
# a GUI backend: it tries to open a window, which fails on a headless server
# and is not safe to call from a web worker thread. "Agg" renders to memory.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (deliberate: after use())
from matplotlib.figure import Figure

FIGSIZE = (9, 4.5)
DPI = 120


def _empty_figure(message: str) -> Figure:
    """A readable placeholder instead of a crash or a blank image.

    Empty input is normal here - `posting_trend` on an empty frame returns
    [], and a chart endpoint should say so rather than 500.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="grey")
    ax.set_axis_off()
    return fig


def trend_figure(trend: list[dict], *, title: str = "Job postings per day") -> Figure:
    """Line chart from `analyse.posting_trend` output.

    Expects [{"date": "2026-10-07", "count": 12}, ...] already in
    chronological order and with empty days filled in - resample does both,
    which is why gaps do not need handling here.
    """
    if not trend:
        return _empty_figure("No postings in range")

    dates = [row["date"] for row in trend]
    counts = [row["count"] for row in trend]

    # The object-oriented API, not the plt.plot() global state machine:
    # that global is shared and not thread-safe, so two concurrent requests
    # in phase 2 could draw onto each other's axes.
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(dates, counts, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("Postings")
    ax.set_ylim(bottom=0)          # a count axis that starts at 40 exaggerates
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def tags_figure(tags: list[dict], *, title: str = "Most requested skills") -> Figure:
    """Horizontal bar chart from `analyse.top_tags` output.

    Horizontal because tag names are words - rotated vertical labels are
    hard to read and get clipped.
    """
    if not tags:
        return _empty_figure("No tags found")

    names = [row["tag"] for row in tags]
    counts = [row["count"] for row in tags]

    # Reverse so the largest ends up at the top: barh draws bottom-up.
    names, counts = names[::-1], counts[::-1]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.0, 0.35 * len(names))), dpi=DPI)
    ax.barh(names, counts)
    ax.set_title(title)
    ax.set_xlabel("Postings mentioning it")
    ax.grid(axis="x", alpha=0.3)
    for y, count in enumerate(counts):
        ax.text(count, y, f" {count}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def salary_coverage_figure(stats: dict) -> Figure:
    """How many postings actually publish a salary - not the salary itself.

    Deliberate choice: coverage is ~3%, so a distribution chart of six
    values would look authoritative and mean nothing. Charting the gap is
    the honest visual.
    """
    n = stats.get("n") or 0
    total = stats.get("total_jobs") or 0
    if total == 0:
        return _empty_figure("No jobs loaded")

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 2.2), dpi=DPI)
    ax.barh(["postings"], [total - n], label="no salary published")
    ax.barh(["postings"], [n], left=[total - n], label=f"salary given (n={n})")
    ax.set_title(
        f"Salary coverage: {n} of {total} postings "
        f"({stats.get('coverage_pct', 0):.1f}%)"
    )
    ax.set_xlabel("Postings")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def save_png(fig: Figure, path: str | Path) -> Path:
    """Write the figure to `path` and close it.

    Closing matters: figures are held by pyplot until closed, so a server
    generating one per request leaks memory until it falls over.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format="png", bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def png_bytes(fig: Figure) -> bytes:
    """The figure as PNG bytes, for serving over HTTP without touching disk.

    Phase 2's /chart endpoint wants this, not a file path.
    """
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", bbox_inches="tight")
    finally:
        plt.close(fig)
    return buffer.getvalue()