"""Phase 1 pipeline: fetch -> normalise -> analyse -> chart.

Run it:
    python -m jobagency.main # fetch live, 1 page of Arbeitnow
    python -m jobagency.main --pages 3 # more Arbeitnow pages
    python -m jobagency.main --offline # reuse the newest saved payloads

The work is split into three functions rather than one script body because
phase 2 imports them: FastAPI will call `load_jobs()` once at startup and
`build_report(df)` per request. A script body cannot be reused that way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from jobagency.analyse import (
    count_by, overview, posting_trend, salary_stats,
    tag_cooccurrence, to_dataframe, top_tags,
)
from jobagency.chart import (
    salary_coverage_figure, save_png, tags_figure, trend_figure,
)
from jobagency.fetch import LAKE_DIR, FetchError, fetch_all, load_raw, save_raw
from jobagency.normalise import normalise_all

CHART_DIR = Path(__file__).parent.parent / "charts"


def _newest(directory: Path, prefix: str) -> Path:
    """The most recent saved payload for one source.

    Filenames end in a UTC timestamp, so sorting by name sorts by time -
    which is the reason for that naming scheme.
    """
    matches = sorted(directory.glob(f"{prefix}_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"no saved {prefix} payload in {directory} - run without --offline first"
        )
    return matches[-1]


def load_jobs(
    *, offline: bool = False, pages: int = 1,
    save: bool = True, lake: Path = LAKE_DIR,
) -> pd.DataFrame:
    """Get job postings as a typed DataFrame, from the network or from disk.

    `offline` reads the newest saved payloads instead of fetching. Explicit
    rather than an automatic fallback: silently serving yesterday's data
    when a fetch fails is how you end up debugging the wrong thing.
    """
    if offline:
        remotive = load_raw(_newest(lake, "remotive"))
        arbeitnow = load_raw(_newest(lake, "arbeitnow"))
    else:
        remotive, arbeitnow = fetch_all(max_pages=pages)
        if save:
            save_raw(remotive, "remotive", lake)
            save_raw(arbeitnow, "arbeitnow", lake)

    return to_dataframe(normalise_all(remotive, arbeitnow))


def build_report(df: pd.DataFrame) -> dict:
    """Every analysis result, in one JSON-serialisable dict.

    This is phase 2's /stats response body, which is why nothing in here is
    a DataFrame.
    """
    return {
        "overview": overview(df),
        "by_category": count_by(df, "category"),
        "by_company": count_by(df, "company", top_n=10),
        "by_job_type": count_by(df, "job_type"),
        "salary_year": salary_stats(df, period="year"),
        "salary_hour": salary_stats(df, period="hour"),
        "top_tags": top_tags(df, top_n=15),
        "trend": posting_trend(df),
        "tag_pairs": tag_cooccurrence(df, max_pairs=10),
    }


def write_charts(report: dict, out_dir: Path = CHART_DIR) -> list[Path]:
    """Render the three charts and return where they landed."""
    return [
        save_png(trend_figure(report["trend"]), out_dir / "trend.png"),
        save_png(tags_figure(report["top_tags"]), out_dir / "tags.png"),
        save_png(
            salary_coverage_figure(report["salary_year"]),
            out_dir / "salary_coverage.png",
        ),
    ]


def print_summary(report: dict, charts: list[Path]) -> None:
    """A readable digest for the terminal. Not the data interface."""
    ov = report["overview"]
    print(f"\n{ov['total_jobs']} postings  "
          f"({ov['first_posted']} to {ov['last_posted']})")
    print(f"  by source : {ov['by_source']}")
    print(f"  remote    : {ov['remote_count']}")

    salary = report["salary_year"]
    print(f"\nAnnual salary: n={salary['n']} "
          f"({salary['coverage_pct']:.1f}% of postings publish one)")
    if salary["n"]:
        print(f"  range  {salary['min']:,.0f} - {salary['max']:,.0f}")
        print(f"  median {salary['median']:,.0f}")
    else:
        print("  not enough data to report a figure")

    print("\nTop skills:")
    for row in report["top_tags"][:8]:
        print(f"  {row['count']:>4}  {row['tag']}")

    if report["tag_pairs"]:
        print("\nRequested together:")
        for row in report["tag_pairs"][:5]:
            print(f"  {row['count']:>4}  {row['tag_a']} + {row['tag_b']}")

    print("\nCharts:")
    for path in charts:
        print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns an exit code rather than calling sys.exit,
    so it can be called from a test."""
    parser = argparse.ArgumentParser(description="Fetch and analyse job postings.")
    parser.add_argument("--offline", action="store_true",
                        help="reuse the newest saved payloads instead of fetching")
    parser.add_argument("--pages", type=int, default=1,
                        help="Arbeitnow pages to fetch (default 1)")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write fetched payloads to the lake")
    parser.add_argument("--out", type=Path, default=CHART_DIR,
                        help="where to write charts")
    args = parser.parse_args(argv)

    try:
        df = load_jobs(offline=args.offline, pages=args.pages, save=not args.no_save)
    except FetchError as error:
        # A failed fetch is an expected outcome, not a bug - report it
        # cleanly instead of dumping a traceback at the user.
        print(f"fetch failed: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    if df.empty:
        print("no postings returned", file=sys.stderr)
        return 1

    report = build_report(df)
    print_summary(report, write_charts(report, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())