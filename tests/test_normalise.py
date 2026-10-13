from datetime import timezone

import pytest

from jobagency.normalise import (
    normalise_all,
    normalise_arbeitnow,
    normalise_remotive,
    parse_salary,
)

COMMON_KEYS = {
    "source", "job_id", "title", "company", "category", "tags", "job_type",
    "posted_at", "location", "remote", "salary_low", "salary_high",
    "salary_period", "description_html", "url",
}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$45-$120/Hour", (45.0, 120.0, "hour")),
        ("$50-$75 /hour", (50.0, 75.0, "hour")),
        ("$20k -$35k", (20_000.0, 35_000.0, "year")),
        ("$10K-$20K", (10_000.0, 20_000.0, "year")),
        ("$170k - $200k", (170_000.0, 200_000.0, "year")),
        ("$100k", (100_000.0, 100_000.0, "year")),
        # the two meanings of a comma - a factor-of-1000 apart
        ("$35,3k- $52k", (35_300.0, 52_000.0, "year")),
        ("$35,000 - $45,000", (35_000.0, 45_000.0, "year")),
    ],
)
def test_parse_salary(raw, expected):
    assert parse_salary(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "Competitive", "DOE"])
def test_parse_salary_unparseable_is_none(raw):
    """Most postings publish no salary. None must mean 'unknown', never 0."""
    assert parse_salary(raw) is None


def test_parse_salary_orders_low_before_high():
    assert parse_salary("$120k - $90k") == (90_000.0, 120_000.0, "year")


def test_arbeitnow_record_has_common_shape(arbeitnow_payload):
    record = normalise_arbeitnow(arbeitnow_payload["data"][0])
    assert set(record) == COMMON_KEYS


def test_both_sources_agree_on_schema(remotive_payload, arbeitnow_payload):
    """The whole point of normalising: one shape, whatever the source."""
    a = normalise_remotive(remotive_payload["jobs"][0])
    b = normalise_arbeitnow(arbeitnow_payload["data"][0])
    assert set(a) == set(b)


def test_dates_are_timezone_aware(remotive_payload, arbeitnow_payload):
    """Mixing naive and aware datetimes raises later, in pandas, far from here."""
    for record in (
            normalise_remotive(remotive_payload["jobs"][0]),
            normalise_arbeitnow(arbeitnow_payload["data"][0]),
    ):
        assert record["posted_at"].tzinfo is not None
        assert record["posted_at"].utcoffset() == timezone.utc.utcoffset(None)


def test_arbeitnow_has_no_category_or_salary(arbeitnow_payload):
    """This source genuinely lacks them. None, not a guess or a placeholder."""
    record = normalise_arbeitnow(arbeitnow_payload["data"][0])
    assert record["category"] is None
    assert record["salary_low"] is None


def test_tags_always_a_list(remotive_payload, arbeitnow_payload):
    assert isinstance(normalise_remotive(remotive_payload["jobs"][0])["tags"], list)
    assert isinstance(normalise_arbeitnow(arbeitnow_payload["data"][0])["tags"], list)


def test_normalise_all_combines_both_sources(remotive_payload, arbeitnow_payload):
    records = normalise_all(remotive_payload, arbeitnow_payload)
    assert {r["source"] for r in records} == {"remotive", "arbeitnow"}


def test_job_ids_are_unique(remotive_payload, arbeitnow_payload):
    records = normalise_all(remotive_payload, arbeitnow_payload)
    ids = [r["job_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_normalise_all_handles_empty_payloads():
    """Edge case that will happen: an API returns nothing."""
    assert normalise_all({}, {}) == []
