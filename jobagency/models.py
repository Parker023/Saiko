from datetime import date, datetime
from pydantic import BaseModel, Field


class Overview(BaseModel):
    total_jobs: int
    by_source: dict[str, int]
    remote_count: int
    first_posted: date
    last_posted: date


class ValueCount(BaseModel):
    value: str
    count: int


class SalaryStats(BaseModel):
    period: str
    n: int
    total_jobs: int
    coverage_pct: float
    min: float | None
    max: float | None
    median: float | None
    p25: float | None
    p75: float | None


class TagCount(BaseModel):
    tag: str
    count: int


class TrendItem(BaseModel):
    date: date
    count: int


class TagPair(BaseModel):
    tag_a: str
    tag_b: str
    count: int


class JobStatsResponse(BaseModel):
    overview: Overview
    by_category: list[ValueCount]
    by_company: list[ValueCount]
    by_job_type: list[ValueCount]
    salary_year: SalaryStats
    salary_hour: SalaryStats
    top_tags: list[TagCount]
    trend: list[TrendItem]
    tag_pairs: list[TagPair]


class JobSummary(BaseModel):
    job_id: str
    title: str
    company: str
    source: str
    category: str | None
    job_type: str | None
    location: str | None
    remote: bool
    tags: list[str]
    posted_at: datetime
    salary_low: float | None
    salary_high: float | None
    salary_period: str | None
    url: str

class JobDetail(JobSummary):
    description_html: str

