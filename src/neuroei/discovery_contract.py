"""Module-local v1 wire models. Integration owns the final common contract."""
from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from .config import StrictModel


class Question(StrictModel):
    id: str
    description: str
    priority: int = Field(gt=0)
    source_refs: list[str] = Field(min_length=1)


class Trigger(StrictModel):
    source_id: str
    version: str
    time: AwareDatetime


class PublicationWindow(StrictModel):
    since: AwareDatetime
    until: AwareDatetime
    date_basis: Literal['published_at']

    @model_validator(mode='after')
    def ordered(self):
        if self.until <= self.since:
            raise ValueError('Window is half-open [since, until)')
        return self


class RecommendationRequest(StrictModel):
    request_id: str
    schema_version: Literal['1']
    user_id: str
    mode: Literal['progress', 'direction_bootstrap', 'direction_daily']
    project_id: str | None
    direction_id: str | None
    as_of: AwareDatetime
    timezone: Literal['Asia/Shanghai']
    trigger: Trigger
    research_questions: list[Question] = Field(min_length=1)
    queries: list[str]
    constraints: dict[str, Any]
    publication_window: PublicationWindow | None
    max_recommendations: int = Field(ge=0, le=20)

    @model_validator(mode='after')
    def identities(self):
        ids = [q.id for q in self.research_questions]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate question IDs')
        return self


class Candidate(StrictModel):
    paper_id: str
    paper_version: str | None
    title: str
    authors: list[str]
    doi: str | None
    arxiv_id: str | None
    venue: str | None
    published_at: str | None
    date_basis: str | None
    abstract: str | None
    original_links: list[str]
    fulltext_links: list[str]
    sources: list[dict[str, Any]]
    versions: list[str]


class DiscoveryResult(StrictModel):
    request_id: str
    schema_version: Literal['1'] = '1'
    user_id: str
    status: Literal['completed', 'partial', 'failed', 'skipped']
    candidates: list[Candidate]
    recommendations: list[dict[str, Any]]
    source_runs: list[dict[str, Any]]
    warnings: list[str]
    # Proposed additive fields; no change to existing PaperRecord/SourceResult.
    decisions: list[dict[str, Any]]
    audit: dict[str, Any]
