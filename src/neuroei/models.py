from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, model_validator

from .config import StrictModel


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceBlock(StrictModel):
    document_id: str
    block_id: str
    text: str
    revision: str
    file: str | None = None
    line_start: int | None = Field(None, ge=1)
    line_end: int | None = Field(None, ge=1)
    section: str | None = None
    order: int | None = Field(None, ge=0)


class ProjectFact(StrictModel):
    id: str
    text: str = Field(min_length=1)
    kind: Literal["goal", "hypothesis", "observation", "question", "constraint", "exclusion"]
    source_blocks: list[str] = Field(min_length=1)
    provenance: Literal["direct", "derived_question"] = "direct"
    source_date: str | None = None
    temporal_status: Literal["current_candidate", "historical", "undated"] = "undated"


class ProfileExtraction(StrictModel):
    facts: list[ProjectFact]
    unresolved: list[str]


class ProjectProfile(StrictModel):
    project_id: str
    version: str
    created_at: str = Field(default_factory=now)
    facts: list[ProjectFact]
    blocks: list[SourceBlock]
    unresolved: list[str] = Field(default_factory=list)
    approved_by: str | None = None
    approved_at: str | None = None

    @model_validator(mode="after")
    def check_sources(self):
        ids = {b.block_id for b in self.blocks}
        if len(ids) != len(self.blocks):
            raise ValueError("Duplicate project source block IDs")
        fact_ids = [f.id for f in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Duplicate project fact IDs")
        for fact in self.facts:
            if not set(fact.source_blocks) <= ids:
                raise ValueError("Project fact references missing source block")
            if fact.provenance == "derived_question" and fact.kind != "question":
                raise ValueError("Derived project items can only be unconfirmed research questions")
        return self


class PaperRecord(StrictModel):
    paper_id: str
    version: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    published: str
    updated: str | None = None
    discovered_at: str = Field(default_factory=now)
    sources: list[str]
    url: str
    fulltext_status: Literal["not_fetched", "available", "unavailable"] = "not_fetched"
    is_preprint: bool = False
    correction_status: str = "not_checked"


class SourceResult(StrictModel):
    source: str
    status: Literal["ok", "empty", "failed", "partial"]
    query: str
    since: str
    until: str
    fetched_at: str = Field(default_factory=now)
    papers: list[PaperRecord] = Field(default_factory=list)
    error: str | None = None
    truncated: bool = False


class RankedPaper(StrictModel):
    paper_id: str
    question_ids: list[str] = Field(min_length=1)
    relevance: int = Field(ge=0, le=3)
    reason: str


class Ranking(StrictModel):
    recommendations: list[RankedPaper]


class EvidenceRecord(StrictModel):
    evidence_id: str
    paper_id: str
    paper_version: str
    file: str
    block_id: str
    page: int | None = Field(None, ge=1)
    section: str | None = None
    text: str = Field(min_length=1)
    parser_version: str
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_locator(self):
        if self.page is None and not self.section:
            raise ValueError("Evidence needs a page or section/anchor")
        return self


class Claim(StrictModel):
    text: str
    kind: Literal["paper_fact", "inference", "experiment_suggestion"]
    evidence_ids: list[str] = Field(min_length=1)
    question_ids: list[str] = Field(min_length=1)
    limitation: str


class Analysis(StrictModel):
    claims: list[Claim]


class Verdict(StrictModel):
    claim_index: int = Field(ge=0)
    support: Literal["supported", "partial", "unsupported"]
    reason: str


class Verification(StrictModel):
    verdicts: list[Verdict]


class FeedbackEvent(StrictModel):
    project_id: str
    user_id: str
    paper_id: str
    action: Literal["relevant", "later", "known", "irrelevant", "method_unsuitable", "fact_correction", "report_quality"]
    reason: str = ""
    occurred_at: datetime
    # Legacy CLI events may lack context; interactive feedback records exactly what was shown.
    paper_version: str | None = None
    profile_version: str | None = None
    workflow_id: str | None = None
    report_id: str | None = None

    @model_validator(mode="after")
    def timezone_required(self):
        if self.occurred_at.tzinfo is None:
            raise ValueError("Feedback time must include timezone")
        return self
