"""Public v1 wire contracts. No model calls or service dependencies."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trigger(Strict):
    source_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    time: AwareDatetime


class Question(Strict):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: int = Field(gt=0)
    source_refs: list[str] = Field(min_length=1)


class PublicationWindow(Strict):
    since: AwareDatetime
    until: AwareDatetime
    date_basis: Literal["published_at"]


class QuestionEvidenceGroup(Strict):
    any_of: list[str] = Field(min_length=1, max_length=4)


class QuestionEvidenceGate(Strict):
    source_refs: list[str] = Field(min_length=1)
    query_refs: list[str] = Field(min_length=1)
    all_of: list[QuestionEvidenceGroup] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def distinct_concepts(self):
        seen = set()
        for group in self.all_of:
            for phrase in group.any_of:
                normalized = " ".join(phrase.split()).casefold()
                if not normalized or len(phrase) > 200 or normalized in seen:
                    raise ValueError("Evidence gate phrases must be nonblank, bounded and distinct")
                seen.add(normalized)
        if len(set(self.source_refs)) != len(self.source_refs) or len(set(self.query_refs)) != len(self.query_refs):
            raise ValueError("Duplicate evidence gate references")
        return self


class Contract(Strict):
    schema_version: Literal["1"]
    request_id: str = Field(min_length=1)


class RecommendationRequest(Contract):
    user_id: str = Field(min_length=1)
    mode: Literal["progress", "direction_bootstrap", "direction_daily"]
    project_id: str | None
    direction_id: str | None
    as_of: AwareDatetime
    timezone: Literal["Asia/Shanghai"]
    trigger: Trigger
    research_questions: list[Question] = Field(min_length=1)
    queries: list[str]
    constraints: dict
    publication_window: PublicationWindow | None
    max_recommendations: int = Field(ge=1)

    @model_validator(mode="after")
    def rules(self):
        if self.mode == "progress" and not self.project_id:
            raise ValueError("progress requires project")
        if self.mode != "progress" and not self.direction_id:
            raise ValueError("direction requires direction id")
        if self.mode == "direction_bootstrap" and self.max_recommendations != 5:
            raise ValueError("bootstrap target must be five; fewer results are valid")
        if len({q.id for q in self.research_questions}) != len(self.research_questions):
            raise ValueError("duplicate question id")
        w = self.publication_window
        if self.mode == "direction_daily" and w is None:
            raise ValueError("daily publication window required")
        if w and w.since >= w.until:
            raise ValueError("invalid half-open publication window")
        gates = self.constraints.get("question_evidence_gates")
        if gates is not None:
            questions = {q.id: q for q in self.research_questions}
            if not isinstance(gates, dict) or not set(gates) <= questions.keys():
                raise ValueError("Evidence gates must name current research questions")
            if not gates:
                return self
            mapping = self.constraints.get("query_question_ids")
            if mapping is None and len(questions) == 1:
                mapping = {query: list(questions) for query in self.queries}
            if not isinstance(mapping, dict):
                raise ValueError("Evidence gates require explicit query/question mapping")
            for qid, value in gates.items():
                gate = QuestionEvidenceGate.model_validate(value)
                if not set(gate.source_refs) <= set(questions[qid].source_refs):
                    raise ValueError("Evidence gate source reference is outside its question")
                if any(query not in self.queries or not isinstance(mapping.get(query), list)
                       or qid not in mapping[query] for query in gate.query_refs):
                    raise ValueError("Evidence gate query reference is outside its question")
        return self


class Candidate(Strict):
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
    sources: list[dict]
    versions: list[str]


class Recommendation(Strict):
    paper_id: str
    question_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    basis: Literal["title", "abstract", "fulltext"]
    ranking_basis: dict
    source_refs: dict[str, list[str]]
    matched_evidence: list[dict] = Field(default_factory=list)
    method_excerpt: dict | None = None
    experiment_excerpt: dict | None = None
    method_routes: list[dict] = Field(default_factory=list)
    applicability_review: dict | None = None
    representative_roles: list[dict] = Field(default_factory=list)


class SourceRun(Strict):
    source: str
    query: str | dict
    since: str | None = None
    until: str | None = None
    status: Literal['ok', 'empty', 'partial', 'failed', 'completed']
    count: int = Field(ge=0)
    error: str | None = None
    truncated: bool
    fetched_at: AwareDatetime
    coverage_complete: bool | None
    effective_requests: list[dict] = Field(default_factory=list)
    source_ref: str | None = None
    search_lanes: list = Field(default_factory=list)
    citation_origins: list = Field(default_factory=list)
    citing_paper_id: str | None = None
    citing_version: str | None = None
    document_identity_declared: dict | str | None = None
    paper_id: str | None = None
    references: list[dict] = Field(default_factory=list)
    requested_document_version: str | None = None
    scope: str | None = None
    snapshot_ref: str | None = None
    url: str | None = None
    version_independently_verified: bool | None = None
    warning: str | None = None


class DiscoveryResult(Contract):
    user_id: str
    status: Literal["completed", "partial", "failed", "skipped"]
    candidates: list[Candidate]
    recommendations: list[Recommendation]
    source_runs: list[SourceRun]
    warnings: list[str]
    decisions: list[dict]
    audit: dict

    @model_validator(mode="after")
    def paper_links(self):
        ids = [p.paper_id for p in self.candidates]
        recommended = [r.paper_id for r in self.recommendations]
        if len(ids) != len(set(ids)) or len(recommended) != len(set(recommended)):
            raise ValueError("Duplicate paper identity")
        if not set(recommended) <= set(ids):
            raise ValueError("Recommendation missing candidate")
        return self


class AnalysisResult(Contract):
    user_id: str
    paper_id: str
    paper_version: str
    status: Literal["completed", "partial", "unavailable", "failed"]
    reading_scope: dict
    methods: list[str]
    findings: list[str]
    limitations: list[str]
    relevance_to_questions: dict
    research_suggestions: list[dict]
    claims: list[dict]
    evidence: list[dict]
    warnings: list[str]
    audit_refs: list[str]
    analyzed_at: AwareDatetime | None = None
    analysis_config: dict | None = None
    cache: dict | None = None
    problem: list[str] = Field(default_factory=list)
    rejected_claims: list[dict] = Field(default_factory=list)
    rejected_suggestions: list[dict] = Field(default_factory=list)
    call_audit: dict | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def evidence_links(self):
        evidence = {e["evidence_id"]: e for e in self.evidence}
        if len(evidence) != len(self.evidence):
            raise ValueError("Duplicate evidence id")
        if any((e['paper_id'], e['paper_version']) != (self.paper_id, self.paper_version) for e in self.evidence):
            raise ValueError("Evidence paper/version mismatch")
        claim_ids = [c.get('claim_id') for c in self.claims]
        if any(not cid for cid in claim_ids) or len(set(claim_ids)) != len(claim_ids):
            raise ValueError("Missing or duplicate claim id")
        for claim in self.claims:
            if not claim.get("supports") or any(s["evidence_id"] not in evidence for s in claim["supports"]):
                raise ValueError("claim missing evidence")
            for support in claim['supports']:
                bound = evidence[support['evidence_id']]
                quote = ' '.join(support.get('quote', '').split())
                if not quote or quote not in ' '.join(bound.get('text', '').split()):
                    raise ValueError("Quote not in bound evidence")
                if not bound.get('document_id') or not bound.get('block_id'):
                    raise ValueError("Referenced evidence requires document and block location")
        for referenced_claims in self.relevance_to_questions.values():
            if not isinstance(referenced_claims, list) or any(cid not in claim_ids for cid in referenced_claims):
                raise ValueError("Question relevance references unknown claims")
        if self.status == "completed" and not self.reading_scope.get("documents"):
            raise ValueError("completed requires read documents")
        return self


CONTRACTS = {c.__name__: c for c in (RecommendationRequest, DiscoveryResult, AnalysisResult)}


def validate(kind, payload):
    return CONTRACTS[kind].model_validate(payload).model_dump(mode="json")


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate a v1 module output offline")
    parser.add_argument("kind", choices=CONTRACTS)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate(args.kind, json.loads(args.path.read_text(encoding="utf-8")))
    print(f"{args.kind}: compatible")
