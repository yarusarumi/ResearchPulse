"""Re-audit saved grounded analyses with the deterministic gates that produced them.

The audit is a pure function: it reads only its arguments and never touches a
Store, the filesystem, a model or the network.  It reports issue codes and
counts instead of evidence/claim text, and ``semantic_acceptance`` is always
false because literal spans and numeric presence cannot prove entailment.  A
``checks_passed`` result therefore means "no deterministic gate objected", not
scientific acceptance.

``accepted_claim_count`` and ``support_span_count`` count only accepted claims
that pass every re-derived gate.  Rejected claims and implementation-review
exclusions are not inspected and are left untouched; the stored
``grounding_check``/``numeric_check`` labels are never trusted.
"""
from collections import Counter

from pydantic import ValidationError

from .grounded import GroundedItem, grounding_check
from .models import Claim, EvidenceRecord, PaperRecord, ProjectProfile, Verdict
from .verification import check_numbers

METHOD = "source-span-label-scope-gates-v2"
GATE_VERSION = "grounding-gates-v3-paraphrase"


def _issue(issues, claim_index, code):
    item = {"claim_index": claim_index, "code": code}
    if item not in issues:
        issues.append(item)


def _locator_issue(raw):
    page = raw.get("page")
    if page is not None:
        # A bool is an int subclass; locators must be genuine positive integers.
        return None if type(page) is int and page >= 1 else "invalid_evidence_locator"
    section = raw.get("section")
    if not isinstance(section, str) or not section.strip():
        return "evidence_locator_missing"
    return None


def _audit_evidence(raw_evidence, paper, issues):
    """Revalidate the saved evidence list; return known IDs, usable records and reasons."""
    ids = []
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            _issue(issues, None, "invalid_evidence_record")
            continue
        evidence_id = raw.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            _issue(issues, None, "invalid_evidence_record")
            continue
        ids.append(evidence_id)
    duplicates = {eid for eid, count in Counter(ids).items() if count > 1}
    if duplicates:
        _issue(issues, None, "duplicate_evidence_id")

    known, usable, unusable = set(), {}, {}
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            continue
        evidence_id = raw.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        known.add(evidence_id)
        if evidence_id in duplicates:
            unusable[evidence_id] = "ambiguous_evidence_reference"
            continue
        if (raw.get("paper_id"), raw.get("paper_version")) != (paper.paper_id, paper.version):
            _issue(issues, None, "evidence_paper_version_mismatch")
            unusable[evidence_id] = "evidence_paper_version_mismatch"
            continue
        locator = _locator_issue(raw)
        if locator:
            _issue(issues, None, locator)
            unusable[evidence_id] = locator
            continue
        try:
            usable[evidence_id] = EvidenceRecord.model_validate(raw)
        except ValidationError:
            _issue(issues, None, "invalid_evidence_record")
            unusable[evidence_id] = "invalid_evidence_record"
    return known, usable, unusable


def _claim_indexes(raw_accepted):
    indexes = []
    for entry in raw_accepted:
        verification = entry.get("verification") if isinstance(entry, dict) else None
        raw_index = verification.get("claim_index") if isinstance(verification, dict) else None
        indexes.append(raw_index if type(raw_index) is int and raw_index >= 0 else None)
    return indexes, Counter(index for index in indexes if index is not None)


def _check_claim(entry, index, claim_ok, identity_ok, known, usable, unusable, question_ids, issues,
                 *, paraphrase=True):
    """Re-derive every gate for one accepted claim; return (ok, support_span_count)."""
    ok = claim_ok and identity_ok
    if not isinstance(entry, dict):
        _issue(issues, index, "invalid_claim_structure")
        return False, 0

    verification = entry.get("verification")
    if not isinstance(verification, dict):
        _issue(issues, index, "invalid_verification_structure")
        ok = False
    else:
        try:
            verdict = Verdict.model_validate(verification)
        except ValidationError:
            _issue(issues, index, "invalid_verification_structure")
            ok = False
        else:
            if verdict.support != "supported":
                _issue(issues, index, "verification_not_supported")
                ok = False

    try:
        claim = Claim.model_validate(entry.get("claim"))
    except ValidationError:
        _issue(issues, index, "invalid_claim_structure")
        return False, 0

    if not set(claim.question_ids) <= question_ids:
        _issue(issues, index, "unknown_question_reference")
        ok = False
    if any(eid not in known for eid in claim.evidence_ids):
        _issue(issues, index, "unknown_evidence_reference")
        ok = False
    for code in sorted({unusable[eid] for eid in claim.evidence_ids if eid in unusable}):
        _issue(issues, index, code)
        ok = False
    if any(eid not in usable for eid in claim.evidence_ids):
        # A missing/unusable record cannot be re-grounded or number-checked.
        return False, 0

    if "grounding" not in entry:
        _issue(issues, index, "missing_grounding")
        return False, 0
    grounding = entry["grounding"]
    if not isinstance(grounding, dict) or "claim" in grounding:
        _issue(issues, index, "invalid_grounding")
        return False, 0
    try:
        # Rebuild from the original grounding; saved check labels are irrelevant.
        item = GroundedItem.model_validate({"claim": claim, **grounding})
    except ValidationError:
        _issue(issues, index, "invalid_grounding")
        return False, 0

    for reason in grounding_check(item, usable, paraphrase=paraphrase)["reasons"]:
        _issue(issues, index, reason)
        ok = False
    numeric = check_numbers(item.claim, [usable[eid] for eid in item.claim.evidence_ids])
    if numeric["status"] == "missing_numeric_literals":
        _issue(issues, index, "missing_numeric_literals")
        ok = False
    return ok, len(item.supports) if ok else 0


def audit_analysis(analysis: dict, profile: ProjectProfile, paper: PaperRecord, *, paraphrase=None) -> dict:
    """Return deterministic re-audit findings for one saved analysis record."""
    issues = []
    if not isinstance(analysis, dict):
        _issue(issues, None, "invalid_analysis_structure")
        return {"status": "needs_review", "accepted_claim_count": 0, "support_span_count": 0,
                "issues": issues, "semantic_acceptance": False}

    identity_ok = True
    if analysis.get("paper_id") != paper.paper_id:
        _issue(issues, None, "analysis_paper_mismatch")
        identity_ok = False
    if analysis.get("paper_version") != paper.version:
        _issue(issues, None, "analysis_paper_version_mismatch")
        identity_ok = False
    if analysis.get("profile_version") != profile.version:
        _issue(issues, None, "analysis_profile_version_mismatch")
        identity_ok = False
    if analysis.get("verification_method") != METHOD:
        _issue(issues, None, "unsupported_verification_method")
        identity_ok = False

    raw_evidence = analysis.get("evidence")
    if not isinstance(raw_evidence, list):
        _issue(issues, None, "invalid_analysis_structure")
        raw_evidence = []
    raw_accepted = analysis.get("accepted_claims")
    if not isinstance(raw_accepted, list):
        _issue(issues, None, "invalid_analysis_structure")
        raw_accepted = []

    known, usable, unusable = _audit_evidence(raw_evidence, paper, issues)
    indexes, index_counts = _claim_indexes(raw_accepted)
    question_ids = {fact.id for fact in profile.facts if fact.kind == "question"}

    if paraphrase is None:
        paraphrase = analysis.get("gate_version") == GATE_VERSION
    passing = support_spans = 0
    for position, entry in enumerate(raw_accepted):
        index = indexes[position]
        if index is None:
            _issue(issues, None, "invalid_claim_index")
        elif index_counts[index] > 1:
            _issue(issues, index, "duplicate_claim_index")
        ok, spans = _check_claim(entry, index, index is not None and index_counts[index] == 1,
                                 identity_ok, known, usable, unusable, question_ids, issues,
                                 paraphrase=paraphrase)
        if ok:
            passing += 1
            support_spans += spans

    return {"status": "needs_review" if issues else "checks_passed",
            "accepted_claim_count": passing, "support_span_count": support_spans,
            "issues": issues, "semantic_acceptance": False}
