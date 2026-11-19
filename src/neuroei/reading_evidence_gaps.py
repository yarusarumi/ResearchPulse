"""Project saved reading evidence into user-visible gaps without rerunning analysis."""
import re


AUTHOR_GAP = re.compile(
    r"^(?:(?:in (?:this|the present) (?:study|paper),\s+)?we\s+"
    r"(?:(?:did|do) not\s+(?:report|disclose|provide|measure|evaluate|test|assess|collect|record|compare|vary)\b"
    r"|have not\s+(?:reported|disclosed|provided|measured|evaluated|tested|assessed|collected|recorded|compared|varied)\b)"
    r"|本文(?:没有|未)(?:报告|披露|提供|测量|评估|检验|测试|采集|记录|比较))",
    flags=re.IGNORECASE,
)
SCOPED_GAP_NOTE = re.compile(r"^(?:本次未(?:定位|核实|找到)|(?:所给|现有|当前)(?:摘录|材料|证据)未(?:见|找到|列出|提供|给出)|"
                            r"所给原文未(?:见|找到|列出|提供)|无法(?:确认|核对|核实))")
EXPERIMENT_FIELDS = ("dataset", "task", "training", "evaluation", "comparator", "metric", "statistical_unit")


def reading_unanswered_notes(scope):
    provider = (scope.get("paperqa_reading") or {}).get("provider_output")
    return (provider.get("unanswered") or ()) if isinstance(provider, dict) else ()


def evidence_gaps(analysis, question_ids=()):
    scope = analysis.get("reading_scope") or {}
    warnings = analysis.get("warnings") or ()
    acquisition = scope.get("fulltext_acquisition")
    if (any(w.startswith("fulltext_unavailable;") for w in warnings)
            or (acquisition in ("failed", "unsupported") and not scope.get("documents"))
            or "fulltext unavailable" in scope.get("missing", ())):
        return [{"kind": "fulltext_unavailable", "acquisition": acquisition}]

    claims = {claim["claim_id"]: claim for claim in analysis.get("claims", ())}
    limitation_blocks = {ref["evidence_id"] for cid in analysis.get("limitations", ()) if cid in claims
                         for ref in claims[cid].get("supports", ())}
    gaps = []
    seen_quotes = set()
    for source in analysis.get("evidence", ()):
        eid = source["evidence_id"]
        section = source.get("section") or ""
        if (source.get("label") == "ocr_text" or (source.get("quality") or {}).get("quality_flags")
                or (eid not in limitation_blocks and not re.search(r"\blimitations?\b|局限", section, re.I))):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", source["text"]):
            quote = sentence.strip()
            if 10 <= len(quote) <= 500 and quote not in seen_quotes and AUTHOR_GAP.match(quote):
                gaps.append({"kind": "author_explicit", "quote": quote, "evidence_id": eid})
                seen_quotes.add(quote)

    checkpoint = scope.get("paperqa_checkpoint") or {}
    review = checkpoint.get("semantic_review_status")
    questions = []
    conditions = []
    if analysis.get("evidence") and review in ("model_checked_not_human_verified", "not_run_no_claims"):
        relevance = analysis.get("relevance_to_questions") or {}
        questions = [qid for qid in dict.fromkeys(question_ids)
                     if not claims.keys() & set(relevance.get(qid, ()))]
        for cid, claim in claims.items():
            values = claim.get("conditions") or {}
            fields = [field for field in EXPERIMENT_FIELDS if field in values and values[field] is None]
            if claim.get("experimental") and fields:
                conditions.append({"claim_id": cid, "fields": fields})
    # A scoped note can describe the final reading even when semantic review is still pending.
    notes = list(dict.fromkeys(note for note in reading_unanswered_notes(scope)
                               if note in warnings and SCOPED_GAP_NOTE.match(note)))
    if questions or conditions or notes:
        gaps.append({"kind": "not_located", "question_ids": questions, "conditions": conditions, "notes": notes})
    return gaps
