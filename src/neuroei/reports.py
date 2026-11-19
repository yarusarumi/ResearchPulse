import hashlib
import re
from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from .models import Claim, EvidenceRecord
from .run_diagnostics import analysis_assurance


def _citation_url(evidence):
    """External locator for an evidence block; HTML block IDs are the parsed anchors."""
    if not evidence.file.startswith("https://"):
        return None
    if evidence.parser_version.startswith("europepmc-oa-jats-"):
        return evidence.file
    anchor = f"page={evidence.page}" if evidence.page else evidence.block_id
    return evidence.file.split("#", 1)[0] + "#" + anchor


SOURCE_LABELS = {"pubmed": "PubMed", "biorxiv": "bioRxiv", "arxiv": "arXiv", "crossref": "Crossref（期刊／会议元数据）"}
STATUS_LABELS = {"ok": "完整", "partial": "不完整", "failed": "失败", "empty": "无结果",
                 "running": "进行中", "pending": "待处理"}
NUMERIC_LABELS = {"literal_numbers_present": "数字已在原文找到",
                  "checked": "数字已在原文找到",
                  "missing": "原文未出现对应数字"}
GROUNDING_LABELS = {"source_spans_present": "已定位原文片段",
                    "supported": "已对应原文",
                    "unsupported": "未找到对应原文"}

_ENVIRONMENT = None
_TEMPLATES = {}


def _environment():
    """One lazily built Jinja environment; filters are the pure module functions above."""
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        env = Environment(undefined=StrictUndefined, autoescape=False)
        env.filters["source_label"] = source_label
        env.filters["status_label"] = status_label
        env.filters["date_only"] = date_only
        env.filters["numeric_label"] = numeric_label
        env.filters["grounding_label"] = grounding_label
        _ENVIRONMENT = env
    return _ENVIRONMENT


def _template(name):
    """Compile each packaged template once; Jinja compilation costs ~9-15 ms per call."""
    template = _TEMPLATES.get(name)
    if template is None:
        source = files("neuroei").joinpath("templates/" + name).read_text(encoding="utf-8")
        template = _environment().from_string(source)
        _TEMPLATES[name] = template
    return template


def source_label(name):
    """Reader-facing source name; journal feeds keep their journal title."""
    raw = str(name or "")
    if raw in SOURCE_LABELS:
        return SOURCE_LABELS[raw]
    if raw.startswith("journal:"):
        return raw[len("journal:"):] + "（期刊）"
    if raw.startswith("rss:"):
        return raw[len("rss:"):]
    return raw


def date_only(value):
    """Timestamp or ISO date to a plain YYYY-MM-DD label."""
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-" else text


def status_label(status):
    return STATUS_LABELS.get(str(status or ""), str(status or ""))


def numeric_label(status):
    """Whether a claim's numbers were found verbatim in the quoted passage."""
    return NUMERIC_LABELS.get(str(status or ""), str(status or ""))


def grounding_label(status):
    """Whether a claim is bound to locatable source spans."""
    return GROUNDING_LABELS.get(str(status or ""), str(status or ""))


def daily_report(*, project, report_date, run_id, sources, papers, recommendations, preview=False,
                 reading_statuses=None, ranking_strategy='exploratory-bm25-flash', experimental=False,
                 context_kind='project', reading_report_included: bool | None = None):
    lookup = {p.paper_id: p for p in papers}
    if reading_statuses is not None:
        reading_statuses = {key: {**value, 'assurance': value.get('assurance') or analysis_assurance(None)}
                            for key, value in reading_statuses.items()}
    text = _template("daily.md.j2").render(project=project, report_date=report_date, run_id=run_id,
        sources=sources, papers=papers, recommendations=recommendations, lookup=lookup, preview=preview,
        reading_statuses=reading_statuses, ranking_strategy=ranking_strategy, experimental=experimental,
        context_kind=context_kind, reading_report_included=reading_report_included)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def reading_report(*, project, report_date, run_id, entries, context_kind='project'):
    """Render only accepted claims with verified identities and real source locators."""
    questions = {f.id: f for f in project.facts if f.kind == "question"}
    blocks = {b.block_id: b for b in project.blocks}
    prepared = []
    for entry in entries:
        paper, analysis = entry["paper"], entry["analysis"]
        available = analysis.get("available_blocks") or 0
        selected = analysis.get("selected_blocks") or 0
        coverage = {"selected": selected, "available": available,
                    "percent": f"{selected / available * 100:.1f}%" if available else "未知"}
        if (analysis["paper_id"], analysis["paper_version"], analysis["profile_version"]) != (
                paper.paper_id, paper.version, project.version):
            raise ValueError("Reading report has mismatched paper/profile versions")
        if analysis.get("verification_method") == "source-span-label-scope-gates-v2":
            from .evidence_audit import audit_analysis
            if audit_analysis(analysis, project, paper)["status"] != "checks_passed":
                raise ValueError("Stored source-span analysis failed deterministic re-audit")
        evidence = {e.evidence_id: e for e in map(EvidenceRecord.model_validate, analysis["evidence"])}
        claims = []
        for index, accepted in enumerate(analysis["accepted_claims"]):
            claim = Claim.model_validate(accepted["claim"])
            if accepted["verification"]["support"] != "supported":
                raise ValueError("Unverified claim cannot enter reading report")
            if accepted.get("numeric_check", {}).get("status") == "missing_numeric_literals":
                raise ValueError("Numerically rejected claim cannot enter reading report")
            if accepted.get("grounding_check", {}).get("status") == "rejected":
                raise ValueError("Source-span rejected claim cannot enter reading report")
            if accepted.get("parser_quality_flags"):
                raise ValueError("Parser quality review required before reporting")
            if not set(claim.evidence_ids) <= evidence.keys() or not set(claim.question_ids) <= questions.keys():
                raise ValueError("Missing reading report source references")
            citations = []
            for eid in claim.evidence_ids:
                e = evidence[eid]
                if e.quality_flags:
                    raise ValueError("Parser quality review required before reporting")
                if (e.paper_id, e.paper_version) != (paper.paper_id, paper.version):
                    raise ValueError("Reading citation belongs to another paper/version")
                # HTML block IDs are the actual parsed anchors; page citations come from PDF parsers.
                url = _citation_url(e)
                citations.append({"id": eid, "label": f"p.{e.page}" if e.page else e.section,
                                  "url": url, "file": e.file,
                                  "kind": "表格" if e.section and "table" in e.section.casefold() else "段落"})
            citation_by_id = {citation["id"]: citation for citation in citations}
            supports = []
            for support in (accepted.get("grounding") or {}).get("supports", []):
                if not isinstance(support, dict):
                    continue
                eid = support.get("evidence_id")
                if eid not in evidence:
                    continue
                quote = str(support.get("quote", "")).strip()
                if not quote:
                    continue
                if len(quote) > 240:
                    quote = quote[:239] + "…"
                cite = citation_by_id.get(eid)
                if cite is None:
                    source = evidence[eid]
                    cite = {"url": _citation_url(source),
                            "kind": "表格" if source.section and "table" in source.section.casefold()
                            else "段落"}
                roles = {"method": "方法", "control": "对照", "result": "结果", "condition": "条件",
                         "limitation": "限制", "source": "来源"}
                role = roles.get(str(support.get("role")), str(support.get("role") or "来源"))
                supports.append({"evidence_id": eid, "quote": quote, "role": role,
                                 "url": cite.get("url"),
                                 "file": evidence[eid].file, "kind": cite.get("kind", "段落")})
            claim_index = (accepted.get("verification") or {}).get("claim_index", index)
            claim_ref = f"{analysis['run_id']}#claim-{claim_index}"
            anchor = 'claim-' + hashlib.sha256(claim_ref.encode()).hexdigest()[:12]
            claims.append({"claim": claim, "citations": citations, "supports": supports,
                           "anchor": anchor, "claim_ref": claim_ref,
                           "numeric_status": accepted.get("numeric_check", {}).get("status", "not_run"),
                           "grounding_status": accepted.get("grounding_check", {}).get("status", "not_run")})
        qids = sorted({qid for item in claims for qid in item["claim"].question_ids})
        project_refs = [{"question": questions[qid], "blocks": [blocks[bid] for bid in questions[qid].source_blocks]}
                        for qid in qids]
        prepared.append({**entry, "claims": claims, "project_refs": project_refs, "coverage": coverage,
                         "rejected": analysis.get("rejected_count", len(analysis.get("rejected_claims") or []))})
    source_dates = [f.source_date for f in project.facts if f.source_date]
    return _template("reading.md.j2").render(
        project=project, report_date=report_date, run_id=run_id, entries=prepared,
        latest_source_date=max(source_dates) if source_dates else "未知", context_kind=context_kind)
