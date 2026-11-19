"""Source-span gates for comparisons and solver assumptions; scientific review remains required."""
import json
import re
import unicodedata
from typing import Literal

from pydantic import Field

from .config import StrictModel
from .llm import ModelFailure
from .models import Claim, EvidenceRecord, Verification
from .verification import check_numbers


class SupportSpan(StrictModel):
    evidence_id: str
    quote: str = Field(min_length=12, max_length=1500)
    role: Literal["method", "control", "result", "condition", "limitation"]


class Comparison(StrictModel):
    intervention: str = Field(min_length=2)
    comparator: str = Field(min_length=2)
    support_index: int = Field(ge=0)


class GroundedItem(StrictModel):
    claim: Claim
    supports: list[SupportSpan] = Field(min_length=1, max_length=6)
    comparison: Comparison | None = None
    assumption_scope: Literal["none", "solver_quality", "policy_optimization", "dynamics_bound"] = "none"


class GroundedAnalysis(StrictModel):
    items: list[GroundedItem] = Field(max_length=10)
    unanswered: list[str] = Field(default_factory=list)


def normalized(text):
    # Preserve case: mW and MW are different units. Only presentation whitespace is relaxed.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


# Copy detection ignores case and punctuation; units still compare through normalized().
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_NON_TOKEN = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+")


def folded(text):
    """Lowercase, punctuation-free, whitespace-collapsed form used only to detect copied spans."""
    return _NON_TOKEN.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _english_words(text):
    """English words of at least three letters; short function words are ignored for overlap."""
    return [word for word in folded(text).split() if word.isascii() and word.isalpha() and len(word) >= 3]


def _han_ratio(text):
    han = len(_CJK.findall(text))
    letters = han + len(re.findall(r"[A-Za-z]", text))
    return han / letters if letters else 0.0


def _max_english_run(text):
    """Longest stretch of consecutive English words, a proxy for 'the claim is an English sentence'."""
    longest = run = 0
    for token in folded(text).split():
        run = run + 1 if token.isascii() and token.isalpha() else 0
        longest = max(longest, run)
    return longest


def _word_overlap(words, quote_text):
    vocabulary = set(folded(quote_text).split())
    return sum(1 for word in words if word in vocabulary) / len(words) if words else 0.0


def paraphrase_reason(claim_text, quotes):
    """Return a rejection reason when claim.text copies its own quotes instead of restating them.

    The claim must be a Chinese restatement. Verbatim reuse is a copy when the folded claim equals a
    folded quote, or when the whole claim (>=12 characters) sits inside a quote. A claim with no
    Chinese at all and >=3 English words is an English sentence, not a restatement. A mixed claim is
    still rejected when >60% of its >=3-letter English words occur in one of its quotes while Chinese
    characters cover <40% of its letters and it reads as running English (>=5 words, a run of >=3);
    a Chinese sentence keeping a few English method/metric names has neither a long run nor that
    little Chinese, so it passes.
    """
    claim = folded(claim_text)
    if not claim:
        return None
    for quote_text in quotes:
        quote = folded(quote_text)
        if quote and (claim == quote or (len(claim) >= 12 and claim in quote)):
            return "claim_is_quote_copy"
    words = _english_words(claim_text)
    if not _CJK.search(claim_text) and len(words) >= 3:
        return "claim_not_paraphrased"
    if (len(words) >= 5 and _max_english_run(claim_text) >= 3 and _han_ratio(claim_text) < 0.40
            and any(_word_overlap(words, quote) > 0.60 for quote in quotes)):
        return "claim_not_paraphrased"
    return None


GATE_VERSION = "grounding-gates-v4-comparison-ellipsis"
EXTRACTION_VERSION = "located-quality-v2"


def _comparison_label_present(label, quote):
    if normalized(label) in quote:
        return True
    # Explicit English shared-noun ellipsis: “absolute instead of relative actions”.
    # Both contrasting adjectives and the shared noun must occur in this one quote.
    for match in re.finditer(r"\b([A-Za-z-]+) (?:instead of|rather than|versus|vs\.?) ([A-Za-z-]+) ([A-Za-z-]+)\b", quote):
        if normalized(label) == match[1] + " " + match[3]:
            return True
    return False


def grounding_check(item, by_id, *, paraphrase=True):
    reasons = []
    cited = set(item.claim.evidence_ids)
    if any(by_id[eid].quality_flags for eid in cited if eid in by_id):
        reasons.append("parser_quality_review_required")
    for support in item.supports:
        if support.evidence_id not in cited or support.evidence_id not in by_id:
            reasons.append("support_outside_claim_citations")
        elif by_id[support.evidence_id].quality_flags:
            reasons.append("parser_quality_review_required")
        elif normalized(support.quote) not in normalized(by_id[support.evidence_id].text):
            reasons.append("quote_not_in_source")
    if paraphrase:
        copied = paraphrase_reason(item.claim.text, [support.quote for support in item.supports])
        if copied:
            reasons.append(copied)
    comparative = re.search(r"相比|相对(?!简单|复杂|容易|困难|有限)|优于|提高|提升|降低|\b(versus|outperform|vs\.?)\b",
                            item.claim.text, re.IGNORECASE)
    solver_claim = re.search(r"求解器|优化器|\b(solver|optimizer)\b", item.claim.text, re.IGNORECASE)
    assumption_claim = re.search(r"假设|前提|要求|单调|\b(assum\w*|monotonic\w*)\b", item.claim.text, re.IGNORECASE)
    if item.claim.kind == "paper_fact" and solver_claim and assumption_claim and item.assumption_scope == "none":
        reasons.append("solver_quality_scope_required")
    if item.claim.kind == "paper_fact" and comparative and item.comparison is None:
        reasons.append("comparison_labels_required")
    if item.comparison:
        comparison = item.comparison
        if comparison.support_index >= len(item.supports):
            reasons.append("comparison_support_missing")
        else:
            quote = normalized(item.supports[comparison.support_index].quote)
            for label in (comparison.intervention, comparison.comparator):
                if not _comparison_label_present(label, quote):
                    reasons.append("comparison_label_not_in_same_source_span")
                if normalized(label) not in normalized(item.claim.text):
                    reasons.append("comparison_label_not_in_claim")
            if normalized(comparison.intervention) == normalized(comparison.comparator):
                reasons.append("comparison_labels_identical")
    if item.assumption_scope != "none":
        conditions = " ".join(normalized(s.quote) for s in item.supports if s.role == "condition").casefold()
        if not conditions:
            reasons.append("assumption_condition_span_required")
        if item.assumption_scope == "solver_quality" and not (
                re.search(r"\b(solver|optimizer)\b", conditions) and re.search(r"\b(monotonic\w*|improv\w*)\b", conditions)
                and re.search(r"\b(budget|computation\w*)\b", conditions)):
            reasons.append("solver_quality_not_grounded_in_solver_budget_span")
        if item.assumption_scope == "policy_optimization" and not re.search(r"\b(ppo|surrogate|advantage\w*)\b", conditions):
            reasons.append("policy_assumption_not_grounded")
        if item.assumption_scope == "dynamics_bound" and not re.search(r"\b(lipschitz|wasserstein|kernel)\b", conditions):
            reasons.append("dynamics_assumption_not_grounded")
    return {"status": "rejected" if reasons else "source_spans_present", "reasons": sorted(set(reasons)),
        "limitation": "Literal spans and labels cannot prove entailment, causal interpretation or complete coverage"}


def regate_analysis(result):
    """Re-evaluate deterministic checks without changing claims or model verdicts."""
    by_id = {e.evidence_id: e for e in map(EvidenceRecord.model_validate, result["evidence"])}
    candidates = sorted(result["accepted_claims"] + result["rejected_claims"],
                        key=lambda item: item["verification"]["claim_index"])
    kept, rejected = [], []
    for original in candidates:
        item = GroundedItem.model_validate({"claim": original["claim"], **original["grounding"]})
        check = grounding_check(item, by_id)
        numeric = check_numbers(item.claim, [by_id[eid] for eid in item.claim.evidence_ids])
        updated = {**original, "grounding_check": check, "numeric_check": numeric}
        if original["verification"]["support"] == "supported" and check["status"] != "rejected" and numeric["status"] != "missing_numeric_literals":
            kept.append(updated)
        else:
            rejected.append(updated)
    return {**result, "accepted_claims": kept, "rejected_claims": rejected, "rejected_count": len(rejected),
            "verification_method": "source-span-label-scope-gates-v2", "gate_version": GATE_VERSION,
            "model_verdicts_unchanged": True}


def _truncated(error: Exception) -> bool:
    message = str(error).lower()
    return "truncated" in message or "incomplete" in message


def repair_support_locations(item, by_id):
    """Repair only unique verbatim matches within the already selected exact-paper evidence."""
    repaired = item.model_copy(deep=True)
    changes = []
    for support in repaired.supports:
        old = by_id.get(support.evidence_id)
        quote = normalized(support.quote)
        if old is None or quote in normalized(old.text):
            continue
        matches = [e for e in by_id.values()
                   if not e.quality_flags and (e.paper_id, e.paper_version) == (old.paper_id, old.paper_version)
                   and quote in normalized(e.text)]
        if len(matches) != 1:
            continue
        changes.append({"from": support.evidence_id, "to": matches[0].evidence_id,
                        "quote": support.quote})
        support.evidence_id = matches[0].evidence_id
        if support.evidence_id not in repaired.claim.evidence_ids:
            repaired.claim.evidence_ids.append(support.evidence_id)
    return repaired, changes


def _remaining(client) -> int:
    getter = getattr(client, "remaining", None)
    return getter() if callable(getter) else 1_000_000


def analyze_grounded(client, profile, paper, evidence, question):
    if not evidence or any((e.paper_id, e.paper_version) != (paper.paper_id, paper.version) for e in evidence):
        raise ValueError("Located evidence for the exact paper version required")
    by_id = {e.evidence_id: e for e in evidence}
    if len(by_id) != len(evidence):
        raise ValueError("Duplicate evidence IDs")
    qids = {f.id for f in profile.facts if f.kind == "question"}
    payload = json.dumps({"question": question,
        "linked_project_background": [f.model_dump() for f in profile.facts if f.id.startswith('project:')],
        "project_questions": [f.model_dump() for f in profile.facts if f.kind == "question"],
        "paper": paper.model_dump(), "evidence": [e.model_dump() for e in evidence],
          "instruction": "关联项目背景保留来源日期与历史属性，只用于判断迁移前提，不能作为论文证据或当前实验状态。"
          "用中文，最多4条，每条只陈述一个可由所引片段完整支持的事实，不把多项方法细节拼成一条。"
          "先回答方法、实验控制和失败/无收益条件；未取得的条件进入unanswered，"
        "不得把缺失证据写成论文不存在局限。每条必须引用原文quote，逐字复制连续原文（不翻译，不加省略号），"
        "quote长度12至240字符，evidence_id须真实。不要复述项目实验数字。"
        "条件与限制用一句话写清（不超过80字）；整个回复不超过1200字，避免逐段复述原文。"
        "claim.text必须是中文转述：保留方法名、指标名、数字等英文专名与缩写，禁止整句照抄原文；"
        "技术描述也须译成中文，例如action-conditioned neural ODE写动作条件神经常微分方程；只保留论文方法专名。"
        "英文原文只放grounding.supports[].quote逐字引用。claim.text中连续英文单词不得超过6个，"
        "若claim基本是英文句子则视为无效，必须先转述为中文。"
        "若给定证据含结果表或定量比较，优先保留最多2条有具体任务、条件、比较对象和原文数值的结果；"
        "表编号不是定量结果。没有足够证据则如实说明，不强行补数值。"
        "比较结论必须填写comparison，intervention和comparator取同一原文span中的原始方法名称，"
        "并在中文claim.text保留相同英文名称；不得替换比较对象或混用不同表的对照。"
        "assumption_scope只标记所述具体理论：solver_quality、policy_optimization、dynamics_bound；"
        "一般可识别性或理想化条件不属于动力学误差界，使用none并照常引用condition；"
        "solver质量随预算改进与PPO更新理论是不同假设，condition原文必须支持对应类型。"
        "不自行计算提升量；没有置信区间不能称显著。推断和建议单列，建议最多1条且不启动实验。"
        "选定结果后必须补读同一段及相邻的实验设置原文、表注和附录细节（已在给定证据中的部分）："
        "记录数据量、独立重复或训练种子数、误差定义（标准差/标准误/置信区间/bootstrap）、基线"
        "以及计算或训练预算，并写入claim或condition。给定证据已写明的内容（如“four seeds”"
        "“per model, modality, and environment”）必须写入claim或条件，不得写成“本次未读取”；"
        "只有给定证据确实未包含的项才写“本次未读取”，任何情况下不得写“论文未报告”。"
        "不得仅凭种子数量推断统计显著性。"}, ensure_ascii=False)
    try:
        analysis = client.structured(payload, GroundedAnalysis, "有原文定位的科研分析")
        degraded = False
    except ModelFailure as error:
        if not _truncated(error) or _remaining(client) < 2:
            raise
        degraded = True
        compact = (payload.replace("最多4条", "最多2条")
                          .replace("quote长度12至240字符", "quote长度12至160字符")
                          .replace("整个回复不超过1200字", "整个回复不超过600字"))
        compact = compact + (
            "\n精简模式：每条claim只陈述一个事实；不要写比较句（不出现“优于/低于/高于/相比”）。"
            "quote必须从给定evidence的text中逐字连续复制，不得改写、翻译、拼接或跨段引用；"
            "找不到可逐字复制的原句就不要写这条claim，把该问题写入unanswered。")
        analysis = client.structured(compact, GroundedAnalysis, "有原文定位的科研分析")
    repairs = []
    for index, item in enumerate(analysis.items):
        repaired, changes = repair_support_locations(item, by_id)
        analysis.items[index] = repaired
        repairs.append(changes)
    checks = []
    for item in analysis.items:
        if not set(item.claim.evidence_ids) <= by_id.keys() or not set(item.claim.question_ids) <= qids:
            raise ValueError("Grounded analysis contains nonexistent evidence/project references")
        checks.append(grounding_check(item, by_id))
    # Check raw evidence again, not only summaries or the quoted spans supplied by the first call.
    verify_prompt = (payload + "\n核验下列items的claim，零起始索引。逐项检查比较对象、控制变量、"
        "假设类型和失败条件，禁止只因数字存在就放行。不能确认即partial。\n" + analysis.model_dump_json())
    try:
        verdicts = client.structured(verify_prompt, Verification, "比较条件与原文证据核验")
    except ModelFailure as error:
        if not _truncated(error) or _remaining(client) < 1:
            raise
        verdicts = client.structured(verify_prompt + "\n输出必须极简：只给verdicts数组，reason不超过20字，不要任何解释。",
                                     Verification, "比较条件与原文证据核验")
    if sorted(v.claim_index for v in verdicts.verdicts) != list(range(len(analysis.items))):
        raise ValueError("Verifier omitted/duplicated/out-of-range item indices")
    kept, rejected = [], []
    for verdict in verdicts.verdicts:
        item = analysis.items[verdict.claim_index]
        numeric = check_numbers(item.claim, [by_id[eid] for eid in item.claim.evidence_ids])
        record = {"claim": item.claim.model_dump(), "verification": verdict.model_dump(), "numeric_check": numeric,
            "grounding": item.model_dump(exclude={"claim"}), "grounding_check": checks[verdict.claim_index],
            "citation_repairs": repairs[verdict.claim_index]}
        if verdict.support == "supported" and numeric["status"] != "missing_numeric_literals" and record["grounding_check"]["status"] != "rejected":
            kept.append(record)
        else:
            rejected.append(record)
    return {"paper_id": paper.paper_id, "paper_version": paper.version, "profile_version": profile.version,
        "accepted_claims": kept, "rejected_claims": rejected, "rejected_count": len(rejected),
        "unanswered": analysis.unanswered, "verification_method": "source-span-label-scope-gates-v2",
        "gate_version": GATE_VERSION, "extraction_version": EXTRACTION_VERSION, "degraded_output": degraded,
        "evidence": [e.model_dump() for e in evidence], "human_review_required": True}
