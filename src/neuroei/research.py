import json
import re
from datetime import datetime

from rank_bm25 import BM25Okapi

from .models import Analysis, EvidenceRecord, FeedbackEvent, Ranking, Verification
from .verification import check_numbers


class RecommendationValidationError(ValueError):
    code = 'invalid_recommendation_references'


def tokenize(text):
    # Latin words + Chinese characters: exploratory lexical retrieval only, no semantic claim.
    return re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", text.lower()) or [""]


def recommend(client, profile, papers, limit=5, *, context_kind='project', retrieval_query=None):
    if not papers:
        return []
    questions = [f for f in profile.facts if f.kind == "question"]
    if not questions:
        raise ValueError("No sourced project questions available")
    bm25 = BM25Okapi([tokenize(p.title + " " + p.abstract) for p in papers])
    # Direction settings may be Chinese while the metadata is English. Reuse the already
    # approved public English queries for retrieval; keep the original facts for model judgment.
    query = retrieval_query or " ".join(f.text for f in profile.facts if f.kind in ("goal", "question"))
    scores = bm25.get_scores(tokenize(query))
    pool = [papers[i] for i in sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)[:30]]
    return rerank_candidates(client, profile, pool, limit=limit, context_kind=context_kind)


def rerank_candidates(client, profile, pool, limit=5, *, context_kind='project'):
    """Rerank an explicitly frozen candidate list without an extra hidden retrieval step."""
    if not pool:
        return []
    if len({p.paper_id for p in pool}) != len(pool):
        raise ValueError("Duplicate reranking candidates")
    questions = [f for f in profile.facts if f.kind == "question"]
    if not questions:
        raise ValueError("No sourced project questions available")
    active = [f for f in profile.facts if f.kind not in ("exclusion", "constraint", "goal")
              or f.temporal_status != "historical"]
    historical = [f for f in profile.facts if f not in active]
    relevance_policy = ("评分依据使用者设置的主题、重点问题和排除内容；2或3须有具体相关方法、结果或研究线索，"
        "只有术语相似只能评1。研究领域由关注设置决定，不额外套用机械臂或BNN领域限制。"
        if context_kind == 'direction' else
        "评分严格：3=摘要给出可直接用于当前问题的方法/评测/失败条件，且与项目方法同构；"
        "2=有部分可迁移内容，但任务、平台或指标不同；1=仅主题或术语相近；0=不相关。"
        "只有主题/术语相似、启发式类比或材料器件应用背景，缺少可迁移计算方法时只能评1。"
        "不要把‘树突神经元’同名材料器件直接当成生物物理神经元模拟或机械臂控制方法。")
    payload = {
        "profile_facts": [f.model_dump() for f in active],
        "historical_context_only": [f.model_dump() for f in historical],
        "candidates": [{"paper_id": p.paper_id, "title": p.title, "abstract": p.abstract} for p in pool],
        "instruction": f"最多推荐{limit}篇，允许零篇；relevance为0-3，2为对当前问题有用，3优先阅读。"
        + relevance_policy +
        "reason 必须写清三件事：可迁移的具体内容；迁移所需前提（任务/平台/指标差异）；"
        "哪一条项目问题因此受益。缺少前提或差异说明的理由视为不完整。"
        "reason 总长度不超过160字，用三句短句分别回答这三件事；不要复述摘要原文，"
        "引用数字必须保留最高/平均等限定及指标、对照对象；不同指标的最佳基线不得混写，篇幅不足则省略数字。"
        "不要引用问题编号（如 direction-q1），需要提及时用问题的文字描述。"
        "仅根据已提供的标题和摘要评相关性，不声称验证全文。摘要为空时只可作为待取得摘要的题目线索，"
        "必须在reason明确未取得摘要，不得推断方法细节、数值、结果、迁移条件或仅凭顶会顶刊名称提高评分。"
        "question_ids只能引用question类型的项目事实。"
        "优先current_candidate问题；historical不代表已解决，但历史失败不得作为永久排除。"}
    ids, qids = {p.paper_id for p in pool}, {q.id for q in questions}
    for attempt in range(2):
        ranking = client.structured(json.dumps(payload, ensure_ascii=False), Ranking, "发现与推荐")
        seen = set()
        invalid = False
        for item in ranking.recommendations:
            if item.paper_id not in ids or not set(item.question_ids) <= qids or item.paper_id in seen:
                invalid = True
            seen.add(item.paper_id)
        if not invalid:
            break
        # A fresh bounded request, without echoing untrusted model text. The normal
        # client budget (including held synthesis calls) still applies.
        if attempt or (hasattr(client, 'remaining') and client.remaining() < 1):
            raise RecommendationValidationError("Recommendation contains invalid or duplicated IDs")
        payload['reference_repair'] = {
            'instruction': '上次推荐引用校验失败。只从下列ID选择，不得重复paper_id；无法推荐时返回空数组。',
            'paper_ids': sorted(ids), 'question_ids': sorted(qids)}
    seen, result = set(), []
    for item in ranking.recommendations:
        seen.add(item.paper_id)
        if item.relevance >= 2:
            result.append(item)
    return sorted(result, key=lambda r: r.relevance, reverse=True)[:limit]


def analyze(client, profile, paper, evidence: list[EvidenceRecord]):
    if not evidence:
        raise ValueError("Full-text evidence is required; cannot substitute abstract")
    if any(e.paper_id != paper.paper_id or e.paper_version != paper.version for e in evidence):
        raise ValueError("Evidence belongs to a different paper/version")
    if len({e.evidence_id for e in evidence}) != len(evidence):
        raise ValueError("Duplicate evidence IDs")
    payload = json.dumps({"project": [f.model_dump() for f in profile.facts if f.kind in ("goal", "question")],
        "paper": paper.model_dump(), "evidence": [e.model_dump() for e in evidence],
        "instruction": "最多12条简短主张，每条正文尽量160汉字内。区分paper_fact、inference、experiment_suggestion。"
        "须覆盖方法及适用前提，而非只罗列成绩。声明条件、局限和项目问题；"
        "paper_fact的每个数字必须出现在所引用证据中；自行计算/跨域解释标为inference。"
        "没有置信区间不能称统计显著。不得把预测窗口、执行窗口和推理延迟混为一谈。"
        "实验建议最多2条，须有对照、指标、失败判据和待实现前提，不立即启动实验。"
        "BNN/ACT未必有flow velocity，不能假定可直接使用FlowRes。项目材料为历史记录，"
        "仅用于定位问题，不复述或补写项目实验数字，不描述当前实验运行状态。不宣称领域问题已解决。"}, ensure_ascii=False)
    analysis = client.structured(payload, Analysis, "科研分析")
    eids = {e.evidence_id for e in evidence}
    qids = {f.id for f in profile.facts if f.kind == "question"}
    for claim in analysis.claims:
        if not set(claim.evidence_ids) <= eids or not set(claim.question_ids) <= qids:
            raise ValueError("Analysis contains nonexistent evidence/project references")
    verdicts = client.structured(payload + "\n逐条核验以下claims（零起始索引），重点检查数字、单位、条件、"
        "引用是否支持；推断和建议是否越界。\n" + analysis.model_dump_json(), Verification, "证据核验")
    indices = [v.claim_index for v in verdicts.verdicts]
    if sorted(indices) != list(range(len(analysis.claims))):
        raise ValueError("Verifier omitted/duplicated/out-of-range claim indices")
    by_id = {e.evidence_id: e for e in evidence}
    kept, rejected = [], []
    for verdict in verdicts.verdicts:
        claim = analysis.claims[verdict.claim_index]
        numeric = check_numbers(claim, [by_id[eid] for eid in claim.evidence_ids])
        parser_flags = sorted({flag for eid in claim.evidence_ids for flag in by_id[eid].quality_flags})
        item = {"claim": claim.model_dump(), "verification": verdict.model_dump(), "numeric_check": numeric,
                "parser_quality_flags": parser_flags}
        if verdict.support == "supported" and numeric["status"] != "missing_numeric_literals" and not parser_flags:
            kept.append(item)
        else:
            rejected.append(item)
    return {"paper_id": paper.paper_id, "paper_version": paper.version,
            "profile_version": profile.version, "accepted_claims": kept,
            "rejected_count": len(analysis.claims) - len(kept),
            "rejected_claims": rejected, "verification_method": "same-model-plus-literal-number-gate-v1",
            "evidence": [e.model_dump() for e in evidence], "human_review_required": True}


def prior_feedback(events: list[FeedbackEvent], project_id: str, cutoff: datetime):
    if cutoff.tzinfo is None:
        raise ValueError("Feedback cutoff needs timezone")
    return [e for e in events if e.project_id == project_id and e.occurred_at < cutoff]
