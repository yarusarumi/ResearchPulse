"""Deterministic seed-scope audit of synthesis text against its cited evidence.

Only direct contradictions between a cited source's stated training-seed scope and
the text are reported. Paraphrase, mixed multi-table statements and unstated
conditions stay unverified; see ``LIMITATION``. This module never infers new
experimental data and never moves a claim across paragraphs.
"""
import re

LIMITATION = ("Per-evidence literal seed-scope audit only: mixed multi-table statements, "
              "paraphrase and unstated conditions are not treated as verified.")

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_CHINESE_NUMBERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_TABLE_BLOCK = re.compile(r"(?:^|[.\-_])T\d+$", re.IGNORECASE)
_EN_SEED = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                      r"(?:random\s+|training\s+|independent\s+|different\s+)?seeds?\b", re.IGNORECASE)
_ZH_SEED = re.compile(r"([0-9]+|[一二两三四五六七八九十])\s*个?\s*(?:随机\s*|训练\s*)?(?:训练\s*)?种子")
_EN_SINGLE = re.compile(r"\b(?:one|single|a single|only one|just one)\s+"
                        r"(?:random\s+|training\s+|independent\s+)?seed\b|"
                        r"\btrained?\s+(?:with|on|using)\s+(?:one|a single)\s+seed\b", re.IGNORECASE)
_ZH_SINGLE = re.compile(r"(?:单(?:一|个|独)?|一个|1\s*个|仅\s*(?:用|使用|采用)?\s*(?:了)?\s*"
                        r"(?:一个|1\s*个))\s*(?:的)?\s*(?:随机)?\s*(?:训练)?\s*种子")
_MEAN_SE = re.compile(r"(?:mean|average)[^.;\n]{0,80}standard\s+error|"
                      r"standard\s+error[^.;\n]{0,80}(?:mean|average)|"
                      r"±\s*(?:s\.?e\.?|standard\s+error)|"
                      r"均值[^。；\n]{0,20}标准误|标准误[^。；\n]{0,20}均值", re.IGNORECASE)
_BOOTSTRAP_TASKS = re.compile(r"bootstrap[^.;\n]{0,60}\b(?:over|across)\b[^.;\n]{0,30}\btasks?\b|"
                              r"bootstrap[^。；\n]{0,40}按?\s*任务|按任务[^。；\n]{0,20}bootstrap", re.IGNORECASE)
_TRAINING_SEED_PHRASE = re.compile(r"training\s+seeds?|训练\s*种子", re.IGNORECASE)
_NO_MULTI = re.compile(
    r"(?:没有|并未|未|无)[^。；;.!?\n]{0,40}"
    r"(?:多(?:个|次|轮)?\s*(?:随机|训练)?\s*种子|多次训练|多轮训练|"
    r"multi[- ]?(?:training[- ]?)?seeds?|multiple\s+(?:training\s+)?(?:seeds|runs)|"
    r"more than one seed)|"
    r"\b(?:not|never|no|does not|do not)\b[^.;!?\n]{0,40}"
    r"(?:multi[- ]?(?:training[- ]?)?seeds?|multiple\s+(?:training\s+)?(?:seeds|runs)|"
    r"more than one seed)", re.IGNORECASE)
_GENERIC_MULTI = re.compile(
    r"多(?:个|次|轮)?\s*(?:随机|训练)?\s*种子|multi[- ]?(?:training[- ]?)?seeds?|"
    r"multiple\s+(?:training\s+|random\s+)?seeds", re.IGNORECASE)
_SCOPE_WORDS = re.compile(r"seed|bootstrap|standard\s+error|std\.?\s*err|confidence|"
                          r"\btasks?\b|episode|trial|\bruns?\b|种子|标准误|误差|置信|任务|实验", re.IGNORECASE)


def _as_number(value):
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value) or _CHINESE_NUMBERS.get(value)


def _sentences(text):
    parts = re.split(r"(?<=[。；;.!?])\s*|\n+", text or "")
    return [part.strip() for part in parts if part.strip()]


def _seed_numbers(text):
    numbers = []
    for match in _EN_SEED.finditer(text):
        number = _as_number(match.group(1))
        if number:
            numbers.append(number)
    for match in _ZH_SEED.finditer(text):
        number = _as_number(match.group(1))
        if number:
            numbers.append(number)
    return numbers


def _scope(text):
    """Literal seed/statistic scope stated by one evidence text; no inference."""
    text = text or ""
    numbers = []
    for sentence in _sentences(text):
        if _BOOTSTRAP_TASKS.search(sentence) and not _TRAINING_SEED_PHRASE.search(sentence):
            continue
        numbers.extend(_seed_numbers(sentence))
    single = bool(_EN_SINGLE.search(text) or _ZH_SINGLE.search(text))
    count = max(numbers) if numbers else None
    if single and (count is None or count == 1):
        count = 1
    statements = [sentence for sentence in _sentences(text) if _SCOPE_WORDS.search(sentence)]
    return {
        "training_seed_count": count,
        "single_training_seed": single or count == 1,
        "mean_and_standard_error": bool(_MEAN_SE.search(text)),
        "bootstrap_over_tasks": bool(_BOOTSTRAP_TASKS.search(text)),
        "scope_statements": statements[:8],
    }


def _field(record, name, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _dump(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _looks_table(block, text):
    if block and _TABLE_BLOCK.search(str(block)):
        return True
    head = (text or "")[:200]
    return " | " in (text or "") or bool(re.match(r"\s*(?:table|表)\s*[\w.\-]+", head, re.IGNORECASE))


def evidence_context(analysis):
    """Map evidence_id to its original block scope and currently citing claims.

    ``text`` is the original evidence text or table caption truncated to 1500
    characters; ``scope`` records verbatim statements, not inferred conditions.
    A claim is attached only to evidence IDs listed in that claim.
    """
    cited = {}
    for group in ("accepted_claims", "rejected_claims"):
        for item in _field(analysis, group, []) or []:
            claim = _field(item, "claim", {}) or {}
            claim_text = _field(claim, "text", "") or ""
            for evidence_id in _field(claim, "evidence_ids", []) or []:
                cited.setdefault(evidence_id, []).append(claim_text)
    contexts = {}
    for record in _field(analysis, "evidence", []) or []:
        evidence_id = _field(record, "evidence_id")
        if not evidence_id:
            continue
        raw = _field(record, "text", "") or ""
        block = _field(record, "block_id")
        contexts[evidence_id] = {
            "evidence_id": evidence_id,
            "block_id": block,
            "file": _field(record, "file"),
            "page": _field(record, "page"),
            "section": _field(record, "section"),
            "kind": "table" if _looks_table(block, raw) else "prose",
            "text": raw[:1500],
            "truncated": len(raw) > 1500,
            "cited_claim_texts": list(cited.get(evidence_id, [])),
            "scope": _scope(raw),
        }
    return contexts


def _facts(contexts):
    if isinstance(contexts, dict):
        items = list(contexts.items())
    else:
        items = [(None, context) for context in (contexts or [])]
    facts = []
    for index, (key, context) in enumerate(items):
        context = _dump(context)
        evidence_id = context.get("evidence_id") or key or f"context-{index}"
        scope = context.get("scope")
        if not isinstance(scope, dict) or not scope:
            scope = _scope(context.get("text") or "")
        facts.append((evidence_id, scope))
    return facts


def _reports_multiseed(scope):
    count = scope.get("training_seed_count") or 0
    return count > 1


def _describe_seeds(scope):
    prefix = "mean and standard error over " if scope.get("mean_and_standard_error") else ""
    return f"{prefix}{scope.get('training_seed_count')} seeds"


def _multi_claim(text):
    numbers = []
    for sentence in _sentences(text):
        if _NO_MULTI.search(sentence):
            continue
        numbers.extend(number for number in _seed_numbers(sentence) if number > 1)
    if numbers:
        return {"n": max(numbers)}
    if _GENERIC_MULTI.search(text) and not _NO_MULTI.search(text):
        return {"n": None}
    return None


def audit_scope(text, contexts):
    """Return reasons for direct seed-scope contradictions; empty otherwise.

    ``contexts`` must be dicts for the evidence actually cited by ``text``
    (typically ``evidence_context(analysis).values()``).
    """
    text = text or ""
    facts = _facts(contexts)
    no_multi = bool(_NO_MULTI.search(text))
    multi = None if no_multi else _multi_claim(text)
    single_mentioned = bool(_EN_SINGLE.search(text) or _ZH_SINGLE.search(text))
    multi_sources = [(eid, scope) for eid, scope in facts if _reports_multiseed(scope)]
    single_sources = [(eid, scope) for eid, scope in facts
                      if scope.get("single_training_seed") and not _reports_multiseed(scope)]
    reasons = []
    if no_multi:
        for evidence_id, scope in multi_sources:
            reasons.append(f"evidence {evidence_id}: source reports {_describe_seeds(scope)}, "
                           "but the text states that no multi-seed results are reported")
    if single_mentioned and set(_seed_numbers(text)) <= {1} and multi_sources and not single_sources:
        for evidence_id, scope in multi_sources:
            reasons.append(f"evidence {evidence_id}: source reports {_describe_seeds(scope)}, "
                           "but the text claims one training seed")
    # Mixed single/multi-table evidence stays unverified instead of guessing which table a claim means.
    if multi and not single_mentioned and single_sources and not multi_sources:
        claim = f"{multi['n']} training seeds" if multi.get("n") else "multiple training seeds"
        for evidence_id, _scope in single_sources:
            reasons.append(f"evidence {evidence_id}: source reports one training seed, "
                           f"but the text claims {claim}")
    return list(dict.fromkeys(reasons))


def absence_scope_issues(text):
    """A limited reading cannot establish that the whole paper omitted a statistic."""
    for clause in re.split(r'[。；;\n]', text or ''):
        match = re.search(r'(?:未|没有)(?:报告|提供|给出|说明).{0,25}'
                          r'(?:预算|种子|显著|置信|区间|标准误|不确定性)', clause)
        if match and not re.search(r'本次|已读|选段|所读|读取|原文明确|作者明确', clause[:match.start()]):
            return ['“论文未报告”需要原文声明或独立确认；仅凭选段不足应写“本次未读取”。']
    return []


_LRC = re.compile(r"(?<![A-Za-z0-9])LRC(?![A-Za-z0-9])|latent[\s-]*rollout\s+consistency",
                  re.IGNORECASE)
_LRC_STEPS = (
    (r'同一(?:记录|条)?(?:的)?动作|记录(?:的)?动作|同一动作序列|same recorded actions?',
     '同一记录动作序列下的比较对象'),
    (r'编码|encode|观测', '观测编码得到的潜变量'),
    (r'预测|predict', '预测的潜变量'),
    (r'时域|horizon|预测步', '按预测时域'),
    (r'R\s*[²2]|决定系数|coefficient\s+of\s+determination|潜坐标|latent\s+coord',
     '每时域每潜坐标的 R²'),
    (r'方差|variance', '按目标方差加权'),
)


def lrc_method_issues(item):
    """A named latent-rollout-consistency metric needs its comparison and aggregation stated."""
    if not isinstance(item, dict) or item.get('category') != 'transferable_method':
        return []
    steps = ' '.join(str(step) for step in (item.get('method_steps') or []))
    if not _LRC.search(str(item.get('text', '')) + ' ' + steps):
        return []
    missing = [label for pattern, label in _LRC_STEPS if not re.search(pattern, steps, re.IGNORECASE)]
    if not missing:
        return []
    return ['LRC 方法步骤缺少复现所需定义：' + '、'.join(missing) +
            '；数据与种子聚合须按来源说明，未取得时写未知；高 LRC 不能单独证明忠实性。']
