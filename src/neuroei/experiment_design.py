"""Small structured experiment-proposal protocol; no statistics engine and no auto-run.

A proposal states what the user would pre-register before running anything. The
renderer is deterministic Markdown and never claims an experiment was executed.
"""
import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, ValidationError, field_validator

from .config import StrictModel

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_FIELDS = ("primary_metric", "improvement_direction", "minimum_effect",
           "pairing_unit", "uncertainty_method", "comparison_regime",
           "primary_control", "secondary_controls")


class ExperimentDesign(StrictModel):
    primary_metric: NonBlankStr
    improvement_direction: Literal["higher_is_better", "lower_is_better"]
    minimum_effect: NonBlankStr
    pairing_unit: NonBlankStr
    uncertainty_method: NonBlankStr
    comparison_regime: Literal["equal_reconstruction_quality", "equal_compute_budget",
                               "equal_training_updates", "equal_demonstration_count", "other"]
    primary_control: NonBlankStr | None = None
    secondary_controls: list[NonBlankStr] = Field(default_factory=list, max_length=4)

    @field_validator("*")
    @classmethod
    def _not_blank(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("field must not be blank")
        return value


_DIRECTIONS = {"higher_is_better": "越高越好", "lower_is_better": "越低越好"}
_REGIMES = {"equal_reconstruction_quality": "同等重建质量",
            "equal_compute_budget": "同等计算预算",
            "equal_training_updates": "同等训练更新次数（不等于计算成本相同）",
            "equal_demonstration_count": "同等示范数量",
            "other": "其他（需在建议中说明）"}
_NUMBER = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*")
# Only the named metric itself counts; a bare "latent rollout" method mention does not.
_LRC_METRIC = re.compile(
    r"(?<![A-Za-z0-9])LRC(?![A-Za-z0-9])|latent[\s-]*rollout\s+consistency", re.IGNORECASE)
_FAILURE = re.compile(r"失败|无效|不支持|拒绝|判据|判定|视为|认为|"
                      r"fail|invalid|reject|not supported|criterion", re.IGNORECASE)
_NEGATION = re.compile(r"不能|不得|不应|不可|不要|禁止|勿|并非|不是|不宜|不以|不把|不用|不作为|"
                       r"不意味着|不代表|不等于|未必|不能说明|不足以|"
                       r"must not|should not|cannot|can't|not use|no ", re.IGNORECASE)
_OVERLAP = (
    re.compile(r"(?:置信区间|CI|区间)\s*(?:是否)?\s*重叠|重叠\s*(?:的)?\s*(?:置信区间|CI)"),
    re.compile(r"方差(?:覆盖|范围)?(?:的)?差异"),
    re.compile(r"confidence intervals?[^.;\n]{0,20}\boverlap|"
               r"overlap[^.;\n]{0,20}confidence intervals?", re.IGNORECASE),
)
# Only comparator identities belong here. Matching constraints (equal demonstration
# count, equal budget, equal updates, equal reconstruction quality) hold a comparison
# fixed; they answer no separate question and are never a second main control.
_IDENTITY_FAMILIES = (
    ("same_generator_zero", re.compile(
        r"α\s*=\s*0|alpha\s*=\s*0|同一(?:个)?生成器|相同生成器|同生成器|"
        r"无多样性|不使用多样性|no diversity|same generator", re.IGNORECASE)),
    ("random_demos", re.compile(
        r"随机(?:采样|抽取|选取|选择|收集)?(?:的)?(?:示范|演示|轨迹|数据)|"
        r"random(?:ly)?\s+(?:collected\s+|sampled\s+)?demonstrations?", re.IGNORECASE)),
    ("existing_demos", re.compile(
        r"(?:原|原有|已有|现有|原始)(?:的)?(?:示范|演示|基线)|(?:示范|演示)基线|"
        r"existing demonstrations?|original demonstrations?", re.IGNORECASE)),
)
_FAMILY_LABELS = {"same_generator_zero": "同一生成器 α=0 条件", "random_demos": "随机示范",
                  "existing_demos": "原有／已有示范"}
_MAIN_MARKER = re.compile(
    r"主对照|主比较|主要对照|主要比较|primary\s+control|main\s+comparison|primary\s+comparison",
    re.IGNORECASE)
# A plain "compare X with Y" states the comparison too, without the word 主比较.
_COMPARE = re.compile(r"比较|对比|相比|与[^。；;\n]{0,20}(?:比较|对照)|compar(?:e|ed|es|ing|ison)",
                      re.IGNORECASE)
_SECONDARY = re.compile(r"次要对照|次级对照|辅助对照|另列|secondary\s+control", re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[。；;！!？?\n，,]")
_NEGATION_MARKER = re.compile(r"未|不|没有|没|非|无")


def _negated(raw, start):
    """True when the nearest clause negates the matched control (e.g. 未控制…, 不使用…).

    “不同随机示范”“无偏随机示范” are descriptions, not negations, so only a
    true negating predicate counts; the check stays inside the current clause.
    """
    clause_start = 0
    for match in _CLAUSE_BREAK.finditer(raw, 0, start):
        clause_start = match.end()
    clause = raw[clause_start:start]
    markers = list(_NEGATION_MARKER.finditer(clause))
    if not markers:
        return False
    tail = clause[markers[-1].start():]
    return not re.match(r"(?:未|不|没有|没|非|无)(?:同|偏|仅)", tail)


def _identities(text):
    found = {}
    raw = str(text or "")
    for name, pattern in _IDENTITY_FAMILIES:
        for match in pattern.finditer(raw):
            # "未控制等计算预算" / "不使用随机示范" are limitations, not declared controls.
            if _negated(raw, match.start()):
                continue
            found[name] = match.group(0)
            break
    return found


def _main_claims(text, ignore=()):
    """Identity families presented as the main comparison; negation-aware.

    A “主对照/主比较” label is the strongest signal, but a plain
    “比较 X 与 Y” sentence also says which comparison the proposal makes; the
    original misreport (“比较定向 RL 生成示范与随机示范” with
    primary_control=已有示范) carries no label and must still be caught.
    Clauses that declare a secondary control are skipped so an already declared
    secondary mention is never re-read as the main comparison. ``ignore`` only
    silences plain secondary mentions: an explicitly labelled 主比较/主对照
    sentence must still agree with the structured primary_control.
    """
    found = {}
    for sentence in re.split(r"[。；;!?！？\n]", str(text or "")):
        labelled = bool(_MAIN_MARKER.search(sentence))
        if not (labelled or _COMPARE.search(sentence)):
            continue
        for clause in re.split(r"[，,]", sentence):
            if _SECONDARY.search(clause):
                continue
            for name, pattern in _IDENTITY_FAMILIES:
                if name in found or (name in ignore and not labelled):
                    continue
                for match in pattern.finditer(clause):
                    if _negated(clause, match.start()):
                        continue
                    found[name] = match.group(0)
                    break
    return found


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER.fullmatch(value)
        if match:
            return float(match.group(1))
    return None


def _interval(design):
    raw = design.get("difference_interval")
    if isinstance(raw, dict):
        raw = [raw.get("low", raw.get("lower")), raw.get("high", raw.get("upper"))]
    if isinstance(raw, str):
        raw = raw.strip().strip("[]()").split(",")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    low, high = _number(raw[0]), _number(raw[1])
    if low is None or high is None:
        return None
    return min(low, high), max(low, high)


def _known(design):
    return {name: design[name] for name in _FIELDS if name in design}


def _validation_detail(error):
    parts = []
    for item in error.errors()[:4]:
        location = ".".join(str(part) for part in item["loc"]) or "design"
        parts.append(f"{location}: {item['msg']}")
    return "；".join(parts)


def _dangerous_sentences(text):
    findings = []
    for sentence in re.split(r"(?<=[。；;!?])\s*|\n+", text or ""):
        sentence = sentence.strip()
        if not sentence or not _FAILURE.search(sentence):
            continue
        for pattern in _OVERLAP:
            for match in pattern.finditer(sentence):
                before = sentence[max(0, match.start() - 25):match.start()]
                after = sentence[match.end():match.end() + 25]
                if _NEGATION.search(before) or _NEGATION.search(after):
                    continue
                findings.append(sentence)
    return list(dict.fromkeys(findings))


def _control_issues(raw, item, scope):
    """One pre-registered primary control; matching constraints are not controls."""
    issues = []
    text = str(item.get("text", ""))
    primary = raw.get("primary_control")
    secondary = raw.get("secondary_controls") or []
    if not isinstance(secondary, list):
        secondary = []
    secondary_text = " ".join(str(value) for value in secondary)
    if not isinstance(primary, str) or not primary.strip():
        issues.append(
            "缺少 primary_control：实验建议必须写明唯一主对照，不能用正文里的对照词表代替；"
            "主对照可以是同一生成器的 α=0 条件、同等数量的已有示范，也可以是冻结编码器等本文自定义基线。")
        return issues
    primary_families = _identities(primary)
    if len(primary_families) > 1:
        issues.append("primary_control 同时描述了多种对照口径（" +
                      "、".join(_FAMILY_LABELS[name] for name in primary_families) +
                      "），请只保留一种：" + primary)
        return issues
    primary_name = next(iter(primary_families), None)
    secondary_families = _identities(secondary_text)
    if primary_name is not None and primary_name in secondary_families:
        issues.append("secondary_controls 重复了主对照口径；其他对照必须检验不同问题。")
    claims = _main_claims(text + " " + str(scope.get("budget_control", "")),
                          ignore=set(secondary_families))
    if primary_name is not None and len(claims) == 1:
        claim_name = next(iter(claims))
        if claim_name != primary_name:
            issues.append("正文把「" + claims[claim_name] + "」写成主对照，与结构化 primary_control（" +
                          primary + "）不一致；请统一主比较，或把另一对照写入 secondary_controls。")
    return issues


def proposal_issues(item):
    """Return deterministic gaps for one insight dict; empty means no known issue."""
    if not isinstance(item, dict) or item.get("category") != "experiment_proposal":
        return []
    issues = []
    scope = item.get("experiment_scope") or {}
    raw = item.get("experiment_design")
    if not raw:
        issues.append("实验建议缺少待预注册的实验设计（primary_metric、improvement_direction、"
                      "minimum_effect、pairing_unit、uncertainty_method、comparison_regime、"
                      "primary_control）。")
    else:
        valid = True
        try:
            ExperimentDesign.model_validate(_known(raw) if isinstance(raw, dict) else raw)
        except ValidationError as error:
            valid = False
            issues.append("实验设计无效（" + _validation_detail(error) + "）；"
                          "多项独立预算比较应拆成独立实验建议，comparison_regime 不能合并为一项。")
        if valid and isinstance(raw, dict):
            metric = str(raw.get("primary_metric", ""))
            # Narrow known-metric guard only; arbitrary metrics keep their own definitions.
            lrc = _LRC_METRIC.search(metric)
            explicit_loss = re.search(r"1\s*[-−]\s*(?:LRC|latent[\s-]*rollout\s+consistency)", metric,
                                      re.IGNORECASE)
            if lrc and not explicit_loss and raw.get("improvement_direction") != "higher_is_better":
                issues.append("LRC（latent rollout consistency）是越高越好的一致性分数；"
                              "不得改称越低越好的误差。若定义1-LRC损失，须明确写出变换并另核对其口径。")
            issues += _control_issues(raw, item, scope)
            pairing = str(raw.get('pairing_unit', ''))
            uncertainty = str(raw.get('uncertainty_method', ''))
            seed_repeats = re.search(r'训练种子|训练运行|training seeds?', pairing, re.IGNORECASE)
            episodes = re.search(r'episode|trial|回合', uncertainty, re.IGNORECASE)
            clustered = re.search(r'分层|hierarchical|cluster|按(?:配对)?(?:训练)?种子[^，。；;]{0,12}(?:抽样|重采样|bootstrap)|'
                                  r'(?:抽样|重采样)单位(?:是|为|：)?(?:配对)?(?:训练)?种子', uncertainty, re.IGNORECASE)
            fixed = re.search(r'固定模型|固定checkpoint|fixed model', uncertainty, re.IGNORECASE)
            limited = re.search(r'不外推|不涵盖|不包含训练随机性|conditional on', uncertainty, re.IGNORECASE)
            if seed_repeats and episodes and not clustered and not (fixed and limited):
                issues.append('训练重复与评估episode的抽样层级未区分：须按种子抽样或分层抽样；'
                              '固定模型的episode区间须明确不外推到训练随机性。')
    for sentence in _dangerous_sentences(item.get("text", "") + '\n' + item.get('conditions_and_limits', '')):
        issues.append("文本把“两组置信区间重叠/方差覆盖差异”当作失败或无效判据：「" + sentence +
                      "」；应改为预注册最小实际差异与差值区间，不能用区间重叠判断。")
    if re.search(r'判读规则|判定规则|判读标准', item.get('text', '')):
        issues.append('实验建议自由文本不得另写判读规则；只写问题、对照和动机，判定由结构化设计统一生成。')
    controls = item.get('text', '') + ' ' + str(scope.get('budget_control', ''))
    if (isinstance(raw, dict) and raw.get('comparison_regime') == 'equal_compute_budget'
            and re.search(r'updates|训练步|更新次', controls, re.IGNORECASE)
            and not re.search(r'FLOPs|浮点运算|固定硬件|墙钟|wall.?time', controls, re.IGNORECASE)):
        issues.append('相同训练更新次数不能推出等计算预算；使用equal_training_updates，或明确定义计算成本口径。')
    return issues


def render_design(design):
    """Deterministic Chinese Markdown for a would-be pre-registration."""
    data = design.model_dump() if isinstance(design, ExperimentDesign) else dict(design or {})
    try:
        model = ExperimentDesign.model_validate(_known(data))
    except ValidationError as error:
        raise ValueError("实验设计无效：" + _validation_detail(error)) from error
    interval = _interval(data)
    minimum = _number(model.minimum_effect)
    lines = [
        "## 待预注册实验设计（建议，未执行）",
        "",
        f"- 主要指标：{model.primary_metric}",
        f"- 改善方向：{_DIRECTIONS[model.improvement_direction]}（差值统一为改善为正）",
        f"- 最小实际差异：{model.minimum_effect}（由使用者预注册，不是论文给出的阈值）",
        f"- 配对单位：{model.pairing_unit}",
        f"- 不确定性方法：{model.uncertainty_method}",
        f"- 比较口径：{_REGIMES[model.comparison_regime]}",
    ]
    if model.primary_control:
        lines.append(f"- 主对照：{model.primary_control}")
        if model.secondary_controls:
            lines.append("- 其他对照（不并入主比较）：" + "；".join(model.secondary_controls))
        lines.append("- 对照解释：主对照只回答一个问题；其他对照须分别报告，不能合并成一个显著性判断。")
    lines += [
        ("- 预设判定规则：改善量为正；差值区间下界高于预注册最小差异，支持改善；"
        "区间上界低于该阈值，不支持达到该实际收益（不等于效应为零）；"
        "其余为证据不足。区间上界低于 0 时另记可能退化。"),
    ]
    if interval is None:
        lines.append("- 差值区间：未提供（本建议尚未执行，不产生观测区间）")
        verdict = "证据不足：尚无差值区间，不能判定方法有效或无效。"
    else:
        sign = 1.0 if model.improvement_direction == "higher_is_better" else -1.0
        low, high = sorted((sign * interval[0], sign * interval[1]))
        lines.append(f"- 差值区间：原始 [{interval[0]:g}, {interval[1]:g}]；"
                     f"统一改善为正后 [{low:g}, {high:g}]")
        if high < 0:
            verdict = ("证据提示可能退化：统一改善为正后区间上界低于 0；"
                       "这不等于方法无效，需按预注册复核。")
        elif minimum is None:
            verdict = "证据不足：最小实际差异尚未预注册，不能判定是否达到实际收益。"
        elif low > minimum:
            verdict = f"支持改善：区间下界 {low:g} 超过最小实际差异 {minimum:g}。"
        elif high < minimum:
            verdict = f"不支持达到该实际收益：区间上界 {high:g} 低于最小实际差异 {minimum:g}。"
        else:
            verdict = (f"证据不足：区间 [{low:g}, {high:g}] 跨越最小实际差异 {minimum:g}，"
                       "不能判定方法有效或无效。")
    lines += [
        f"- 判定：{verdict}",
        "",
        ("> 本内容为待预注册建议：未执行实验，也未做统计功效验证；"
        "不能用两组置信区间是否重叠作为失败或成功判据。"),
    ]
    return "\n".join(lines)
