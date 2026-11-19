"""Read-only interpretation of stored ResearchPulse run records.

``describe_run`` and ``analysis_assurance`` are pure functions: they take an
already-loaded record dict, never touch the store, clock, network or model, and
never rewrite history. Missing fields stay ``None``/``unknown`` instead of being
guessed, and a stored ``status`` of ``completed`` is never upgraded into
full-text, table or supplementary-material human verification.

Expected run keys (all optional; older records may lack them): ``status``,
``read_limit``, ``recommendations``, ``reading_plan``, ``papers``, ``sources``
(or ``source_summary``) and ``synthesis``. Newer records may attach a precise
``reading_skip_reason`` per paper (or a ``reading_skip_reasons`` mapping from
paper id to reason) such as ``model_budget`` or ``abstract_only``. Without that
field the cause is reported as ``not_recorded``; it is never assumed to be a
model budget or full-text problem. A plan that reads fewer papers than were
recommended is a configured reading limit, not a failed run.
"""
_EXECUTION = {
    'running': ('running', '运行中'),
    'starting': ('running', '运行中'),
    'awaiting_remote_read': ('awaiting', '等待远端读取结果'),
    'awaiting_remote_rank': ('awaiting', '等待远端排序结果'),
    'completed': ('ended', '执行已结束'),
    'partial': ('ended', '执行已结束'),
    'failed': ('failed', '执行失败'),
    'cancelled': ('cancelled', '已取消'),
    'requires_attention': ('requires_attention', '需要人工检查'),
}
_UNKNOWN_EXECUTION = ('unknown', '状态未知')

_REASON_ALIASES = {
    'budget': 'model_budget',
    'llm_budget': 'model_budget',
    'model_call_budget': 'model_budget',
    'max_llm_requests': 'model_budget',
    'no_budget': 'model_budget',
    'read_limit_reached': 'read_limit',
    'reading_limit': 'read_limit',
    'planned_limit': 'read_limit',
    'no_fulltext': 'fulltext_unavailable',
    'fulltext_missing': 'fulltext_unavailable',
    'abstract': 'abstract_only',
}


def _int_or_none(value):
    return value if type(value) is int and value >= 0 else None


def _execution(run):
    return _EXECUTION.get(str(run.get('status') or '').strip().lower(), _UNKNOWN_EXECUTION)


def _recommendation_count(run):
    recommendations = run.get('recommendations')
    if isinstance(recommendations, list):
        return len(recommendations)
    return _int_or_none(run.get('recommended_count'))


def _paper_list(run):
    papers = run.get('papers')
    if not isinstance(papers, list):
        return None
    return [item for item in papers if isinstance(item, dict)]


def _sources(run):
    """Return usable source summaries, or ``None`` when coverage is unknown."""
    for key in ('sources', 'source_summary'):
        value = run.get(key)
        if isinstance(value, list):
            cleaned = [item for item in value if isinstance(item, dict)]
            if cleaned:
                return cleaned
    return None


def _planned_count(run):
    plan = run.get('reading_plan')
    if not isinstance(plan, dict):
        return None
    items = plan.get('items')
    return len(items) if isinstance(items, list) else None


def _reason(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    token = raw.strip().lower().replace('-', '_').replace(' ', '_')
    return _REASON_ALIASES.get(token, token)


def _skip_reason_for(run, paper):
    explicit = paper.get('reading_skip_reason') or paper.get('skip_reason')
    if explicit:
        return _reason(explicit)
    mapping = run.get('reading_skip_reasons')
    if isinstance(mapping, dict):
        return _reason(mapping.get(paper.get('paper_id')))
    return None


def _reading_summary(run, papers, target):
    completed = None
    skips = {}
    incomplete = 0
    if papers is None:
        return _int_or_none(run.get('reading_completed_count')), skips, incomplete
    completed = sum(1 for paper in papers if paper.get('status') == 'completed')
    for paper in papers:
        status = paper.get('status')
        if status == 'completed':
            continue
        reason = _skip_reason_for(run, paper)
        if status == 'unavailable':
            incomplete += 1
        elif status not in ('skipped', 'abstract_only', 'partial') and reason is None:
            continue
        reason = reason or 'not_recorded'
        skips[reason] = skips.get(reason, 0) + 1
    planned = _planned_count(run)
    if planned is not None and target is not None and target > planned:
        skips['read_limit'] = max(skips.get('read_limit', 0), target - planned)
    return completed, skips, incomplete


def _synthesis_failed(value):
    if not isinstance(value, dict):
        return True
    if value.get('status') in ('failed', 'unavailable') or value.get('error'):
        return True
    gaps = value.get('evidence_gaps')
    return isinstance(gaps, list) and any(
        isinstance(gap, str) and gap.startswith('科研综合未完成') for gap in gaps)


def describe_run(run: dict) -> dict:
    """Summarize one stored run without inferring causes it did not record.

    ``execution_state`` is a stable English enum (``running``, ``awaiting``,
    ``ended``, ``failed``, ``cancelled``, ``requires_attention`` or ``unknown``);
    ``execution_label`` is its Chinese label. ``coverage_incomplete`` is ``None``
    when no usable source summary was stored, so the UI can say unknown instead
    of displaying a gap it never saw. ``reading_skips`` counts non-completed
    papers by recorded reason; ``not_recorded`` marks legacy records, and
    ``issues`` only carries codes that the stored fields actually support.
    """
    run = run if isinstance(run, dict) else {}
    state, label = _execution(run)
    target = _recommendation_count(run)
    papers = _paper_list(run)
    completed, skips, incomplete = _reading_summary(run, papers, target)

    issues = []
    coverage = None
    sources = _sources(run)
    if sources is not None:
        failed = [s for s in sources if s.get('status') == 'failed']
        truncated = [s for s in sources if s.get('status') != 'failed'
                     and (s.get('status') == 'partial' or s.get('truncated') is True)]
        coverage = bool(failed or truncated)
        if failed:
            issues.append({'code': 'source_failed', 'severity': 'error',
                           'label': f'{len(failed)} 个来源检索失败；失败来源的未覆盖部分不能据此判断没有相关研究'})
        if truncated:
            issues.append({'code': 'source_truncated', 'severity': 'warning',
                           'label': f'{len(truncated)} 个来源结果被截断（partial）；仅覆盖已返回的部分'})

    if run.get('synthesis_unavailable') or ('synthesis' in run and _synthesis_failed(run.get('synthesis'))):
        issues.append({'code': 'synthesis_unavailable', 'severity': 'error',
                       'label': '科研综合未完成或缺失；不能把未完成的综合当作没有证据'})

    planned = _planned_count(run)
    if skips.get('not_recorded'):
        issues.append({'code': 'reading_skip_reason_not_recorded', 'severity': 'info',
                       'label': f"有 {skips['not_recorded']} 篇未完成阅读且旧记录未标原因；"
                                '不能据此推断为模型预算不足、全文不可得或读取失败'})
    if incomplete:
        issues.append({'code': 'reading_unavailable', 'severity': 'warning',
                       'label': f'{incomplete} 篇论文未完成阅读；原因见 reading_skips，未逐项核对全文不可得性'})
    if skips.get('model_budget'):
        issues.append({'code': 'model_budget_insufficient', 'severity': 'warning',
                       'label': f"{skips['model_budget']} 篇因记录明确的模型预算不足未阅读；不代表论文不重要"})
    if planned is not None and target is not None and target > planned:
        issues.append({'code': 'read_limit_planned', 'severity': 'info',
                       'label': f"阅读上限预设为 {run.get('read_limit')} 篇：{target} 篇推荐中计划仅读 {planned} 篇；"
                                '这是配置而非执行失败'})
    elif skips.get('read_limit'):
        issues.append({'code': 'read_limit_planned', 'severity': 'info',
                       'label': f"{skips['read_limit']} 篇按预设阅读上限仅保留摘要；这是配置而非执行失败"})
    if papers is None and target:
        issues.append({'code': 'reading_status_not_recorded', 'severity': 'info',
                       'label': '旧记录缺少逐篇阅读状态；阅读完成数与跳过原因未知，不能据此判断执行失败或预算不足'})

    return {
        'execution_state': state,
        'execution_label': label,
        'reading_completed_count': completed,
        'reading_target_count': target,
        'coverage_incomplete': coverage,
        'reading_skips': skips,
        'issues': issues,
    }


def analysis_assurance(analysis: dict | None) -> dict:
    """Describe how far one stored reading analysis actually got.

    ``level`` is ``located_excerpt_auto_checks`` when the record carries real
    block/claim counts and ``unknown`` otherwise; a ``status`` of ``completed``
    on its own never raises it. Numeric handling is reported as
    ``literal_presence_only`` and independent numeric review as
    ``not_recorded``; unknown fields are ``None``.
    """
    if not isinstance(analysis, dict):
        return {
            'selected_blocks': None,
            'available_blocks': None,
            'accepted_count': None,
            'rejected_count': None,
            'degraded': False,
            'level': 'unknown',
            'numeric_audit': 'literal_presence_only',
            'independent_numeric_review': 'not_recorded',
        }
    accepted = analysis.get('accepted_claims')
    accepted_count = len(accepted) if isinstance(accepted, list) else None
    rejected_count = _int_or_none(analysis.get('rejected_count'))
    if rejected_count is None:
        rejected_claims = analysis.get('rejected_claims')
        rejected_count = len(rejected_claims) if isinstance(rejected_claims, list) else None
    selected = _int_or_none(analysis.get('selected_blocks'))
    available = _int_or_none(analysis.get('available_blocks'))
    located = any(value is not None for value in (selected, available, accepted_count, rejected_count)) \
        or bool(analysis.get('evidence'))
    return {
        'selected_blocks': selected,
        'available_blocks': available,
        'accepted_count': accepted_count,
        'rejected_count': rejected_count,
        'degraded': bool(analysis.get('degraded_output')),
        'source_integrity': analysis.get('source_integrity') or {},
        'level': 'located_excerpt_auto_checks' if located else 'unknown',
        'numeric_audit': 'literal_presence_only',
        'independent_numeric_review': 'not_recorded',
    }
