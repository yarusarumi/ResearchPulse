"""Weekly roll-up of saved direction daily evidence; no new retrieval, reading or lab writes."""
import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import code_revision
from .direction_queries import ERROR_MESSAGES, DirectionChanged
from .directions import load_direction, optional, require_direction_project, resolve_direction_context
from .execution import project_lock
from .llm import FlashClient, ModelFailure
from .models import PaperRecord, now
from .run_diagnostics import describe_run
from .store import Store
from .synthesis import render_synthesis, synthesize

ZONE = ZoneInfo('Asia/Shanghai')
STRATEGY = 'direction-weekly-v2'
MAX_MODEL_CALLS = 2
ELIGIBLE_STATUSES = ('completed', 'partial')
BLOCKING_ATTEMPT_STATUSES = ('running', 'failed', 'cancelled')
COVERAGE_LABELS = {'complete': '完整', 'incomplete': '有缺口', 'unknown': '覆盖未知'}
TRIGGER_LABELS = {'manual': '手动触发', 'scheduled': '定时触发'}
UNFINISHED_STATUSES = ('running', 'starting', 'awaiting_remote_read', 'awaiting_remote_rank')
SETTINGS_KEYS = ('direction_id', 'user_id', 'title', 'description', 'questions', 'exclusions',
                 'public_queries', 'project_id')
MISSING_REASONS = {
    'no_run': '生成时未收录运行',
    'run_failed': '运行失败',
    'run_cancelled': '运行已取消',
    'run_unfinished': '运行未结束',
    'settings_mismatch': '设置版本不同，未混合',
    'no_report': '运行无可读报告',
    'no_eligible_report': '没有合格报告',
    'run_ineligible': '运行状态不合格',
}


class DirectionWeeklyClient(FlashClient):
    """Flash client that re-checks the direction settings before every model request."""

    def check_current(self):
        return None

    def complete(self, *args, **kwargs):
        self.check_current()
        return super().complete(*args, **kwargs)


def _iso_day(value):
    if not isinstance(value, str):
        raise TypeError('ISO date string required')
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError('Invalid ISO date: ' + value[:40]) from None


def _moment_value(value):
    """Parse an ISO timestamp; naive values are treated as UTC, invalid values are ignored."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _moment(run):
    parsed = _moment_value(run.get('started_at'))
    return parsed or datetime.min.replace(tzinfo=UTC)


def _run_cutoff(run):
    """The latest known moment a daily run contributed, as the stored ISO text."""
    for key in ('ended_at', 'started_at'):
        if _moment_value(run.get(key)) is not None:
            return str(run.get(key))
    return None


def _input_cutoff(runs):
    """Latest ended_at (else started_at) across included daily runs; None without runs."""
    best = None
    for run in runs:
        value = _run_cutoff(run)
        moment = _moment_value(value)
        if moment is not None and (best is None or moment > best[0]):
            best = (moment, value)
    return best[1] if best else None


def _evidence_cutoff(pairs):
    """Latest evidence timestamp (report revision, report creation, else run cutoff) among inputs."""
    best = None
    for run, report in pairs:
        for value in (report.get('revised_at'), report.get('created_at'), _run_cutoff(run)):
            moment = _moment_value(value)
            if moment is not None and (best is None or moment > best[0]):
                best = (moment, str(value))
    return best[1] if best else None


def _shanghai_text(value):
    """Render a stored ISO timestamp in the local report zone, or None when absent/invalid."""
    moment = _moment_value(value)
    if moment is None:
        return None
    return moment.astimezone(ZONE).strftime('%Y-%m-%d %H:%M') + '（Asia/Shanghai）'


def _run_reference(run):
    return {'run_id': str(run.get('run_id') or ''), 'date': str(run.get('date') or ''),
            'status': run.get('status')}


def _run_sort_key(run):
    return (str(run.get('date') or ''),
            _moment_value(run.get('started_at')) or datetime.min.replace(tzinfo=UTC),
            str(run.get('run_id') or ''))


def _freshness_run(store, run, direction, context_version):
    """Runs count as evidence even before they stored a report; only settings compatibility matters.

    A pre-context-version run whose settings version is the current one is treated conservatively
    as comparable (it may be a legacy run); runs of an older or unknown version are not guessed.
    """
    if run.get('status') not in ELIGIBLE_STATUSES:
        return False
    if _compatible_run(store, run, direction, context_version):
        return True
    if run.get('context_version'):
        return False
    return (run.get('direction_version') == direction.get('version')
            and _saved_version(store, direction.get('direction_id'), run.get('direction_version')) is not None)


def _report_snapshot(row):
    return {'report_id': str(row.get('report_id') or ''), 'run_id': str(row.get('run_id') or ''),
            'date': str(row.get('date') or ''), 'revised_at': row.get('revised_at') or row.get('created_at')}


def weekly_freshness(store, report):
    """Compare a stored weekly snapshot with the eligible evidence it should have included.

    Staleness is decided by identity, not by comparing run.started_at with the weekly
    created_at: a late-published run stored before generation still counts as missed, and a
    revision of a run that was never included is reported as a pending run, never as a
    revision of an included daily report.
    """
    period = report.get('period') or {}
    generated_at = report.get('created_at') or period.get('generated_at')
    result = {'generated_at': generated_at, 'input_cutoff': period.get('input_cutoff'),
              'is_stale': None, 'pending_runs': [], 'newer_runs': [], 'pending_revisions': []}
    generated = _moment_value(generated_at)
    user = report.get('user_id')
    did = report.get('direction_id') or period.get('direction_id')
    since, until = str(period.get('since') or ''), str(period.get('until') or '')
    if generated is None or not user or not did or not since or not until:
        return result
    if 'source_run_ids' not in period:
        return result
    try:
        direction = load_direction(store, user, did)
    except ValueError:
        return result
    context, _project_context = resolve_direction_context(store, direction)
    runs = [run for run in store.list('direction_run')
            if run.get('user_id') == user and run.get('direction_id') == did]
    eligible = [run for run in runs if _freshness_run(store, run, direction, context.version)]
    included = {str(item) for item in period.get('source_run_ids') or []}
    input_reports = {str(item) for item in period.get('input_report_ids')
                     or report.get('input_report_ids') or []}
    windowed = [run for run in eligible if since <= str(run.get('date') or '') <= until]
    latest = _latest_by_date(windowed)
    pending = [_run_reference(run) for _day, run in sorted(latest.items())
               if str(run.get('run_id')) not in included]
    pending_ids = {row['run_id'] for row in pending}
    # A run that started before generation but only ended afterwards is still missed evidence.
    for run in windowed:
        run_id = str(run.get('run_id'))
        cutoff = _moment_value(_run_cutoff(run))
        if run_id not in included and run_id not in pending_ids and cutoff is not None \
                and cutoff > generated:
            pending.append(_run_reference(run))
            pending_ids.add(run_id)
    pending.sort(key=_run_sort_key)
    newer = [_run_reference(run) for run in eligible if str(run.get('date') or '') > until]
    newer.sort(key=_run_sort_key)
    included_runs = {str(run.get('run_id')): run for run in runs
                     if str(run.get('run_id')) in included}
    revisions = {}
    for rep in store.list('direction_report'):
        run_id = str(rep.get('run_id') or '')
        if run_id not in included_runs:
            continue
        report_id = str(rep.get('report_id') or '')
        if not report_id:
            continue
        row = _report_snapshot(rep)
        if report_id in input_reports:
            revised = _moment_value(rep.get('revised_at'))
            if revised is not None and revised > generated:
                revisions[(run_id, report_id, 'revised')] = row
            continue
        current = str(included_runs[run_id].get('report_id') or '')
        revised = _moment_value(rep.get('revised_at') or rep.get('created_at'))
        if (revised is not None and revised > generated) or report_id == current:
            revisions[(run_id, report_id, 'new')] = row
    revisions = sorted(revisions.values(),
                       key=lambda row: (row['date'], str(row.get('revised_at') or ''), row['run_id']))
    result['pending_runs'], result['newer_runs'] = pending, newer
    result['pending_revisions'] = revisions
    result['is_stale'] = bool(pending or newer or revisions)
    return result


def _latest_by_date(runs):
    latest = {}
    for run in runs:
        day = str(run.get('date') or '')
        if day not in latest or (_moment(run), str(run.get('run_id'))) > (
                _moment(latest[day]), str(latest[day].get('run_id'))):
            latest[day] = run
    return latest


def _load_report(store, run, user, did, version):
    try:
        report = store.get('direction_report', run['report_id'])
    except ValueError:
        raise ValueError('Selected direction run points to a missing report; weekly report not generated') \
            from None
    expected = {'user_id': user, 'direction_id': did, 'direction_version': version, 'date': run.get('date')}
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError('Direction report is not the same owner, direction, version or date')
    if report.get('report_id') not in (None, run['report_id']):
        raise ValueError('Direction report identity does not match the selected run')
    if report.get('run_id') not in (None, run['run_id']):
        raise ValueError('Direction report belongs to another run')
    return {**report, 'report_id': run['report_id']}


def _canonical_settings(value):
    return json.dumps({key: value.get(key) for key in SETTINGS_KEYS}, sort_keys=True, ensure_ascii=False)


def _saved_version(store, did, version):
    if not version:
        return None
    return optional(store, 'direction_version', f'{did}:{version}')


def _compatible_settings(body, direction):
    """Only enabled/priority may differ; every research setting must match the immutable version."""
    if body is None:
        return False
    if _canonical_settings(body) != _canonical_settings(direction):
        return False
    return {key: body.get(key) for key in ('direction_id', 'user_id')} == {
        key: direction.get(key) for key in ('direction_id', 'user_id')}


def _compatible_run(store, run, direction, context_version):
    """A run is reusable when its saved settings differ only by enabled/priority from the current ones."""
    if run.get('context_version') != context_version:
        return False
    body = _saved_version(store, direction['direction_id'], run.get('direction_version'))
    return _compatible_settings(body, direction)


def _eligible_run(store, run, direction, context_version):
    return (run.get('status') in ELIGIBLE_STATUSES and bool(run.get('report_id'))
            and _compatible_run(store, run, direction, context_version))


def _run_references(run, report):
    revision = report.get('analysis_ids')
    if revision is not None:
        ids = [str(item) for item in revision]
    else:
        ids = [item['analysis_id'] for item in run.get('papers', [])
               if item.get('status') == 'completed' and item.get('analysis_id')]
    recommended = None
    if 'recommendations' in run:
        recommended = {str(item.get('paper_id')) for item in run.get('recommendations') or []}
    return list(dict.fromkeys(ids)), recommended


def _resolve_entry(store, run, analysis_id, recommended, did, context):
    try:
        analysis = store.get('direction_analysis', analysis_id)
    except ValueError:
        raise ValueError('A selected daily report references a missing analysis; weekly report not generated') \
            from None
    paper_id, paper_version = analysis.get('paper_id'), analysis.get('paper_version')
    if not paper_id or not paper_version:
        raise ValueError('Direction analysis lacks a paper identity; exact version cannot be resolved')
    if analysis.get('profile_version') != context.version:
        raise ValueError('Direction analysis does not match the current context version; no mixing')
    if analysis.get('direction_id') != did:
        raise ValueError('Direction analysis belongs to another direction or has no direction identity')
    if recommended is not None and paper_id not in recommended:
        return None
    if not run.get('pool_id'):
        raise ValueError('Direction run has no stored paper pool; exact paper version cannot be resolved')
    try:
        pool = store.get('pool', run['pool_id'])
    except ValueError:
        raise ValueError('Direction run paper pool is missing; exact paper version cannot be resolved') \
            from None
    match = [item for item in pool.get('papers', [])
             if item.get('paper_id') == paper_id and item.get('version') == paper_version]
    if not match:
        raise ValueError('Paper pool does not contain the analysed paper version; no version substitution')
    source_run_id = str(analysis.get('run_id') or '').split(':', 1)[0]
    source_run = optional(store, 'direction_run', source_run_id) if source_run_id else None
    return {'paper': PaperRecord.model_validate(match[0]), 'analysis': analysis, 'run_id': run['run_id'],
            'date': run['date'], 'started': _moment(run), 'reading_reused': True,
            'analysis_date': (source_run or {}).get('date')}


def _deduplicate(entries):
    chosen = {}
    for entry in entries:
        key = (entry['paper'].paper_id, entry['paper'].version)
        prior = chosen.get(key)
        if prior is None or (entry['started'], entry['analysis']['run_id']) >= (
                prior['started'], prior['analysis']['run_id']):
            chosen[key] = entry
    return [chosen[key] for key in sorted(chosen)]


def _reusable_judgments(selected, entries):
    """A one-paper window needs a coverage ledger, not another paraphrase of the same claims."""
    if len(entries) != 1:
        return None
    analysis = entries[0]['analysis']
    accepted = {analysis['run_id'] + '#claim-' + str(item['verification']['claim_index']): item['claim']
                for item in analysis.get('accepted_claims', [])}
    from .experiment_design import proposal_issues
    for run in sorted(selected, key=_moment, reverse=True):
        result = run.get('synthesis') or {}
        lookup = result.get('claim_lookup') or {}
        if not result.get('insights') or not lookup or not set(lookup) <= accepted.keys():
            continue
        if any(source.get('claim') != accepted[ref] for ref, source in lookup.items()):
            continue
        if any(not item.get('claim_refs') or not set(item['claim_refs']) <= lookup.keys()
               or proposal_issues(item) for item in result['insights']):
            continue
        return deepcopy(result), run['report_id']
    return None


def _longest_streak(days):
    best = current = 0
    previous = None
    for day in sorted(days):
        ordinal = date.fromisoformat(day).toordinal()
        current = current + 1 if previous is not None and ordinal == previous + 1 else 1
        best = max(best, current)
        previous = ordinal
    return best


def _cross_day_duplicates(referenced):
    dates = {}
    for paper_id, paper_version, day in referenced:
        dates.setdefault((paper_id, paper_version), set()).add(day)
    return [{'paper_id': paper_id, 'paper_version': paper_version, 'dates': sorted(values)}
            for (paper_id, paper_version), values in sorted(dates.items()) if len(values) > 1]


def _daily_coverage(run):
    described = describe_run(run)
    coverage = described['coverage_incomplete']
    coverage_state = 'unknown' if coverage is None else 'incomplete' if coverage else 'complete'
    issues = [{'code': item['code'], 'severity': item['severity'], 'label': item['label']}
              for item in described['issues']]
    return {'date': str(run.get('date') or ''), 'run_id': str(run.get('run_id') or ''),
            'status': run.get('status'), 'execution_state': described['execution_state'],
            'execution_label': described['execution_label'], 'coverage_state': coverage_state,
            'reading_completed_count': described['reading_completed_count'],
            'reading_target_count': described['reading_target_count'],
            'reading_skips': described['reading_skips'], 'issues': issues,
            'gap_codes': [item['code'] for item in issues if item['severity'] in ('error', 'warning')]}


def _coverage_ok(row):
    """A daily report counts as complete coverage only when it says so explicitly."""
    return (row['coverage_state'] == 'complete' and row['execution_state'] == 'ended'
            and row['status'] == 'completed' and not row['gap_codes'])


def _reading_text(row):
    completed, target = row['reading_completed_count'], row['reading_target_count']
    return f"{'未知' if completed is None else completed}/{'未知' if target is None else target}"


def _missing_dates(store, window, covered, runs, direction, context_version):
    covered = set(covered)
    rows = []
    for day in window:
        if day in covered:
            continue
        same = [run for run in runs if str(run.get('date') or '') == day]
        if not same:
            rows.append({'date': day, 'reason': 'no_run'})
            continue
        latest = max(same, key=lambda run: (_moment(run), str(run.get('run_id'))))
        status = latest.get('status')
        if status in ELIGIBLE_STATUSES:
            if not _compatible_run(store, latest, direction, context_version):
                reason = 'settings_mismatch'
            elif not latest.get('report_id'):
                reason = 'no_report'
            else:
                reason = 'no_eligible_report'
        elif status == 'failed':
            reason = 'run_failed'
        elif status == 'cancelled':
            reason = 'run_cancelled'
        elif status in UNFINISHED_STATUSES:
            reason = 'run_unfinished'
        else:
            reason = 'run_ineligible'
        row = {'date': day, 'reason': reason}
        if status:
            row['status'] = str(status)
        if latest.get('direction_version'):
            row['direction_version'] = str(latest['direction_version'])
        rows.append(row)
    return rows


def _render_report(direction, user, since, until, period, pairs, entries, synthesis, llm_requests, context):
    generated_text = _shanghai_text(period.get('generated_at')) or '未记录'
    lines = ['# 关注方向周报：' + direction['title'], '',
             f'{since} 至 {until} · {user}',
             f'生成时间：{generated_text}',
             f'数据截至：{_shanghai_text(period.get("input_cutoff")) or "无"}',
             f'证据更新至：{_shanghai_text(period.get("evidence_cutoff")) or "无"}',
             '',
             '本报告整编已保存的方向日报与正文分析，不重新检索、不重新精读。',
             '',
             '**覆盖范围**：' + period['coverage_note'],
             '']
    if period['missing_dates']:
        labels = [f"{row['date']}（{MISSING_REASONS.get(row['reason'], row['reason'])}）"
                  for row in period['missing_dates']]
        lines += ['**缺少的日期**：' + '；'.join(labels) + '。', '']
    if period['cross_day_duplicate_papers']:
        lines += [(f"**重复出现的论文**：有 {len(period['cross_day_duplicate_papers'])} 篇论文在多个日期被引用，"
                   '已按论文的最新分析去重。'), '']
    if not period['direction_enabled']:
        lines += ['**方向状态**：该方向当前已暂停，本报告只整理暂停前的日报，不会触发新的检索或阅读。', '']
    note = period.get('settings_compatibility_note')
    if note:
        lines += ['**设置变化**：' + note, '']
    lines += ['## 每日覆盖', '',
              '| 日期 | 状态 | 来源覆盖 | 阅读 | 说明 |',
              '| --- | --- | --- | --- | --- |']
    for row in period['daily_coverage']:
        issues = '；'.join(item['label'] for item in row['issues']) or '未记录问题'
        lines.append(f"| {row['date']} | {row['execution_label']} | "
                     f"{COVERAGE_LABELS[row['coverage_state']]} | {_reading_text(row)} | {issues} |")
    lines += ['', '「覆盖未知」表示旧记录没有保存来源摘要，不能当作完整检索证据；「有缺口」指来源失败或被截断。', '',
              '本报告是生成时的快照；之后新增的日报不会自动改写它。', '']
    content = chr(10).join(lines)
    if period.get('synthesis_failure'):
        content += '\n## 周报综合未完成\n\n' + period['synthesis_failure'] + '\n'
    else:
        content += render_synthesis(synthesis, context_kind='direction', profile=context)
    if period.get('compact_retry'):
        content += '\n本期综合首次输出截断，已在剩余预算内精简重试一次。\n'
    content += '\n## 输入日报\n\n'
    for run, report in pairs:
        content += (run['date'] + ' 日报\n\n<details><summary>追溯详情</summary>\n\n日报编号：`'
                    + str(report['report_id']) + '`\n\n</details>\n\n')
    if period.get('judgment_mode') == 'reused_daily_judgments':
        content += chr(10) + '本期沿用日报中已保存的判断，未新增科研推断。' + chr(10)
    mode = ('综合未完成，仅保留覆盖与输入记录' if period.get('synthesis_failure') else
            '沿用已保存日报判断' if period.get('judgment_mode') == 'reused_daily_judgments' else
            '基于已保存日报与正文分析的模型综合')
    content += ('\n触发方式：' + TRIGGER_LABELS.get(period.get('trigger'), '未记录')
                + '；汇总方式：' + mode + f"；实际覆盖：{period['covered_days']} 天。\n")
    return content


def run_direction_weekly(config, user, direction_id, until, *, since=None, progress=lambda _: None,
                         allow_model=True, trigger='manual'):
    """Compile a read-only weekly report from saved daily evidence for one user direction."""
    if trigger not in TRIGGER_LABELS:
        raise ValueError('Unknown weekly trigger')
    if not isinstance(user, str) or not user.strip() or len(user.strip()) > 100:
        raise ValueError('User name required for a direction weekly report')
    user = user.strip()
    until_date = _iso_day(until)
    if until_date > datetime.now(ZONE).date():
        raise ValueError('Direction weekly report cannot be future dated')
    since_date = _iso_day(since) if since is not None else until_date - timedelta(days=6)
    window_days = (until_date - since_date).days + 1
    if not 1 <= window_days <= 7:
        raise ValueError('Weekly window must cover 1-7 calendar days')
    since_text, until_text = since_date.isoformat(), until_date.isoformat()
    window = [(since_date + timedelta(days=offset)).isoformat() for offset in range(window_days)]
    store = Store(config.data_dir)
    direction = load_direction(store, user, direction_id)
    did = direction['direction_id']
    with project_lock(store.root, 'direction-weekly:' + did):
        current = load_direction(store, user, did)
        if current['version'] != direction['version']:
            raise DirectionChanged('Direction settings changed before weekly compilation; refresh and retry')
        return _run_locked(store, config, user, current, since_text, until_text, window,
                           progress, allow_model, trigger)


def _run_locked(store, config, user, direction, since, until, window, progress, allow_model, trigger):
    did, version = direction['direction_id'], direction['version']
    context, project_context = resolve_direction_context(store, direction)
    runs = [run for run in store.list('direction_run')
            if run.get('user_id') == user and run.get('direction_id') == did
            and since <= str(run.get('date') or '') <= until]
    eligible = [run for run in runs if _eligible_run(store, run, direction, context.version)]
    if not eligible:
        raise ValueError('No completed or partial direction daily reports in the requested period; '
                         'weekly report not generated')
    latest = _latest_by_date(eligible)
    pairs = [(latest[day], _load_report(store, latest[day], user, did,
                                        latest[day]['direction_version'])) for day in sorted(latest)]
    selected = [run for run, _report in pairs]
    entries, referenced, report_ids = [], [], []
    for run, report in pairs:
        analysis_refs, recommended = _run_references(run, report)
        for analysis_id in analysis_refs:
            entry = _resolve_entry(store, run, analysis_id, recommended, did, context)
            if entry is None:
                continue
            referenced.append((entry['paper'].paper_id, entry['paper'].version, run['date']))
            entries.append(entry)
        report_ids.append(str(report['report_id']))
    entries = _deduplicate(entries)
    if not entries:
        raise ValueError('No completed paper analyses in the eligible daily reports; '
                         'weekly report not generated')
    reusable = _reusable_judgments(selected, entries)
    if not reusable and not allow_model:
        raise ValueError('No matching saved judgments for reuse-only weekly report; no model call made')
    analysis_ids = [entry['analysis']['run_id'] for entry in entries]
    identity = {'strategy': STRATEGY, 'user_id': user, 'direction_id': did,
                'direction_version': version, 'context_version': context.version,
                'period': {'since': since, 'until': until},
                'input_report_ids': report_ids, 'analysis_ids': analysis_ids}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
    report_id = 'directionweekly-' + digest
    existing = optional(store, 'direction_weekly', report_id)
    if existing:
        return {**existing, 'reused': True}
    target = store.root / 'directions' / did / (report_id + '.md')
    if target.exists():
        raise ValueError('Weekly markdown exists without a stored direction_weekly report; '
                         'needs inspection before any model call')
    attempts = [attempt for attempt in store.list('direction_weekly_run')
                if attempt.get('report_id') == report_id]
    if any(attempt.get('status') in BLOCKING_ATTEMPT_STATUSES for attempt in attempts):
        raise ValueError('Previous direction weekly attempt needs inspection; no automatic model redispatch')
    if any(attempt.get('status') in ELIGIBLE_STATUSES for attempt in attempts):
        raise ValueError('Completed direction weekly attempt has no stored report object; needs inspection')
    covered_dates = sorted(str(run['date']) for run in selected)
    missing = _missing_dates(store, window, covered_dates, runs, direction, context.version)
    duplicates = _cross_day_duplicates(referenced)
    streak = _longest_streak(covered_dates)
    daily_coverage = [_daily_coverage(run) for run in selected]
    coverage_gaps = [row for row in daily_coverage if not _coverage_ok(row)]
    note = (f"实际覆盖 {len(covered_dates)} 个日期（请求窗口 {len(window)} 天，最长连续 {streak} 天）；"
            f"来源覆盖完整的日报 {len(daily_coverage) - len(coverage_gaps)}/{len(daily_coverage)} 天；"
            '覆盖日期数不等于连续 7 天自动运行。')
    input_versions = [{'date': str(run['date']), 'run_id': run['run_id'],
                       'report_id': str(report['report_id']),
                       'direction_version': str(run.get('direction_version'))}
                      for run, report in pairs]
    older_versions = sorted({row['direction_version'] for row in input_versions} - {version})
    settings_note = ('所有输入日报均使用当前设置版本。' if not older_versions else
                     f"输入日报中有 {len(older_versions)} 个旧设置版本（{'、'.join(older_versions)}）；"
                     f"与当前 {version} 仅启用状态或优先级不同，主题、问题与检索设置一致，"
                     '已按不可变设置版本逐一核对后整编，未混入主题或其他设置变更。')
    source_run_ids = sorted(run['run_id'] for run in selected)
    generated_at = now()
    period = {'since': since, 'until': until, 'strategy': STRATEGY, 'trigger': trigger,
              'requested_days': len(window),
              'covered_days': len(covered_dates), 'actual_dates': covered_dates,
              'missing_dates': missing, 'longest_consecutive_covered_days': streak,
              'continuous_seven_day_run': False, 'daily_coverage': daily_coverage,
              'source_coverage_complete': not coverage_gaps, 'coverage_gap_count': len(coverage_gaps),
              'coverage_note': note, 'cross_day_duplicate_papers': duplicates,
              'analysis_identities': [{'analysis_id': entry['analysis']['run_id'],
                                       'paper_id': entry['paper'].paper_id,
                                       'paper_version': entry['paper'].version} for entry in entries],
              'input_report_ids': report_ids, 'source_run_ids': source_run_ids,
              'generated_at': generated_at, 'input_cutoff': _input_cutoff(selected),
              'evidence_cutoff': _evidence_cutoff(pairs),
              'settings_version': version, 'context_version': context.version,
              'input_settings_versions': input_versions, 'settings_version_changes': older_versions,
              'settings_compatibility_note': settings_note,
              'direction_enabled': bool(direction.get('enabled'))}
    period['judgment_mode'] = 'reused_daily_judgments' if reusable else 'model_synthesis'
    period['reused_synthesis_report_id'] = reusable[1] if reusable else None
    status = 'completed' if len(covered_dates) == len(window) and not coverage_gaps else 'partial'
    attempt_id = 'directionweekly-run-' + uuid4().hex
    attempt = {'attempt_id': attempt_id, 'report_id': report_id, 'user_id': user, 'direction_id': did,
               'direction_version': version, 'context_version': context.version, 'date': until,
               'period': period, 'strategy': STRATEGY,
               'input_report_ids': report_ids, 'analysis_ids': analysis_ids, 'started_at': now(),
               'ended_at': None, 'status': 'running', 'phase': 'synthesize', 'llm_requests': 0,
               'max_llm_requests': min(MAX_MODEL_CALLS, config.llm.max_calls_per_run),
               'code_revision': code_revision(),
               'error_type': None, 'scientific_acceptance': False}
    if project_context:
        attempt['project_context'] = project_context
    client = None

    def save(**changes):
        attempt.update(changes)
        if client is not None:
            attempt['llm_requests'] = client.calls
        store.put('direction_weekly_run', attempt_id, attempt)
        progress({key: attempt.get(key) for key in ('attempt_id', 'report_id', 'status', 'phase',
                                                    'llm_requests')})

    def current():
        latest_direction = load_direction(store, user, did)
        if latest_direction['version'] != version:
            raise DirectionChanged('Direction settings changed during weekly synthesis; no report written')
        if project_context:
            require_direction_project(store, latest_direction)
            if store.get('active_profile', project_context['project_id'])['key'] != project_context['profile_key']:
                raise DirectionChanged('Linked project profile changed during weekly synthesis')

    save()
    try:
        current()
        if reusable:
            synthesis = reusable[0]
        else:
            client = DirectionWeeklyClient(config.llm.model_copy(update={
                'max_calls_per_run': min(MAX_MODEL_CALLS, config.llm.max_calls_per_run),
                'request_attempts': 1, 'schema_repair_attempts': 0}), store, attempt_id)
            client.check_current = current
            try:
                try:
                    synthesis = synthesize(client, context, entries, period, context_kind='direction')
                except ModelFailure as exc:
                    if not any(word in str(exc).lower() for word in ('truncated', 'incomplete')):
                        raise
                    if client.calls >= client.config.max_calls_per_run:
                        raise
                    period['compact_retry'] = True
                    synthesis = synthesize(client, context, entries, period,
                                           context_kind='direction', compact=True)
            except (ModelFailure, ValueError) as exc:
                status = 'failed'
                reason = ('综合输出未通过引用或结构校验，未发布科研判断' if isinstance(exc, ValueError) else
                          '精简重试仍未完成' if period.get('compact_retry') else
                          '首次输出截断，剩余调用预算不足，未进行精简重试' if
                          any(word in str(exc).lower() for word in ('truncated', 'incomplete')) else
                          '模型请求失败，未进行截断恢复')
                period['synthesis_failure'] = (reason + '。'
                    '本报告仅保留覆盖范围和输入日报，不包含可用的周报综合；原日报和单篇分析仍保留。')
                synthesis = {'insights': [], 'claim_lookup': {}, 'evidence_gaps': []}
                save(error_type=type(exc).__name__)
        current()
        calls = client.calls if client else 0
        save(phase='render')
        content = _render_report(direction, user, since, until, period, pairs, entries, synthesis,
                                 calls, context)
        if project_context:
            content += ('\n关联项目研究档案：' + project_context['project_id'] + '；来源日期截至 '
                        + (project_context['source_date'] or '未知')
                        + '。历史实验记录不代表当前状态；只合并同一档案版本的日报证据。\n')
        save(phase='publish')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            stream.write(content)
        body = {'kind': 'direction-weekly', 'report_id': report_id, 'run_id': report_id, 'user_id': user,
                'direction_id': did, 'direction_version': version, 'context_version': context.version,
                'date': until, 'period': period, 'input_report_ids': report_ids, 'analysis_ids': analysis_ids,
                'input_settings_versions': input_versions, 'covered_dates': covered_dates,
                'missing_dates': missing,
                'source_run_ids': source_run_ids, 'content': content,
                'path': str(target), 'llm_requests': calls, 'status': status, 'strategy': STRATEGY,
                'error_type': attempt['error_type'],
                'created_at': generated_at, 'ended_at': now(),
                'code_revision': attempt['code_revision'],
                'scientific_acceptance': False, 'synthesis': synthesis}
        if project_context:
            body['project_context'] = project_context
        store.put('direction_weekly', report_id, body, immutable=True)
        save(status=status, phase='finished', ended_at=now())
        return {**body, 'reused': False}
    except DirectionChanged:
        save(status='cancelled', phase='finished', ended_at=now(), error_type='DirectionChanged')
        return {**attempt, 'summary': ERROR_MESSAGES['DirectionChanged']}
    except Exception as exc:  # noqa: BLE001 - sanitized failure record, no prompts or response bodies
        save(status='failed', phase='finished', ended_at=now(), error_type=type(exc).__name__)
        return {**attempt, 'summary': ERROR_MESSAGES.get(
            type(exc).__name__, '周报处理未完成；请检查失败记录，不应据此判断没有科研进展。')}
    finally:
        if client is not None:
            client.close()
