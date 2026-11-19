"""Persist manual requests; one explicitly configured worker processes each job once."""
import json
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from zoneinfo import ZoneInfo

from . import accounts
from .models import now
from .recommendation_events import authorize, has_resolvable_versions
from .recommendation_pipeline import saved_direction_configuration


def understanding_model_config(max_calls=1):
    from .config import LLMConfig

    return LLMConfig(max_calls_per_run=max_calls, request_attempts=1, schema_repair_attempts=0,
                     max_tokens=131072, timeout_seconds=1200)


def understanding_profile():
    from .project_understanding import POLICY
    from .project_understanding_spans import SPAN_PROVIDER

    from . import ai_backend

    config = ai_backend.model_config(understanding_model_config(), ai_backend.selected())
    return {**{key: getattr(config, key) for key in ('model', 'reasoning_effort', 'max_tokens', 'timeout_seconds')},
            'provider_protocol': SPAN_PROVIDER, 'extraction_policy': POLICY}


def worker_status(store):
    try:
        row = store.get('recommendation_worker', 'current')
    except ValueError:
        return {'status': 'unconfigured', 'paid_model_calls_enabled': False}
    age = (datetime.now(UTC) - datetime.fromisoformat(row['updated_at'])).total_seconds()
    if row['status'] == 'ready' and not 0 <= age <= 20:
        return {**row, 'status': 'offline'}
    return row


def understanding_budget_details(store, job):
    if not job.get('event_id'):
        return {}
    try:
        event = store.get('recommendation_event', job['event_id'])
    except ValueError:
        return {}
    if event['user_id'] != job['user_id'] or event['project_id'] != job['snapshot']['project_id']:
        return {}
    return {key: event['input'][key] for key in ('required_calls', 'allowed_calls') if key in event['input']}


def retry_call_limit(store, job, worker):
    if job['reason'] == 'model_budget_insufficient' and not job.get('understanding_run_id'):
        required = understanding_budget_details(store, job).get('required_calls')
        budget = worker.get('understanding_call_limit')
        if required is not None and budget is not None:
            if budget < required:
                raise ValueError('这批资料一次处理不完。请选择较少的日报后重新分析。')
            return budget
    return None


def legacy_feishu_daily(snapshot):
    from .feishu_folders import DAILY_SCOPE_POLICY

    return (snapshot.get('feishu') and snapshot.get('analysis_mode') == 'daily'
            and snapshot.get('feishu_daily_policy') != DAILY_SCOPE_POLICY)


def member_job(job):
    private = {'understanding_model_config', 'understanding_policy', 'retry_call_limit', 'feishu_daily_policy',
               'feishu_grant_id', 'scheduled_project_daily'}
    internal = {'checkpoint', 'understanding_run_id', 'understanding_phase', 'automatic_followup'}
    return {**{key: value for key, value in job.items() if key not in internal},
            'snapshot': {key: value for key, value in job['snapshot'].items() if key not in private}}


def member_jobs(store, jobs, worker):
    def can_retry(job):
        if (job['action'] != 'project_understanding' or not job.get('understanding_phase')
                or job['status'] not in ('failed', 'partial') or job['snapshot'].get('image_analysis_pending')
                or (job['status'] == 'partial' and job['snapshot'].get('batch_gaps'))
                or job['reason'] == 'feishu_documents_unavailable'
                or job['snapshot'].get('daily_scope_unverified') or legacy_feishu_daily(job['snapshot'])):
            return False
        if job.get('understanding_run_id'):
            if job['snapshot'].get('understanding_model_config') != understanding_profile():
                return False
        elif job['reason'] not in ('model_budget_insufficient', 'model_not_configured'):
            return False
        try:
            budget = retry_call_limit(store, job, worker)
        except ValueError:
            return False
        retries = [other for other in jobs if other['snapshot'].get('retry_job_id') == job['job_id']]
        return not any(budget is None or other['snapshot'].get('retry_call_limit') == budget
                       or other['status'] in ('queued', 'processing', 'completed') for other in retries)

    return [{**member_job(job), 'retry_available': can_retry(job)}
            if job['action'] == 'project_understanding' else member_job(job) for job in jobs]


def enqueue(store, account, payload, *, as_of=None, resume_feishu_job=None, scheduled_project_daily=False):
    created_at = as_of or now()
    clock = datetime.fromisoformat(created_at)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError('Timezone-aware scheduling time required')
    action = payload.get('action')
    if action == 'project_understanding_retry':
        if set(payload) != {'action', 'job_id'}:
            raise ValueError('请选择需要恢复的分析任务。')
        previous = store.get('recommendation_job', payload['job_id'])
        if previous['user_id'] != account['user_id']:
            raise PermissionError('Job owner mismatch')
        if previous['action'] != 'project_understanding' or previous['status'] not in ('failed', 'partial'):
            raise ValueError('只有首次失败的分析可以显式重试一次。')
        if previous['snapshot'].get('image_analysis_pending'):
            raise ValueError('请重新发起项目分析，继续处理其余图片。')
        if previous['status'] == 'partial' and previous['snapshot'].get('batch_gaps'):
            raise ValueError('已保存通过校验的资料。请用“更新项目背景”补做剩余日报。')
        if previous['reason'] == 'feishu_documents_unavailable':
            raise ValueError('请再次发起项目分析，重新读取未能获取的日报。')
        if legacy_feishu_daily(previous['snapshot']):
            raise ValueError('日报分析范围已更新，请重新点击“分析昨日日报”。')
        if previous['snapshot'].get('daily_scope_unverified'):
            raise ValueError('请先确认日报的日期标题，或用“更新项目背景”建立资料基线。')
        from .research_workspace import project_for

        project_for(store, account, previous['snapshot']['project_id'])
        run_id = previous.get('understanding_run_id')
        if run_id and previous['snapshot'].get('understanding_model_config') != understanding_profile():
            raise ValueError('分析方式已更新，请展开“项目资料与背景”重新发起分析。')
        if not run_id and previous['reason'] not in ('model_budget_insufficient', 'model_not_configured'):
            raise ValueError('请联系管理员核对这次分析的恢复条件。')
        snapshot = {**previous['snapshot'], 'retry_run_id': run_id, 'retry_job_id': previous['job_id'],
                    'retry_phase': previous['understanding_phase'], 'analysis_as_of': previous['created_at']}
        identity = [action, account['user_id'], previous['job_id']]
        if (budget := retry_call_limit(store, previous, worker_status(store))) is not None:
            snapshot['retry_call_limit'] = budget
            identity.append(budget)
    elif action == 'project_understanding':
        allowed_fields = {'action', 'project_id', 'analysis_mode', 'manual_selection', 'auto_dispatch', 'auto_email'}
        if not {'action', 'project_id'} <= set(payload) <= allowed_fields:
            raise ValueError('请选择需要分析的项目。')
        if any(key in payload and not isinstance(payload[key], bool) for key in ('auto_dispatch', 'auto_email')):
            raise ValueError('自动流程参数无效。')
        analysis_mode = payload.get('analysis_mode', 'imported')
        if analysis_mode not in ('imported', 'daily', 'manual'):
            raise ValueError('请选择导入分析、昨日检查或手动选择。')
        from .research_workspace import analysis_snapshot

        snapshot = analysis_snapshot(store, account, payload['project_id'])
        snapshot['analysis_mode'] = analysis_mode
        snapshot['auto_dispatch'] = bool(payload.get('auto_dispatch', False))
        snapshot['auto_email'] = bool(payload.get('auto_email', False))
        day = clock.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
        from .project_understanding import POLICY
        from .project_understanding import identity as understanding_identity

        if analysis_mode in ('daily', 'manual') and not any(
                key == understanding_identity([account['user_id'], snapshot['project_id']])
                for key, _ in store.items('understanding_head')):
            raise ValueError('请先点击“生成项目背景”完成首次分析。')

        snapshot['understanding_policy'] = POLICY
        snapshot['understanding_model_config'] = understanding_profile()
        identity = [action, account['user_id'], snapshot['project_id'], snapshot['journal_versions'],
                    day, analysis_mode, POLICY, snapshot['understanding_model_config'], snapshot['auto_dispatch']]
        from . import ai_backend

        if analysis_mode == 'imported':
            from .project_brief import POLICY as BRIEF_POLICY
            identity.append(BRIEF_POLICY)

        if analysis_mode == 'daily' and ai_backend.selected() == 'deepseek':
            from .daily_scope_backend import selected, uses_personal_selection

            if uses_personal_selection():
                snapshot['daily_scope_backend'] = selected(store, account['user_id'])
                if snapshot['daily_scope_backend'] != 'deepseek':
                    identity.append(snapshot['daily_scope_backend'])
        if analysis_mode == 'imported' and snapshot.get('background_edited'):
            background = snapshot['background']
            identity.extend([background['blocks'][0]['revision'], background['available_at']] if background else [None])
        if snapshot.get('feishu'):
            from .feishu_folders import (
                DAILY_SCOPE_POLICY,
                MANUAL_SELECTION_POLICY,
                MAX_MANUAL_CHARS,
                MAX_MANUAL_LINES,
                FolderUnavailable,
                freeze_folder,
                manual_preview,
                manual_row,
            )

            try:
                if analysis_mode == 'manual':
                    selection = payload.get('manual_selection')
                    if not isinstance(selection, dict) or set(selection) != {'source_id', 'revision', 'line_start', 'line_end'}:
                        raise ValueError('请选择要分析的日报文件与起止行。')
                    start, end = int(selection['line_start']), int(selection['line_end'])
                    if start < 1 or end < start or end - start + 1 > MAX_MANUAL_LINES:
                        raise ValueError('每次最多选择 400 行日报内容。')
                    try:
                        preview = manual_preview(store, account, snapshot['feishu'], selection['source_id'],
                                                 offset=start - 1, limit=end - start + 1)
                    except FolderUnavailable as exc:
                        raise ValueError(str(exc))
                    if end > preview['line_count']:
                        raise ValueError('所选行超出日报末尾，请重新预览后选择。')
                    if preview['revision'] != selection['revision']:
                        raise ValueError('文件内容已更新，请重新预览后选择行号。')
                    chars = sum(len(line['text']) + 1 for line in preview['lines'])
                    if preview['truncated'] or chars > MAX_MANUAL_CHARS:
                        raise ValueError('所选内容超过 2 万字符，请缩小行范围。')
                    image_blocks = [image['block_id'] for image in preview.get('images', [])
                                    if start <= image['after_line'] <= end]
                    snapshot['manual_selection'] = {'source_id': selection['source_id'],
                                                    'revision': preview['revision'],
                                                    'line_start': start, 'line_end': end,
                                                    'name': preview['name'],
                                                    'image_block_ids': image_blocks}
                    try:
                        snapshot['feishu_manifest'] = [manual_row(store, account, snapshot['feishu'],
                                                                  selection['source_id'])]
                    except FolderUnavailable as exc:
                        raise ValueError(str(exc))
                    snapshot['feishu_daily_policy'] = MANUAL_SELECTION_POLICY
                    identity.append(['manual', snapshot['manual_selection']])
                else:
                    manifest = freeze_folder(store, account, snapshot['feishu'], created_at, analysis_mode)
                    snapshot['feishu_manifest'] = manifest
                    identity.extend([snapshot['feishu'], snapshot['feishu_grant_id'], manifest])
                    if analysis_mode == 'daily':
                        snapshot['feishu_daily_policy'] = DAILY_SCOPE_POLICY
                        identity.append(DAILY_SCOPE_POLICY)
            except FolderUnavailable:
                manifest = None
                snapshot['feishu_manifest'] = manifest
    elif action == 'progress':
        if not {'action', 'request_id'} <= set(payload) <= {'action', 'request_id', 'question_ids', 'auto_email'}:
            raise ValueError('Explicit request required')
        if 'auto_email' in payload and not isinstance(payload['auto_email'], bool):
            raise ValueError('自动邮件参数无效。')
        delivery = store.get('recommendation_delivery', payload['request_id'])
        allowed = accounts.projects_for(store, account)
        authorize(delivery['request'], account['user_id'],
                  [delivery['project_id']] if allowed is None else allowed, ())
        snapshot = delivery
        from .recommendation_handoff import initial_review_questions, select_review_questions

        suggested = initial_review_questions(store, delivery)
        if 'question_ids' in payload:
            chosen = payload['question_ids']
        elif suggested:
            chosen = suggested
        else:
            chosen = None
        if chosen is not None:
            request = select_review_questions(delivery['request'], chosen)
            snapshot = {**delivery, 'request': request, 'review_question_ids': [q['id'] for q in request['research_questions']]}
        if payload.get('auto_email'):
            snapshot = {**snapshot, 'auto_email': True}
        identity = ['progress', account['user_id'], payload['request_id'], bool(payload.get('auto_email', False))]
    elif action in ('reading_recovery', 'discovery_recovery'):
        if set(payload) != {'action', 'event_id'}:
            raise ValueError('请选择需要补读的推送。')
        event = store.get('recommendation_event', payload['event_id'])
        allowed = accounts.projects_for(store, account)
        authorize(event['input'], account['user_id'],
                  [event['project_id']] if allowed is None else allowed,
                  [event['direction_id']] if event['direction_id'] else [])
        if event['direction_id']:
            saved_direction_configuration(store, account, event['direction_id'])
        if event['status'] not in ('partial', 'failed') or not event.get('discovery'):
            raise ValueError('本次推送当前不需要补读。')
        if action == 'reading_recovery':
            selected = {r['paper_id'] for r in event['discovery']['recommendations']}
            worker = worker_status(store)
            if not any(p['paper_id'] in selected and p['paper_version'] in p['versions']
                       for p in event['discovery']['candidates']) and not (
                           has_resolvable_versions(event, open_access=worker.get('open_access_fulltexts_enabled', False))
                           and worker.get('fulltext_download_enabled')):
                raise ValueError('论文版本尚未核实，暂不能补读；请先核对原始论文资料。')
        attempt_key = 'recovery_attempts' if action == 'reading_recovery' else 'discovery_recovery_attempts'
        if action == 'discovery_recovery':
            from .recommendation_events import only_unresolved_recommendations

            if (event['analyses'] or (event['discovery']['recommendations'] and not only_unresolved_recommendations(event))
                    or event['discovery']['status'] == 'completed' or not event['discovery']['source_runs']):
                raise ValueError('仅没有可阅读结果的未完成检索可以重试。')
        snapshot = {'project_id': event['project_id'], 'direction_id': event['direction_id'],
                    'event_id': event['event_id']}
        identity = [action, account['user_id'], event['event_id']]
        if attempt := event.get(attempt_key, 0):
            identity.append(attempt)
    elif action in ('direction_bootstrap', 'direction_daily'):
        if set(payload) != {'action', 'direction_id'}:
            raise ValueError('Explicit direction required')
        snapshot = saved_direction_configuration(store, account, payload['direction_id'])
        allowed = accounts.projects_for(store, account)
        if snapshot['project_id'] and allowed is not None and snapshot['project_id'] not in allowed:
            raise PermissionError('Project access required')
        day = clock.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
        identity = [action, account['user_id'], snapshot['direction_id'], snapshot['version']]
        if action == 'direction_daily':
            identity.append(day)
    else:
        raise ValueError('Unknown recommendation action')
    if scheduled_project_daily:
        snapshot['scheduled_project_daily'] = True
    key = sha256(json.dumps(identity).encode()).hexdigest()[:32]
    job = {'job_id': key, 'user_id': account['user_id'], 'action': action, 'snapshot': snapshot,
           'status': 'queued', 'created_at': created_at, 'queued_at': now(), 'event_id': None, 'reason': None}
    with store.transaction() as db:
        from .scope_deletion import require_active

        db.execute('BEGIN IMMEDIATE')
        require_active(store, project_id=snapshot.get('project_id'), direction_id=snapshot.get('direction_id'), db=db)
        if action == 'project_understanding':
            matched_resume = False
            while row := db.execute("SELECT body FROM objects WHERE kind='recommendation_job' AND id=?", (key,)).fetchone():
                prior = json.loads(row[0])
                if prior['status'] == 'partial' and prior['reason'] == 'feishu_attachments_pending':
                    snapshot['feishu_attachment_continuation'] = True
                    snapshot['feishu_attachment_previous_pending'] = prior['snapshot'].get('feishu_attachments_pending', 0)
                    matched_resume |= prior['job_id'] == resume_feishu_job
                    key = sha256((key+':continue-feishu-attachments').encode()).hexdigest()[:32]
                elif prior['status'] == 'canceled':
                    key = sha256((key+':restart-stopped-analysis').encode()).hexdigest()[:32]
                elif prior['status'] == 'partial' and prior['snapshot'].get('image_analysis_pending'):
                    key = sha256((key+':continue-image-analysis').encode()).hexdigest()[:32]
                elif (prior['status'] == 'partial' and prior['snapshot'].get('batch_gaps')
                      and snapshot.get('analysis_mode') in ('imported', 'daily')):
                    matched_resume |= (prior['job_id'] == resume_feishu_job
                                       and prior['snapshot']['batch_gaps'].get('stop_reason') == 'model_budget_insufficient')
                    key = sha256((key+':continue-journal-analysis').encode()).hexdigest()[:32]
                elif (snapshot.get('feishu') and prior['status'] in ('failed', 'partial')
                      and prior['reason'] in ('source_failed', 'feishu_documents_unavailable')):
                    if prior['reason'] == 'feishu_documents_unavailable':
                        snapshot.pop('feishu_attachment_continuation', None)
                        snapshot.pop('feishu_attachment_previous_pending', None)
                    key = sha256((key+':source-retry').encode()).hexdigest()[:32]
                elif (not resume_feishu_job and snapshot.get('analysis_mode') == 'imported'
                      and prior['status'] in ('completed', 'skipped', 'partial')
                      and prior['snapshot'].get('project_brief_status') not in
                      (None, 'current', 'updated', 'no_evidence', 'canceled')):
                    key = sha256((key+':project-brief-retry').encode()).hexdigest()[:32]
                else:
                    break
            if resume_feishu_job and not matched_resume:
                return None
            job['job_id'] = key
        if action in ('direction_bootstrap', 'direction_daily'):
            while row := db.execute("SELECT body FROM objects WHERE kind='recommendation_job' AND id=?", (key,)).fetchone():
                prior = json.loads(row[0])
                if prior['status'] in ('queued', 'running', 'processing'):
                    break
                # An explicit click is a fresh run, including after completed or
                # partial results. Only an actually active job stays idempotent.
                snapshot['retry_token'] = prior['job_id']
                key = sha256((key+f':{action}-retry:{prior["job_id"]}').encode()).hexdigest()[:32]
            job['job_id'] = key
        db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)', ('recommendation_job', key, json.dumps(job)))
    saved = store.get('recommendation_job', key)
    if saved['user_id'] != account['user_id']:
        raise PermissionError('Job owner mismatch')
    if 'question_ids' in payload and saved['snapshot'].get('review_question_ids') != snapshot.get('review_question_ids'):
        raise ValueError('本次检索已经提交，不能修改已确认的问题。')
    return saved


def visible_jobs(store, account):
    from .scope_deletion import deleted_scopes, scope_deleted

    allowed = accounts.projects_for(store, account)
    deleted = deleted_scopes(store)
    return [j for j in store.list('recommendation_job') if j['user_id'] == account['user_id']
            and not scope_deleted(j['snapshot'], deleted)
            and (not j['snapshot'].get('project_id') or allowed is None
                 or j['snapshot']['project_id'] in allowed)]


def persist_job(store, job):
    """Persist a worker's job state without resurrecting a user-canceled job."""
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        saved = db.execute("SELECT body FROM objects WHERE kind='recommendation_job' AND id=?",
                           (job['job_id'],)).fetchone()
        current = json.loads(saved[0]) if saved else {}
        if current.get('status') == 'canceled':
            job.update(status='canceled', reason=current.get('reason') or 'canceled_by_user')
            if current.get('canceled_at'):
                job['canceled_at'] = current['canceled_at']
        db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',
                   ('recommendation_job', job['job_id'], json.dumps(job)))
    return job


def save_job_progress(store, job, stage, percent, **counts):
    job['checkpoint'] = {'stage': stage, 'percent': percent, 'updated_at': now(), **counts}
    persist_job(store, job)


def jobs_with_progress(store, jobs, *, worker_ready, clock=None):
    """Project saved milestones and recent same-mode durations for already-authorized jobs."""
    clock = clock or datetime.now(UTC)
    durations = {}
    modes = ('direction_bootstrap', 'direction_daily', 'progress', 'project_understanding',
             'project_understanding_retry', 'reading_recovery', 'discovery_recovery')

    journal_actions = ('project_understanding', 'project_understanding_retry')

    def timing_key(job):
        action = job['action']
        snapshot = job.get('snapshot', {})
        total = None
        if action in journal_actions:
            checkpoint = job.get('checkpoint', {})
            if checkpoint.get('stage') != 'analyzing_journals' or not checkpoint.get('total'):
                return None
            total = checkpoint['total']
        return (job['user_id'], action, snapshot.get('analysis_mode'),
                snapshot.get('project_id'), snapshot.get('direction_id'), total)

    past = sorted((job for job in jobs if job['action'] in modes and job.get('started_at') and job.get('ended_at')
                   and (job['status'] == 'completed'
                        or job['status'] == 'partial' and job['action'] not in journal_actions)),
                  key=lambda job: job['started_at'], reverse=True)
    for job in past:
        if (key := timing_key(job)) is None:
            continue
        samples = durations.setdefault(key, [])
        if len(samples) < 6:
            seconds = (datetime.fromisoformat(job['ended_at']) - datetime.fromisoformat(job['started_at'])).total_seconds()
            if seconds > 0:
                samples.append(seconds)

    def project(job):
        mode = job['action']
        if mode not in modes or job['status'] not in ('queued', 'processing'):
            return job
        progress = {'stage': 'queued', 'percent': 0, 'eta_status': 'queued'}
        if job['status'] == 'processing':
            event = None
            if (key := job.get('event_id')) and mode in ('direction_bootstrap','direction_daily','progress','discovery_recovery'):
                try:
                    event = store.get('recommendation_event', key)
                except ValueError:
                    pass  # The worker links the ID immediately before creating the event.
                if event and event['user_id'] != job['user_id']:
                    event = None
                if event and mode == 'discovery_recovery' and not event.get('discovery_recovery_attempts'):
                    event = None
            progress.update(stage='preparing', percent=None, eta_status='insufficient_history')
            checkpoint = job.get('checkpoint', {})
            progress.update(checkpoint)
            updated_at = progress.pop('updated_at', None)
            if event:
                discovery = event.get('discovery')
                total = len(discovery['recommendations']) if discovery else 0
                done = min(total, len(event['analyses']) + len(event['errors']))
                if event['status'] != 'processing' or (discovery and done == total):
                    progress.update(stage='finalizing', percent=None)
                elif discovery:
                    progress.update(stage='reading', percent=None, done=done, total=total)
                elif not checkpoint:
                    progress.update(stage='searching')
            if updated_at and progress['stage'] == checkpoint.get('stage'):
                progress['checkpoint_minutes'] = max(0, int((clock - datetime.fromisoformat(updated_at)).total_seconds() // 60))
            elapsed = max(0, (clock - datetime.fromisoformat(job['started_at'])).total_seconds())
            progress['elapsed_minutes'] = int(elapsed // 60)
            samples = durations.get(timing_key(job), [])
            if mode in journal_actions and not progress.get('done'):
                samples = []
            if len(samples) >= 2:
                latest = ceil((max(samples) - elapsed) / 60)
                earliest = max(1, ceil((min(samples) - elapsed) / 60))
                if latest <= 0:
                    progress['eta_status'] = 'beyond_history'
                elif latest - earliest <= max(5, earliest):
                    progress.update(eta_status='estimated', eta_minutes=[earliest, latest])
                else:
                    progress['eta_status'] = 'variable_history'
        if not worker_ready:
            progress['eta_status'] = 'worker_unavailable'
            progress.pop('eta_minutes', None)
        return {**job, 'progress': progress}

    return [project(job) for job in jobs]


def cancel(store, account, job_id):
    """Stop a queued or processing job. The worker observes the flag at the next safe point."""
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT body FROM objects WHERE kind='recommendation_job' AND id=?",
                         (str(job_id),)).fetchone()
        job = json.loads(row[0]) if row else None
        if not job or job.get('user_id') != account['user_id']:
            raise PermissionError('Job not found')
        if job.get('status') not in ('queued', 'processing'):
            raise ValueError('Job already finished')
        previous = job['status']
        job.update(status='canceled', reason='canceled_by_user', canceled_at=now())
        db.execute("UPDATE objects SET body=? WHERE kind='recommendation_job' AND id=?",
                   (json.dumps(job), job['job_id']))
    event_id = job.get('event_id')
    if event_id:
        try:
            event = store.get('recommendation_event', event_id)
        except ValueError:
            event = None
        if event and event.get('status') in ('queued', 'processing'):
            event.update(status='partial', reason='canceled_by_user', ended_at=now())
            warnings = list(event.get('warnings') or [])
            if 'canceled_by_user' not in warnings:
                warnings.append('canceled_by_user')
            event['warnings'] = warnings
            store.put('recommendation_event', event_id, event)
            store.audit(event_id, {'event': 'event_canceled', 'job_id': job['job_id']})
    store.audit(job['job_id'], {'event': 'job_canceled', 'previous_status': previous})
    return job


def automatic_project_enabled(store, job):
    snapshot = job['snapshot']
    if not snapshot.get('scheduled_project_daily'):
        return True
    rows = store.select('research_project', ('automation',), key=snapshot['project_id'], decode=('automation',))
    return bool(rows and (rows[0]['automation'] or {}).get('daily_enabled'))


def _same_day_manual_request(store, job):
    snapshot = job['snapshot']
    zone = ZoneInfo('Asia/Shanghai')
    day = datetime.fromisoformat(job['created_at']).astimezone(zone).date()
    scope = ('project_id', 'feishu', 'feishu_grant_id', 'feishu_manifest')
    matches = sorted((prior for prior in store.list('recommendation_job')
                      if prior['user_id'] == job['user_id'] and prior['action'] == 'project_understanding'
                      and prior.get('understanding_run_id') and prior['status'] in ('completed', 'partial')
                      and prior['snapshot'].get('analysis_mode') == 'daily'
                      and not prior['snapshot'].get('auto_dispatch')
                      and datetime.fromisoformat(prior['created_at']).astimezone(zone).date() == day
                      and all(prior['snapshot'].get(field) == snapshot.get(field) for field in scope)),
                     key=lambda prior: datetime.fromisoformat(prior['created_at']), reverse=True)
    for prior in matches:
        rows = store.select('understanding_run', ('request',), key=prior['understanding_run_id'], decode=('request',))
        if rows and (request := rows[0]['request']):
            if store.select('recommendation_delivery', ('request_id',), key=request['request_id']):
                return request['request_id']
    return None


def _automatic_followup(store, job, account=None):
    followup = job.get('automatic_followup')
    if not followup:
        return
    account = account or accounts.find(store, job['user_id'])
    if account and not account['disabled'] and job['status'] != 'canceled' and automatic_project_enabled(store, job):
        for request_id in followup.get('request_ids', []):
            try:
                delivery = store.get('recommendation_delivery', request_id)
                if delivery['project_id'] != job['snapshot']['project_id'] or delivery['user_id'] != account['user_id']:
                    raise PermissionError('Automatic review scope mismatch')
                question_ids = [question['id'] for question in delivery['request'].get('research_questions', [])]
                enqueue(store, account, {'action': 'progress', 'request_id': request_id,
                                         'question_ids': question_ids, 'auto_email': bool(job['snapshot'].get('auto_email'))},
                        as_of=job['created_at'],
                        scheduled_project_daily=job['snapshot'].get('scheduled_project_daily', False))
            except (ValueError, PermissionError):
                store.audit(job['job_id'], {'event': 'automatic_progress_enqueue_failed',
                                             'request_id': request_id})
        if followup.get('email'):
            from .project_daily_email import send_daily_report

            send_daily_report(store, account, job)
    del job['automatic_followup']
    persist_job(store, job)


def process_one(store, *, execute):
    """Claim atomically; interrupted jobs remain processing for explicit inspection."""
    with store.transaction() as db:
        row = db.execute("SELECT id,body FROM objects WHERE kind='recommendation_job' "
                         "AND json_extract(body,'$.status')='queued' ORDER BY json_extract(body,'$.created_at') LIMIT 1").fetchone()
        if not row:
            return None
        key, body = row
        job = json.loads(body)
        job.update(status='processing', started_at=now())
        claimed = db.execute("UPDATE objects SET body=? WHERE kind='recommendation_job' AND id=? "
                             "AND json_extract(body,'$.status')='queued'", (json.dumps(job), key)).rowcount
    if not claimed:
        return None
    if not automatic_project_enabled(store, job):
        return persist_job(store, {**job, 'status': 'skipped', 'reason': 'automation_disabled', 'ended_at': now()})
    account = None
    auto_requests = []
    try:
        from .scope_deletion import require_active

        account = accounts.find(store, job['user_id'])
        if not account or account['disabled']:
            raise PermissionError('Account unavailable')
        allowed = accounts.projects_for(store, account)
        project = job['snapshot'].get('project_id')
        require_active(store, project_id=project, direction_id=job['snapshot'].get('direction_id'))
        if project and allowed is not None and project not in allowed:
            raise PermissionError('Project access revoked')
        if job['action'].startswith('direction_'):
            # Recheck ownership without replacing the historical settings snapshot.
            saved_direction_configuration(store, account, job['snapshot']['direction_id'])
        from . import ai_backend

        def report_fallback():
            job['model_fallback'] = 'sol_quota_deepseek'
            persist_job(store, job)

        with ai_backend.task_selection(report_fallback):
            event = execute(job, account)
        auto_requests = event.pop('_auto_review_requests', [])
        if event['user_id'] != account['user_id']:
            raise PermissionError('Worker result owner mismatch')
        try:
            canceled = (store.get('recommendation_job', key) or {}).get('status') == 'canceled'
        except ValueError:
            canceled = False
        if canceled:
            job.update(status='canceled', reason='canceled_by_user', event_id=event['event_id'])
        else:
            job.update(status=event['status'], event_id=event['event_id'], reason=event['reason'])
        job.update({key: event[key] for key in ('understanding_run_id', 'understanding_phase') if key in event})
        if (job['snapshot'].get('auto_dispatch') and job['status'] == 'skipped'
                and job['reason'] in ('no_update', 'no_new_or_updated_journals_in_window')):
            if not auto_requests and (request_id := _same_day_manual_request(store, job)):
                auto_requests = [request_id]
            if auto_requests:
                job.update(status='completed', reason='awaiting_review', event_id=None)
    except PermissionError:
        job.update(status='failed', reason='access_denied')
    except Exception:  # noqa: BLE001 -- persist safe worker failure without leaking provider exceptions
        job.update(status='failed', reason='worker_failed')
    job['ended_at'] = now()
    if auto_requests and job['status'] in ('completed', 'partial'):
        job['automatic_followup'] = {'request_ids': list(dict.fromkeys(auto_requests))}
    elif account and job['snapshot'].get('auto_email') and job.get('event_id') and job['status'] != 'canceled':
        job['automatic_followup'] = {'email': True}
    persist_job(store, job)
    _automatic_followup(store, job, account)
    pending = job['snapshot'].get('feishu_attachments_pending', 0)
    previous_pending = job['snapshot'].get('feishu_attachment_previous_pending')
    snapshot = job['snapshot']
    gaps = snapshot.get('batch_gaps') or {}
    attachment_progress = (job['reason'] == 'feishu_attachments_pending' and pending
                           and (previous_pending is None or pending < previous_pending))
    journal_progress = (snapshot.get('feishu') and snapshot.get('analysis_mode') == 'imported'
                        and gaps.get('stop_reason') == 'model_budget_insufficient'
                        and gaps.get('accepted', 0) > 0 and gaps.get('unprocessed', 0) > 0)
    if job['status'] == 'partial' and (attachment_progress or journal_progress):
        try:
            enqueue(store, account, {'action': 'project_understanding', 'analysis_mode': 'imported',
                    'project_id': job['snapshot']['project_id']}, as_of=job['created_at'],
                    resume_feishu_job=job['job_id'])
        except (ValueError, PermissionError):
            # The completed batch is durable. The owner can retry when Feishu is available.
            pass
    return job


def reconcile_interrupted(store):
    """Restore committed outcomes and finish queued handoffs after acquiring the worker lock."""
    for job in store.list('recommendation_job'):
        if job['status'] == 'canceled' and job.get('started_at') and not job.get('ended_at'):
            persist_job(store, {**job, 'ended_at': now()})
        if job['status'] != 'processing':
            _automatic_followup(store, job)
            continue
        event = None
        if job.get('event_id'):
            try:
                event = store.get('recommendation_event', job['event_id'])
            except ValueError:
                pass  # Worker stopped before creating its event.
        if event and event['user_id'] != job['user_id']:
            raise PermissionError('Job checkpoint owner mismatch')
        if event and event['status'] != 'processing':
            job.update(status=event['status'], reason=event['reason'])
            if job['snapshot'].get('auto_email'):
                job['automatic_followup'] = {'email': True}
        else:
            job.update(status='failed', reason='interrupted')
            if event:
                event.update(status='failed', reason='interrupted')
                store.put('recommendation_event', event['event_id'], event)
        job['ended_at'] = now()
        store.put('recommendation_job', job['job_id'], job)
        store.audit(job['job_id'], {'event': 'interrupted_job_reconciled', 'status': job['status']})
        _automatic_followup(store, job)
