"""Bounded unattended entry and observed calendar-day coverage, separate from scientific acceptance."""
import uuid
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .execution import WorkflowBusy
from .models import now
from .store import Store
from .workflow import run_daily, run_weekly

ZONE = ZoneInfo('Asia/Shanghai')
TERMINAL = {'completed','partial'}
OBSERVABLE = {'awaiting_remote_read','awaiting_remote_rank'}


def trial_status(store, project, *, until=None, since=None):
    observed_at = now()
    today = datetime.fromisoformat(observed_at).astimezone(ZONE).date()
    requested_until = date.fromisoformat(until) if until else today
    end = min(requested_until,today)
    start = date.fromisoformat(since) if since else end-timedelta(days=6)
    if start > requested_until:
        raise ValueError('Valid trial coverage window required')
    days = {}
    excluded = []
    reports = {r['report_id']:r for r in store.list('report')}
    for w in store.list('workflow'):
        if w['project_id'] != project or w.get('experimental') or w['status'] not in TERMINAL:
            continue
        report = reports.get(w.get('report_id'))
        if (not report or report.get('kind') != 'workflow-daily' or report.get('workflow_id') != w['run_id']
                or (report.get('project_id'),report.get('profile_version')) != (project,w['profile_version'])):
            continue
        try:
            stamp = datetime.fromisoformat(w['ended_at'])
            if stamp.tzinfo is None:
                raise ValueError('Timezone missing')
            executed = stamp.astimezone(ZONE).date()
        except (KeyError,ValueError,TypeError):
            excluded.append({'workflow_id':w['run_id'],'reason':'missing_valid_completion_time'})
            continue
        if w['date'] != executed.isoformat() or executed > today:
            excluded.append({'workflow_id':w['run_id'],'reason':'backfill_or_future_date'})
            continue
        if start <= executed <= end:
            row = days.setdefault(executed.isoformat(),{'workflows':[], 'reports':[], 'quality':'not_accepted'})
            row['workflows'].append(w['run_id'])
            row['reports'].append(report['report_id'])
    expected = []
    day = start
    while day <= end:
        expected.append(day.isoformat())
        day += timedelta(days=1)
    consecutive = 0
    for day in reversed(expected):
        if day not in days:
            break
        consecutive += 1
    feedback = [e for e in store.list('feedback') if e['project_id']==project]
    return {'project_id':project,'observed_at':observed_at,'timezone':'Asia/Shanghai',
            'since':start.isoformat(),'requested_until':requested_until.isoformat(),'observed_until':end.isoformat(),
            'days':days,'days_with_drafts':len(days),'consecutive_days_ending_at_observed_until':consecutive,
            'missing_dates':[d for d in expected if d not in days],'excluded_runs':excluded,
            'explicit_feedback_events':len(feedback),'scientific_quality':'not_accepted',
            'seven_day_acceptance':False,'thirty_day_acceptance':False}


def controlled_run(config, project, *, report_date=None, reader_release, read_limit=2, weekly=False, progress=print):
    """No new versions or blind retries; reobserve only known remote-job checkpoints."""
    report_date = report_date or datetime.now(ZONE).date().isoformat()
    if date.fromisoformat(report_date) > datetime.now(ZONE).date():
        raise ValueError('Unattended runs cannot create future-dated reports')
    store = Store(config.data_dir)
    attempt = {'attempt_id':'operation-'+uuid.uuid4().hex,'project_id':project,'date':report_date,
               'started_at':now(),'status':'running','reader_release':reader_release,'read_limit':read_limit,
               'policy':'bounded-baseline-draft-v1','automatic_new_version':False,'feishu_upload':False}
    attempt['code_revision'] = os.environ.get('NEUROEI_CODE_REVISION','unrecorded-working-tree')
    store.put('operation',attempt['attempt_id'],attempt)
    try:
        latest = {}
        for w in sorted(store.list('workflow'),key=lambda w:w['started_at']):
            if w['project_id']==project:
                latest[w['slot']] = w
        pending = sorted([w for w in latest.values() if w['status'] not in TERMINAL],key=lambda w:w['started_at'])
        for old in pending:
            if old['status'] not in OBSERVABLE or not old.get('reader_release') or not old.get('read_limit'):
                attempt.update(status='requires_attention',workflow_id=old['run_id'],
                               detail='Interrupted/legacy checkpoint requires inspection; no automatic restart')
                return attempt
            original = {'reader_release':old['reader_release'],'read_limit':old['read_limit'],
                        'resume':old['run_id'],'progress':progress}
            if old.get('experimental'):
                original.update(ranking_strategy='paperflow-experimental',interest_mapping=old['interest_mapping'])
            recovered = run_daily(config,project,old['date'],**original)
            if recovered['status'] not in TERMINAL:
                attempt.update(status=recovered['status'],workflow_id=recovered['run_id'],
                               detail='Observed original checkpoint once; no replacement task')
                return attempt
        result = run_daily(config,project,report_date,reader_release=reader_release,
                           read_limit=read_limit,progress=progress)
        attempt.update(status=result['status'],workflow_id=result['run_id'],
                       report_id=result.get('report_id'),reused_existing_workflow=result.get('reused_existing_workflow',False))
        if weekly and result['status'] in TERMINAL:
            report = run_weekly(config,project,report_date)
            attempt['weekly_report_id'] = report['report_id']
    except WorkflowBusy:
        attempt.update(status='busy',detail='Another process owns the project lock; no duplicate execution')
    except Exception as exc:
        attempt.update(status='failed',error_type=type(exc).__name__)
    finally:
        attempt['ended_at'] = now()
        store.put('operation',attempt['attempt_id'],attempt)
    return attempt
