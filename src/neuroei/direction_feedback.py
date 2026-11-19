"""Explicit direction feedback; automatic learning and ranking updates stay disabled.

Only immutable direction_feedback objects are appended. Project profiles, laboratory
feedback, direction settings, runs and reports are never rewritten here, and no model
call reinterprets an explicit user label.
"""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from .config import StrictModel
from .directions import load_direction, optional, require_direction_project
from .execution import project_lock
from .models import now

FeedbackAction = Literal['relevant', 'later', 'known', 'irrelevant',
                         'method_unsuitable', 'fact_correction', 'report_quality']

# "相关" is about the declared direction, never about a laboratory project.
ACTION_LABELS = {
    'relevant': '与关注方向相关',
    'later': '稍后再看',
    'known': '已了解该工作',
    'irrelevant': '与关注方向无关',
    'method_unsuitable': '方法不适用',
    'fact_correction': '事实纠错',
    'report_quality': '报告质量反馈',
}
ACTION_NOTE = '“相关”指是否贴合本关注方向，不是实验室项目相关性判断；仅相关/不相关可用于后续兴趣评估；当前未应用排序更新。'
OTHER_CHANNELS = ('later', 'known', 'method_unsuitable', 'fact_correction', 'report_quality')


class DirectionFeedbackRequest(StrictModel):
    event_id: UUID
    user_id: str = Field(min_length=1, max_length=100)
    direction_id: UUID
    direction_version: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1, max_length=100)
    report_id: str = Field(min_length=1, max_length=200)
    paper_id: str = Field(min_length=1, max_length=300)
    paper_version: str = Field(min_length=1, max_length=40)
    action: FeedbackAction
    reason: str = Field(default='', max_length=4000)

    @field_validator('user_id')
    @classmethod
    def nonblank_user(cls, value):
        if not value.strip():
            raise ValueError('Nonblank user name required')
        return value.strip()

    @field_validator('reason')
    @classmethod
    def trimmed_reason(cls, value):
        return value.strip()

    @model_validator(mode='after')
    def explain_corrections(self):
        if self.action in ('fact_correction', 'report_quality') and not self.reason:
            raise ValueError('Corrections and report quality feedback need a reason')
        return self


def _user(value):
    if not isinstance(value, str):
        raise TypeError('Explicit user name required')
    user = value.strip()
    if not user or len(user) > 100:
        raise ValueError('User name required')
    return user


def _pool_paper(store, run, paper_id):
    # The frozen run display names the version; the immutable pool is the only authority for it.
    displayed = next((item for item in run.get('papers') or []
                      if item.get('paper_id') == paper_id), None)
    if displayed is None or not displayed.get('version'):
        raise ValueError('Paper is missing from the run display; refresh the report')
    if not run.get('pool_id'):
        raise ValueError('Direction run has no pool; feedback is closed for this run')
    pool = optional(store, 'pool', run['pool_id'])
    if pool is None:
        raise ValueError('Direction run pool is missing; refresh the report')
    version = displayed['version']
    matches = [item for item in pool.get('papers') or []
               if item.get('paper_id') == paper_id and item.get('version') == version]
    if not matches:
        raise ValueError('Paper version is not in the run pool; refresh the report')
    return matches[0]


def _run_report(store, run_id, user, direction_id, version, report_id):
    report = store.get('direction_report', report_id)
    if report.get('run_id') != run_id:
        raise ValueError('Direction report does not belong to this run')
    if report.get('user_id') != user:
        raise ValueError('Direction report belongs to another user')
    if report.get('direction_id') != direction_id:
        raise ValueError('Direction report belongs to another direction')
    if report.get('direction_version') != version:
        raise ValueError('Direction report cites another settings version')
    if report.get('project_context'):
        require_direction_project(store, {'user_id': user, 'project_id': report['project_context']['project_id']})
    return report


def _parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        raise TypeError('Direction feedback needs an occurred_at time')
    if parsed.tzinfo is None:
        raise ValueError('Direction feedback time needs timezone')
    return parsed


def _cutoff(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise TypeError('Feedback cutoff must be a datetime')
    if value.tzinfo is None:
        raise ValueError('Feedback cutoff needs timezone')
    return value


def save_direction_feedback(store, payload):
    request = DirectionFeedbackRequest.model_validate(payload)
    user = request.user_id
    direction = load_direction(store, user, str(request.direction_id))
    did = direction['direction_id']
    run = store.get('direction_run', request.run_id)
    if run.get('user_id') != user:
        raise ValueError('Direction run belongs to another user')
    if run.get('direction_id') != did:
        raise ValueError('Direction run belongs to another direction')
    if run.get('direction_version') != request.direction_version:
        raise ValueError('Feedback must cite the settings version used by this run')
    # Historical settings versions stay feedbackable, but the report must belong to this run.
    _run_report(store, request.run_id, user, did, request.direction_version, request.report_id)
    recommended = next((item for item in run.get('recommendations') or []
                        if item.get('paper_id') == request.paper_id), None)
    if recommended is None:
        raise ValueError('Paper is not a recommendation in this direction run')
    paper = _pool_paper(store, run, request.paper_id)
    if paper['version'] != request.paper_version:
        raise ValueError('Paper version changed; refresh the displayed report')
    key = str(request.event_id)
    with project_lock(store.root, 'direction-feedback:' + did):
        prior = optional(store, 'direction_feedback', key)
        occurred_at = prior['occurred_at'] if prior else now()
        body = {
            'feedback_id': key, 'user_id': user, 'direction_id': did,
            'direction_version': request.direction_version, 'run_id': request.run_id,
            'report_id': request.report_id, 'paper_id': request.paper_id,
            'paper_version': paper['version'], 'action': request.action,
            'reason': request.reason, 'occurred_at': occurred_at,
            'provenance': 'explicit_user_direction_feedback',
        }
        if prior is not None and prior != body:
            raise ValueError('Feedback ID already records a different action')
        store.put('direction_feedback', key, body, immutable=True)
    return {'feedback_id': key, 'saved': prior is None, 'reused': prior is not None,
            'occurred_at': occurred_at, 'learning_enabled': False}


def direction_feedback_state(store, user, direction_id, *, run_id=None, report_id=None):
    user = _user(user)
    direction = load_direction(store, user, direction_id)
    did = direction['direction_id']
    events = [event for event in store.list('direction_feedback')
              if event.get('user_id') == user and event.get('direction_id') == did]
    events.sort(key=lambda event: (_parse_time(event.get('occurred_at')),
                                   event.get('feedback_id', '')), reverse=True)
    state = {'user_id': user, 'direction_id': did, 'settings_version': direction['version'],
             'feedback': events, 'action_labels': dict(ACTION_LABELS), 'action_note': ACTION_NOTE,
             'other_channels': list(OTHER_CHANNELS), 'learning_enabled': False}
    if report_id and not run_id:
        raise ValueError('Specify the direction run before selecting its report')
    if run_id:
        run = store.get('direction_run', run_id)
        if run.get('user_id') != user:
            raise ValueError('Direction run belongs to another user')
        if run.get('direction_id') != did:
            raise ValueError('Direction run belongs to another direction')
        version = run.get('direction_version')
        selected = report_id or run.get('report_id')
        if selected:
            _run_report(store, run_id, user, did, version, selected)
        papers = []
        for rec in run.get('recommendations') or []:
            paper = _pool_paper(store, run, rec['paper_id'])
            displayed = next((item for item in run.get('papers') or []
                              if item.get('paper_id') == rec['paper_id']), {})
            papers.append({'paper_id': rec['paper_id'], 'version': paper['version'],
                           'title': paper.get('title') or displayed.get('title')})
        state.update(run_id=run_id, direction_version=version, report_id=selected, papers=papers)
    return state


def feedback_snapshot(store, user, direction_id, cutoff):
    user = _user(user)
    when = _cutoff(cutoff)
    direction = load_direction(store, user, direction_id)
    did = direction['direction_id']
    version = direction['version']
    eligible = []
    for event in store.list('direction_feedback'):
        if event.get('user_id') != user or event.get('direction_id') != did:
            continue
        if event.get('direction_version') != version:
            continue
        occurred = _parse_time(event.get('occurred_at'))
        if occurred >= when:
            continue
        eligible.append((occurred, event))
    eligible.sort(key=lambda row: (row[0], row[1].get('feedback_id', '')))
    explicit = {}
    channels = {name: [] for name in OTHER_CHANNELS}
    for occurred, event in eligible:
        action = event.get('action')
        if action in ('relevant', 'irrelevant'):
            key = (event.get('paper_id'), event.get('paper_version'))
            prior = explicit.get(key)
            if prior and prior[0] == occurred and prior[1].get('action') != action:
                raise ValueError('Conflicting same-time direction interest feedback requires review')
            explicit[key] = (occurred, event)
        elif action in channels:
            channels[action].append(event)
        else:
            raise ValueError('Unknown direction feedback action')
    positive = sorted(
        ({'paper_id': pid, 'paper_version': pver, 'feedback_id': event.get('feedback_id'),
          'occurred_at': event.get('occurred_at')}
         for (pid, pver), (_, event) in explicit.items() if event.get('action') == 'relevant'),
        key=lambda item: (item['paper_id'], item['paper_version']))
    negative = sorted(
        ({'paper_id': pid, 'paper_version': pver, 'feedback_id': event.get('feedback_id'),
          'occurred_at': event.get('occurred_at')}
         for (pid, pver), (_, event) in explicit.items() if event.get('action') == 'irrelevant'),
        key=lambda item: (item['paper_id'], item['paper_version']))
    return {'user_id': user, 'direction_id': did, 'direction_version': version,
            'cutoff_exclusive': when.isoformat(), 'eligible_event_count': len(eligible),
            'positive': positive, 'negative': negative, 'other_channels': channels,
            'policy': {'interest_scope': 'explicit relevant/irrelevant for this declared direction only',
                       'relevant_means': '与本关注方向相关，不是实验室项目相关性',
                       'not_negative': list(OTHER_CHANNELS)},
            'learning_enabled': False, 'ranking_update_applied': False}
