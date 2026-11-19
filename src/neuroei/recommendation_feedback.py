"""Explicit feedback bound to the displayed recommendation and paper snapshot."""
import hashlib
import json
from uuid import UUID

from pydantic import Field, model_validator

from .config import StrictModel
from .direction_feedback import FeedbackAction
from .models import now
from .recommendation_events import member_event
from .scope_deletion import deleted_scopes, scope_deleted


class ReadingFeedbackRequest(StrictModel):
    feedback_id: UUID
    event_id: str = Field(min_length=1, max_length=200)
    paper_id: str = Field(min_length=1, max_length=300)
    paper_version: str = Field(min_length=1, max_length=100)
    snapshot_version: str = Field(pattern=r'^[a-f0-9]{64}$')
    action: FeedbackAction
    reason: str = Field(default='', max_length=4000)

    @model_validator(mode='after')
    def require_explanation(self):
        self.reason = self.reason.strip()
        if self.action in ('fact_correction', 'report_quality') and not self.reason:
            raise ValueError('纠错和报告质量反馈需要填写说明。')
        return self


class ReadingSnapshotChanged(ValueError):
    pass


def reading_context(store, account, projects, event_id, paper_id):
    event = store.get('recommendation_event', event_id)
    if (event.get('user_id') != account['user_id'] or scope_deleted(event, deleted_scopes(store))
            or (event.get('project_id') and projects is not None and event['project_id'] not in projects)):
        raise PermissionError('不能访问这份论文记录。')
    discovery = event.get('discovery') or {}
    paper = next((p for p in discovery.get('candidates', []) if p['paper_id'] == paper_id), None)
    recommendation = next((r for r in discovery.get('recommendations', []) if r['paper_id'] == paper_id), None)
    if paper is None or recommendation is None:
        raise ValueError('这篇论文不在本次推荐中。')
    analysis = next((a for a in event.get('analyses', []) if a['paper_id'] == paper_id), None)
    if analysis and analysis.get('paper_version') not in (None, paper.get('paper_version')):
        raise ValueError('论文与分析版本不一致。')
    # Keep native evidence/rendering fields; exclude provider diagnostics and cost/call metadata.
    displayed_analysis = {key: value for key, value in analysis.items() if key in (
        'paper_id', 'paper_version', 'status', 'reading_scope', 'methods', 'findings', 'limitations',
        'research_suggestions', 'claims', 'evidence', 'warnings', 'analyzed_at', 'problem',
        'rejected_claims', 'rejected_suggestions', 'relevance_to_questions')} if analysis else None
    member = member_event(event)
    if displayed_analysis:
        scope = analysis.get('reading_scope') or {}
        displayed_analysis['reading_scope'] = {key: scope[key] for key in (
            'missing', 'fulltext_acquisition', 'targeted_evidence', 'visual_review') if key in scope}
        gaps = next(a['reading_scope']['evidence_gaps'] for a in member['analyses'] if a['paper_id'] == paper_id)
        displayed_analysis['reading_scope'].update(
            evidence_gaps=gaps,
            documents=[None] * len(scope.get('documents', [])),
            parser={'scan_fallback': bool(scope.get('parser', {}).get('scan_fallback'))},
            agent_review=bool(scope.get('agent_review')),
            paperqa_checkpoint={key: [None] * len(scope.get('paperqa_checkpoint', {}).get(key, [])) for key in (
                'provider_validation_rejections', 'condition_validation_rejections', 'semantic_validation_rejections')})
        for key in ('rejected_claims', 'rejected_suggestions'):
            displayed_analysis[key] = [None] * len(analysis.get(key, []))
    displayed = {'paper': paper, 'recommendation': recommendation, 'analysis': displayed_analysis,
                 'input': member['input']}
    snapshot = {**displayed, 'analysis': analysis, 'input': event.get('input') or {}}
    version = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                       separators=(',', ':')).encode()).hexdigest()
    context = {'event_id': event_id, 'paper_id': paper_id, 'paper_version': paper.get('paper_version'),
               'snapshot_version': version, 'mode': event['mode'], 'project_id': event.get('project_id'),
               'direction_id': event.get('direction_id'), 'date': event['date']}
    known_version = paper.get('paper_version') and paper['paper_version'] in paper.get('versions', [])
    journal_refs = (event.get('basis') or {}).get('source_refs') or {}
    linked = set(recommendation.get('question_ids', ()))
    question_refs = {ref for question in displayed['input'].get('research_questions', [])
                     if question['id'] in linked for ref in question.get('source_refs', [])}
    return {**displayed, 'context': context, 'can_submit': bool(known_version), 'learning_enabled': False,
            'journal_sources': {ref: journal_refs[ref] for ref in question_refs if ref in journal_refs}}


def reading_state(store, account, projects, event_id, paper_id):
    state = reading_context(store, account, projects, event_id, paper_id)
    state['feedback'] = sorted((row for row in store.list('recommendation_feedback')
        if row['user_id'] == account['user_id'] and row['event_id'] == event_id and row['paper_id'] == paper_id),
        key=lambda row: row['occurred_at'], reverse=True)
    return state


def save_reading_feedback(store, account, projects, payload):
    request = ReadingFeedbackRequest.model_validate(payload).model_dump(mode='json')
    key = request['feedback_id']
    # One SQLite transaction guards both the selected snapshot and the immutable feedback ID.
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        state = reading_context(store, account, projects, request['event_id'], request['paper_id'])
        old = db.execute("SELECT body FROM objects WHERE kind='recommendation_feedback' AND id=?", (key,)).fetchone()
        if old:
            prior = json.loads(old[0])
            if prior['user_id'] != account['user_id'] or any(prior.get(k) != v for k, v in request.items()):
                raise ValueError('这个反馈编号已经用于另一条评价。')
            return {'feedback_id': key, 'saved': True, 'reused': True, 'learning_enabled': False}
        context = state['context']
        if not state['can_submit'] or any(request[k] != context[k] for k in ('paper_version', 'snapshot_version')):
            raise ReadingSnapshotChanged('论文或解读版本已更新。说明已保留，请刷新核对后重新提交。')
        body = {**context, **request, 'user_id': account['user_id'], 'occurred_at': now(),
                'provenance': 'explicit_user_recommendation_feedback'}
        db.execute('INSERT INTO objects VALUES (?,?,?)', ('recommendation_feedback', key,
                   json.dumps(body, ensure_ascii=False, sort_keys=True)))
    return {'feedback_id': key, 'saved': True, 'reused': False, 'learning_enabled': False}
