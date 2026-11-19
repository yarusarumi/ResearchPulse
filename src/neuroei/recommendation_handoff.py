"""Durable, owner-checked boundary between understanding and reviewed dispatch."""
import json

from .contracts import RecommendationRequest
from .models import now
from .project_understanding import acknowledge_recommendation_request, pending_recommendation_requests
from .recommendation_events import authorize, event_id


def initial_review_questions(store, delivery):
    """Suggest a short list for the first imported journals after background setup."""
    request = delivery['request']
    questions = request['research_questions']
    if len(questions) <= 4 or delivery.get('trigger_details', {}).get('selection') != 'explicit_import':
        return []
    try:
        state = store.get('understanding_state', request['trigger']['version'])
        parent = store.get('understanding_run', state.get('parent_version'))
    except ValueError:
        return []
    owner = (delivery['user_id'], delivery['project_id'])
    if parent.get('reason') not in ('bootstrap_completed', 'background_context_saved') or any(
            (row.get('user_id'), row.get('project_id')) != owner for row in (state, parent)):
        return []
    mapping = request['constraints'].get('query_question_ids', {})
    terms = {q['id']: {' '.join(query.casefold().split()) for query in request['queries']
                       if q['id'] in mapping.get(query, [])} for q in questions}
    if not all(terms.values()):
        return []
    selected, used = [], set()
    for question in sorted(questions, key=lambda q: -q['priority']):
        qid = question['id']
        if used.isdisjoint(terms[qid]):
            selected.append(qid)
            used.update(terms[qid])
            if len(selected) == 4:
                break
    return selected


def select_review_questions(request, question_ids):
    questions = request['research_questions']
    if (not isinstance(question_ids, list) or not question_ids
            or not all(isinstance(qid, str) for qid in question_ids)
            or len(set(question_ids)) != len(question_ids)
            or not set(question_ids) <= {q['id'] for q in questions}):
        raise ValueError('请选择至少一个本次研究问题。')
    selected = set(question_ids)
    constraints = request['constraints']
    mapping = constraints.get('query_question_ids', {})
    links = {query: [qid for qid in mapping.get(query, []) if qid in selected] for query in request['queries']}
    links = {query: ids for query, ids in links.items() if ids}
    if {qid for ids in links.values() for qid in ids} != selected:
        raise ValueError('所选问题缺少对应检索词。')
    updated = {**constraints, 'query_question_ids': links}
    if 'question_evidence_gates' in constraints:
        updated['question_evidence_gates'] = {qid: gate for qid, gate in constraints['question_evidence_gates'].items()
                                              if qid in selected}
    return RecommendationRequest.model_validate({
        **request, 'research_questions': [q for q in questions if q['id'] in selected],
        'queries': list(links), 'constraints': updated,
    }).model_dump(mode='json')


def receive_pending(store, *, user_id, project_id, projects):
    authorize({'user_id': user_id, 'project_id': project_id, 'direction_id': None}, user_id, projects, ())
    deliveries = []
    for run in pending_recommendation_requests(store, user_id=user_id, project_id=project_id):
        request = RecommendationRequest.model_validate(run['request']).model_dump(mode='json')
        authorize(request, user_id, projects, ())
        refs = run['source_refs']
        if any(ref not in refs for q in request['research_questions'] for ref in q['source_refs']):
            raise ValueError('Unresolvable request evidence')
        key = request['request_id']
        delivery = {'request_id': key, 'event_id': event_id(request), 'user_id': user_id,
                    'receipt_id': 'recommendation_delivery:'+key,
                    'project_id': project_id, 'request': request, 'source_refs': refs,
                    'trigger_details': run['trigger_details'], 'status': 'awaiting_review', 'received_at': now()}
        # Persist before ACK. A restart can safely run the same INSERT again.
        with store.transaction() as db:
            db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)',
                       ('recommendation_delivery', key, json.dumps(delivery)))
        saved = store.get('recommendation_delivery', key)
        if saved['user_id'] != user_id or saved['project_id'] != project_id or saved['request'] != request:
            raise ValueError('Delivery identity collision')
        acknowledge_recommendation_request(store, user_id=user_id, project_id=project_id,
                                           request_id=key, event_id=saved.get('receipt_id', 'recommendation_delivery:'+key))
        deliveries.append(saved)
    return deliveries


def dispatch_reviewed(store, request_id, *, user_id, projects, discover, analyze=None, question_ids=None,
                      should_stop=None, synthesize=None, automatic=False):
    """Explicit content approval; authorization precedes all external callbacks."""
    from .recommendation_events import run_event

    saved = store.get('recommendation_delivery', request_id)
    authorize(saved['request'], user_id, projects, ())
    if saved['user_id'] != user_id:
        raise PermissionError('Delivery owner mismatch')
    request = saved['request'] if question_ids is None else select_review_questions(saved['request'], question_ids)
    refs = saved['source_refs']
    if question_ids is not None:
        used = {ref for question in request['research_questions'] for ref in question['source_refs']}
        refs = {key: value for key, value in refs.items() if key in used}
    result = run_event(store, request, user_id=user_id, projects=projects,
                       discover=discover, analyze=analyze, should_stop=should_stop, synthesize=synthesize,
                       automatic=automatic,
                       basis={'source_refs': refs, 'trigger_details': saved['trigger_details']})
    store.put('recommendation_delivery', request_id, {**saved, 'status': 'dispatched',
                                                    'event_id': result['event_id']})
    _complete_review_jobs(store, request_id)
    return result


def _complete_review_jobs(store, request_id):
    """The delivery is resolved; its understanding job is no longer awaiting review."""
    from .recommendation_jobs import persist_job

    receipt_id = 'recommendation_delivery:' + request_id
    for job in store.list('recommendation_job'):
        snapshot = job.get('snapshot') or {}
        if snapshot.get('receipt_id') != receipt_id:
            continue
        if job.get('status') in ('completed', 'failed', 'canceled', 'cancelled'):
            continue
        persist_job(store, {**job, 'status': 'completed', 'reason': None, 'ended_at': now(),
                            'checkpoint': {**(job.get('checkpoint') or {}), 'stage': 'completed'}})


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    from .store import Store

    parser = argparse.ArgumentParser(description='Receive completed understanding requests into an isolated review queue')
    parser.add_argument('--store', type=Path, required=True, help='Existing isolated integration Store')
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--project-id', required=True)
    args = parser.parse_args()
    if not (args.store / 'neuroei.sqlite3').is_file():
        parser.error('An existing integration Store is required')
    store = Store(args.store)
    try:
        rows = receive_pending(store, user_id=args.user_id, project_id=args.project_id,
                               projects=[args.project_id])
        print(json.dumps({'received': len(rows), 'requests': [r['request_id'] for r in rows],
                          'state': 'awaiting_review', 'model_calls': 0}, ensure_ascii=False))
    finally:
        store.close()
