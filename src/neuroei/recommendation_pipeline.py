"""Explicit isolated orchestration. No clients, scheduler, downloads or model defaults."""
import json
from datetime import datetime, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from .discovery import discover
from .models import now
from .project_understanding import understand
from .recommendation_events import authorize, event_id, record_gate, run_event

TZ = ZoneInfo('Asia/Shanghai')


class QueryPlanningError(ValueError):
    """User-correctable query input; safe to display without provider exceptions."""


def save_direction_for_account(store, account, payload):
    from .directions import DirectionSave, save_direction

    values = dict(payload)
    links = values.pop('query_question_links', None)
    expected_query_version = values.pop('expected_query_version', None)
    values['user_id'] = account['display_name']
    checked = DirectionSave.model_validate(values)
    values = checked.model_dump(mode='json')
    from .execution import project_lock

    with project_lock(store.root, 'direction-query-save:'+str(checked.direction_id)):
        if links is not None:
            from .directions import load_direction, optional

            pointer = optional(store, 'direction_current', str(checked.direction_id))
            current = load_direction(store, account['display_name'], str(checked.direction_id),
                                     check_project_access=False) if pointer else None
            prior_plan = load_query_plan(store, current, account) if current else None
            if (expected_query_version != (prior_plan['version'] if prior_plan else None)
                    and (not prior_plan or links != prior_plan['query_question_links'])):
                raise ValueError('Query settings changed; refresh before saving')
        if len(set(checked.questions)) != len(checked.questions):
            raise ValueError('Duplicate research questions')
        if len(set(checked.public_queries)) != len(checked.public_queries):
            raise ValueError('Duplicate public queries')
        if links is not None:
            if not isinstance(links, dict) or set(links) != set(checked.public_queries):
                raise ValueError('Each explicit query needs its question links')
            linked = set()
            for indices in links.values():
                if not isinstance(indices, list) or not indices or any(
                        type(i) is not int or not 1 <= i <= len(checked.questions) for i in indices):
                    raise ValueError('Invalid question numbers')
                linked.update(indices)
            if linked != set(range(1, len(checked.questions)+1)):
                raise ValueError('Every research question needs an explicit query')
        from .discovery_aliases import literal_query_support

        query_links = links or ({q: [1] for q in checked.public_queries} if len(checked.questions) == 1 else None)
        if query_links is None:
            raise QueryPlanningError('请为每组检索词指定对应的研究问题。')
        for query, indices in query_links.items():
            if not literal_query_support(query, [checked.questions[i-1] for i in indices])['supported']:
                raise QueryPlanningError('检索词“'+query+'”暂不支持。请使用完整英文短语；具体模型名需同时写入对应研究问题。')
        saved = save_direction(store, values, allow_research_project=True)
        if links is not None:
            version = sha256(json.dumps(links, sort_keys=True).encode()).hexdigest()[:16]
            key = saved['direction_id']+':'+saved['version']+':'+version
            try:
                store.get('direction_query_plan', key)
            except ValueError:
                store.put('direction_query_plan', key, {'user_id': account['user_id'], 'version': version,
                          'saved_at': now(), 'query_question_links': links}, immutable=True)
            store.put('direction_query_current', saved['direction_id']+':'+saved['version'], {'key': key})
            saved['query_plan_version'] = version
        return saved


def load_query_plan(store, setting, account):
    try:
        pointer = store.get('direction_query_current', setting['direction_id']+':'+setting['version'])
    except ValueError:
        return None
    plan = store.get('direction_query_plan', pointer['key'])
    if plan['user_id'] != account['user_id']:
        raise PermissionError('Query plan owner mismatch')
    return plan


def saved_direction_configuration(store, account, direction_id):
    """Map the existing owner-checked settings to the common request boundary."""
    from .directions import load_direction

    saved = load_direction(store, account['display_name'], direction_id)
    if not saved['enabled']:
        raise ValueError('Direction is paused')
    if not saved['questions'] or not saved['public_queries']:
        raise QueryPlanningError('请先在「方向设置」填写研究问题和英文检索词，再发起推荐。')
    configuration = {'direction_id': saved['direction_id'], 'title': saved['title'], 'user_id': account['user_id'],
            'project_id': saved['project_id'], 'version': saved['version'], 'updated_at': saved['saved_at'],
            'research_questions': [{'id': 'q-'+sha256(q.encode()).hexdigest()[:16], 'description': q,
                'priority': {1: 3, 2: 2, 3: 1}[saved['priority']],
                'source_refs': [f"{direction_id}:{saved['version']}:question:{i}"]}
                for i, q in enumerate(saved['questions'])],
            'queries': saved['public_queries'], 'constraints': {'exclusions': saved['exclusions']}}
    plan = load_query_plan(store, saved, account)
    if plan:
        configuration['version'] += ':queries:'+plan['version']
        configuration['updated_at'] = plan['saved_at']
        ids = [q['id'] for q in configuration['research_questions']]
        configuration['constraints']['query_question_ids'] = {
            query: [ids[i-1] for i in indices] for query, indices in plan['query_question_links'].items()}
    return configuration


def progress(store, batch, *, extractor, sources, user_id, project_id, projects,
             as_of, ranker=None, analyze=None, source_names=('arxiv',), max_calls=12, require_review=True, bootstrap=False,
             imported_at=None, retry_of=None, on_progress=None, model_selection=None, manual_selection=None,
             project_context=None, should_stop=None, delivery_window_end=None):
    from .recommendation_handoff import receive_pending

    authorize({'user_id': user_id, 'project_id': project_id, 'direction_id': None}, user_id, projects, ())
    if any((r.user_id, r.project_id) != (user_id, project_id) for r in batch.records):
        raise PermissionError('Journal account/project mismatch')
    recovered = receive_pending(store, user_id=user_id, project_id=project_id, projects=projects)
    result = understand(store, extractor, batch, user_id=user_id, project_id=project_id,
                        as_of=as_of, max_calls=max_calls, bootstrap=bootstrap, imported_at=imported_at,
                        retry_of=retry_of, on_progress=on_progress, model_selection=model_selection,
                        manual_selection=manual_selection, project_context=project_context,
                        should_stop=should_stop)
    if result['reason'] == 'background_context_saved':
        return {'understanding': result, 'event': None}
    deliveries = recovered + receive_pending(store, user_id=user_id, project_id=project_id, projects=projects)
    if delivery_window_end:
        deliveries = [delivery for delivery in deliveries
                      if (window := delivery['trigger_details'].get('window'))
                      and datetime.fromisoformat(window['until']).astimezone(TZ).date() == delivery_window_end]
    if require_review and (result.get('request') or (deliveries and result['status'] != 'failed')):
        return {'understanding': result, 'event': None, 'deliveries': deliveries,
                'review_required': True}
    request = result.get('request')
    if not result.get('dispatch_recommendation'):
        if request:
            # Repeated synchronization must never re-dispatch a model call.
            try:
                event = store.get('recommendation_event', event_id(request))
            except ValueError:
                event = None
            return {'understanding': result, 'event': event}
        status = result['status']
        if status not in ('skipped', 'failed', 'partial'):
            return {'understanding': result, 'event': None}
        reason = ('canceled_by_user' if (result.get('batch_gaps') or {}).get('stop_reason') == 'canceled_by_user' else
                  'image_analysis_pending' if result['reason'] == 'image_analysis_pending' else
                  'extraction_partial' if result['reason'] == 'extraction_partial' else
                  'model_balance_insufficient' if result['reason'] == 'model_balance_insufficient' else
                  'context_too_large' if result['reason'] == 'model_context_budget_exceeded' else
                  'budget_exhausted' if 'budget' in result['reason'] else
                  'no_imported_update' if result['reason'] == 'no_changed_imported_journals' else
                  'no_update' if result['reason'] == 'no_new_or_updated_journals_in_window' else
                  'source_failed' if result['reason'] == 'source_read_failure' else
                  'feishu_documents_unavailable' if result['reason'] == 'feishu_documents_unavailable' else
                  'initialization_required' if result['reason'] == 'bootstrap_required' else 'upstream_failed')
        versions = [{'source_id': r.source_id, 'content_version': r.content_version} for r in batch.records]
        gate_version = result.get('run_id') or sha256(json.dumps({
            'journals': sorted((v['source_id'], v['content_version']) for v in versions), 'sources': sorted((s.source_id, s.status) for s in batch.source_runs),
            'reason': result['reason'], 'configuration': result.get('execution_metadata', {}).get('configuration_version'),
            'allowed_calls': result.get('allowed_calls')}, sort_keys=True).encode()).hexdigest()
        snapshot = {**result, 'journal_input': batch.model_dump(mode='json'), 'journal_versions': versions}
        event = record_gate(store, user_id=user_id, projects=projects, project_id=project_id,
                            date=datetime.fromisoformat(as_of).astimezone(TZ).date().isoformat(),
                            source_version=gate_version, status=status,
                            input_snapshot=snapshot, reason=reason, upstream_reason=result['reason'])
        return {'understanding': result, 'event': event}
    refs = result['source_refs']
    if any(ref not in refs for q in request['research_questions'] for ref in q['source_refs']):
        raise ValueError('Unresolvable research question evidence')
    event = run_event(store, request, user_id=user_id, projects=projects,
                      discover=lambda req: discover(req, sources, source_names=source_names,
                                                     ranker=ranker, max_queries=24, store=store).model_dump(mode='json'),
                      analyze=analyze, basis={'source_refs': refs, 'trigger_details': result['trigger_details']})
    return {'understanding': result, 'event': event}


def direction_recommendation_history(store, request):
    from .discovery import paper_history_keys

    if request['mode'] != 'direction_daily':
        return ()
    keys = set()
    for row in store.select('recommendation_event', ('user_id', 'direction_id', 'mode', 'discovery'),
                            decode=('discovery',)):
        if (row['user_id'], row['direction_id']) != (request['user_id'], request['direction_id']):
            continue
        discovery = row['discovery'] or {}
        if discovery.get('request_id') == request['request_id']:
            continue
        selected = {r['paper_id'] for r in discovery.get('recommendations', [])}
        for paper in discovery.get('candidates', []):
            if paper['paper_id'] in selected:
                keys.update(paper_history_keys(paper))
    return tuple(sorted(keys))


def discover_direction(request, sources, *, source_names, ranker, store, web_fallback=None,
                       wechat_lookup=None, saved_wechat=None, on_progress=None, should_stop=None,
                       adaptive=None):
    if on_progress and wechat_lookup and not saved_wechat:
        on_progress('searching_wechat')
    runs, articles = saved_wechat or (wechat_lookup(request) if wechat_lookup and
                                      not (should_stop and should_stop()) else ([], []))
    leads = [{'source_ref': article['url'], 'identifiers': article['identifiers']}
             for article in articles if article['identifiers']]
    history = direction_recommendation_history(store, request)
    wire = discover(request, sources, source_names=source_names, ranker=ranker, max_queries=24,
                    store=store, web_fallback=web_fallback, media_leads=leads, on_progress=on_progress,
                    previously_recommended=history, should_stop=should_stop).model_dump(mode='json')
    if adaptive:
        wire = adaptive(request, wire, history)
    if runs:
        wire['source_runs'].extend(runs)
        wire['audit']['wechat_account_articles'] = articles
        if any(run['status'] in ('partial', 'failed') for run in runs):
            wire['warnings'].append('wechat_account_source_incomplete')
            if wire['status'] == 'completed':
                wire['status'] = 'partial'
    return wire


def direction(store, configuration, *, user_id, directions, projects, as_of, bootstrap,
              sources, ranker=None, analyze=None, source_names=('arxiv',), web_fallback=None, wechat_lookup=None,
              on_progress=None, should_stop=None, adaptive=None, synthesize=None):
    if configuration['user_id'] != user_id:
        raise PermissionError('Direction owner mismatch')
    req = direction_request(configuration, as_of=as_of, bootstrap=bootstrap)
    return run_event(store, req, user_id=user_id, projects=projects, directions=directions,
                     discover=lambda r: discover_direction(r, sources, source_names=source_names, ranker=ranker,
                         store=store, web_fallback=web_fallback, wechat_lookup=wechat_lookup, on_progress=on_progress,
                         should_stop=should_stop, adaptive=adaptive),
                     analyze=analyze, should_stop=should_stop, synthesize=synthesize,
                     basis={'direction_configuration': configuration})


def direction_request(configuration, *, as_of, bootstrap):
    clock = datetime.fromisoformat(as_of)
    if clock.tzinfo is None:
        raise ValueError('Timezone required')
    end = clock.astimezone(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    mode = 'direction_bootstrap' if bootstrap else 'direction_daily'
    retry_token = configuration.get('retry_token')
    retry_suffix = f":retry:{retry_token}" if retry_token else ''
    trigger = {'source_id': configuration['direction_id'], 'version': configuration['version'],
               'time': configuration['updated_at']}
    recent_days = 90
    constraints = configuration['constraints']
    if not bootstrap:
        constraints = {**constraints, 'window_policy': 'recent_90d', 'window_days': recent_days}
    return {'schema_version': '1', 'request_id': f"{configuration['direction_id']}:{mode}:{end.date()}:{configuration['version']}{retry_suffix}",
           'user_id': configuration['user_id'], 'mode': mode, 'project_id': configuration.get('project_id'),
           'direction_id': configuration['direction_id'], 'as_of': as_of, 'timezone': 'Asia/Shanghai',
           'trigger': trigger,
           'research_questions': configuration['research_questions'], 'queries': configuration['queries'],
           'constraints': constraints, 'max_recommendations': 5,
           'publication_window': None if bootstrap else {'since': (end-timedelta(days=recent_days)).isoformat(),
                                                       'until': end.isoformat(), 'date_basis': 'published_at'}}
