"""Immutable event snapshots; authorization is checked before any module call."""
import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx

from .contracts import AnalysisResult, Candidate, DiscoveryResult, RecommendationRequest
from .reading_evidence_gaps import evidence_gaps


def event_id(request):
    # Match run_event's wire representation before writing a worker checkpoint.
    # Pydantic serializes +00:00 as Z; hashing the raw payload points at a different row.
    request = RecommendationRequest.model_validate(request).model_dump(mode="json")
    identity = [request[k] for k in ("user_id", "project_id", "direction_id", "mode", "trigger")]
    if request["mode"] in ("direction_bootstrap", "direction_daily"):
        identity.append(request["request_id"])
    if request["mode"] == "direction_daily":
        identity.append(request["publication_window"])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]


def reviewed_event_id(store, request, *, automatic=False):
    key = event_id(request)
    if not automatic:
        return key
    rows = store.select('recommendation_event', ('input',), key=key, decode=('input',))
    questions = sorted(q['id'] for q in request['research_questions'])
    if rows and questions != sorted(q['id'] for q in (rows[0]['input'] or {}).get('research_questions', [])):
        # Keep the manually selected subset in history and search all questions separately.
        return hashlib.sha256(json.dumps([key, questions]).encode()).hexdigest()[:32]
    return key


def authorize(request, user_id, projects, directions):
    if request["user_id"] != user_id:
        raise PermissionError("Owner mismatch")
    if request["project_id"] and request["project_id"] not in projects:
        raise PermissionError("Project access required")
    if request["direction_id"] and request["direction_id"] not in directions:
        raise PermissionError("Direction access required")


def read_paper(analyze, paper, request, *, question_ids=None, journal_sources=None):
    questions = {q['id'] for q in request['research_questions']}
    if question_ids is not None:
        selected = set(question_ids)
        if not selected or not selected <= questions:
            raise ValueError('Analysis question mismatch')
        if selected != questions:
            constraints = dict(request['constraints'])
            if 'question_evidence_gates' in constraints:
                constraints['question_evidence_gates'] = {
                    qid: gate for qid, gate in constraints['question_evidence_gates'].items() if qid in selected
                }
            queries = request['queries']
            mapping = constraints.get('query_question_ids')
            if isinstance(mapping, dict):
                links = {query: [qid for qid in mapping[query] if qid in selected]
                         for query in queries if isinstance(mapping.get(query), list)}
                constraints['query_question_ids'] = {query: ids for query, ids in links.items() if ids}
                queries = [query for query in queries if query in constraints['query_question_ids']]
            request = {**request, 'queries': queries, 'constraints': constraints, 'research_questions': [
                q for q in request['research_questions'] if q['id'] in selected]}
        questions = selected
    if request['mode'] == 'progress' and journal_sources:
        # Only named excerpts from the already authorized event, with no links or locators.
        excerpts = {}
        remaining = 4800
        for question in request['research_questions']:
            quotes = []
            for ref in question['source_refs']:
                quote = (journal_sources.get(ref) or {}).get('quote', '').strip()
                if not quote or remaining <= 0:
                    continue
                excerpt = quote[:min(800, remaining)]
                if excerpt not in quotes:
                    quotes.append(excerpt)
                    remaining -= len(excerpt)
            if quotes:
                excerpts[question['id']] = quotes
        if excerpts:
            request = {**request, 'constraints': {**request['constraints'], '_journal_advice_excerpts': excerpts}}
    reading = AnalysisResult.model_validate(analyze(paper, request)).model_dump(mode='json')
    if (reading['request_id'], reading['user_id'], reading['paper_id'], reading['paper_version']) != (
            request['request_id'], request['user_id'], paper['paper_id'], paper['paper_version']):
        raise ValueError('Analysis identity mismatch')
    if not set(reading['relevance_to_questions']) <= questions:
        raise ValueError('Analysis question mismatch')
    return reading


def resolve_paper_version(store, event, paper, analyze):
    """Enrich only document identity at the reader boundary; keep original recommendation facts."""
    resolve = getattr(analyze, 'resolve_version', None)
    if paper['paper_version'] in paper['versions'] or resolve is None:
        return
    check = {'paper_id': paper['paper_id'], 'checked_at': datetime.now(UTC).isoformat()}
    try:
        wire = resolve(deepcopy(paper), deepcopy(event['input']))
        if hasattr(wire, 'model_dump'):
            wire = wire.model_dump(mode='json')
        updated = Candidate.model_validate(wire).model_dump(mode='json')
        mutable = {'paper_version', 'versions', 'fulltext_links', 'sources'}
        if any(updated[k] != v for k, v in paper.items() if k not in mutable):
            raise ValueError('Version resolution changed original recommendation facts')
        if updated['paper_version'] is None or updated['paper_version'] not in updated['versions']:
            check.update(status='unresolved', reason='version_evidence_unavailable')
        else:
            event.setdefault('version_resolution_originals', {}).setdefault(paper['paper_id'], deepcopy(paper))
            paper.update(updated)
            check.update(status='resolved', paper_version=paper['paper_version'])
    except httpx.HTTPError as exc:
        check.update(status='failed', reason='official_source_unavailable')
        if isinstance(exc, httpx.HTTPStatusError):
            check['http_status'] = exc.response.status_code
    except (ValueError, TypeError):
        check.update(status='failed', reason='version_evidence_invalid')
    except (OSError, RuntimeError):
        check.update(status='failed', reason='version_source_error')
    event.setdefault('version_checks', []).append(check)
    store.audit(event['event_id'], {'event': 'paper_version_resolution', **check})
    store.put('recommendation_event', event['event_id'], event)


def run_event(store, payload, *, user_id, projects, directions=(), discover, analyze=None, basis=None,
              should_stop=None, synthesize=None, automatic=False):
    request = RecommendationRequest.model_validate(payload).model_dump(mode="json")
    authorize(request, user_id, projects, directions)
    key = reviewed_event_id(store, request, automatic=automatic)
    # A unique database row acts as an idempotency lock, including across processes.
    event = {"event_id": key, "user_id": user_id, "project_id": request["project_id"],
             "created_at": request['as_of'],
             "direction_id": request["direction_id"], "mode": request["mode"],
             "date": datetime.fromisoformat(request["as_of"]).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat(),
             "status": "processing", "reason": "processing", "input": request,
             "discovery": None, "analyses": [], "errors": [], "basis": basis}
    with store.transaction() as db:
        inserted = db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)",
                              ("recommendation_event", key, json.dumps(event))).rowcount
    if not inserted:
        return store.get("recommendation_event", key)
    return _complete_event(store, event, discover=discover, analyze=analyze, should_stop=should_stop,
                           synthesize=synthesize)


def _mark_incomplete_synthesis(event):
    if (event['status'] == 'completed' and event.get('reason') == 'results'
            and (event.get('synthesis') or {}).get('status') == 'failed'):
        event.update(status='partial', reason='synthesis_incomplete')


def _complete_event(store, event, *, discover, analyze=None, should_stop=None, synthesize=None):
    request, key, user_id = event['input'], event['event_id'], event['user_id']
    try:
        result = DiscoveryResult.model_validate(discover(request)).model_dump(mode="json")
        if (result["request_id"], result["user_id"]) != (request["request_id"], user_id):
            raise ValueError("Discovery request mismatch")
        event["discovery"] = result
        store.put("recommendation_event", key, event)
        if should_stop and should_stop():
            event.update(status="partial", reason="canceled")
            event["warnings"] = list(dict.fromkeys([*event.get("warnings", []), "canceled_by_user"]))
            store.put("recommendation_event", key, event)
            return event
        papers = {p["paper_id"]: p for p in result["candidates"]}
        questions = {q["id"] for q in request["research_questions"]}
        question_refs = {q['id']: set(q['source_refs']) for q in request['research_questions']}
        for recommendation in result["recommendations"]:
            if should_stop and should_stop():
                event.update(status="partial", reason="canceled")
                event["warnings"] = list(dict.fromkeys([*event.get("warnings", []), "canceled_by_user"]))
                store.put("recommendation_event", key, event)
                return event
            paper = papers[recommendation["paper_id"]]
            if not set(recommendation["question_ids"]) <= questions:
                raise ValueError("Unknown question reference")
            for qid in recommendation['question_ids']:
                refs = recommendation['source_refs'].get(qid, [])
                if not refs or not set(refs) <= question_refs[qid]:
                    raise ValueError('Recommendation question evidence mismatch')
            if analyze:
                resolve_paper_version(store, event, paper, analyze)
                if paper['paper_version'] not in paper['versions']:
                    event['errors'].append({'paper_id': paper['paper_id'], 'reason': 'paper_version_unresolved'})
                else:
                    try:
                        reading = read_paper(analyze, paper, request, question_ids=recommendation['question_ids'],
                                             journal_sources=(event.get('basis') or {}).get('source_refs'))
                        event["analyses"].append(reading)
                    except Exception:  # noqa: BLE001 -- isolate third-party failures per paper
                        event["errors"].append({"paper_id": paper["paper_id"], "reason": "analysis_failed"})
                store.put("recommendation_event", key, event)
        if should_stop and should_stop():
            event.update(status="partial", reason="canceled")
            event["warnings"] = list(dict.fromkeys([*event.get("warnings", []), "canceled_by_user"]))
            store.put("recommendation_event", key, event)
            return event
        if synthesize:
            try:
                if summary := synthesize(request, event["analyses"]):
                    event["synthesis"] = summary
            except Exception:  # noqa: BLE001 -- optional synthesis must not discard validated results
                event["synthesis"] = {"status": "failed", "reason": "synthesis_failed", "insights": []}
        if should_stop and should_stop():
            event.update(status="partial", reason="canceled")
            event["warnings"] = list(dict.fromkeys([*event.get("warnings", []), "canceled_by_user"]))
            store.put("recommendation_event", key, event)
            return event
        event["status"] = result["status"]
        event["reason"] = ("no_results" if not result["recommendations"] else "results") if result["status"] == "completed" else "discovery_incomplete"
        if (result["status"] == "completed" and result["recommendations"]
                and (not analyze or event["errors"] or any(a["status"] != "completed" for a in event["analyses"]))):
            event.update(status="partial", reason="analysis_incomplete")
        if any('model_budget_exhausted' in a['warnings'] for a in event['analyses']):
            event.update(status='partial', reason='budget_exhausted')
        _mark_incomplete_synthesis(event)
    except Exception:  # noqa: BLE001 -- persist a safe failure at the module boundary
        event.update(status="failed", reason="module_failed")
    store.put("recommendation_event", key, event)
    return event


def only_unresolved_recommendations(event):
    discovery = event.get('discovery') or {}
    selected = {r['paper_id'] for r in discovery.get('recommendations', [])}
    unresolved = {e['paper_id'] for e in event.get('errors', []) if e['reason'] == 'paper_version_unresolved'}
    papers = {p['paper_id']: p for p in discovery.get('candidates', [])}
    return bool(selected) and not event['analyses'] and selected <= unresolved and all(
        pid in papers and papers[pid]['paper_version'] not in papers[pid]['versions'] for pid in selected)


def has_resolvable_versions(event, *, open_access=False):
    """Supported publisher identity only; no promise of successful official access."""
    from .pmc_open_pdf import supports_open_pdf
    from .recommendation_reader import crossref_aircc_pdf_url

    selected = {r['paper_id'] for r in event['discovery']['recommendations']}
    return any(p['paper_id'] in selected and p['paper_version'] not in p['versions'] and (
               (re.fullmatch(r'doi:10\.3390/[a-z0-9.-]+', p['paper_id']) and p['paper_id'] == 'doi:'+str(p['doi']))
               or any(s.get('selected_document') and str(s.get('source', '')).startswith(
                   ('official-rss:', 'pmlr:', 'neurips:', 'journal-detail:')) for s in p.get('sources', []))
               or crossref_aircc_pdf_url(p)
               or open_access and supports_open_pdf(p))
               for p in event['discovery']['candidates'])


def recover_discovery(store, key, *, user_id, projects, directions=(), discover, analyze=None):
    """Retry explicit unsuccessful discovery while retaining every previous attempt."""
    event = store.get('recommendation_event', key)
    request = RecommendationRequest.model_validate(event['input']).model_dump(mode='json')
    authorize(request, user_id, projects, directions)
    attempts = event.get('discovery_recovery_attempts', 0)
    old = event['discovery']
    if (event['status'] not in ('failed', 'partial') or event['analyses']
            or (old and ((old['recommendations'] and not only_unresolved_recommendations(event))
                         or old['status'] == 'completed'))):
        raise ValueError('Only unsuccessful discovery without paper results can be retried')
    with store.transaction() as db:
        claimed = db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)',
            ('recommendation_discovery_revision', f'{key}:{attempts}', json.dumps(event))).rowcount
    if not claimed:
        raise ValueError('Discovery recovery already claimed; inspect its checkpoint')
    event.update(status='processing', reason='recovering_discovery', discovery=None, errors=[],
                 discovery_recovery_attempts=attempts+1)
    store.put('recommendation_event', key, event)
    recovered = _complete_event(store, event, discover=discover, analyze=analyze)
    recovered['discovery_recovered_at'] = datetime.now(UTC).isoformat()
    store.put('recommendation_event', key, recovered)
    return recovered


def recover_analysis(store, key, *, user_id, projects, directions=(), analyze, max_attempts=None, report=None):
    """Recover unfinished readings explicitly, retaining useful results and prior attempts."""
    event = store.get('recommendation_event', key)
    request = event['input']
    authorize(request, user_id, projects, directions)
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
        raise ValueError('Recovery attempts must be positive')
    attempts = event.get('recovery_attempts', 0)
    if event['reason'] == 'recovering':
        raise ValueError('Recovery in progress or interrupted; inspect its checkpoint')
    if max_attempts is not None and attempts >= max_attempts:
        raise ValueError('Recovery attempt budget exhausted')
    if not event['discovery']:
        raise ValueError('No discovery checkpoint; do not retry downstream')
    def needs_reading(a):
        if a['status'] == 'partial' and not a['claims'] and 'semantic_review_unavailable' in a['warnings']:
            return True
        return a['status'] in ('failed', 'unavailable') or (a['status'] == 'partial' and not a['claims']
            and any(w.startswith(('model_reading_not_run;', 'model_budget_exhausted', 'stage_failed:',
                                  'input_budget_exceeded;')) for w in a['warnings']))
    kept = [a for a in event['analyses'] if not needs_reading(a)]
    finished = {a['paper_id'] for a in kept}
    papers = {p['paper_id']: p for p in event['discovery']['candidates']}
    question_ids = {r['paper_id']: r.get('question_ids') for r in event['discovery']['recommendations']}
    pending = [papers[pid] for pid in question_ids if pid not in finished]
    if not pending:
        return event
    if getattr(analyze, 'resolve_version', None) is None:
        pending = [p for p in pending if p['paper_version'] in p['versions']]
    if not pending:
        raise ValueError('论文版本尚未核实，暂不能补读；请先核对原始论文资料。')
    # INSERT establishes a cross-process claim before the first explicit callback.
    with store.transaction() as db:
        inserted = db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)',
                              ('recommendation_revision', f'{key}:{attempts}', json.dumps(event))).rowcount
    if not inserted:
        raise ValueError('Recovery already claimed; inspect its checkpoint')
    event.update(analyses=kept, recovery_attempts=attempts+1, status='processing', reason='recovering')
    store.put('recommendation_event', key, event)
    if report:
        report(0, len(pending))
    for done, paper in enumerate(pending, 1):
        event['errors'] = [e for e in event['errors'] if e['paper_id'] != paper['paper_id']]
        resolve_paper_version(store, event, paper, analyze)
        if paper['paper_version'] not in paper['versions']:
            event['errors'].append({'paper_id': paper['paper_id'], 'reason': 'paper_version_unresolved'})
        else:
            try:
                event['analyses'].append(read_paper(
                    analyze, paper, request, question_ids=question_ids[paper['paper_id']],
                    journal_sources=(event.get('basis') or {}).get('source_refs')))
            except Exception:  # noqa: BLE001 -- preserve other successful paper checkpoints
                event['errors'].append({'paper_id': paper['paper_id'], 'reason': 'analysis_failed'})
        store.put('recommendation_event', key, event)
        if report:
            report(done, len(pending))
    complete = (event['discovery']['status'] == 'completed' and not event['errors']
                and all(a['status'] == 'completed' for a in event['analyses']))
    event.update(status='completed' if complete else 'partial', reason='results' if complete else 'analysis_incomplete')
    _mark_incomplete_synthesis(event)
    event['recovered_at'] = datetime.now(UTC).isoformat()
    store.put('recommendation_event', key, event)
    return event


def record_gate(store, *, user_id, projects, project_id, date, source_version, status, input_snapshot,
                reason=None, upstream_reason=None):
    """Record upstream no-update/read-failure without invoking discovery or analysis."""
    if status not in ("skipped", "failed", "partial"):
        raise ValueError("Gate must explicitly describe skipped, failed or partial input")
    if project_id not in projects:
        raise PermissionError("Project access required")
    key = hashlib.sha256(json.dumps([user_id, project_id, date, source_version]).encode()).hexdigest()[:32]
    event = {"event_id": key, "user_id": user_id, "project_id": project_id, "direction_id": None,
             "created_at": input_snapshot.get('as_of', date+'T00:00:00+08:00'),
             "mode": "progress", "date": date, "status": status,
             "reason": reason or ("no_update" if status == "skipped" else "source_failed"),
             "upstream_reason": upstream_reason,
             "input": input_snapshot, "discovery": None, "analyses": [], "errors": []}
    with store.transaction() as db:
        db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)',
                   ('recommendation_event', key, json.dumps(event)))
    return store.get('recommendation_event', key)


def visible_events(store, *, user_id, projects):
    from .recommendation_jobs import worker_status
    from .scope_deletion import deleted_scopes, scope_deleted

    worker = worker_status(store)
    open_access = worker['status'] == 'ready' and worker.get('open_access_fulltexts_enabled', False)
    deleted = deleted_scopes(store)
    rows = [e for e in store.list("recommendation_event")
            if e["user_id"] == user_id and not scope_deleted(e, deleted)
            and (not e["project_id"] or projects is None or e["project_id"] in projects)]
    for event in rows:
        if event.get('discovery') and has_resolvable_versions(event, open_access=open_access):
            event['version_resolution_available'] = True
        if event.get('recovery_attempts'):
            original = store.get('recommendation_revision', event['event_id']+':0')
            if original['user_id'] != user_id or original['project_id'] != event['project_id']:
                raise PermissionError('Event revision owner mismatch')
            event['before_recovery'] = original
        if event.get('discovery_recovery_attempts'):
            original = store.get('recommendation_discovery_revision', event['event_id']+':0')
            if original['user_id'] != user_id or original['project_id'] != event['project_id']:
                raise PermissionError('Discovery revision owner mismatch')
            event['before_discovery_recovery'] = original
    return sorted(rows, key=lambda e: (datetime.fromisoformat(
        e.get('created_at', e['date']+'T00:00:00+08:00')).timestamp(), e["event_id"]), reverse=True)


def member_event(event):
    row = dict(event)
    _mark_incomplete_synthesis(row)
    internal = {'required_calls', 'allowed_calls', 'model_calls', 'execution_metadata', 'provider_audit_refs',
                'validation_error'}
    row['input'] = {key: value for key, value in (event.get('input') or {}).items() if key not in internal}
    if discovery := event.get('discovery'):
        audit = discovery.get('audit', {})
        coverage = ('official_source_plan', 'journal_source_plan', 'web_fallback_runs', 'wechat_account_articles')
        row['discovery'] = {**discovery, 'audit': {key: audit[key] for key in coverage if key in audit}}
    if analyses := event.get('analyses'):
        questions = {r['paper_id']: r.get('question_ids', ()) for r in
                     (event.get('discovery') or {}).get('recommendations', [])}
        row['analyses'] = [
            {**analysis, 'reading_scope': {**(analysis.get('reading_scope') or {}),
             'evidence_gaps': evidence_gaps(analysis, questions.get(analysis['paper_id'], ()))}}
            for analysis in analyses
        ]
    for key in ('before_recovery', 'before_discovery_recovery'):
        if key in event:
            row[key] = member_event(event[key])
    return row
