import json
from pathlib import Path

import pytest

from neuroei.contracts import RecommendationRequest
from neuroei.recommendation_events import (
    record_gate,
    recover_analysis,
    recover_discovery,
    run_event,
    visible_events,
)
from neuroei.store import Store


def request():
    return json.loads(Path('tests/fixtures/paper_analysis_input.json').read_text(encoding='utf-8'))['request']


def empty(req):
    return {'schema_version': '1', 'request_id': req['request_id'], 'status': 'completed',
            'candidates': [], 'recommendations': [], 'source_runs': [], 'warnings': [],
            'user_id': req['user_id'], 'decisions': [], 'audit': {}}


def test_duplicate_and_snapshot(tmp_path):
    store = Store(tmp_path)
    calls = []
    req = request()
    def discover(r):
        calls.append(r)
        return empty(r)
    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=discover)
    req['research_questions'][0]['description'] = 'changed after event'
    second = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=discover)
    assert first == second
    assert len(calls) == 1
    assert first['status'] == 'completed' and first['reason'] == 'no_results'
    assert visible_events(Store(tmp_path), user_id=req['user_id'], projects=['demo']) == [first]
    assert visible_events(store, user_id='another', projects=['demo']) == []
    assert visible_events(store, user_id=req['user_id'], projects=[]) == []


def test_raw_request_checkpoint_matches_persisted_event_after_time_serialization(tmp_path):
    from neuroei.recommendation_events import event_id

    req = request()
    req['trigger']['time'] = '2026-11-03T01:00:00+00:00'
    checkpoint = event_id(req)
    event = run_event(Store(tmp_path), req, user_id=req['user_id'], projects=['demo'], discover=empty)
    assert event['input']['trigger']['time'] == '2026-11-03T01:00:00Z'
    assert checkpoint == event['event_id']


@pytest.mark.parametrize('owner,projects', [('another', ['demo']), ('offline-reviewer', [])])
def test_unauthorized_never_calls(tmp_path, owner, projects):
    with pytest.raises(PermissionError):
        run_event(Store(tmp_path), request(), user_id=owner, projects=projects,
                  discover=lambda r: pytest.fail('unauthorized call'))


def test_contract_rejects_naive_time_and_wrong_version():
    for field, value in [('as_of', '2026-11-03T22:00:00'), ('schema_version', '2')]:
        req = request()
        req[field] = value
        with pytest.raises(ValueError):
            RecommendationRequest.model_validate(req)


def test_missing_analysis_retains_paper(tmp_path):
    req = request()
    result = empty(req)
    result['candidates'] = [{'paper_id': 'doi:example', 'paper_version': 'v1', 'title': 'Example',
        'authors': [], 'doi': None, 'arxiv_id': None, 'venue': None, 'published_at': None,
        'date_basis': None, 'abstract': None, 'original_links': [], 'fulltext_links': [], 'sources': [], 'versions': ['v1']}]
    result['recommendations'] = [{'paper_id': 'doi:example', 'question_ids': ['q1'], 'reason': 'specific',
        'basis': 'title', 'ranking_basis': {}, 'source_refs': {'q1': ['offline-fixture:q1']}}]
    event = run_event(Store(tmp_path), req, user_id=req['user_id'], projects=['demo'], discover=lambda r: result)
    assert event['status'] == 'partial'
    assert len(event['discovery']['recommendations']) == 1
    assert not event['analyses']


def test_module_failure_is_not_empty_success(tmp_path):
    def failed(r):
        raise RuntimeError('secret provider exception')
    req = request()
    event = run_event(Store(tmp_path), req, user_id=req['user_id'], projects=['demo'], discover=failed)
    assert event['status'] == 'failed'
    assert 'secret' not in json.dumps(event)


def test_gate_distinguishes_failure_and_no_update(tmp_path):
    store = Store(tmp_path)
    args = {'user_id': 'u', 'projects': ['p'], 'project_id': 'p', 'date': '2026-11-03', 'input_snapshot': {}}
    skipped = record_gate(store, **args, source_version='v1', status='skipped')
    failed = record_gate(store, **args, source_version='v2', status='failed')
    assert skipped['reason'] == 'no_update'
    assert failed['reason'] == 'source_failed'
    assert len(visible_events(store, user_id='u', projects=['p'])) == 2
    repeated = record_gate(store, **{**args, 'input_snapshot': {'as_of': '2026-11-03T12:00:00+08:00'}},
                           source_version='v1', status='skipped')
    assert repeated == skipped  # Preserve the first snapshot when checked again later that day.


def test_direction_daily_windows_have_independent_events(tmp_path):
    store = Store(tmp_path)
    req = request()
    req.update(mode='direction_daily', direction_id='d', project_id=None)
    results = []
    for since, until in [('2026-11-01', '2026-11-02'), ('2026-11-02', '2026-11-03')]:
        req['publication_window'] = {'since': since+'T00:00:00+08:00',
                                     'until': until+'T00:00:00+08:00', 'date_basis': 'published_at'}
        results.append(run_event(store, req, user_id=req['user_id'], projects=[], directions=['d'], discover=empty))
    assert results[0]['event_id'] != results[1]['event_id']
    with pytest.raises(PermissionError):
        run_event(store, req, user_id=req['user_id'], projects=[], directions=[], discover=empty)


def test_partial_ranking_is_not_scientific_zero(tmp_path):
    req = request()
    result = empty(req)
    result.update(status='partial', warnings=['ranking_unavailable'])
    event = run_event(Store(tmp_path), req, user_id=req['user_id'], projects=['demo'], discover=lambda r: result)
    assert event['status'] == 'partial'
    assert event['reason'] == 'discovery_incomplete'


@pytest.mark.parametrize('initial_unavailable', [False, True, 'parsed_only'])
def test_recovery_does_not_rediscover_or_drop_prior_results(tmp_path, initial_unavailable):
    store = Store(tmp_path)
    req = request()
    result = empty(req)
    candidate = json.loads(Path('examples/discovery/output.json').read_text(encoding='utf-8'))['candidates'][0]
    second = {**candidate, 'paper_id': 'doi:another'}
    result['candidates'] = [candidate, second]
    result['recommendations'] = [{'paper_id': p['paper_id'], 'question_ids': ['q1'], 'reason': 'fixture',
        'basis': 'title', 'ranking_basis': {}, 'source_refs': {'q1': req['research_questions'][0]['source_refs']}}
        for p in result['candidates']]
    def reading(p, r):
        out = json.loads(Path('tests/fixtures/contracts_v1/AnalysisResult.json').read_text(encoding='utf-8'))
        return {**out, 'status': 'partial', 'paper_id': p['paper_id'], 'paper_version': p['paper_version']}
    def initial(p, r):
        if p['paper_id'] == second['paper_id']:
            if initial_unavailable:
                from neuroei.paper_analysis import analyze

                output = analyze(p, r, cache_dir=tmp_path/'no-fulltext')
                if initial_unavailable == 'parsed_only':
                    output.update(status='partial', warnings=['model_reading_not_run; parsing/retrieval only'])
                return output
            raise RuntimeError('injected single paper failure')
        return reading(p, r)
    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=lambda r: result, analyze=initial)
    assert len(first['analyses']) == (2 if initial_unavailable else 1)
    assert len(first['errors']) == (0 if initial_unavailable else 1)
    called = []
    def retry(p, r):
        called.append(p['paper_id'])
        return reading(p, r)
    recovered = recover_analysis(store, first['event_id'], user_id=req['user_id'], projects=['demo'], analyze=retry)
    assert called == [second['paper_id']]
    assert recovered['analyses'][0] == first['analyses'][0]
    assert len(recovered['analyses']) == 2 and not recovered['errors']
    assert store.get('recommendation_revision', first['event_id']+':0') == first
    with pytest.raises(ValueError, match='budget'):
        recover_analysis(store, first['event_id'], user_id=req['user_id'], projects=['demo'], analyze=retry, max_attempts=1)


def test_each_paper_reads_only_associated_questions_before_and_after_recovery(tmp_path):
    store, req = Store(tmp_path), request()
    ids = [f'q{i}' for i in range(1, 11)]
    req['research_questions'] = [{**req['research_questions'][0], 'id': qid, 'source_refs': [qid + ':note']}
                                 for qid in ids]
    req['queries'] = [f'query for {qid}' for qid in ids]
    req['constraints'] = {
        'query_question_ids': dict(zip(req['queries'], ([qid] for qid in ids))),
        'question_evidence_gates': {qid: {'source_refs': [qid + ':note'], 'query_refs': [query],
            'all_of': [{'any_of': ['robot']}, {'any_of': ['control']}]} for qid, query in zip(ids, req['queries'])},
    }
    candidate = json.loads(Path('examples/discovery/output.json').read_text(encoding='utf-8'))['candidates'][0]
    other = {**candidate, 'paper_id': 'doi:other'}
    associated = {candidate['paper_id']: ['q1', 'q7'], other['paper_id']: ['q10']}
    result = empty(req)
    result['candidates'] = [candidate, other]
    result['recommendations'] = [{'paper_id': pid, 'question_ids': qids, 'reason': 'fixture',
        'basis': 'title', 'ranking_basis': {}, 'source_refs': {qid: [qid + ':note'] for qid in qids}}
        for pid, qids in associated.items()]
    calls = []

    def reading(paper, scoped_request):
        scoped = RecommendationRequest.model_validate(scoped_request)
        qids = associated[paper['paper_id']]
        assert [question.id for question in scoped.research_questions] == qids
        assert scoped.queries == [f'query for {qid}' for qid in qids]
        assert set(scoped.constraints['question_evidence_gates']) == set(qids)
        calls.append(paper['paper_id'])
        out = json.loads(Path('tests/fixtures/contracts_v1/AnalysisResult.json').read_text(encoding='utf-8'))
        return {**out, 'status': 'completed', 'paper_id': paper['paper_id'], 'paper_version': paper['paper_version'],
                'reading_scope': {'documents': ['offline-fixture']}}

    def initial(paper, scoped_request):
        out = reading(paper, scoped_request)
        if paper['paper_id'] == other['paper_id']:
            out['relevance_to_questions'] = {'q1': []}
        return out

    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=lambda _: result, analyze=initial)
    assert ([q['id'] for q in first['input']['research_questions']], first['input']['queries']) == (ids, req['queries'])
    assert [analysis['paper_id'] for analysis in first['analyses']] == [candidate['paper_id']]
    assert first['errors'] == [{'paper_id': other['paper_id'], 'reason': 'analysis_failed'}]
    recovered = recover_analysis(store, first['event_id'], user_id=req['user_id'], projects=['demo'], analyze=reading)
    assert calls == [candidate['paper_id'], other['paper_id'], other['paper_id']]
    assert recovered['analyses'][0] == first['analyses'][0]
    assert recovered['status'] == 'completed' and not recovered['errors']


@pytest.mark.parametrize('still_failed', [False, True])
def test_discovery_recovery_keeps_failed_snapshot_and_original_request(tmp_path, still_failed):
    store, req = Store(tmp_path), request()
    failed = {**empty(req), 'status': 'failed', 'warnings': ['source unavailable']}
    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=lambda _: failed)
    calls = []
    def retry(r):
        calls.append(r)
        return failed if still_failed else empty(r)
    with pytest.raises(PermissionError):
        recover_discovery(store, first['event_id'], user_id='another', projects=['demo'], discover=retry)
    assert not calls
    recovered = recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'], discover=retry)
    assert calls == [first['input']]
    assert recovered['event_id'] == first['event_id'] and recovered['input'] == first['input']
    assert recovered['status'] == ('failed' if still_failed else 'completed')
    assert recovered['discovery_recovered_at']
    assert store.get('recommendation_discovery_revision', first['event_id']+':0') == first
    shown = visible_events(store, user_id=req['user_id'], projects=['demo'])[0]
    assert shown['before_discovery_recovery'] == first
    assert 'before_discovery_recovery' not in store.get('recommendation_event', first['event_id'])
    assert run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=retry) == recovered
    assert len(calls) == 1
    if still_failed:
        second = recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'], discover=retry)
        assert second['discovery_recovery_attempts'] == 2 and len(calls) == 2
        assert store.get('recommendation_discovery_revision', first['event_id']+':1') == recovered
        assert store.get('recommendation_discovery_revision', first['event_id']+':0') == first
    else:
        with pytest.raises(ValueError, match='without paper results'):
            recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'], discover=retry)
        assert len(calls) == 1


def test_discovery_recovery_snapshots_only_unresolved_initial_selections(tmp_path):
    store, req = Store(tmp_path), request()
    output = empty(req)
    candidate = json.loads(Path('examples/discovery/output.json').read_text(encoding='utf-8'))['candidates'][0]
    output.update(status='partial', candidates=[{**candidate, 'paper_version': None, 'versions': []}],
                  recommendations=[{'paper_id': candidate['paper_id'], 'question_ids': ['q1'],
                      'reason': 'Title-only lead', 'basis': 'title', 'ranking_basis': {},
                      'source_refs': {'q1': req['research_questions'][0]['source_refs']}}])
    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=lambda _: output,
                      analyze=lambda *_: pytest.fail('Unresolved version reached reading'))
    assert first['errors'] == [{'paper_id': candidate['paper_id'], 'reason': 'paper_version_unresolved'}]
    seen = []
    def retry(original):
        seen.append(original)
        return empty(req)
    recovered = recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'], discover=retry)
    assert seen == [first['input']]
    assert recovered['input'] == first['input'] and recovered['errors'] == []
    original = store.get('recommendation_discovery_revision', first['event_id']+':0')
    assert original == first and original['discovery']['recommendations']
    assert original['errors'][0]['reason'] == 'paper_version_unresolved'
    with pytest.raises(ValueError, match='without paper results'):
        recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'], discover=retry)


def test_discovery_recovery_never_reranks_existing_recommendations_or_normal_zero(tmp_path):
    store, req = Store(tmp_path), request()
    first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=empty)
    with pytest.raises(ValueError, match='without paper results'):
        recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'],
                          discover=lambda _: pytest.fail('normal zero was retried'))
    # A previously selected paper is retained, even when the surrounding run failed.
    first.update(status='failed', discovery={**first['discovery'], 'status': 'failed',
                                            'recommendations': [{'paper_id': 'kept'}]})
    store.put('recommendation_event', first['event_id'], first)
    with pytest.raises(ValueError, match='without paper results'):
        recover_discovery(store, first['event_id'], user_id=req['user_id'], projects=['demo'],
                          discover=lambda _: pytest.fail('existing paper was reranked'))
    assert store.get('recommendation_event', first['event_id']) == first


@pytest.mark.parametrize('retry_succeeds', [False, True])
def test_zero_claim_stage_failure_keeps_other_readings_history_and_bounded_recovery(tmp_path, retry_succeeds):
    """Production failure shape replay; synthetic claims test persistence, not scientific quality."""
    store, req = Store(tmp_path), request()
    result = empty(req)
    candidate = json.loads(Path('examples/discovery/output.json').read_text(encoding='utf-8'))['candidates'][0]
    good_before, bad, good_after, unresolved = ['doi:good-before', 'doi:stage-failed', 'doi:good-after', 'doi:unresolved']
    result.update(status='partial', warnings=['source coverage incomplete'])
    result['candidates'] = [{**candidate, 'paper_id': pid} for pid in (good_before, bad, good_after)]
    result['candidates'].append({**candidate, 'paper_id': unresolved, 'paper_version': None, 'versions': []})
    result['recommendations'] = [{'paper_id': p['paper_id'], 'question_ids': ['q1'], 'reason': 'offline boundary fixture',
        'basis': 'title', 'ranking_basis': {}, 'source_refs': {'q1': req['research_questions'][0]['source_refs']}}
        for p in result['candidates']]
    calls = []

    def reading(p, r, *, failed=False):
        out = json.loads(Path('tests/fixtures/contracts_v1/AnalysisResult.json').read_text(encoding='utf-8'))
        out.update(request_id=r['request_id'], user_id=r['user_id'], paper_id=p['paper_id'],
                   paper_version=p['paper_version'], status='partial',
                   reading_scope={'documents': ['offline-document:'+p['paper_id']], 'fulltext_acquisition': 'available'},
                   warnings=['stage_failed:ValidationError'] if failed else ['synthetic reading; semantics not assessed'])
        if not failed:
            text = 'The offline fixture method is simple.'
            out.update(claims=[{'claim_id': 'c1', 'text': text, 'supports': [{'evidence_id': 'e1', 'quote': text}]}],
                       evidence=[{'evidence_id': 'e1', 'paper_id': p['paper_id'], 'paper_version': p['paper_version'],
                                  'document_id': 'offline-document:'+p['paper_id'], 'block_id': 'paragraph:1', 'text': text}])
        return out

    def initial(p, r):
        calls.append(p['paper_id'])
        return reading(p, r, failed=p['paper_id'] == bad)

    try:
        first = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=lambda _: result, analyze=initial)
        assert calls == [good_before, bad, good_after]  # A returned failed reading cannot stop later papers.
        assert first['status'] == 'partial'
        readings = {a['paper_id']: a for a in first['analyses']}
        assert len(readings) == 3 and len(first['discovery']['recommendations']) == 4
        assert readings[bad]['reading_scope']['fulltext_acquisition'] == 'available'
        assert readings[bad]['claims'] == [] and readings[bad]['warnings'] == ['stage_failed:ValidationError']
        assert all(readings[pid]['claims'] and readings[pid]['evidence'] for pid in (good_before, good_after))
        assert first['errors'] == [{'paper_id': unresolved, 'reason': 'paper_version_unresolved'}]
        store.close()
        store = Store(tmp_path)
        assert visible_events(store, user_id=req['user_id'], projects=['demo']) == [first]
        assert run_event(store, req, user_id=req['user_id'], projects=['demo'],
                         discover=lambda _: pytest.fail('Refresh rediscovered'), analyze=initial) == first
        assert calls == [good_before, bad, good_after]
        retried = []

        def retry(p, r):
            retried.append(p['paper_id'])
            assert store.get('recommendation_revision', first['event_id']+':0') == first
            return reading(p, r, failed=not retry_succeeds)

        recovered = recover_analysis(store, first['event_id'], user_id=req['user_id'], projects=['demo'], analyze=retry)
        assert retried == [bad]  # Kept claims and unresolved versions must never consume reader calls.
        assert recovered['input'] == first['input'] and recovered['discovery'] == first['discovery']
        assert recovered['status'] == 'partial' and recovered['errors'] == first['errors']
        after = {a['paper_id']: a for a in recovered['analyses']}
        assert all(after[pid] == readings[pid] for pid in (good_before, good_after))
        assert bool(after[bad]['claims']) is retry_succeeds
        store.close()
        store = Store(tmp_path)
        shown = visible_events(store, user_id=req['user_id'], projects=['demo'])[0]
        assert shown['before_recovery'] == first and shown['analyses'] == recovered['analyses']
        assert shown['before_recovery']['analyses'][1]['claims'] == []
        assert visible_events(store, user_id='another', projects=['demo']) == []
        with pytest.raises(ValueError, match='budget'):
            recover_analysis(store, first['event_id'], user_id=req['user_id'], projects=['demo'], analyze=retry, max_attempts=1)
        assert retried == [bad]
        assert run_event(store, req, user_id=req['user_id'], projects=['demo'],
                         discover=lambda _: pytest.fail('Refresh rediscovered'), analyze=initial) == recovered
    finally:
        store.close()



def test_optional_synthesis_failure_preserves_validated_event(tmp_path):
    req = request()

    def broken_synthesis(*_):
        raise RuntimeError('Optional provider unavailable')

    store = Store(tmp_path)
    result = run_event(store, req, user_id=req['user_id'], projects=['demo'], discover=empty,
                       synthesize=broken_synthesis)
    assert result['status'] == 'completed'
    assert result['reason'] == 'no_results'
    assert result['synthesis']['status'] == 'failed'
    assert store.get('recommendation_event', result['event_id']) == result


@pytest.mark.parametrize('unexpected_exception', [False, True])
def test_failed_synthesis_marks_event_partial_and_retains_completed_readings(tmp_path, unexpected_exception):
    from neuroei.project_daily_email import _body
    from neuroei.recommendation_events import member_event

    req = request()
    discovery = empty(req)
    candidate = json.loads(Path('examples/discovery/output.json').read_text(encoding='utf-8'))['candidates'][0]
    discovery['candidates'] = [candidate, {**candidate, 'paper_id': 'doi:another'}]
    discovery['recommendations'] = [
        {'paper_id': p['paper_id'], 'question_ids': ['q1'], 'reason': 'fixture', 'basis': 'title',
         'ranking_basis': {}, 'source_refs': {'q1': req['research_questions'][0]['source_refs']}}
        for p in discovery['candidates']
    ]
    fixture = json.loads(Path('tests/fixtures/contracts_v1/AnalysisResult.json').read_text(encoding='utf-8'))

    def read(paper, _request):
        return {**fixture, 'paper_id': paper['paper_id'], 'paper_version': paper['paper_version'],
                'status': 'completed', 'reading_scope': {'documents': ['fixture']}, 'warnings': []}

    def synthesize(_request, analyses):
        assert [a['status'] for a in analyses] == ['completed', 'completed']
        if unexpected_exception:
            raise RuntimeError('provider details')
        return {'status': 'failed', 'reason': 'synthesis_model_failed', 'insights': []}

    store = Store(tmp_path)
    result = run_event(store, req, user_id=req['user_id'], projects=['demo'],
                       discover=lambda _: discovery, analyze=read, synthesize=synthesize)
    assert (result['status'], result['reason']) == ('partial', 'synthesis_incomplete')
    assert [a['status'] for a in result['analyses']] == ['completed', 'completed']
    assert result['synthesis']['status'] == 'failed'
    assert store.get('recommendation_event', result['event_id']) == result
    previous = {**result, 'status': 'completed', 'reason': 'results'}
    assert member_event(previous)['status'] == 'partial'
    assert previous['status'] == 'completed'
    email = _body({'title': 'fixture'}, result, {}, 'https://example.invalid')
    assert '本次状态：部分完成' in email
    assert '跨论文综合未完成，单篇深读结果已保留。' in email
    assert 'provider details' not in email
