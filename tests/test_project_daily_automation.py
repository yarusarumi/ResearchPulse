"""Exercise the 05:00 daily chain with an isolated store and no external services."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date
from threading import Event

import pytest

from neuroei import accounts, project_daily_email, project_daily_schedule, recommendation_jobs as jobs
from neuroei.project_understanding import JournalBatch, identity
from neuroei.recommendation_handoff import dispatch_reviewed
from neuroei.recommendation_pipeline import progress
from neuroei.recommendation_worker import execution
from neuroei.research_workspace import create_project, update_project_automation
from neuroei.store import Store
from test_recommendation_events import empty, request

STAMP = '2026-11-19T05:00:00+08:00'


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr('neuroei.feishu_user_auth.current_grant_id', lambda *args: 'test-grant')
    monkeypatch.setattr('neuroei.feishu_folders.freeze_folder', lambda *args: [{'token': 'journal'}])
    with closing(Store(tmp_path)) as store:
        account = accounts.create(store, 'researcher', 'Researcher', 'test-password-123', email='qa@example.invalid')
        project = create_project(store, account, {'title': 'QA project', 'background': 'Example context'})
        pid = project['project_id']
        row = store.get('research_project', pid)
        store.put('research_project', pid, {**row, 'feishu': {
            'url': 'https://example.feishu.cn/drive/folder/journal', 'selection': [],
        }})
        store.put('understanding_head', identity([account['user_id'], pid]), {'key': 'initialized'})
        update_project_automation(store, account, {'project_id': pid, 'enabled': True})
        yield store, account, pid


@pytest.fixture
def mailbox(monkeypatch):
    messages = []

    class Mailer:
        origin = 'https://example.invalid'

        def send_daily_report(self, *message):
            messages.append(message)

    monkeypatch.setattr(project_daily_email.account_email.SMTPMailer, 'configured', Mailer)
    return messages


def schedule(store, stamp=STAMP, **options):
    return project_daily_schedule.run(store, as_of=stamp, **options)['projects'][0]


def delivery(store, account, project):
    payload = {**request(), 'request_id': 'daily-analysis', 'as_of': STAMP,
               'user_id': account['user_id'], 'project_id': project}
    other = {**payload['research_questions'][0], 'id': 'q2'}
    payload['research_questions'] = [*payload['research_questions'], other]
    payload['queries'] = ['robot control']
    payload['constraints'] = {**payload['constraints'], 'query_question_ids': {
        query: ['q1', 'q2'] for query in payload['queries']
    }}
    store.put('recommendation_delivery', payload['request_id'], {
        'request': payload, 'request_id': payload['request_id'], 'user_id': account['user_id'],
        'project_id': project, 'receipt_id': 'recommendation_delivery:' + payload['request_id'],
        'source_refs': {}, 'trigger_details': {},
    })
    return payload


def saved_event(store, job, account):
    event = {'event_id': job['job_id'] + ':event', 'user_id': account['user_id'],
             'project_id': job['snapshot']['project_id'], 'status': 'skipped', 'reason': 'no_update'}
    store.put('recommendation_event', event['event_id'], event)
    return event


@pytest.mark.parametrize('reason, recovered_delivery', [
    ('no_update', False), ('no_new_or_updated_journals_in_window', False),
    ('no_new_or_updated_journals_in_window', True),
])
def test_manual_analysis_does_not_suppress_scheduled_email(workspace, mailbox, reason, recovered_delivery):
    store, account, pid = workspace
    manual = jobs.enqueue(store, account, {'action': 'project_understanding', 'project_id': pid,
                                           'analysis_mode': 'daily'}, as_of='2026-11-19T04:00:00+08:00')
    payload = delivery(store, account, pid)
    store.put('understanding_run', 'same-day-manual', {'request': payload})
    jobs.process_one(store, execute=lambda job, owner: {
        'event_id': None, 'user_id': owner['user_id'], 'status': 'completed',
        'reason': 'awaiting_review', 'understanding_run_id': 'same-day-manual',
    })
    automatic = schedule(store)
    assert automatic['job_id'] != manual['job_id']
    assert schedule(store)['job_id'] == automatic['job_id']
    def skipped(job, owner):
        if recovered_delivery:
            return {'user_id': owner['user_id'], 'event_id': None, 'status': 'skipped', 'reason': reason,
                    '_auto_review_requests': [payload['request_id']]}
        return {**saved_event(store, job, owner), 'reason': reason}

    jobs.process_one(store, execute=skipped)
    downstream, = [job for job in store.list('recommendation_job') if job['action'] == 'progress']
    assert downstream['snapshot']['review_question_ids'] == ['q1', 'q2']
    assert downstream['snapshot']['auto_email'] and downstream['snapshot']['scheduled_project_daily']
    assert not mailbox
    jobs.process_one(store, execute=lambda job, owner: saved_event(store, job, owner))
    assert len(mailbox) == 1 and mailbox[0][0] == 'qa@example.invalid'


def test_schedule_recovers_only_the_current_daily_window(workspace, monkeypatch):
    store, account, pid = workspace
    pending = [
        {'request_id': 'older', 'trigger_details': {'window': {'until': '2026-11-18T00:00:00+08:00'}}},
        {'request_id': 'manual-range', 'trigger_details': {'window': None, 'selection': 'explicit_manual'}},
        {'request_id': 'current', 'trigger_details': {'window': {'until': '2026-11-19T00:00:00+08:00'}}},
    ]
    deliveries = iter([pending, []])
    monkeypatch.setattr('neuroei.recommendation_handoff.receive_pending', lambda *args, **kwargs: next(deliveries))
    monkeypatch.setattr('neuroei.recommendation_pipeline.understand', lambda *args, **kwargs: {
        'status': 'skipped', 'reason': 'no_new_or_updated_journals_in_window',
    })
    batch = JournalBatch.model_validate({'records': [], 'source_runs': [{'source_id': 'journal', 'status': 'read'}]})
    result = progress(store, batch, extractor=None, sources=None, user_id=account['user_id'], project_id=pid,
                      projects=[pid], as_of=STAMP, delivery_window_end=date(2026, 11, 19))
    assert result['review_required'] and result['deliveries'] == pending[-1:]


def test_local_schedule_selects_all_questions_without_sending_email(workspace, mailbox):
    store, account, pid = workspace
    scheduled = schedule(store, email=False)
    assert schedule(store, email=False)['job_id'] == scheduled['job_id']
    payload = delivery(store, account, pid)
    jobs.process_one(store, execute=lambda job, owner: {
        'event_id': None, 'user_id': owner['user_id'], 'status': 'completed',
        'reason': 'awaiting_review', '_auto_review_requests': [payload['request_id']]})
    downstream, = [job for job in store.list('recommendation_job') if job['action'] == 'progress']
    assert downstream['snapshot']['review_question_ids'] == ['q1', 'q2']
    assert downstream['snapshot']['scheduled_project_daily']
    assert not downstream['snapshot'].get('auto_email')
    jobs.process_one(store, execute=lambda job, owner: saved_event(store, job, owner))
    assert not mailbox


@pytest.mark.parametrize('handoff', ['search', 'email'])
def test_restart_finishes_durable_daily_followup(workspace, mailbox, monkeypatch, handoff):
    store, account, pid = workspace
    parent = schedule(store)
    payload = delivery(store, account, pid)
    persist = jobs.persist_job

    def stop_after_commit(store, job):
        persist(store, job)
        raise SystemExit('worker interrupted after commit')

    def execute(job, owner):
        if handoff == 'email':
            return saved_event(store, job, owner)
        return {'event_id': None, 'user_id': owner['user_id'], 'status': 'completed', 'reason': 'awaiting_review',
                '_auto_review_requests': [payload['request_id']]}

    with monkeypatch.context() as interrupt:
        interrupt.setattr(jobs, 'persist_job', stop_after_commit)
        with pytest.raises(SystemExit):
            jobs.process_one(store, execute=execute)
    assert 'automatic_followup' in store.get('recommendation_job', parent['job_id'])
    assert not mailbox
    jobs.reconcile_interrupted(store)
    jobs.reconcile_interrupted(store)
    assert 'automatic_followup' not in store.get('recommendation_job', parent['job_id'])
    if handoff == 'search':
        child, = [job for job in store.list('recommendation_job') if job['action'] == 'progress']
        assert child['snapshot']['review_question_ids'] == ['q1', 'q2']
        jobs.process_one(store, execute=lambda job, owner: saved_event(store, job, owner))
    assert len(mailbox) == 1


def test_opt_out_skips_scheduled_work_already_queued(workspace, mailbox):
    store, account, pid = workspace
    schedule(store)
    update_project_automation(store, account, {'project_id': pid, 'enabled': False})
    job = jobs.process_one(store, execute=lambda *_: pytest.fail('Disabled schedule ran'))
    assert (job['status'], job['reason']) == ('skipped', 'automation_disabled')
    assert not mailbox
    assert project_daily_schedule.run(store, as_of=STAMP)['projects'] == []
    schedule(store, '2026-11-20T05:00:00+08:00', project_id=pid)
    jobs.process_one(store, execute=lambda job, owner: saved_event(store, job, owner))
    assert len(mailbox) == 1  # explicit one-project run still works with recurring automation off


def test_auto_email_serializes_overlapping_delivery(workspace, monkeypatch):
    store, account, _ = workspace
    job = store.get('recommendation_job', schedule(store)['job_id'])
    job['event_id'] = saved_event(store, job, account)['event_id']
    sending, release = Event(), Event()
    recipients = []

    class Mailer:
        origin = 'https://example.invalid'

        def send_daily_report(self, email, subject, body):
            recipients.append(email)
            sending.set()
            assert release.wait(5)

    monkeypatch.setattr(project_daily_email.account_email.SMTPMailer, 'configured', Mailer)

    def dispatch():
        with closing(Store(store.root)) as connection:
            return project_daily_email.send_daily_report(connection, account, job)

    with ThreadPoolExecutor(1) as pool:
        completed = pool.submit(dispatch)
        try:
            assert sending.wait(5)
            assert dispatch()['status'] == 'sending'
        finally:
            release.set()
        assert completed.result()['status'] == 'sent'
    assert dispatch()['status'] == 'sent'
    assert recipients == ['qa@example.invalid']


def test_automatic_search_preserves_manual_subset_and_reuses_full_results(workspace, mailbox, monkeypatch):
    store, account, pid = workspace
    payload = delivery(store, account, pid)
    parent = schedule(store)
    manual = dispatch_reviewed(store, payload['request_id'], user_id=account['user_id'], projects=[pid],
                               discover=empty, question_ids=['q1'])
    jobs.process_one(store, execute=lambda job, owner: {
        'event_id': None, 'user_id': owner['user_id'], 'status': 'completed', 'reason': 'awaiting_review',
        '_auto_review_requests': [payload['request_id']],
    })
    searches = []

    def discover(wire, *_args, **_kwargs):
        from neuroei.contracts import DiscoveryResult

        searches.append([q['id'] for q in wire['research_questions']])
        return DiscoveryResult.model_validate(empty(wire))

    monkeypatch.setattr('neuroei.recommendation_worker.discover', discover)
    child = jobs.process_one(store, execute=execution(store, sources=object(), ranker_factory=lambda *_: None))
    full = store.get('recommendation_event', child['event_id'])
    assert full['event_id'] != manual['event_id']
    assert searches == [['q1', 'q2']]
    assert store.get('recommendation_event', manual['event_id']) == manual
    assert dispatch_reviewed(store, payload['request_id'], user_id=account['user_id'], projects=[pid],
                             discover=lambda _: pytest.fail('Repeated search'), question_ids=['q1', 'q2'],
                             automatic=True) == full
    assert len(mailbox) == 1
    assert store.get('recommendation_job', parent['job_id'])['status'] == 'completed'
