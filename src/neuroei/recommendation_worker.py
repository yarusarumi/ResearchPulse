"""Explicit foreground worker. No scheduler registration or implicit paid clients."""
import argparse
import json
import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from . import accounts, ai_backend
from .discovery import discover
from .discovery_paperflow import RANKING_AUDIT_POLICY
from .discovery_phrases import PHRASE_POLICY
from .discovery_sources import DiscoverySources
from .models import now
from .recommendation_handoff import dispatch_reviewed
from .recommendation_jobs import (
    persist_job,
    process_one,
    reconcile_interrupted,
    save_job_progress,
    understanding_model_config,
)
from .recommendation_pipeline import direction, direction_request, discover_direction
from .store import Store
from .wechat_discovery import discover_wechat_accounts


class RequestRanker:
    """Bind an existing CPU encoder to each request's explicit question/query plan."""
    paid_model_calls = 0

    def __init__(self, store, encoder, *, revision, audit_ref, aliases=None, review_calls=0, fulltext_rescue=False):
        if review_calls not in (0, 1):
            raise ValueError('Discovery review call cap must be zero or one')
        self.store, self.encoder, self.revision, self.audit_ref = store, encoder, revision, audit_ref
        self.review_calls = review_calls
        self.fulltext_rescue = fulltext_rescue and bool(review_calls) and ai_backend.selected() == 'sol-high'
        self.aliases = aliases
        self.model_calls, self.embedding_runs, self.decisions, self.ranking_audit = 0, [], {}, []
        self.review_errors = []
        self.should_stop = None
        self.policy = {'adapter': 'request-query-gates-v3-concepts', 'model_revision': revision,
                       'phrase_policy': PHRASE_POLICY,
                       'ranking_audit_policy': RANKING_AUDIT_POLICY,
                       'explicit_aliases': aliases, 'embedding_audit_ref': audit_ref,
                       'max_candidates': 1200, 'max_input_characters': 4_000_000}
        if review_calls:
            from .discovery_review import REVIEW_ISOLATION_POLICY
            from .discovery_review_provider import REVIEW_PROVIDER_POLICY
            from .discovery_reviewed_ranker import REVIEWED_SHORTLIST_POLICY
            self.policy['review'] = {'provider': REVIEW_PROVIDER_POLICY, 'selection': REVIEWED_SHORTLIST_POLICY,
                                    'isolation_policy': REVIEW_ISOLATION_POLICY,
                                    'max_candidates': 40, 'max_calls': 1, 'max_input_characters': 160_000,
                                    **ai_backend.model_config(discovery_review_config(), ai_backend.selected()).model_dump(mode='json')}
        if self.fulltext_rescue:
            from .discovery_fulltext_review import POLICY, review_config

            self.policy['fulltext_rescue'] = {'policy': POLICY, 'mode': 'progress',
                **ai_backend.model_config(review_config(), 'sol-high').model_dump(mode='json')}

    def __call__(self, request, papers):
        from .discovery_aliases import aliases_from_queries
        from .discovery_paperflow import PaperFlowSemanticRanker

        if self.model_calls:
            raise ValueError('Per-request embedding budget exhausted')
        if self.aliases is None:
            plan = aliases_from_queries(request.model_dump(mode='json'))
            self.store.audit(request.request_id, {'event': 'question_query_plan', **plan})
            if plan['status'] != 'ready':
                raise ValueError('query_question_mapping_required')
            aliases = plan['aliases']
        else:
            aliases = self.aliases
        ranker = PaperFlowSemanticRanker(self.encoder, aliases, model_revision=self.revision,
                                         embedding_audit_ref=self.audit_ref, max_calls=1,
                                         max_candidates=self.policy['max_candidates'],
                                         max_input_characters=self.policy['max_input_characters'])
        client = fulltext_client = None
        if self.review_calls:
            from .discovery_review_provider import FlashAbstractReviewer
            from .discovery_reviewed_ranker import ReviewedPaperFlowRanker
            client = ai_backend.text_client(discovery_review_config(), self.store,
                                            'discovery-review:' + self.audit_ref, backend=ai_backend.selected())
            client.should_stop = self.should_stop
            reviewer = FlashAbstractReviewer(client, max_calls=1,
                                             max_input_characters=self.policy['review']['max_input_characters'],
                                             max_tokens=client.config.max_tokens)
            fulltext_review = None
            if self.fulltext_rescue and request.mode == 'progress':
                from .discovery_fulltext_review import FulltextCandidateReviewer, review_config

                fulltext_client = ai_backend.text_client(review_config(), self.store,
                    'discovery-fulltext-review:' + self.audit_ref, backend='sol-high')
                fulltext_client.should_stop = self.should_stop
                fulltext_review = FulltextCandidateReviewer(self.store, fulltext_client, should_stop=self.should_stop)
            ranker = ReviewedPaperFlowRanker(ranker, reviewer, max_candidates=40, fulltext_review=fulltext_review)
        try:
            return ranker(request, papers)
        finally:
            self.model_calls = ranker.model_calls
            self.paid_model_calls = getattr(ranker, 'paid_model_calls', 0)
            self.embedding_runs = ranker.embedding_runs
            self.decisions = ranker.decisions
            self.ranking_audit = getattr(ranker, 'ranking_audit', [])
            self.review_errors = getattr(ranker, 'review_errors', [])
            if client:
                client.close()
            if fulltext_client:
                fulltext_client.close()


def discovery_review_config():
    from .config import LLMConfig
    return LLMConfig(max_tokens=131_072, timeout_seconds=900, request_attempts=1,
                     schema_repair_attempts=0, max_calls_per_run=1)


def configured_web_fallback(store, job, request, api_key, *, search_limit=32, fetch_limit=64, max_usd=.25):
    """Job-scoped paid web allowance; audits are durable before every provider request."""
    from .discovery_web import DeepSeekWeb

    if request['user_id'] != job['user_id']:
        raise PermissionError('Web fallback owner mismatch')
    def audit(entry):
        store.audit('web-fallback:'+job['job_id'], {'event': 'web_tool_attempt',
            'user_id': job['user_id'], 'request_id': request['request_id'], **entry})
    return DeepSeekWeb(api_key, enabled=True, search_limit=search_limit, fetch_limit=fetch_limit,
                       max_usd=max_usd, audit_callback=audit, source_scoped=True)


def saved_source_plan(discovery):
    runs = (discovery or {}).get('source_runs', [])
    names = tuple(dict.fromkeys(r['source'] for r in runs
                               if r['source'] not in ('deepseek_search_fetch', 'pubmed_exact', 'media_identifier')
                               and not r['source'].startswith('wechat:')))
    articles = (discovery or {}).get('audit', {}).get('wechat_account_articles', [])
    return names, ([r for r in runs if r['source'].startswith('wechat:')], articles)


def execution(store, *, sources, ranker_factory, analyze=None, reader_factory=None, understanding_calls=0,
              source_planner=None, web_factory=None, journal_planner=None, wechat_sources=False, adaptive_search=False):
    def execute(job, account):
        snapshot = job['snapshot']

        def stop_requested():
            try:
                return (store.get('recommendation_job', job['job_id']) or {}).get('status') == 'canceled'
            except ValueError:
                return False
        def report_discovery(stage, **counts):
            save_job_progress(store, job, stage, None, **counts)

        if job['action'] in ('project_understanding', 'project_understanding_retry'):
            return execute_understanding(store, job, account, max_calls=understanding_calls, should_stop=stop_requested)
        if job['action'] in ('reading_recovery', 'discovery_recovery'):
            if reader_factory is None:
                raise ValueError('Reading service required')
            job['event_id'] = snapshot['event_id']
            persist_job(store, job)
            retry_discovery = None
            if job['action'] == 'discovery_recovery':
                previous = store.get('recommendation_event', snapshot['event_id'])
                names, wechat = saved_source_plan(previous['discovery'])
                ranker = ranker_factory(None, 'discovery-recovery:'+job['job_id'])
                def retry_discovery(request):
                    plan = previous['discovery'].get('audit', {}).get('official_source_plan')
                    if source_planner is not None:
                        sources.configure_official_plan(plan)
                    if journal_planner is not None:
                        sources.configure_journal_plan(previous['discovery'].get('audit', {}).get('journal_source_plan'))
                    return discover_direction(request, sources, source_names=names, ranker=ranker,
                                              store=store, web_fallback=web, saved_wechat=wechat, on_progress=report_discovery)
            web = web_factory(job, previous['input']) if web_factory and retry_discovery else None
            def report_reading(done, total):
                save_job_progress(store, job, 'reading' if done < total else 'finalizing',
                                  15 + 80 * done // total, done=done, total=total)
            try:
                return recover_readings(store, snapshot['event_id'], account['user_id'],
                                        reader_factory=reader_factory, discover=retry_discovery, report=report_reading)
            finally:
                if web:
                    web.close()
        from .recommendation_events import reviewed_event_id

        request = (snapshot['request'] if job['action'] == 'progress' else
                   direction_request(snapshot, as_of=job['created_at'],
                                     bootstrap=job['action'] == 'direction_bootstrap'))
        automatic = bool(snapshot.get('auto_email') or snapshot.get('scheduled_project_daily')) and job['action'] == 'progress'
        job['event_id'] = reviewed_event_id(store, request, automatic=automatic)
        persist_job(store, job)
        allowed = accounts.projects_for(store, account)
        projects = allowed if allowed is not None else [snapshot.get('project_id')]
        reader = reader_factory(job) if reader_factory else analyze
        web = None
        try:
            web = web_factory(job, request) if web_factory else None
            from .discovery_web import DeepSeekWeb

            if isinstance(web, DeepSeekWeb):
                web.should_stop = stop_requested
            if reader_client := getattr(reader, 'client', None):
                reader_client.should_stop = stop_requested

            def adaptive(req, first, history=()):
                from .discovery_adaptive import FlashQueryPlanner, expand_empty_discovery
                from .discovery_sources import DiscoverySources

                if not adaptive_search or not isinstance(sources, DiscoverySources):
                    return first

                def supplemental(patched, limit):
                    ranker = ranker_factory(None, job['job_id']+':adaptive')
                    if isinstance(ranker, RequestRanker):
                        ranker.should_stop = stop_requested
                    lane = DiscoverySources(max_results=sources.max_results, max_pages=sources.max_pages,
                                            concept_categories=sources.concept_categories)
                    lane.coordination_dir = sources.coordination_dir
                    try:
                        return discover(patched, lane, source_names=('arxiv', 'crossref'), ranker=ranker,
                                        max_queries=limit, store=store, previously_recommended=history,
                                        should_stop=stop_requested, on_progress=report_discovery).model_dump(mode='json')
                    finally:
                        lane.close()

                strategy = os.environ.get('NEUROEI_DISCOVERY_FOLLOWUP', 'per_question')
                planner = FlashQueryPlanner(store, 'adaptive-query:'+job['job_id'], should_stop=stop_requested)
                return expand_empty_discovery(req, first, plan=planner,
                                              run=supplemental, should_stop=stop_requested, on_progress=report_discovery,
                                              strategy=strategy)

            def synthesize(req, analyses):
                from .recommendation_synthesis import synthesize_recommendations

                return synthesize_recommendations(req, analyses, getattr(reader, 'client', None))

            names = ('arxiv', 'crossref') if job['action'] == 'progress' else ('arxiv',)
            if source_planner is not None:
                plan = source_planner(request)
                sources.configure_official_plan(plan)
                names = tuple(dict.fromkeys((*names, *plan['source_names'])))
                job['official_source_plan'] = plan
                persist_job(store, job)
            if journal_planner is not None:
                journal_plan = journal_planner(request)
                sources.configure_journal_plan(journal_plan)
                if journal_plan.get('pubmed_exact') and 'crossref' not in names:
                    names += ('crossref',)
                names = tuple(dict.fromkeys((*names, *journal_plan['source_names'])))
                job['journal_source_plan'] = journal_plan
                persist_job(store, job)
            if job['action'] == 'progress':
                req = snapshot['request']
                ranker = ranker_factory(req, job['job_id'])
                if isinstance(ranker, RequestRanker):
                    ranker.should_stop = stop_requested
                return dispatch_reviewed(store, req['request_id'], user_id=account['user_id'], projects=projects,
                    discover=lambda request: adaptive(request, discover(request, sources, source_names=names, ranker=ranker,
                        max_queries=24, store=store, web_fallback=web,
                        on_progress=report_discovery, should_stop=stop_requested).model_dump(mode='json')),
                    analyze=reader, question_ids=snapshot.get('review_question_ids'), should_stop=stop_requested,
                    synthesize=synthesize, automatic=automatic)
            # Ranker aliases are configured by stable question ID, preserving explicit provenance.
            ranker = ranker_factory(snapshot, job['job_id'])
            if isinstance(ranker, RequestRanker):
                ranker.should_stop = stop_requested
            def wechat_lookup(wire):
                return discover_wechat_accounts(wire, web, should_stop=stop_requested)
            return direction(store, snapshot, user_id=account['user_id'], projects=projects,
                             directions=[snapshot['direction_id']], as_of=job['created_at'],
                             bootstrap=job['action'] == 'direction_bootstrap', sources=sources,
                             ranker=ranker, analyze=reader, source_names=names, web_fallback=web,
                             wechat_lookup=wechat_lookup if wechat_sources and web else None,
                             on_progress=report_discovery, should_stop=stop_requested,
                             adaptive=adaptive, synthesize=synthesize)
        finally:
            if web:
                web.close()
            if reader_factory:
                reader.close()
    return execute


def execute_understanding(store, job, account, *, max_calls, should_stop=None):
    from .project_understanding import TZ, JournalBatch, JournalRecord, identity
    from .project_understanding_spans import SpanSelectingUnderstandingExtractor
    from .recommendation_events import record_gate
    from .recommendation_pipeline import progress
    from .research_workspace import project_for

    snapshot = job['snapshot']
    project = project_for(store, account, snapshot['project_id'])
    if project['owner_id'] != account['user_id']:
        raise PermissionError('Project input owner mismatch')
    analysis_as_of = snapshot.get('analysis_as_of', job['created_at'])
    daily_feishu = snapshot.get('analysis_mode') == 'daily' and snapshot.get('feishu')
    manual_selection = snapshot.get('manual_selection') if snapshot.get('analysis_mode') == 'manual' else None
    head = identity([account['user_id'], snapshot['project_id']])
    pointer = dict(store.items('understanding_head')).get(head)
    if snapshot.get('scheduled_project_daily') and snapshot.get('analysis_mode') == 'daily' and 'analysis_as_of' not in snapshot:
        # Keep the scheduled Beijing date, but use the actual observation time for a catch-up within that day.
        slot = datetime.fromisoformat(analysis_as_of).astimezone(TZ)
        day_end = (slot + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        observed = datetime.fromisoformat(job['started_at'])
        if pointer:
            baseline = datetime.fromisoformat(store.get('understanding_state', pointer['key'])['as_of'])
            if baseline >= day_end:
                return {'user_id': account['user_id'], 'event_id': None,
                        'status': 'skipped', 'reason': 'scheduled_window_superseded'}
            observed = max(observed, baseline)
        analysis_as_of = max(slot, min(observed, day_end - timedelta(microseconds=1))).astimezone(TZ).isoformat()
        snapshot['analysis_as_of'] = analysis_as_of
    background_record = snapshot['background']
    background_changed = False
    if background_record and snapshot.get('analysis_mode') == 'imported' and snapshot.get('background_edited'):
        background = JournalRecord.model_validate(background_record)
        prior = store.get('understanding_state', pointer['key'])['sources'] if pointer else {}
        background_changed = prior.get(background.source_id, {}).get('version') != background.content_version
    if should_stop and should_stop():
        return {'user_id': account['user_id'], 'event_id': None, 'status': 'canceled', 'reason': 'canceled_by_user'}
    save_job_progress(store, job, 'reading_journals', None)
    if snapshot.get('feishu') and not snapshot.get('feishu_loaded'):
        from .feishu_folders import folder_batch

        attachments_only = bool(snapshot.get('feishu_attachment_continuation'))
        saved_sources = store.get('understanding_state', pointer['key'])['sources'] if pointer else {}
        completed_attachments = {key: (block['revision'], block['section']) for key, value in saved_sources.items()
                                 if ':attachment:' in key and value.get('record', {}).get('blocks')
                                 and not value.get('deferred_dated_blocks') for block in value['record']['blocks'][:1]}
        image_options = {}
        image_client = None
        if not attachments_only and (max_calls or os.environ.get('NEUROEI_FEISHU_IMAGE_OCR') == '1' or (
                manual_selection and manual_selection.get('image_block_ids'))):
            from .config import LLMConfig
            from .feishu_image_ocr import CODEX_SECTION, transcribe, transcribe_sol

            image_options['image_reader'] = transcribe
            if max_calls and ai_backend.selected() != 'deepseek':
                image_client = ai_backend.journal_image_client(LLMConfig(
                    max_calls_per_run=max_calls, request_attempts=1, schema_repair_attempts=0,
                    timeout_seconds=180), store, 'feishu-image:' + job['job_id'],
                    backend=ai_backend.selected(), should_stop=should_stop)
                if getattr(image_client, 'supports_images', True):
                    image_options['sol_image_reader'] = partial(transcribe_sol, client=image_client)
                    if os.environ.get('NEUROEI_DESKTOP') == '1':
                        config = image_client.config
                        image_options.update(sol_policy=f'codex-transcript-v1:{config.model}:{config.reasoning_effort}',
                                             sol_section=CODEX_SECTION, image_workers=config.max_concurrency)
            image_options['on_image_progress'] = lambda done, total, reused, document: save_job_progress(
                store, job, 'reading_journal_images', None, done=done, total=total, reused=reused, document=document)
            if snapshot.get('analysis_mode') == 'daily':
                image_options['image_day'] = datetime.fromisoformat(analysis_as_of).astimezone(TZ).date() - timedelta(days=1)
        try:
            batch = folder_batch(store, account, snapshot['project_id'], snapshot['feishu'], snapshot['feishu_manifest'],
                                 analysis_as_of, grant_id=snapshot.get('feishu_grant_id', ''),
                                 include_attachments=manual_selection is None,
                                 completed_attachments=completed_attachments, attachments_only=attachments_only,
                                 only_image_ids=(manual_selection or {}).get('image_block_ids'),
                                 **({'should_stop': should_stop} if should_stop else {}), **image_options)
        finally:
            if image_client:
                image_client.close()
        if should_stop and should_stop():
            return {'user_id': account['user_id'], 'event_id': None, 'status': 'canceled', 'reason': 'canceled_by_user'}
        snapshot['feishu_attachment_failures'] = batch.pop('attachment_failures', [])
        snapshot['feishu_document_failures'] = batch.pop('document_failures', [])
        snapshot['feishu_attachments_pending'] = sum(int(w.rsplit(':', 1)[1]) for w in batch.get('warnings', [])
                                                     if w.startswith('feishu_attachments_over_limit:'))
        for field, rows in batch.items():
            snapshot['batch'].setdefault(field, []).extend(rows)
        snapshot['feishu_loaded'] = True
        persist_job(store, job)
    batch = JournalBatch.model_validate(snapshot['batch'])
    source_failed = batch.blocked
    if manual_selection and not snapshot['batch']['records'] and not source_failed:
        return {'user_id': account['user_id'], 'event_id': None, 'status': 'failed',
                'reason': 'manual_selection_empty', 'understanding_phase': 'incremental'}
    if ((snapshot.get('feishu') or not background_record) and not batch.records
            and not source_failed and not background_changed):
        if snapshot.get('feishu_attachment_failures'):
            return {'user_id': account['user_id'], 'event_id': None, 'status': 'partial',
                    'reason': 'feishu_attachments_pending', 'understanding_phase': 'incremental'}
        reason = 'no_update' if snapshot.get('analysis_mode') == 'daily' else 'no_imported_update'
        event = record_gate(store, user_id=account['user_id'], projects=[project['project_id']],
                            project_id=project['project_id'], date=datetime.fromisoformat(analysis_as_of).astimezone(TZ).date().isoformat(),
                            source_version=job['job_id'], status='skipped', reason=reason,
                            input_snapshot={'as_of': analysis_as_of, 'journal_input': snapshot['batch']})
        return {'user_id': account['user_id'], 'event_id': event['event_id'], 'status': 'skipped', 'reason': reason}
    backend = ai_backend.selected()
    client = (ai_backend.text_client(understanding_model_config(max_calls), store,
                                     'understanding-job:' + job['job_id'], backend=backend)
              if max_calls and not source_failed else None)
    if client:
        client.should_stop = should_stop
    extractor = SpanSelectingUnderstandingExtractor(client) if client else None
    common = {'extractor': extractor, 'sources': None, 'user_id': account['user_id'],
              'project_id': snapshot['project_id'], 'projects': [snapshot['project_id']],
              'as_of': analysis_as_of, 'require_review': True, 'should_stop': should_stop}
    try:
        if background_record and not source_failed and not pointer:
            save_job_progress(store, job, 'analyzing_background', None)
            background = JournalBatch.model_validate({'records': [background_record],
                'source_runs': [{'source_id': background_record['source_id'], 'status': 'read'}]})
            retry = snapshot.get('retry_run_id') if snapshot.get('retry_phase') == 'bootstrap' else None
            outcome = progress(store, background, bootstrap=True, retry_of=retry,
                               max_calls=client.remaining() if client else 0, **common)
            result = outcome['understanding']
            # A saved project description is enough context to begin reading journals.
            if result['status'] != 'completed' and result['reason'] != 'background_context_saved':
                return {'user_id': account['user_id'], 'event_id': (outcome.get('event') or {}).get('event_id'),
                        'status': result['status'], 'reason': result.get('failure_code', result['reason']),
                        'understanding_phase': 'bootstrap', 'understanding_run_id': result.get('run_id')}
        if pointer and background_changed and not source_failed:
            batch = JournalBatch.model_validate({'warnings': batch.warnings,
                'records': [background_record, *batch.records],
                'source_runs': [*batch.source_runs, {'source_id': background.source_id, 'status': 'read'}]})
        imported_at = analysis_as_of if snapshot.get('analysis_mode') == 'imported' else None
        retry = snapshot.get('retry_run_id') if snapshot.get('retry_phase') == 'incremental' else None
        if not source_failed:
            save_job_progress(store, job, 'analyzing_journals', None)
        def report_journals(done, total):
            save_job_progress(store, job, 'analyzing_journals', None, done=done, total=total)

        selection = None
        if daily_feishu:
            from .daily_scope_review import MAX_REVIEW_CALLS, review_delta
            from .feishu_folders import DAILY_SCOPE_POLICY, daily_model_records

            def select_material(records, sources, day):
                manifest = snapshot.get('feishu_manifest') or []
                scope_calls = 0

                def review(record, candidates, previous, target_day):
                    nonlocal scope_calls
                    remaining = client.remaining() if client else 0
                    allowance = min(MAX_REVIEW_CALLS - scope_calls, remaining - 1)
                    if allowance <= 0 or (should_stop and should_stop()):
                        return [], [block.block_id for block in candidates]
                    save_job_progress(store, job, 'confirming_daily_scope', None)
                    from .daily_scope_backend import scope_client, uses_personal_selection
                    from .llm import ModelUnavailable

                    primary = ai_backend.selected()
                    personal = primary == 'deepseek' and uses_personal_selection()
                    backend = snapshot.get('daily_scope_backend', 'deepseek') if personal else primary
                    used, failure = 0, None
                    try:
                        reviewer = (scope_client(store, account['user_id'], job['job_id'], backend, client, should_stop)
                                    if personal else client)
                        before = reviewer.calls
                        kept, pending = review_delta(reviewer, record, candidates, previous, target_day,
                                                     max_calls=allowance + 1)
                        used = reviewer.calls - before
                    except ModelUnavailable:
                        kept, pending = [], [block.block_id for block in candidates]
                        failure = 'scope_backend_unavailable'
                    scope_calls += used
                    snapshot.setdefault('daily_scope_reviews', []).append(
                        {'candidates': len(candidates), 'included': len(kept), 'unresolved': len(pending),
                         'calls': used, 'backend': backend, 'failure': failure})
                    save_job_progress(store, job, 'analyzing_journals', None)
                    return kept, pending

                material, unverified, deferred, skipped = daily_model_records(records, sources, day, manifest, review=review)
                source_ids = {'feishu:' + row['token'] for row in manifest}
                snapshot['daily_text_blocks'] = sum(len(r.blocks) for r in material if r.source_id in source_ids)
                snapshot['daily_scope_unverified'] = len(unverified)
                snapshot['daily_scope_skipped'] = len(skipped)
                return material, unverified, deferred

            selection = (DAILY_SCOPE_POLICY, select_material)
        context = background_record['blocks'][0]['text'] if background_record else snapshot.get('project_context')
        if snapshot.get('auto_dispatch') and snapshot['analysis_mode'] == 'daily':
            common['delivery_window_end'] = datetime.fromisoformat(analysis_as_of).astimezone(TZ).date()
        outcome = progress(store, batch, max_calls=client.remaining if client else 0,
                           imported_at=imported_at, retry_of=retry, on_progress=report_journals,
                           model_selection=selection, manual_selection=manual_selection,
                           project_context=context, **common)
        result = outcome['understanding']
        if (max_calls and not source_failed and snapshot.get('analysis_mode') == 'imported'
                and result['status'] in ('completed', 'skipped') and result.get('state_version')
                and (ai_backend.deepseek_available() if backend == 'deepseek' else ai_backend.runner())):
            from .project_brief import update_brief

            snapshot['project_brief_status'] = update_brief(
                store, account, snapshot['project_id'], result['state_version'], job['job_id'],
                should_stop=should_stop, on_start=lambda: save_job_progress(store, job, 'synthesizing_background', None))
        if pending_images := result.get('image_analysis_pending'):
            snapshot['image_analysis_pending'] = pending_images
        if gaps := result.get('batch_gaps'):
            snapshot['batch_gaps'] = gaps
        if (snapshot.get('analysis_mode') == 'imported' and result['status'] in ('completed', 'skipped')
                and (snapshot.get('feishu_attachments_pending') or snapshot.get('feishu_attachment_failures'))):
            return {'user_id': account['user_id'], 'event_id': (outcome.get('event') or {}).get('event_id'),
                    'status': 'partial', 'reason': 'feishu_attachments_pending',
                    'understanding_phase': 'incremental', 'understanding_run_id': result.get('run_id')}
        returned = {'user_id': account['user_id'], 'event_id': (outcome.get('event') or {}).get('event_id'),
                    'status': result['status'], 'reason': ('source_failed' if source_failed else
                        'awaiting_review' if outcome.get('review_required') and result['status'] == 'completed' else
                        result.get('failure_code', result['reason'])),
                    'understanding_phase': 'incremental', 'understanding_run_id': result.get('run_id')}
        if snapshot.get('auto_dispatch') and outcome.get('deliveries'):
            returned['_auto_review_requests'] = [delivery['request_id'] for delivery in outcome['deliveries']]
        return returned
    finally:
        if client:
            client.close()


def recover_readings(store, event_id, user_id, *, reader_factory, discover=None, report=None):
    """Explicit one-attempt recovery; discovery retry is restricted to empty failures."""
    from .directions import load_direction
    from .recommendation_events import recover_analysis, recover_discovery

    owner = accounts.find(store, user_id)
    if not owner or owner['disabled']:
        raise PermissionError('Enabled account required')
    event = store.get('recommendation_event', event_id)
    if event['user_id'] != user_id:
        raise PermissionError('Event owner mismatch')
    allowed = accounts.projects_for(store, owner)
    projects = allowed if allowed is not None else [event['project_id']]
    if event['project_id'] and event['project_id'] not in projects:
        raise PermissionError('Project access required')
    directions = []
    if event['direction_id']:
        if not load_direction(store, owner['display_name'], event['direction_id'])['enabled']:
            raise PermissionError('Direction is paused')
        directions = [event['direction_id']]
    reader_job = {'job_id': ('recover-discovery:' if discover else 'recover-reading:')+event_id,
                  'user_id': user_id}
    attempt_key = 'discovery_recovery_attempts' if discover else 'recovery_attempts'
    if attempt := event.get(attempt_key, 0):
        reader_job['job_id'] += f':{attempt+1}'
    if discover is None:
        reader_job['reading_recovery_attempt'] = f"{event_id}:{event.get('recovery_attempts', 0) + 1}"
    reader = reader_factory(reader_job)
    try:
        if discover is None:
            recovered = recover_analysis(store, event_id, user_id=user_id, projects=projects,
                                         directions=directions, analyze=reader, report=report)
        else:
            recovered = recover_discovery(store, event_id, user_id=user_id, projects=projects,
                                          directions=directions, discover=discover, analyze=reader)
        for job in store.list('recommendation_job'):
            if job.get('event_id') == event_id and job['user_id'] == user_id:
                store.audit(job['job_id'], {'event': 'discovery_recovered' if discover else 'reading_recovered', 'previous_status': job['status'],
                                             'previous_reason': job['reason']})
                job.update(status=recovered['status'], reason=recovered['reason'], ended_at=now())
                persist_job(store, job)
        return recovered
    finally:
        reader.close()


def main(argv=None, *, shutdown=None):
    from .discovery_web import MAX_WEB_FETCHES, MAX_WEB_SEARCHES
    from .recommendation_reader import MAX_READING_CALLS, ConfiguredReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--model-path', type=Path, help='Existing local BGE CPU model')
    parser.add_argument('--model-revision')
    parser.add_argument('--aliases', type=Path, help='Optional operator override; default uses explicit request queries')
    parser.add_argument('--fetch-fulltexts', action='store_true', help='Fetch exact versioned official arXiv PDFs, one attempt; no paid calls')
    parser.add_argument('--open-access-fulltexts', action='store_true', help='Allow verified PMC published open-access PDFs for DOI papers')
    parser.add_argument('--journal-sources', action='store_true', help='Enable the bounded official journal source plan')
    parser.add_argument('--wechat-sources', action='store_true', help='Search curated public WeChat accounts for directions')
    parser.add_argument('--fulltext-manifest', type=Path, help='User and paper keyed authorized local documents')
    parser.add_argument('--artifacts-path', type=Path, help='Existing local Docling CPU artifacts')
    parser.add_argument('--max-understanding-calls', type=int, default=0, choices=range(101),
                        help='Explicit project understanding call cap per job, default zero')
    parser.add_argument('--max-model-calls', type=int, default=0, choices=range(MAX_READING_CALLS + 1),
                        help='Explicit paid reading call cap per job, default zero; no retries')
    parser.add_argument('--max-discovery-review-calls', type=int, default=0, choices=(0, 1),
                        help='Explicit abstract review call cap per job, default zero; no retries')
    parser.add_argument('--web-fallback', action='store_true', help='Use paid Search/Fetch only after source failures')
    parser.add_argument('--web-search-limit', type=int, default=32, choices=range(MAX_WEB_SEARCHES + 1))
    parser.add_argument('--web-fetch-limit', type=int, default=64, choices=range(MAX_WEB_FETCHES + 1))
    parser.add_argument('--web-max-usd', type=float, default=.25)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--recover-discovery-event', help='Retry one empty failed discovery using its saved source plan')
    recovery.add_argument('--recover-event', help='Explicit one-attempt failed-reading recovery; no rediscovery')
    parser.add_argument('--user-id', help='Authorized stable account ID for recovery')
    parser.add_argument('--once', action='store_true', help='Process at most one queued request and exit')
    args = parser.parse_args(argv)
    desktop = os.environ.get('NEUROEI_DESKTOP') == '1'
    if bool(args.model_path) != bool(args.model_revision) or (not desktop and args.model_path is None):
        parser.error('--model-path and --model-revision are required for paper recommendations')
    if not args.model_path and (args.fetch_fulltexts or args.max_model_calls or args.max_discovery_review_calls):
        parser.error('Install the local reading assets before enabling paper recommendations')
    if args.wechat_sources and not args.web_fallback:
        parser.error('--wechat-sources requires the budgeted Search/Fetch provider')
    if args.fetch_fulltexts and (not args.artifacts_path or not args.artifacts_path.is_dir()):
        parser.error('--fetch-fulltexts requires an existing local --artifacts-path for PDF parsing')
    recovery_key = args.recover_event or args.recover_discovery_event
    if bool(recovery_key) != bool(args.user_id):
        parser.error('A recovery event and --user-id must be provided together')
    if not (args.store / 'neuroei.sqlite3').is_file():
        parser.error('Existing isolated account Store required')
    backend = ai_backend.selected()
    allow_web_fallback = args.web_fallback and (desktop or backend == 'deepseek')
    paid = args.max_model_calls or args.max_understanding_calls or args.max_discovery_review_calls
    if not desktop and backend == 'deepseek' and (paid or allow_web_fallback) and not os.environ.get('DEEPSEEK_API_KEY'):
        parser.error('Explicit paid reading requires DEEPSEEK_API_KEY in the worker environment')
    encoder = None
    if args.model_path:
        import torch

        from .readers import bge_cpu

        torch.set_num_threads(2)
        encoder = bge_cpu(str(args.model_path.resolve(strict=True)))
        encoder.max_seq_length = 1024
    aliases = json.loads(args.aliases.read_text(encoding='utf-8')) if args.aliases else None
    store = Store(args.store)
    sources = DiscoverySources(max_results=10, max_pages=1)
    from .discovery_official_plan import plan_official_sources
    journal_planner = None
    if args.journal_sources:
        from functools import partial

        from .discovery_journals import plan_journal_sources
        journal_planner = partial(plan_journal_sources, pubmed_exact=True)
    sources.coordination_dir = store.root / 'arxiv-coordination'
    def factory(request, job_id):
        return RequestRanker(store, encoder, aliases=aliases, revision=args.model_revision, audit_ref='job:'+job_id,
                             review_calls=args.max_discovery_review_calls,
                             fulltext_rescue=args.fetch_fulltexts and not job_id.endswith(':adaptive'))
    manifest = json.loads(args.fulltext_manifest.read_text(encoding='utf-8')) if args.fulltext_manifest else {}
    def reader_factory(job):
        return ConfiguredReader(store, job, manifest, artifacts_path=args.artifacts_path,
                                embedding_model_path=str(args.model_path.resolve(strict=True)) if args.model_path else None,
                                max_model_calls=args.max_model_calls, fetch_fulltexts=args.fetch_fulltexts,
                                open_access_fallback=args.open_access_fulltexts)
    def web_factory(job, request):
        key = ai_backend.deepseek_key() if ai_backend.selected() == 'deepseek' else None
        return (configured_web_fallback(store, job, request, key, search_limit=args.web_search_limit,
                    fetch_limit=args.web_fetch_limit, max_usd=args.web_max_usd) if key else None)
    if not allow_web_fallback:
        web_factory = None
    if recovery_key:
        from .execution import project_lock

        try:
            with project_lock(store.root, 'recommendation-worker'):
                retry_discovery = None
                if args.recover_discovery_event:
                    previous = store.get('recommendation_event', recovery_key)
                    names, wechat = saved_source_plan(previous['discovery'])
                    if not names:
                        parser.error('No saved source plan; inspect the failed event before retrying')
                    ranker = factory(None, 'discovery-recovery:'+recovery_key)
                    web = web_factory({'job_id': 'cli-recover:'+recovery_key, 'user_id': args.user_id}, previous['input']) if web_factory else None
                    def retry_discovery(request):
                        audit = previous['discovery'].get('audit', {})
                        sources.configure_official_plan(audit.get('official_source_plan'))
                        if journal_planner is not None:
                            sources.configure_journal_plan(audit.get('journal_source_plan'))
                        return discover_direction(request, sources, source_names=names, ranker=ranker,
                                                  store=store, web_fallback=web, saved_wechat=wechat)
                try:
                    event = recover_readings(store, recovery_key, args.user_id, reader_factory=reader_factory,
                                             discover=retry_discovery)
                finally:
                    if args.recover_discovery_event and web:
                        web.close()
                print(json.dumps({k: event[k] for k in ('event_id', 'status', 'reason')}))
        finally:
            sources.close()
            store.close()
        return
    execute = execution(store, sources=sources, ranker_factory=factory if encoder is not None else None, reader_factory=reader_factory,
                        understanding_calls=args.max_understanding_calls, source_planner=plan_official_sources,
                        web_factory=web_factory, journal_planner=journal_planner, wechat_sources=args.wechat_sources,
                        adaptive_search=bool(args.max_discovery_review_calls))
    run_loop(store, sources, execute, once=args.once, paid_model_calls_enabled=bool(args.max_model_calls),
             fetch_fulltexts=args.fetch_fulltexts, understanding_enabled=bool(args.max_understanding_calls),
             understanding_call_limit=args.max_understanding_calls,
             official_sources_enabled=True, web_fallback_enabled=allow_web_fallback,
             open_access_fulltexts_enabled=args.fetch_fulltexts and args.open_access_fulltexts,
             journal_sources_enabled=journal_planner is not None, wechat_sources_enabled=args.wechat_sources,
             recommendations_enabled=encoder is not None, shutdown=shutdown)


def run_loop(store, sources, execute, *, once=False, paid_model_calls_enabled=False, fetch_fulltexts=False,
             understanding_enabled=False, official_sources_enabled=False, web_fallback_enabled=False,
             journal_sources_enabled=False, open_access_fulltexts_enabled=False, wechat_sources_enabled=False,
             understanding_call_limit=None, recommendations_enabled=True, shutdown=None):
    import threading

    from .execution import project_lock

    # The OS releases the singleton lock on exit, including a crashed worker.
    with project_lock(store.root, 'recommendation-worker'):
        reconcile_interrupted(store)
        stopped = threading.Event()
        shutdown = shutdown or threading.Event()
        def heartbeat():
            try:
                while not stopped.is_set():
                    store.put('recommendation_worker', 'current', {'updated_at': now(), 'status': 'ready',
                               'mode': 'cpu_recommendations', 'paid_model_calls_enabled': paid_model_calls_enabled,
                               'fulltext_download_enabled': fetch_fulltexts, 'understanding_enabled': understanding_enabled,
                               'understanding_call_limit': understanding_call_limit,
                               'recommendations_enabled': recommendations_enabled,
                               'official_sources_enabled': official_sources_enabled, 'web_fallback_enabled': web_fallback_enabled,
                               'journal_sources_enabled': journal_sources_enabled,
                               'wechat_sources_enabled': wechat_sources_enabled,
                               'open_access_fulltexts_enabled': open_access_fulltexts_enabled})
                    stopped.wait(5)
            finally:
                store.close()
        pulse = threading.Thread(target=heartbeat, daemon=True)
        pulse.start()
        try:
            while not shutdown.is_set():
                result = process_one(store, execute=execute)
                if result:
                    print(json.dumps({k: result[k] for k in ('job_id', 'status', 'reason')}, ensure_ascii=False), flush=True)
                if once:
                    break
                if not result:
                    shutdown.wait(2)
        finally:
            stopped.set()
            pulse.join(timeout=10)
            store.put('recommendation_worker', 'current', {'updated_at': now(), 'status': 'stopped',
                       'mode': 'cpu_recommendations', 'paid_model_calls_enabled': paid_model_calls_enabled,
                       'fulltext_download_enabled': fetch_fulltexts, 'understanding_enabled': understanding_enabled,
                       'official_sources_enabled': official_sources_enabled, 'web_fallback_enabled': web_fallback_enabled})
            sources.close()
            store.close()



if __name__ == '__main__':
    main()
