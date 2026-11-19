"""A bounded manual daily run for independent user directions; no Feishu or scheduler writes."""
import re
from datetime import date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import code_revision
from .direction_feedback import feedback_snapshot
from .direction_queries import (
    ERROR_MESSAGES,
    SEARCH_POLICY_VERSION,
    DirectionChanged,
    resolve_direction_queries,
    search_result_summary,
)
from .directions import (
    direction_paper_history,
    load_direction,
    optional,
    require_direction_project,
    resolve_direction_context,
)
from .evidence_selection import select_evidence
from .execution import project_lock
from .grounded import EXTRACTION_VERSION, analyze_grounded
from .html_reader import fetch_arxiv_html
from .llm import FlashClient
from .media import media_snapshot, render_media_digest
from .media_links import analyze_media, render_media_links
from .models import EvidenceRecord, SourceResult, now
from .reports import daily_report, reading_report
from .research import recommend
from .run_diagnostics import analysis_assurance
from .sources import Sources, deduplicate
from .store import Store
from .synthesis import render_synthesis, synthesize
from .workflow import latest_papers

ZONE = ZoneInfo('Asia/Shanghai')


class DirectionClient(FlashClient):
    def complete(self, *args, **kwargs):
        self.check_current()
        return super().complete(*args, **kwargs)


MAX_READS_PER_RUN = 3
SYNTHESIS_RESERVE = 2


def clean_report_text(content, facts=()):
    """Hide internal run identifiers while keeping claim headings and their anchors intact."""
    parts = re.split(r'(<details><summary>追溯详情</summary>[\s\S]*?</details>)', content)
    if len(parts) > 1:
        return ''.join(part if part.startswith('<details><summary>追溯详情</summary>')
                       else clean_report_text(part, facts) for part in parts)
    content = re.sub(r'directionrun-[a-f0-9]{32}', '本次运行', content)
    content = re.sub(r'workflow-[a-f0-9]{8,}', '一次检索', content)
    content = re.sub(r'证据[0-9a-f]{8,}', '相关证据', content)
    content = content.replace('### 综合判断的证据核查', '#### 证据原文摘录')
    content = re.sub(r'^追溯编号：.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\[([^\]]{1,40})\]\(相关证据\)', r'\1', content)
    content = re.sub(r'（arxiv:[^）]*:html:[^）]*）', '', content)
    content = re.sub(r' · arxiv:[\w.:-]+', '', content)
    content = re.sub(r'claim-ref：[^；\n]*；本地锚点：[^\n]*', '对应原文摘录：见下方', content)
    content = re.sub(r'claim-\d+(?![0-9a-f])', '该条判断', content)
    content = content.replace('数字检查：literal_numbers_present', '数字检查：数字已在原文找到')
    content = content.replace('数字检查：checked', '数字检查：数字已在原文找到')
    content = content.replace('全文精读', '正文选段阅读')
    content = content.replace('原文绑定：source_spans_present', '原文绑定：已定位原文片段')
    content = content.replace('原文绑定：supported', '原文绑定：已对应原文')
    content = re.sub(r'\n{3,}', '\n\n', content)
    for fact in facts:
        if fact.kind == 'question' and fact.id in content:
            label = fact.text if len(fact.text) <= 40 else fact.text[:40] + '…'
            content = content.replace(fact.id, '「' + label + '」')
    return content


def keep_renderable_entries(context, day, run_id, entries):
    """Drop readings that the report audit would reject, so one stale reading cannot fail the run."""
    kept = []
    for entry in entries:
        try:
            reading_report(project=context, report_date=day, run_id=run_id, entries=[entry],
                           context_kind='direction')
        except ValueError:
            continue
        kept.append(entry)
    return kept


def cached_analysis_usable(analysis, paper, context):
    """A cached reading may be stale after a direction/profile change; drop it instead of failing the report."""
    if (analysis.get('paper_id'), analysis.get('paper_version'), analysis.get('profile_version')) != (
            paper.paper_id, paper.version, context.version):
        return False
    questions = {f.id for f in context.facts if f.kind == 'question'}
    evidence = {e.get('evidence_id') for e in analysis.get('evidence') or []}
    for accepted in analysis.get('accepted_claims') or []:
        claim = accepted.get('claim') or {}
        if not set(claim.get('evidence_ids') or []) <= evidence:
            return False
        if not set(claim.get('question_ids') or []) <= questions:
            return False
        if accepted.get('parser_quality_flags'):
            return False
    return True


def run_direction_daily(config, user, direction_id, *, report_date=None, pool_id=None, read_limit=1,
                        run_id=None, progress=print, rerun_of=None):
    if not 0 <= read_limit <= MAX_READS_PER_RUN:
        raise ValueError('Direction run allows at most three new reads')
    if run_id is not None and not re.fullmatch(r'directionrun-[a-f0-9]{32}', run_id):
        raise ValueError('Invalid direction run identity')
    user = user.strip()
    day = report_date or datetime.now(ZONE).date().isoformat()
    if date.fromisoformat(day) > datetime.now(ZONE).date():
        raise ValueError('Direction report cannot be future dated')
    store = Store(config.data_dir)
    direction = load_direction(store, user, direction_id)
    if not direction['enabled']:
        raise ValueError('This direction is paused')
    did = direction['direction_id']
    with project_lock(store.root, 'direction-run:' + did):
        context, project_context = resolve_direction_context(store, direction)
        prior_runs = [r for r in store.list('direction_run') if r['direction_id'] == did and r['user_id'] == user]
        same = [r for r in prior_runs if (r['date'], r['direction_version']) == (day, direction['version'])
                and r.get('search_policy_version') == SEARCH_POLICY_VERSION]
        if same:
            prior = max(same, key=lambda r: r['started_at'])
            if project_context and prior.get('context_version') != context.version:
                raise ValueError('Linked project profile changed since this daily run; inspect before paid rerun')
            if prior['status'] not in ('completed', 'partial'):
                raise ValueError('Existing direction run needs inspection; no automatic redispatch')
            if rerun_of is None:
                return {**prior, 'reused': True}
            if rerun_of != prior['run_id']:
                raise ValueError('Requested rerun is no longer the latest run; no redispatch')
        elif rerun_of is not None:
            raise ValueError('Requested rerun has no matching same-day run')
        social = media_snapshot(store, user, did, report_date=day)
        record = {'run_id': run_id or 'directionrun-' + uuid4().hex, 'user_id': user, 'direction_id': did,
            'direction_version': direction['version'], 'context_version': context.version, 'date': day,
            'started_at': now(), 'status': 'running', 'phase': 'collect', 'llm_requests': 0,
            'code_revision': code_revision(),
            'max_llm_requests': min(20, config.llm.max_calls_per_run), 'read_limit': read_limit, 'papers': [],
            'scientific_acceptance': False, 'generated_feishu_upload_enabled': False,
            'search_policy_version': SEARCH_POLICY_VERSION,
            'media_versions': [item['version_id'] for item in social['items']],
            'media_selection': social['selection'], 'media_captured_at': social['captured_at']}
        if rerun_of is not None:
            record['rerun_of'] = rerun_of
        if project_context:
            record['project_context'] = project_context
        record['feedback_snapshot'] = feedback_snapshot(
            store, user, did, datetime.fromisoformat(record['started_at']))
        client = DirectionClient(config.llm.model_copy(update={'max_calls_per_run': record['max_llm_requests'],
                                  'request_attempts': 1, 'schema_repair_attempts': 1}), store, record['run_id'])
        client.hold(SYNTHESIS_RESERVE)

        def save():
            record['llm_requests'] = client.calls
            store.put('direction_run', record['run_id'], record)
            progress({k: record[k] for k in ('run_id', 'status', 'phase', 'llm_requests')})

        def current():
            latest = load_direction(store, user, did)
            if not latest['enabled'] or latest['version'] != direction['version']:
                raise DirectionChanged('Direction paused or changed during run; no more model requests')
            if project_context:
                require_direction_project(store, latest)
                if store.get('active_profile', project_context['project_id'])['key'] != project_context['profile_key']:
                    raise DirectionChanged('Linked project profile changed during run')

        client.check_current = current
        save()
        try:
            if pool_id:
                pool = store.get('pool', pool_id)
                if pool['until'] > day:
                    raise ValueError('Stored pool is from a future reporting window')
                record.update(pool_id=pool_id, source_mode='explicit_public_paper_pool_replay')
            else:
                if not config.sources.enabled and not config.sources.journals:
                    raise ValueError('No enabled public sources')
                record['phase'] = 'plan_queries'
                save()
                plan = resolve_direction_queries(client, store, direction)
                record.update(effective_queries=plan['queries'], query_plan_origin=plan['origin'],
                              query_plan_id=plan['plan_id'], phase='collect', sources_completed=0,
                              sources_planned=len(plan['queries']) * len(config.sources.enabled) + len(config.sources.journals))
                save()
                since = (date.fromisoformat(day) - timedelta(days=6)).isoformat()
                results = []
                adapter = Sources(min(config.sources.max_results, 30))
                adapter.coordination_dir = store.root
                try:
                    for query in plan['queries']:
                        for source in config.sources.enabled:
                            current()
                            result = adapter.collect(source, query, since, day)
                            results.append(result)
                            record['source_summary'] = [{k: getattr(r, k) for k in
                                ('source', 'query', 'status', 'error')} | {'count': len(r.papers)} for r in results]
                            record['sources_completed'] = len(results)
                            record['sources_planned'] = len(plan['queries']) * len(config.sources.enabled) + len(config.sources.journals)
                            save()
                finally:
                    adapter.close()
                # Journal feeds have no query endpoint: collect once per run, then let
                # the same relevance ranking evaluate them alongside queried metadata.
                from .journals import collect_journal
                for journal in config.sources.journals:
                    current()
                    results.append(collect_journal(journal, since, day))
                    record['source_summary'] = [{k: getattr(r, k) for k in
                        ('source', 'query', 'status', 'error')} | {'count': len(r.papers)} for r in results]
                    record['sources_completed'] = len(results)
                    save()
                pid = 'directionpool-' + uuid4().hex
                pool = {'pool_id': pid, 'since': since, 'until': day,
                        'sources': [result.model_dump() for result in results],
                        'papers': [p.model_dump() for p in deduplicate([p for r in results for p in r.papers])],
                        'direction_id': did, 'user_id': user}
                store.put('pool', pid, pool, immutable=True)
                record.update(pool_id=pid, source_mode='live_direction_public_queries')
            papers = latest_papers(pool)
            ranking_query = ' '.join(record.get('effective_queries') or
                                     list(dict.fromkeys(s['query'] for s in pool['sources'])))
            record['ranking_query'] = ranking_query
            record['source_summary'] = [{k: s.get(k) for k in ('source', 'query', 'status', 'error')}
                | {'count': len(s['papers'])} for s in pool['sources']]
            record.update(phase='rank', candidate_count=len(papers))
            # Deterministic association/screening only; media text never enters paper analysis.
            media_links = analyze_media(store, user, did, pool_id=pool['pool_id'], report_date=day)
            record['media_links'] = {'pool_id': media_links['pool_id'], **media_links['counts']}
            save()
            current()
            ranking = recommend(client, context, papers, limit=3, context_kind='direction',
                                retrieval_query=ranking_query)
            record['recommendations'] = [r.model_dump() for r in ranking]
            record['result_kind'], record['summary'] = search_result_summary(pool['sources'], len(papers), len(ranking))
            history = direction_paper_history(store, prior_runs, user, did, day)
            known = history['first_recommended']
            previous = max([r for r in prior_runs if r['status'] in ('completed', 'partial') and r['date'] <= day],
                           key=lambda r: (r['date'], r['started_at']), default=None)
            record.update(previous_run_id=previous['run_id'] if previous else None, new_papers=[],
                          new_fulltext_analyses=[], first_seen_recommendations=[], phase='read',
                          history_policy_version='candidate-history-v1',
                          candidate_history_complete=history['candidate_history_complete'])
            save()
            entries, attempted = [], 0
            lookup = {paper.paper_id: paper for paper in papers}
            for rec in ranking:
                current()
                paper = lookup[rec.paper_id]
                item = {'paper_id': paper.paper_id, 'version': paper.version, 'title': paper.title,
                        'status': 'abstract_only', 'new_to_direction': (paper.paper_id, paper.version) not in known}
                identity = (paper.paper_id, paper.version)
                seen_before = history['first_seen'].get(identity)
                item.update(first_seen_date=seen_before or (day if history['candidate_history_complete'] else None),
                            first_recommended_date=known.get(identity, day),
                            first_seen_this_run=False if seen_before else
                            True if history['candidate_history_complete'] else None)
                if item['first_seen_this_run'] is True:
                    record['first_seen_recommendations'].append({'paper_id': paper.paper_id, 'version': paper.version})
                if item['new_to_direction']:
                    record['new_papers'].append({'paper_id': paper.paper_id, 'version': paper.version})
                cached = [a for a in store.list('direction_analysis')
                          if (a['paper_id'], a['paper_version'], a['profile_version'], a.get('direction_id')) ==
                             (paper.paper_id, paper.version, context.version, did)
                          and a.get('retrieval', {}).get('strategy') in
                              {'bm25-section-coverage-v2', 'bm25-section-coverage-v3'}]
                superseded = {a.get('supersedes_analysis_id') for a in cached}
                cached = [a for a in cached if a['run_id'] not in superseded
                          and (a.get('accepted_claims') or a.get('extraction_version') == EXTRACTION_VERSION)]
                run_times = {r['run_id']: r['started_at'] for r in prior_runs}
                analysis = max(cached, key=lambda a: (
                    run_times.get(a['run_id'].split(':', 1)[0], ''), a['run_id'])) if cached else None
                if analysis is None and attempted < read_limit and client.remaining() >= 4:
                    attempted += 1
                    try:
                        if paper.paper_id.startswith('arxiv:'):
                            payload = fetch_arxiv_html(paper)
                        else:
                            from .pmc_reader import fetch_pmc_xml
                            payload = fetch_pmc_xml(paper)
                        evidence = [EvidenceRecord.model_validate(e) for e in payload['evidence']]
                        selected, retrieval = select_evidence(evidence, ranking_query or
                                                              ' '.join(f.text for f in context.facts),
                                                              payload['section_context'])
                        current()
                        analysis = analyze_grounded(client, context, paper, selected,
                            'Explain methods, controlled results and limitations relevant to the user-declared direction.')
                        analysis.update(run_id=record['run_id'] + ':' + paper.paper_id, direction_id=did,
                                        retrieval=retrieval, limitations=payload['limitations'],
                                        source_integrity=payload.get('source_integrity'),
                                        code_revision=record['code_revision'], available_blocks=len(evidence),
                                        selected_blocks=len(selected))
                        store.put('direction_analysis', analysis['run_id'], analysis, immutable=True)
                        record['new_fulltext_analyses'].append(paper.paper_id)
                    except DirectionChanged:
                        raise
                    except Exception as exc:  # noqa: BLE001 - preserve a draft with explicit reading gaps
                        item.update(status='unavailable', error_type=type(exc).__name__,
                                    reading_skip_reason='reading_error')
                        record.setdefault('reading_errors', {})[paper.paper_id] = (
                            type(exc).__name__ + ': ' + str(exc)[:160])
                elif analysis is None:
                    item['reading_skip_reason'] = ('read_limit' if attempted >= read_limit else 'model_budget')
                if analysis:
                    source_run = optional(store, 'direction_run', analysis['run_id'].split(':', 1)[0])
                    item['analysis_date'] = (source_run or {}).get('date') or (None if cached else day)
                    item.update(status='completed', analysis_id=analysis['run_id'],
                                reader='cached_direction_analysis' if cached else 'official_public_text',
                                assurance=analysis_assurance(analysis))
                    entries.append({'paper': paper, 'analysis': analysis,
                                    'analysis_date': item['analysis_date'], 'reading_reused': bool(cached)})
                record['papers'].append(item)
                save()
            record['reading_attempts'] = attempted
            report_entries = keep_renderable_entries(context, day, record['run_id'], entries)
            for entry in entries:
                if entry in report_entries:
                    continue
                for item in record['papers']:
                    if item['paper_id'] == entry['paper'].paper_id:
                        item.update(status='unavailable', reading_skip_reason='stale_analysis')
            current()
            record['phase'] = 'report'
            save()
            client.release(SYNTHESIS_RESERVE)
            try:
                record['synthesis'] = synthesize(client, context, report_entries, {
                    'since': pool['since'], 'until': pool['until'],
                    'previous_progress': {'run_id': previous['run_id'], 'date': previous['date'],
                        'evidence_gaps': previous.get('synthesis', {}).get('evidence_gaps', [])} if previous else None,
                    'new_paper_ids': [p['paper_id'] for p in record['first_seen_recommendations']],
                    'first_recommended_paper_ids': [p['paper_id'] for p in record['new_papers']],
                    'new_fulltext_analysis_ids': record['new_fulltext_analyses'],
                    'scope': 'User-declared direction; distinguish newly found items, deeper readings and repeated findings.'},
                    context_kind='direction')
            except DirectionChanged:
                raise
            except Exception as exc:  # noqa: BLE001 - a budget or synthesis failure preserves located readings
                record['synthesis'] = {'insights': [], 'claim_lookup': {}, 'evidence_gaps': []}
                record['synthesis_unavailable'] = True
                record['synthesis_failure'] = type(exc).__name__ + ': ' + str(exc)[:120]
            content = f"**关注方向：** {direction['title']}\n\n"
            content += f"{day} · {user}\n\n"
            if project_context:
                content += ('已结合关联项目 ' + project_context['project_id'] + ' 的已启用研究档案（'
                            + str(project_context['fact_count']) + ' 条事实、目标或问题；来源日期截至 '
                            + (project_context['source_date'] or '未知')
                            + '）。历史实验记录不代表当前实验状态，未读取的图片和附件仍不作为证据。\n\n')
            content += '**本期结论：' + record['summary'] + '**\n\n'
            if record.get('effective_queries'):
                content += '检索词：' + '；'.join(record['effective_queries']) + '。\n\n'
            content += ('上一期报告：' + previous['date'] + '\n\n') if previous else '这是该方向的第一份报告。\n\n'
            if record['source_mode'] == 'explicit_public_paper_pool_replay':
                content += '本次显式复用既有公开论文候选池，未重新采集；范围：' + pool['since'] + ' 至 ' + pool['until'] + '。\n\n'
            content += daily_report(project=context, report_date=day, run_id=record['run_id'],
                sources=[SourceResult.model_validate(s) for s in pool['sources']], papers=papers,
                recommendations=ranking, context_kind='direction',
                reading_statuses={p['paper_id']: p for p in record['papers']},
                reading_report_included=bool(report_entries))
            if record.get('synthesis_unavailable'):
                content += ('\n## 科研综合未完成\n\n本期科研综合没有跑通（' + record.get('synthesis_failure', '原因未记录')
                            + '）。推荐、阅读与单篇分析的记录都已保留；这一步失败说明跨文献综合不可用，'
                            '不代表本期没有进展，也不应据此判断某个方向没有新发现。\n\n')
                gaps = [gap for gap in (record['synthesis'].get('evidence_gaps') or []) if gap]
                if gaps:
                    content += '\n'.join('- 证据缺口：' + gap for gap in gaps) + '\n\n'
            elif record['synthesis'].get('insights') or record['synthesis'].get('evidence_gaps'):
                content += render_synthesis(record['synthesis'], context_kind='direction', profile=context)
            else:
                content += ('\n## 科研综合\n\n本期没有形成可支持的跨文献判断，也没有记录到明确的证据缺口；'
                            '这不等于没有进展，请结合上面的推荐与阅读记录人工判断。\n\n')
            if report_entries:
                content += reading_report(project=context, report_date=day, run_id=record['run_id'],
                                          entries=report_entries, context_kind='direction')
            content = clean_report_text(content, context.facts)
            if social:
                content += render_media_digest(social)
            if media_links:
                content += render_media_links(media_links)
            folder = store.root / 'directions' / did
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (record['run_id'] + '.md')
            with target.open('x', encoding='utf-8') as stream:
                stream.write(content)
            store.put('direction_report', record['run_id'], {'run_id': record['run_id'], 'user_id': user,
                'direction_id': did, 'direction_version': direction['version'], 'date': day,
                'content': content, 'path': str(target), 'media_versions': record['media_versions'],
                **({'project_context': project_context} if project_context else {})}, immutable=True)
            record.update(report_id=record['run_id'], report_path=str(target), phase='finished',
                status='failed' if record['result_kind'] == 'search_failed' else
                'partial' if any(s['status'] in ('partial', 'failed') for s in pool['sources'])
                or any(p['status'] != 'completed' and p.get('reading_skip_reason') != 'read_limit'
                       for p in record['papers'])
                or record.get('synthesis_unavailable') else 'completed')
        except DirectionChanged:
            record.update(status='cancelled', error_type='DirectionChanged', summary=ERROR_MESSAGES['DirectionChanged'])
        except Exception as exc:  # noqa: BLE001 - persist sanitized state without model bodies or credentials
            code = type(exc).__name__
            record.update(status='failed', error_type=code, result_kind='execution_failed',
                          summary=ERROR_MESSAGES.get(code, '运行未完成；请查看失败阶段，不应据此判断没有相关研究。'))
            if getattr(exc, 'code', None) == 'invalid_recommendation_references':
                record['error_code'] = exc.code
                record['summary'] = '推荐引用校验未通过；未发布不可靠推荐，请稍后重试或联系维护者。'
        finally:
            record['ended_at'] = now()
            save()
            client.close()
        return record
