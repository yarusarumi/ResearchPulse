"""A resumable draft workflow: acquire -> rank -> read -> daily/weekly evidence reports."""
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from .evidence_selection import select_evidence
from .execution import project_lock
from .feishu import inspect_inputs
from .grounded import analyze_grounded
from .html_reader import fetch_arxiv_html
from .journals import FEEDS, collect_journal
from .llm import FlashClient
from .models import EvidenceRecord, PaperRecord, SourceResult, now
from .project_updates import input_update_summary
from .projects import load_profile
from .remote_ranking import rank_remote
from .remote_reader import read_remote
from .reports import daily_report, reading_report
from .research import recommend, rerank_candidates
from .sources import Sources, deduplicate
from .store import Store
from .synthesis import render_synthesis, synthesize

PUBLIC_QUERIES = ['all:robot AND all:"imitation learning"', 'all:neuron AND (all:dendritic OR all:spiking)']

# New workflows freeze an explicit read allocation plan. Records without this marker keep the
# historical ranking[:read_limit] behavior and are never rewritten.
READING_ALLOCATION_POLICY = 'cached-retained-new-reads-v1'
CACHED_ANALYSIS_METHOD = 'source-span-label-scope-gates-v2'


def collect_pool(config, store, project, until, progress=print):
    since=(date.fromisoformat(until)-timedelta(days=6)).isoformat()
    results=[]

    def collect_one(source, query):
        adapter=Sources(config.sources.max_results)
        adapter.coordination_dir=store.root
        try:
            return adapter.collect(source,query,since,until)
        finally:
            adapter.close()

    jobs=[('arxiv',query) for query in PUBLIC_QUERIES]
    jobs += [('pubmed','"robot learning" OR "dendritic computation"'),('biorxiv','robot neuron dendritic spiking')]
    # arXiv queries stay sequential; the independent publishers and other APIs overlap.
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures=[workers.submit(collect_journal,name,since,until) for name in FEEDS]
        futures += [workers.submit(collect_one,*job) for job in jobs if job[0]!='arxiv']
        for source,query in jobs:
            if source=='arxiv':
                if results:
                    time.sleep(3)
                results.append(collect_one(source,query))
        for future in as_completed(futures):
            item=future.result()
            results.append(item)
            progress({'phase':'source','source':item.source,'status':item.status,'count':len(item.papers)})
    papers=deduplicate([p for r in results for p in r.papers])
    pool_id=uuid.uuid4().hex
    body={'pool_id':pool_id,'project_id':project,'created_at':now(),'since':since,'until':until,
          'query':'reviewed-public-workflow-queries-v1','sources':[r.model_dump() for r in results],
          'papers':[p.model_dump() for p in papers]}
    store.put('pool',pool_id,body,immutable=True)
    return body


def latest_papers(pool):
    latest={}
    for raw in pool['papers']:
        p=PaperRecord.model_validate(raw)
        v=int(p.version[1:]) if p.version.startswith('v') and p.version[1:].isdigit() else 0
        if p.paper_id not in latest or v>latest[p.paper_id][0]:
            latest[p.paper_id]=(v,p)
    return [row[1] for row in latest.values()]


def cached_analysis(store, profile, paper):
    candidates=[a for a in store.list('analysis') if (a.get('paper_id'),a.get('paper_version'),a.get('profile_version'))==
        (paper.paper_id,paper.version,profile.version) and a.get('verification_method')==CACHED_ANALYSIS_METHOD]
    # Preserve implementation-review exclusions when a reviewed derivative is available.
    return max(candidates,key=lambda a:('implementation_review_exclusions' in a,a['run_id'])) if candidates else None


def _build_reading_plan(store, profile, ranking, lookup, read_limit):
    items=[]
    new_reads=0
    for rec in ranking:
        paper=lookup[rec.paper_id]
        analysis=cached_analysis(store,profile,paper)
        if analysis:
            # Every valid cached analysis is retained in ranking order and consumes no new-read slot.
            items.append({'paper_id':paper.paper_id,'version':paper.version,'kind':'cached_analysis',
                          'analysis_id':analysis['run_id'],'profile_version':profile.version,
                          'verification_method':CACHED_ANALYSIS_METHOD})
        elif new_reads<read_limit:
            # Failed attempts keep their slot; later recommendations never backfill it.
            items.append({'paper_id':paper.paper_id,'version':paper.version,'kind':'new_read'})
            new_reads+=1
    return {'policy_version':READING_ALLOCATION_POLICY,'read_limit':read_limit,
            'frozen_at':now(),'items':items}


def _frozen_analysis(store, profile, paper, planned):
    """Resolve a frozen cache entry without ever substituting a fresh read."""
    if (planned.get('paper_id'),planned.get('version'))!=(paper.paper_id,paper.version):
        raise ValueError('Reading plan paper version changed; inspection required')
    try:
        analysis=store.get('analysis',planned['analysis_id'])
    except ValueError:
        raise ValueError('Frozen reading-plan analysis is missing; inspection required') from None
    expected={'paper_id':paper.paper_id,'paper_version':paper.version,'profile_version':profile.version,
              'verification_method':CACHED_ANALYSIS_METHOD,'run_id':planned.get('analysis_id')}
    if any(analysis.get(key)!=value for key,value in expected.items()):
        raise ValueError('Frozen reading-plan analysis identity mismatch; inspection required')
    return analysis


def _allocation_version(record):
    """Only a record without the field is legacy; unknown versions require inspection."""
    if 'reading_allocation_version' not in record:
        return None
    version=record['reading_allocation_version']
    if version != READING_ALLOCATION_POLICY:
        raise ValueError('Unknown reading allocation policy; inspection required')
    return version


def write_report(store, profile, report_date, run_id, kind, content, **extra):
    folder=store.root/'reports'
    folder.mkdir(exist_ok=True)
    target=folder/(kind+'-'+report_date+'-'+run_id+'.md')
    if target.exists() and target.read_text(encoding='utf-8')!=content:
        raise ValueError('Report version already exists with different contents')
    if not target.exists():
        target.write_text(content,encoding='utf-8')
    report_id=f'{profile.project_id}:{report_date}:{kind}:{run_id}'
    body={'report_id':report_id,'run_id':run_id,'project_id':profile.project_id,'profile_version':profile.version,
          'date':report_date,'kind':kind,'path':str(target),'content':content,'status':'draft',
          'human_review_required':True,**extra}
    store.put('report',report_id,body,immutable=True)
    store.put('report_latest',f'{profile.project_id}:{report_date}:{kind}',{'report_id':report_id})
    return body


def run_daily(config, project, report_date, *, pool_id=None, reader_release=None, read_limit=3,
              new_version=False, resume=None, progress=print, ranking_strategy='baseline', interest_mapping=None):
    with project_lock(config.data_dir, project):
        return _run_daily(config, project, report_date, pool_id=pool_id, reader_release=reader_release,
                          read_limit=read_limit, new_version=new_version, resume=resume, progress=progress,
                          ranking_strategy=ranking_strategy, interest_mapping=interest_mapping)


def _run_daily(config, project, report_date, *, pool_id=None, reader_release=None, read_limit=3,
               new_version=False, resume=None, progress=print, ranking_strategy='baseline', interest_mapping=None):
    if not 1<=read_limit<=5:
        raise ValueError('Daily deep-read limit must be 1–5')
    date.fromisoformat(report_date)
    store=Store(config.data_dir)
    profile=load_profile(store,project)
    reader_backend = config.reader_backend
    slot=f'{project}:{report_date}:{profile.version}'
    experimental = ranking_strategy == 'paperflow-experimental'
    if ranking_strategy not in ('baseline', 'paperflow-experimental'):
        raise ValueError('Unknown workflow ranking strategy')
    mapping = None
    if experimental:
        from .paperflow_adapter import validate_mapping
        if not interest_mapping or not reader_release:
            raise ValueError('Experimental PaperFlow needs an explicit mapping and deployed release')
        mapping = validate_mapping(profile, store.get('interest_mapping', interest_mapping))
        slot += ':paperflow:'+mapping.mapping_version
    previous=[x for x in store.list('workflow') if x['slot']==slot]
    if resume:
        record=store.get('workflow',resume)
        if record['slot']!=slot:
            raise ValueError('Resume project/date/profile mismatch')
        if reader_release and reader_release != record.get('reader_release'):
            raise ValueError('Resume must use the original deployed release')
        if record.get('read_limit', read_limit) != read_limit:
            raise ValueError('Resume must use the original reading limit')
        if record.get('reader_backend', 'ssh') != reader_backend:
            raise ValueError('Resume must use the original reader backend; inspect before controller migration')
        reader_release = record.get('reader_release')
    elif previous and not new_version:
        record=sorted(previous,key=lambda x:x['started_at'])[-1]  # noqa: FURB192 - preserve the last stored item on timestamp ties
        if experimental and record.get('interest_mapping') != interest_mapping:
            raise ValueError('Existing experiment uses another interest mapping identity')
        if pool_id and record.get('pool_id') and record['pool_id'] != pool_id:
            raise ValueError('Existing workflow uses another pool; request an explicit new version')
        if record['status'] in ('completed','partial'):
            _allocation_version(record)
            return {**record,'reused_existing_workflow':True}
        raise ValueError('Existing unfinished workflow; use --resume with its run ID')
    else:
        record={'run_id':'workflow-'+uuid.uuid4().hex,'slot':slot,'project_id':project,'date':report_date,
                'profile_version':profile.version,'started_at':now(),'status':'running',
                'code_revision':os.environ.get('NEUROEI_CODE_REVISION','unrecorded-working-tree'),
                'papers':[],'failures':[],'remote_llm_requests':0,'desktop_llm_requests':0,
                'quality':'draft_not_scientifically_accepted','reader_release':reader_release,
                'read_limit':read_limit,
                'reading_allocation_version':READING_ALLOCATION_POLICY,
                'reader_backend':reader_backend,
                'ranking_backend':reader_backend if experimental else 'not_used',
                'ranking_strategy':'paperflow-topics-bge-flash-experimental' if experimental else 'exploratory-bm25-flash',
                'experimental': experimental, 'interest_mapping': interest_mapping, 'public_query_policy':PUBLIC_QUERIES}
    allocation=_allocation_version(record)
    if experimental and record.get('interest_mapping') != interest_mapping:
        raise ValueError('Existing experiment uses another interest mapping identity')
    client=FlashClient(config.llm,store,record['run_id'])
    client.calls=record['desktop_llm_requests']

    def save():
        record['desktop_llm_requests']=client.calls
        store.put('workflow',record['run_id'],record)
        progress({'run_id':record['run_id'],'phase':record.get('phase'),'status':record['status'],
                  'desktop_calls':client.calls,'remote_calls':record['remote_llm_requests']})

    try:
        record.update(status='running',phase='collect')
        save()
        if 'project_inputs' not in record:
            record['project_inputs']=inspect_inputs(config.feishu,store)
            save()
        if 'project_input_update' not in record:
            record['project_input_update']=input_update_summary(store,profile,record['project_inputs'])
            save()
        if 'pool_id' not in record:
            pool=store.get('pool',pool_id) if pool_id else collect_pool(config,store,project,report_date,progress)
            if pool['project_id']!=project:
                raise ValueError('Candidate pool belongs to another project')
            record['pool_id']=pool['pool_id']
            save()
        pool=store.get('pool',record['pool_id'])
        papers=latest_papers(pool)
        lookup={p.paper_id:p for p in papers}
        record['phase']='rank'
        if 'recommendations' not in record:
            save()
            if experimental and papers:
                if 'ranking_features_id' not in record:
                    newly_assigned_rank = 'ranking_job_id' not in record
                    if newly_assigned_rank:
                        record['ranking_job_id'] = 'rankjob-'+uuid.uuid4().hex
                        save()
                    options = {'release':reader_release, 'job_id':record['ranking_job_id'],
                               'resume_only':not newly_assigned_rank}
                    if reader_backend == 'server-local':
                        from .local_ranking import rank_local
                        features = rank_local(config, profile, papers, mapping, **options)
                    else:
                        features = rank_remote(profile, papers, mapping, **options)
                    record['ranking_observation'] = {k:v for k,v in features.items()
                        if k in ('job_id','status','live_verified','error_type','detail')}
                    if features['status'] == 'awaiting_result':
                        record.update(status='awaiting_remote_rank', phase='rank')
                        return record
                    if features['status'] != 'completed':
                        raise ValueError('Remote PaperFlow feature job failed; no automatic strategy fallback')
                    store.put('ranking_features', features['job_id'], features, immutable=True)
                    record['ranking_features_id'] = features['job_id']
                    save()
                features = store.get('ranking_features', record['ranking_features_id'])
                ids = features['candidate_ids']
                ranking = rerank_candidates(client, profile, [lookup[pid] for pid in ids], limit=5)
            else:
                ranking=recommend(client,profile,papers,limit=5)
            record['recommendations']=[r.model_dump() for r in ranking]
            save()
        from .models import RankedPaper
        ranking=[RankedPaper.model_validate(r) for r in record['recommendations']]
        record['phase']='read'
        if allocation==READING_ALLOCATION_POLICY:
            if 'reading_plan' not in record:
                record['reading_plan']=_build_reading_plan(store,profile,ranking,lookup,read_limit)
                # The plan is frozen before any reader/fetch, so resume cannot reallocate slots.
                save()
            plan=record['reading_plan']
            if plan.get('policy_version')!=READING_ALLOCATION_POLICY or \
                    plan.get('read_limit')!=record.get('read_limit'):
                raise ValueError('Reading plan does not match this workflow; inspection required')
            planned_items=plan['items']
        else:
            planned_items=[{'paper_id':rec.paper_id,'kind':'legacy_ranking_slice'} for rec in ranking[:read_limit]]
        for planned in planned_items:
            if planned['paper_id'] not in lookup:
                raise ValueError('Reading plan paper is missing from the frozen pool; inspection required')
            frozen=None
            if planned.get('kind')=='cached_analysis':
                frozen=_frozen_analysis(store,profile,lookup[planned['paper_id']],planned)
            item=next((x for x in record['papers'] if x['paper_id']==planned['paper_id']),None)
            if frozen is not None and item and item.get('remote_job_id'):
                # The original handle owns any remote budget; a cache hit must not overwrite it.
                raise ValueError('Frozen cached reading has an assigned remote job; observation or inspection required')
            if frozen is not None and item and item.get('status')=='completed' and \
                    item.get('analysis_id')!=frozen['run_id']:
                raise ValueError('Completed reading does not match the frozen plan; inspection required')
            if item and item['status'] not in ('pending','awaiting_remote_result'):
                continue
            if item and item['status']=='awaiting_remote_result' and not item.get('remote_job_id'):
                # Legacy records did not persist an input identity before dispatch. Never invent a replacement.
                item['error_type']='LegacyReaderHandleRequiresInspection'
                record.update(status='awaiting_remote_read',phase='read')
                return record
            paper=lookup[planned['paper_id']]
            if planned.get('version') not in (None,paper.version):
                raise ValueError('Reading plan paper version changed; inspection required')
            record['current_paper']=paper.paper_id
            save()
            if item is None:
                item={'paper_id':paper.paper_id,'version':paper.version,'status':'pending','model_calls':0}
                record['papers'].append(item)
            # An assigned job is always reobserved before considering another reader/cache.
            if planned.get('kind')=='cached_analysis':
                analysis=frozen
            elif planned.get('kind')=='new_read':
                analysis=None
            else:
                analysis=None if item.get('remote_job_id') else cached_analysis(store,profile,paper)
            if analysis:
                item.update(status='completed',analysis_id=analysis['run_id'],reader='cached_checked_analysis')
            else:
                remaining=config.llm.max_calls_per_run-client.calls-record['remote_llm_requests']
                if item.get('remote_job_id') or (reader_release and remaining>=2 and paper.paper_id.startswith('arxiv:')):
                    newly_assigned = 'remote_job_id' not in item
                    if newly_assigned:
                        item.update(remote_job_id='job-'+uuid.uuid4().hex,remote_budget=min(4,remaining),
                                    remote_accounted_calls=min(4,remaining))
                        record['remote_llm_requests']+=item['remote_accounted_calls']
                        # Persist job and conservative budget reservation before remote work starts.
                        save()
                    reader = read_remote
                    if reader_backend == 'server-local':
                        from .local_reader import read_local
                        reader = read_local
                    remote=reader(config,profile,paper,release=reader_release,
                                       budget=item['remote_budget'],job_id=item['remote_job_id'],
                                       resume_only=not newly_assigned)
                    accounted=item['remote_budget'] if remote.get('billing_unknown') else remote['llm_requests']
                    record['remote_llm_requests']+=accounted-item['remote_accounted_calls']
                    item['remote_accounted_calls']=accounted
                    item['remote_result']={k:v for k,v in remote.items() if k!='analysis'}
                    analysis=remote.get('analysis') if remote['status']=='completed' else None
                    if remote['status']=='awaiting_result':
                        item.update(status='awaiting_remote_result',error_type='ObservationPendingNoAutomaticRestart')
                        record.update(status='awaiting_remote_read',phase='read')
                        return record
                    if remote.get('billing_unknown'):
                        item.update(status='unavailable',error_type=remote.get('error_type','UnknownRemoteBilling'))
                    else:
                        item['status']='pending'
                    if analysis:
                        item['reader']='paperqa-docling-cpu'
                client.config=client.config.model_copy(update={'max_calls_per_run':max(1,config.llm.max_calls_per_run-record['remote_llm_requests'])})
                if analysis is None and item['status']=='pending' and client.calls+record['remote_llm_requests']+2<=config.llm.max_calls_per_run:
                    try:
                        if paper.paper_id.startswith('arxiv:'):
                            payload=fetch_arxiv_html(paper)
                            reader_label='official-html-explicit-fallback'
                        else:
                            from .pmc_reader import fetch_pmc_xml
                            payload=fetch_pmc_xml(paper)
                            reader_label='europepmc-oa-xml'
                        evidence=[EvidenceRecord.model_validate(e) for e in payload['evidence']]
                        supplementary=None
                        section_context=dict(payload['section_context'])
                        if payload.get('provenance',{}).get('service')=='europepmc' and payload.get('supplementary_materials'):
                            from .supplement_materials import acquire_supplement_text
                            try:
                                supplementary=acquire_supplement_text(payload,config.data_dir/'supplements')
                                evidence.extend(EvidenceRecord.model_validate(e) for e in supplementary['evidence'])
                                section_context.update(supplementary['section_context'])
                            except Exception as exc:  # noqa: BLE001 - keep body evidence when supplementary access fails
                                supplementary={'status':'unavailable','error_type':type(exc).__name__,'files':[],
                                    'evidence':[],'limitations':['Supplementary acquisition was not completed: '+type(exc).__name__]}
                        query=' '.join(f.text for f in profile.facts if f.kind=='question')
                        selected,retrieval=select_evidence(evidence,query,section_context)
                        before=client.calls
                        analysis=analyze_grounded(client,profile,paper,selected,
                            'Identify methods, controlled results, limitations and relevance to project questions.')
                        analysis.update(run_id=record['run_id']+':'+paper.paper_id+':'+paper.version,
                            code_revision=record['code_revision'],available_blocks=len(evidence),selected_blocks=len(selected),
                            retrieval=retrieval,limitations=list(payload['limitations']),
                            source_integrity=payload.get('source_integrity'))
                        if 'provenance' in payload:
                            analysis.update(acquisition=payload['provenance'],
                                            supplementary_materials=payload.get('supplementary_materials',[]))
                        if supplementary is not None:
                            analysis['supplementary_acquisition']={k:v for k,v in supplementary.items()
                                if k not in ('evidence','section_context')}
                            analysis['limitations']+=supplementary['limitations']
                            if supplementary['status']!='unavailable':
                                analysis['limitations']=[x for x in analysis['limitations']
                                    if x!='supplementary files are recorded only and were not fetched']
                        item.update(reader=reader_label,model_calls=client.calls-before)
                    except Exception as exc:  # noqa: BLE001 - one unavailable paper must not discard the draft
                        item.update(status='unavailable',error_type=type(exc).__name__)
                if analysis:
                    store.put('analysis',analysis['run_id'],analysis,immutable=True)
                    item.update(status='completed',analysis_id=analysis['run_id'])
                    item.pop('error_type',None)
                elif item['status']=='pending':
                    item.update(status='unavailable',error_type='NoFulltextOrBudget')
            if analysis and 'supplementary_acquisition' in analysis:
                supp=analysis['supplementary_acquisition']
                item.update(supplementary_status=supp['status'],
                            supplement_files_parsed=sum(f['status'] in ('parsed_text','partial_text') for f in supp['files']))
            save()
        entries=[{'paper':lookup[p['paper_id']], 'analysis':store.get('analysis',p['analysis_id'])}
                 for p in record['papers'] if p['status']=='completed']
        record['phase']='report'
        save()
        if 'synthesis' not in record:
            try:
                record['synthesis']=synthesize(client,profile,entries,{'since':pool['since'],'until':pool['until']})
            except Exception as exc:  # noqa: BLE001 - preserve explicit synthesis failure without upstream bodies
                record['synthesis']={'insights':[],'claim_lookup':{},'evidence_gaps':['科研综合未完成：'+type(exc).__name__]}
            save()
        content=daily_report(project=profile,report_date=report_date,run_id=record['run_id'],
            sources=[SourceResult.model_validate(s) for s in pool['sources']],papers=papers,recommendations=ranking,
            reading_statuses={p['paper_id']:p for p in record['papers']},
            reading_report_included=bool(entries),
            ranking_strategy=record['ranking_strategy'], experimental=experimental)
        content+=render_synthesis(record['synthesis'], profile=profile)
        content+='\n## 项目输入同步\n\n当前分析使用已生效档案 '+profile.version+'；新抓取文档不自动改写项目事实。\n\n'
        for source in record.get('project_inputs',[]):
            content+='- '+(source.get('title') or source['wiki_node'])+'：'+source['status']+'；'+source.get('detail',source.get('profile_update',''))+'\n'
        update=record.get('project_input_update')
        if update:
            counts=update['decisions']
            content+='\n本次来源的增量提取：'+str(update['candidate_blocks'])+'个差异块；支持已有项'+str(counts.get('supports_existing',0)) \
                +'、可能新增'+str(counts.get('new_fact',0))+'、其他项目'+str(counts.get('unrelated',0)) \
                +'、上下文不足'+str(counts.get('needs_context',0))+'。状态：'+update['status'] \
                +'。此分类不代表科研验收，生效档案未改变。追溯键：'+update['update_key']+'\n'
        if entries:
            content+='\n'+reading_report(project=profile,report_date=report_date,run_id=record['run_id'],entries=entries)
        content+='\n## 本次读取状态\n\n'+ '\n'.join('- '+p['paper_id']+'：'+p['status']+'；'+p.get('reader',p.get('error_type','')) for p in record['papers'])+'\n'
        content+='\nSupplementary获取与解析状态见各篇阅读记录；机构订阅尚未接通，未读取的材料不作为证据。\n'
        report=write_report(store,profile,report_date,record['run_id'],
            'workflow-experimental-daily' if experimental else 'workflow-daily',content,
            pool_id=record['pool_id'],analysis_ids=[e['analysis']['run_id'] for e in entries],
            synthesis=record['synthesis'],workflow_id=record['run_id'], experimental=experimental,
            ranking_strategy=record['ranking_strategy'],
            **({'project_input_update':update} if update else {}))
        record.update(report_id=report['report_id'],report_path=report['path'],
            status='partial' if any(p['status']!='completed' for p in record['papers']) or
            any(s['status'] in ('failed','partial') for s in pool['sources']) else 'completed',phase='finished',ended_at=now())
    except Exception as exc:  # noqa: BLE001 - persist a sanitized resumable workflow checkpoint
        record.update(status='failed',error_type=type(exc).__name__,ended_at=now())
        if type(exc) is ValueError:
            record['error_detail']=str(exc)[:300]
    finally:
        save()
        client.close()
    return record


def run_weekly(config,project,until,*,since=None,new_version=False):
    with project_lock(config.data_dir, project):
        return _run_weekly(config,project,until,since=since,new_version=new_version)


def _run_weekly(config,project,until,*,since=None,new_version=False):
    store=Store(config.data_dir)
    profile=load_profile(store,project)
    since=since or (date.fromisoformat(until)-timedelta(days=6)).isoformat()
    reports=[r for r in store.list('report') if r.get('kind')=='workflow-daily' and r['project_id']==project
             and r['profile_version']==profile.version and since<=r['date']<=until]
    latest={}
    for report in reports:
        pointer=store.get('report_latest',f"{project}:{report['date']}:workflow-daily")
        if report['report_id']==pointer['report_id']:
            latest[report['date']]=report
    selected=sorted(latest.values(),key=lambda r:r['date'])
    if not selected:
        raise ValueError('No evidence-bearing workflow daily reports in the requested period')
    inputs=[r['report_id'] for r in selected]
    old=[r for r in store.list('report') if r.get('kind')=='workflow-weekly' and r.get('input_report_ids')==inputs
         and r.get('period')=={'since':since,'until':until}]
    if old and not new_version:
        return old[-1]
    poolpapers={(p['paper_id'],p['version']):PaperRecord.model_validate(p) for report in selected
                for p in store.get('pool',report['pool_id'])['papers']}
    ids=list(dict.fromkeys(aid for report in selected for aid in report['analysis_ids']))
    entries=[{'analysis':store.get('analysis',aid)} for aid in ids]
    for entry in entries:
        entry['paper']=poolpapers[(entry['analysis']['paper_id'],entry['analysis']['paper_version'])]
    run_id='weekly-'+uuid.uuid4().hex
    client=FlashClient(config.llm,store,run_id)
    try:
        result=synthesize(client,profile,entries,{'since':since,'until':until})
    finally:
        client.close()
    content=f'# ResearchPulse 项目研究周报草稿 · {since} — {until}\n\n项目：{project}；档案：{profile.version}。\n\n'
    content+=f'实际覆盖 {len(selected)} 个日报日期、{len(entries)} 份论文分析；不足7天不代表已连续运行7天。所有综合判断待审阅。\n\n'
    content+='输入日报：\n\n'+'\n'.join('- ['+r['date']+']('+Path(r['path']).as_posix()+')' for r in selected)+'\n'
    content+=render_synthesis(result, profile=profile)
    return write_report(store,profile,until,run_id,'workflow-weekly',content,input_report_ids=inputs,
                        period={'since':since,'until':until},synthesis=result,llm_requests=client.calls)
