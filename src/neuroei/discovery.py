"""Request -> bounded discovery -> exact identity union -> PaperFlow shortlist.

No implicit model/network invocation: sources and ranking are explicit capabilities.
"""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

import httpx

from .discovery_contract import Candidate, DiscoveryResult, RecommendationRequest
from .discovery_sources import publication_basis
from .sources import normalize_doi

TZ = ZoneInfo('Asia/Shanghai')
POLICY = 'discovery-v1.8-query-pilot'


def keys(paper):
    result = set()
    if paper.doi:
        result.add('doi:'+normalize_doi(paper.doi))
    if paper.paper_id.startswith('arxiv:'):
        result.add(re.sub(r'v\d+$', '', paper.paper_id))
    if paper.paper_id.startswith('doi:'):
        result.add('doi:'+normalize_doi(paper.paper_id[4:]))
    return result or {paper.paper_id}


def merge(papers):
    """Transitive exact union. Titles only flag uncertainty, never silently merge."""
    groups = []
    for paper in papers:
        matches = [g for g in groups if keys(paper) & set().union(*(keys(p) for p in g))]
        group = [paper]
        for old in matches:
            groups.remove(old)
            group.extend(old)
        groups.append(group)
    return groups


def candidate(group, fulltext_candidates=None, official_evidence=None):
    fulltext_candidates = fulltext_candidates or {}

    def version(paper):
        return paper.version if paper.paper_id.startswith('arxiv:') and re.fullmatch(r'v\d+', paper.version) else None

    def links(paper):
        if paper.paper_id.startswith('arxiv:'):
            selected = version(paper)
            return ['https://arxiv.org/pdf/'+re.sub(r'v\d+$', '', paper.paper_id[6:])+selected] if selected else []
        return fulltext_candidates.get(paper.url, fulltext_candidates.get(paper.paper_id, []))

    # Prefer a concrete arXiv version, newest first; otherwise an official PDF.
    # Metadata revision hashes are never treated as document versions.
    def recovered_abstract(paper):
        return bool(paper.abstract and any(s.startswith('pubmed-exact:') for s in paper.sources))

    has_primary_abstract = any(p.abstract and not recovered_abstract(p) for p in group)
    ordered = sorted(group, key=lambda p: (not bool(version(p)),
        -int(version(p)[1:]) if version(p) else 0, not bool(links(p)),
        not (not has_primary_abstract and recovered_abstract(p)), p.paper_id, p.url))
    first = ordered[0]
    identifiers = set().union(*(keys(p) for p in group))
    arxiv = next((k[6:] for k in sorted(identifiers) if k.startswith('arxiv:')), None)
    doi = next((k[4:] for k in sorted(identifiers) if k.startswith('doi:')), None)
    dated = [p for p in ordered if publication_basis(p) and p.published]
    # Earliest known publication across manifestations prevents a later journal issue
    # turning a known preprint into a new daily paper.
    dated.sort(key=lambda p: p.published)
    date_paper = dated[0] if dated else None
    canonical_id = 'arxiv:'+arxiv if arxiv else 'doi:'+doi if doi else first.paper_id
    return Candidate(paper_id=canonical_id, paper_version=version(first),
                     title=first.title, authors=first.authors,
                     doi=doi, arxiv_id=arxiv,
                     venue=next((s.split(':', 1)[1] for p in ordered for s in p.sources
                                 if s.startswith(('crossref:', 'official-rss:', 'pmlr:', 'neurips:', 'journal-detail:', 'pubmed-exact:'))), None),
                     published_at=date_paper.published if date_paper else None,
                     date_basis=publication_basis(date_paper) if date_paper else None,
                     abstract=first.abstract or None,
                     original_links=sorted({p.url for p in group if p.url}),
                     fulltext_links=links(first),
                     sources=[{'source': s, 'paper_id': p.paper_id, 'version': p.version,
                               'url': p.url, 'published_at': p.published or None,
                               'updated_at': p.updated,
                               'paper_version': version(p), 'fulltext_links': links(p),
                               'selected_document': p is first,
                               **({'pubmed_abstract': p.abstract} if p.abstract and s.startswith('pubmed-exact:') else {}),
                               **({'official_metadata': official_evidence[p.url]} if official_evidence and p.url in official_evidence else {}),
                               'date_basis': publication_basis(p)} for p in group for s in p.sources],
                     versions=sorted({p.version for p in group}))


def window(request):
    if request.mode == 'direction_daily':
        until = request.as_of.astimezone(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        if request.constraints.get('window_policy') == 'recent_90d':
            days = request.constraints.get('window_days', 90)
            since = until - timedelta(days=int(days))
            if request.publication_window and (request.publication_window.since != since
                                              or request.publication_window.until != until):
                raise ValueError('Recent direction window must match its declared rolling range')
            return since, until
        since = until - timedelta(days=1)
        if request.publication_window and (request.publication_window.since != since
                                          or request.publication_window.until != until):
            raise ValueError('Daily window must equal yesterday in Asia/Shanghai')
        return since, until
    if request.publication_window:
        return request.publication_window.since, request.publication_window.until
    return None


def in_window(paper, bounds, as_of):
    for source in paper.sources:
        if (source.get('selected_document') and source['source'] == 'deepseek-original-page'
                and source.get('paper_version') not in (None, 'v1') and not source.get('updated_at')):
            return False, 'version_time_unverified'
        if source.get('selected_document') and source.get('updated_at'):
            updated = datetime.fromisoformat(source['updated_at'])
            if len(source['updated_at']) == 10:
                if updated.date() > as_of.astimezone(TZ).date():
                    return False, 'version_after_as_of'
                if updated.date() == as_of.astimezone(TZ).date():
                    return False, 'version_time_unverified'
            if updated.tzinfo is not None and updated > as_of:
                return False, 'version_after_as_of'
    if not paper.published_at or not paper.date_basis:
        return (False, 'publication_date_unverified') if bounds else (True, None)
    try:
        value = datetime.fromisoformat(paper.published_at)
    except ValueError:
        return False, 'publication_date_unverified'
    if len(paper.published_at) == 10:
        if value.date() > as_of.astimezone(TZ).date():
            return False, 'future_publication'
        if bounds and (bounds[1] - bounds[0]).days >= 7:
            # A rolling multi-month window can place a date-only journal record
            # on its publication date without inventing a Beijing timestamp.
            return (True, None) if bounds[0].date() <= value.date() < bounds[1].date() else (False, 'date_filtered')
        # Date-only publisher metadata has no timezone. Do not invent one at a
        # Beijing day boundary; strict daily selection requires an absolute timestamp.
        return (False, 'publication_timezone_unknown') if bounds else (True, None)
    if value.tzinfo is None:
        return False, 'publication_timezone_unknown'
    if value > as_of:
        return False, 'future_publication'
    if not bounds:
        return True, None
    return (True, None) if bounds[0] <= value < bounds[1] else (False, 'date_filtered')


def paper_history_keys(paper):
    """Use only verified identity fields. A title is never a history key."""
    versions = set()
    for row in [paper, *paper.get('sources', [])]:
        paper_id = row.get('paper_id') or ''
        doi = row.get('doi') or (paper_id[4:] if paper_id.startswith('doi:') else '')
        arxiv_id = row.get('arxiv_id') or (paper_id[6:] if paper_id.startswith('arxiv:') else '')
        if doi:
            versions.add(('doi:'+normalize_doi(doi), ''))
        if arxiv_id and (row is paper or row.get('selected_document')):
            arxiv_id = arxiv_id.lower()
            match = re.search(r'v\d+$', arxiv_id)
            version = match[0] if match else row.get('paper_version') or row.get('version') or ''
            versions.add(('arxiv:'+re.sub(r'v\d+$', '', arxiv_id), version if re.fullmatch(r'v\d+', version) else ''))
    return versions


def already_recommended(paper, history):
    identities = paper_history_keys(paper.model_dump(mode='json'))
    for key, version in identities:
        if key.startswith('arxiv:') and version:
            prior = [old_version for old_key, old_version in history if old_key == key]
            if prior and all(prior):
                # A shared DOI identifies the work, not the selected arXiv revision.
                # Unknown historical versions retain the conservative identity fallback.
                return int(version[1:]) <= max(int(value[1:]) for value in prior)
    if identities.intersection(history):
        return True
    return any(key == old_key and (not version or not old_version)
               for key, version in identities for old_key, old_version in history)


def select_recommendations(request, rows):
    """PaperFlow order remains the tie-breaker; bootstrap first covers evidenced routes."""
    limit = min(request.max_recommendations, 5) if request.mode == 'direction_bootstrap' else request.max_recommendations
    if request.mode != 'direction_bootstrap':
        primary = [row for row in rows if row.get('applicability_review', {}).get('applicability') != 'reference']
        # A transfer from different conditions should not displace a directly
        # relevant result or fill empty positions in the recommendation list.
        return (primary if primary else rows[:1])[:limit]
    remaining, ordered, covered = list(rows), [], set()
    # First recommendations should orient the user: keep one evidenced survey/review
    # ahead of the diversity fill when the pool contains one.
    review = next((row for row in remaining if row.get('review_like')), None)
    if review is not None and len(ordered) < limit:
        ordered.append(review)
        remaining.remove(review)
    # Only explicitly reviewed evidence supplies these roles. Publication age,
    # citation frequency and PaperFlow scores never imply a historical role.
    for role in ('basic_method', 'methodological_progress'):
        match = next((row for row in remaining if any(r['role'] == role for r in row.get('representative_roles', []))), None)
        if match is not None and len(ordered) < limit:
            ordered.append(match)
            remaining.remove(match)
            covered.update(route['id'] for route in match.get('method_routes', []))
    while remaining and len(ordered) < limit:
        gains = [len({route['id'] for route in row.get('method_routes', [])}-covered) for row in remaining]
        if not max(gains, default=0):
            break
        row = remaining.pop(gains.index(max(gains)))
        ordered.append(row)
        covered.update(route['id'] for route in row.get('method_routes', []))
    if not ordered:
        for question in sorted(request.research_questions, key=lambda q: -q.priority):
            match = next((r for r in remaining if question.id in r['question_ids']), None)
            if match and len(ordered) < limit:
                ordered.append(match)
                remaining.remove(match)
    ordered.extend(remaining)
    return ordered[:limit]


def discover(request, sources, *, ranker=None, source_names=('arxiv', 'crossref'),
             max_queries=3, media_leads=(), known_ids=(), store=None, web_fallback=None,
             citation_expander=None, abstract_review=None, on_progress=None,
             previously_recommended=(), should_stop=None):
    req = RecommendationRequest.model_validate(request)
    official_plan = getattr(sources, 'official_plan', None)
    if official_plan and (official_plan['request_id'] != req.request_id or official_plan['user_id'] != req.user_id
            or official_plan['as_of'] != req.as_of.isoformat() or official_plan['mode'] != req.mode
            or official_plan['queries'] != req.queries[:24]):
        raise ValueError('Configure official plan for this exact request before discovery')
    journal_plan = getattr(sources, 'journal_plan', None)
    if journal_plan and (journal_plan['request_id'] != req.request_id or journal_plan['user_id'] != req.user_id
            or journal_plan['as_of'] != req.as_of.isoformat() or journal_plan['mode'] != req.mode
            or journal_plan['queries'] != req.queries[:24]):
        raise ValueError('Configure journal plan for this exact request before discovery')
    bounds = window(req)
    sources.sort_by = 'submittedDate' if req.mode == 'direction_daily' else 'relevance'
    sources.discovery_mode = req.mode
    if not 1 <= max_queries <= 24:
        raise ValueError('max_queries must be 1..24')
    if store and ranker is not None and not getattr(ranker, 'policy', None):
        raise ValueError('Cached ranking requires explicit ranker.policy identity')
    fingerprint = hashlib.sha256(json.dumps({'request': req.model_dump(mode='json'),
        'sources': source_names, 'max_queries': max_queries, 'media': list(media_leads),
        'known_ids': list(known_ids), 'ranker': getattr(ranker, 'policy', 'unavailable'),
        'source_policy': getattr(sources, 'policy', None),
        'fallback_policy': getattr(web_fallback, 'policy', None),
        'citation_policy': getattr(citation_expander, 'policy', None),
        'abstract_review': abstract_review, 'previously_recommended': sorted(previously_recommended),
        'policy': POLICY}, sort_keys=True).encode()).hexdigest()
    cache_id = req.user_id+':'+fingerprint
    if store:
        try:
            return DiscoveryResult.model_validate(store.get('discovery_result_v1', cache_id))
        except ValueError as exc:
            if not str(exc).startswith('Missing '):
                raise
    runs, warnings, decisions, raw, page_evidence = [], [], [], [], {}
    web_fallback_runs = []
    index_pages = {}
    query_pilots, pilot_ids = {}, {}
    if official_plan:
        warnings.extend(dict.fromkeys(official_plan.get('warnings', [])))
    if journal_plan:
        warnings.extend(dict.fromkeys(journal_plan.get('warnings', [])))
    citation_documents, citation_origins = [], {}

    def cite_leads(papers, query):
        for paper in papers:
            if len([d for d in citation_documents if d['query'] == query]) >= 2:
                break
            url = paper.url.replace('http://arxiv.org/', 'https://arxiv.org/', 1)
            if (paper.paper_id.startswith('arxiv:') and url.startswith('https://arxiv.org/abs/')
                    and not any(d['url'] == url for d in citation_documents)):
                citation_documents.append({'url': url, 'query': query})
    if len(req.queries) > max_queries:
        warnings.append('query_limit_truncated')
    since = bounds[0].date().isoformat() if bounds else '1900-01-01'
    until = bounds[1].date().isoformat() if bounds else req.as_of.astimezone(TZ).date().isoformat()
    queries = req.queries[:max_queries]
    total_searches = len(source_names) * len(queries)
    searches_done = 0
    stopped = should_stop or (lambda: False)
    from .discovery_sources import DiscoverySources

    if isinstance(sources, DiscoverySources):
        sources.source_stop_requested = stopped
    if on_progress and type(sources) is DiscoverySources and {'arxiv', 'crossref'} <= set(source_names):
        on_progress('searching', done=0, total=total_searches)
    prefetched = (sources.prefetch_independent(source_names, queries, since, until, stopped)
                  if type(sources) is DiscoverySources else {})
    for name in source_names:
        if journal_plan and journal_plan.get('pubmed_exact') and name in journal_plan['source_names']:
            sources.journal_doi_leads = tuple(p.doi for p in raw if p.doi and any(s.startswith('crossref:') for s in p.sources))
        for query in queries:
            if stopped() and (name, query) not in prefetched:
                break
            if on_progress:
                on_progress('searching', done=searches_done, total=total_searches, source=name)
            if (name, query) in prefetched:
                run, sources.request_log, sources.source_errors, sources.lane_runs = prefetched[name, query]
                sources.journal_recovery = None
            else:
                run = sources.collect(name, query, since, until)
            raw.extend(run.papers)
            if name in ('arxiv', 'crossref') and (run.status in ('ok', 'empty') or run.papers):
                pilot_ids.setdefault(query, set()).update(paper.paper_id for paper in run.papers)
                hits = query_pilots.setdefault(query, [])
                known = {hit['paper_id'] for hit in hits}
                hits.extend({'paper_id': paper.paper_id, 'title': paper.title[:180],
                             'abstract_excerpt': (paper.abstract or '')[:360]}
                            for paper in run.papers[:3] if paper.paper_id not in known)
            cite_leads(run.papers, query)
            lanes = list(getattr(sources, 'lane_runs', []))
            requests = getattr(sources, 'request_log', [])
            recovery = getattr(sources, 'journal_recovery', None)
            pubmed_requests = []
            if recovery:
                pubmed_requests = [r for r in requests if r['endpoint'].startswith('https://eutils.ncbi.nlm.nih.gov/')]
                requests = [r for r in requests if r not in pubmed_requests]
                if recovery[1]['status'] == 'skipped':
                    lanes.append({'pubmed_exact': recovery[1]})
            if any(lane.get('status') == 'skipped' for lane in lanes):
                warnings.append('optional_concept_lane_skipped')
            primary_index = len(runs)
            runs.append({'source': name, 'query': query, 'since': since, 'until': until,
                         'status': run.status, 'count': len(run.papers), 'error': run.error,
                         'truncated': run.truncated, 'fetched_at': run.fetched_at,
                         'effective_requests': requests, 'search_lanes': lanes,
                         'coverage_complete': False})
            exact_recovered = bool(recovery and recovery[0] and recovery[1]['status'] == 'completed')
            if recovery and recovery[1]['status'] != 'skipped':
                from .discovery_pubmed_exact import MAX_INDEX_PAGE_FETCHES, recover_index_abstract

                papers, pubmed_audit, cache_hit = recovery
                page_runs = []
                if web_fallback is not None and req.mode != 'direction_daily':
                    for paper in papers:
                        if stopped():
                            break
                        if (not paper.abstract and paper.url not in index_pages
                                and len(index_pages) < MAX_INDEX_PAGE_FETCHES):
                            index_pages[paper.url] = recover_index_abstract(paper, web_fallback)
                    page_runs = [index_pages[p.url][1] if p.url in index_pages else
                                 {'doi': p.doi, 'url': p.url, 'status': 'skipped', 'reason': 'index_page_limit'}
                                 for p in papers if not p.abstract]
                    papers = [index_pages[p.url][0] if p.url in index_pages else p for p in papers]
                raw.extend(papers)
                runs.append({'source': 'pubmed_exact', 'query': query, 'since': since, 'until': until,
                             'status': pubmed_audit['status'],
                             'count': len(papers), 'error': pubmed_audit.get('error'),
                             'truncated': bool(pubmed_audit['skipped_dois'] or pubmed_audit.get('search_truncated')
                                               or any(page['status'] == 'skipped' for page in page_runs)),
                             'fetched_at': run.fetched_at, 'coverage_complete': False, 'effective_requests': pubmed_requests,
                             'search_lanes': [{**pubmed_audit, 'primary_source': name, 'primary_source_run_index': primary_index,
                                               'cache_hit': cache_hit, 'index_page_runs': page_runs,
                                               'publisher_web_fallback_skipped': bool(web_fallback and exact_recovered)}]})
            if web_fallback is not None and not exact_recovered and not stopped():
                from .discovery_web import fallback_after_failure, original_page_record
                fallback = fallback_after_failure(run, web_fallback,
                    partial_failure=bool(getattr(sources, 'source_errors', [])))
                if fallback['status'] != 'skipped':
                    extra = []
                    for document in fallback['documents']:
                        record = original_page_record(document)
                        if record:
                            extra.append(record)
                            page_evidence[record.url] = document
                    raw.extend(extra)
                    cite_leads(extra, query)
                    runs.append({'source': 'deepseek_search_fetch', 'query': query,
                        'since': since, 'until': until, 'status': 'failed' if fallback['status'] == 'failed' else 'partial',
                        'count': len(extra), 'error': ';'.join(fallback['warnings']) or None,
                        'truncated': fallback.get('fetch_limit_truncated', len(fallback['leads']) > len(fallback['documents'])),
                         'fetched_at': datetime.now(TZ).isoformat(), 'coverage_complete': False})
                    web_fallback_runs.append({'source_run_index': len(runs)-1,
                        'primary_source_run_index': primary_index, 'query': query,
                        'primary_failure': fallback['primary_failure'], 'fallback_result': fallback})
                    warnings.append('fallback_used_original_page_fields_require_review')
            searches_done += 1
    # Imported messages are identifier leads only, never abstracts. Resolve identifiers
    # absent from the current pool as well. Hard cap keeps imports from unbounded fetches.
    leads = [(lead['source_ref'], x) for lead in media_leads for x in lead['identifiers']]
    if len(leads) > 10:
        warnings.append('media_identifier_limit_truncated')
    for ref, identifier in leads[:10]:
        if stopped():
            break
        try:
            found = sources.resolve(identifier['kind'], identifier['value'])
            for paper in found:
                paper.sources.append('media_lead:'+str(ref))
            raw.extend(found)
            runs.append({'source': 'media_identifier', 'source_ref': ref, 'query': identifier,
                         'since': None, 'until': None, 'status': 'ok' if found else 'empty',
                         'count': len(found), 'truncated': False, 'error': None,
                         'fetched_at': datetime.now(TZ).isoformat(), 'coverage_complete': False})
        except (ValueError, RuntimeError, OSError, httpx.HTTPError, ET.ParseError, KeyError, TypeError) as exc:
            runs.append({'source': 'media_identifier', 'source_ref': ref, 'query': identifier,
                         'status': 'failed', 'count': 0, 'error': type(exc).__name__,
                         'truncated': False, 'since': None, 'until': None,
                         'fetched_at': datetime.now(TZ).isoformat(), 'coverage_complete': False})
    if citation_expander is not None and req.mode == 'direction_bootstrap' and not stopped():
        expansion = citation_expander(citation_documents, sources)
        raw.extend(expansion['papers'])
        runs.extend(expansion['source_runs'])
        warnings.extend(expansion['warnings'])
        if expansion['unscheduled_identifiers']:
            warnings.append('citation_identifier_limit_truncated')
        if expansion['unscheduled_documents']:
            warnings.append('citation_document_limit_truncated')
        for run in expansion['source_runs']:
            if run.get('paper_id'):
                citation_origins[run['paper_id']] = run['citation_origins']
    if on_progress:
        on_progress('screening')
    groups = merge(raw)
    candidates = [candidate(g, getattr(sources, 'fulltext_candidates', {}), getattr(sources, 'official_evidence', {})) for g in groups]
    for paper in candidates:
        for origin in paper.sources:
            if origin['url'] in page_evidence:
                origin['page_evidence'] = page_evidence[origin['url']]
            if origin['paper_id'] in citation_origins:
                origin['citation_origins'] = citation_origins[origin['paper_id']]
    eligible = []
    for group, paper in zip(groups, candidates):
        for duplicate in group[1:]:
            decisions.append({'paper_id': duplicate.paper_id, 'canonical_id': paper.paper_id,
                              'reason': 'deduplicated', 'version': duplicate.version})
        keep, reason = in_window(paper, bounds, req.as_of)
        text = (paper.title+' '+(paper.abstract or '')).casefold()
        exclusions = req.constraints.get('exclusions', [])
        if any(str(x).casefold() in text for x in exclusions if str(x).strip()):
            keep, reason = False, 'explicit_exclusion'
        if keep and previously_recommended and already_recommended(paper, previously_recommended):
            keep, reason = False, 'already_recommended_in_direction'
        if keep:
            eligible.append(paper)
        else:
            decisions.append({'paper_id': paper.paper_id, 'reason': reason})
            if reason in ('publication_date_unverified', 'publication_timezone_unknown', 'version_after_as_of', 'version_time_unverified'):
                warnings.append(reason+':'+paper.paper_id)
    # Flag uncertain title matches and suppress repeat pushes until identity is checked.
    uncertain = set()
    for i, paper in enumerate(eligible):
        for other in eligible[:i]:
            if SequenceMatcher(None, paper.title.casefold(), other.title.casefold()).ratio() >= .94:
                uncertain.update((paper.paper_id, other.paper_id))
    for pid in uncertain:
        decisions.append({'paper_id': pid, 'reason': 'identity_uncertain_title_similarity'})
    if uncertain:
        warnings.append('uncertain_identities_withheld')
    eligible = [p for p in eligible if p.paper_id not in uncertain]
    recommendations = []
    ranking_audit = []
    review_stage_errors = []
    ranking_failed = False
    if not req.max_recommendations:
        decisions.extend({'paper_id': p.paper_id, 'reason': 'recommendation_limit_zero'} for p in eligible)
    if eligible and req.max_recommendations and not stopped():
        if ranker is None:
            ranking_failed = True
            warnings.append('paperflow_ranking_unavailable: explicit provisioned embeddings required')
            decisions.extend({'paper_id': p.paper_id, 'reason': 'ranking_unavailable'} for p in eligible)
        else:
            try:
                if on_progress:
                    on_progress('ranking', candidates=len(eligible))
                rows = ranker(req, eligible)
                ranking_audit = getattr(ranker, 'ranking_audit', [])
                issues = getattr(ranker, 'review_errors', [])
                review_stage_errors = [issue for issue in issues if issue.get('stage') and not issue.get('paper_id')]
                warnings.extend('review_partial:'+stage for stage in dict.fromkeys(
                    issue['stage'] for issue in issues if issue.get('stage')))
                if any(issue.get('paper_id') for issue in issues) or any(row.get('review_error') for row in ranking_audit):
                    warnings.append('review_partial:per_paper_evidence_errors')
                review_decisions = {}
                if abstract_review is not None:
                    from .discovery_review import apply_abstract_review
                    rows, review_decisions = apply_abstract_review(req, eligible, rows, abstract_review)
                by_id = {p.paper_id: p for p in eligible}
                question_ids = {q.id for q in req.research_questions}
                seen = set()
                for row in rows:
                    if row['paper_id'] in seen or row['paper_id'] not in by_id:
                        raise ValueError('Ranker returned unknown/duplicate identity')
                    if not row['question_ids'] or not set(row['question_ids']) <= question_ids:
                        raise ValueError('Ranker question provenance invalid')
                    if not row.get('reason') or row.get('basis') not in ('title', 'abstract', 'fulltext'):
                        raise ValueError('Ranker must state reason and actual reading basis')
                    if row['basis'] == 'fulltext':
                        from .discovery_fulltext_review import evidence_matches_paper

                        refs = row.get('matched_evidence', [])
                        if (not evidence_matches_paper(by_id[row['paper_id']], refs)
                                or {ref.get('question_id') for ref in refs} != set(row['question_ids'])):
                            raise ValueError('Ranking evidence needs the verified PDF version and page')
                    seen.add(row['paper_id'])
                if req.mode == 'direction_bootstrap':
                    warnings.append('bootstrap_representativeness_requires_review: question coverage is not historical impact')
                recommendations = [{key: value for key, value in row.items() if key != 'review_like'}
                                   for row in select_recommendations(req, rows)]
                selected = {r['paper_id'] for r in recommendations}
                decisions.extend({'paper_id': p.paper_id, 'reason': 'selected' if p.paper_id in selected
                                  else 'ranking_cutoff' if p.paper_id in seen else 'relevance_insufficient'}
                                 for p in eligible)
                for decision in decisions:
                    if decision['reason'] == 'relevance_insufficient':
                        decision['reason'] = review_decisions.get(decision['paper_id'],
                            getattr(ranker, 'decisions', {}).get(decision['paper_id'], decision['reason']))
            except (ValueError, RuntimeError, ImportError, OSError, KeyError, TypeError) as exc:
                ranking_audit = getattr(ranker, 'ranking_audit', [])
                ranking_failed = True
                warnings.append('ranking_failed:'+type(exc).__name__)
                reason = 'embedding_budget_insufficient' if type(exc).__name__ == 'EmbeddingBudgetExceeded' else 'ranking_failed'
                decisions.extend({'paper_id': p.paper_id, 'reason': reason} for p in eligible)
    present = set().union(*(keys(p) for p in raw)) if raw else set()
    failed = any(r['status'] == 'failed' or (r['status'] == 'partial' and r.get('error')) for r in runs)
    for pid in known_ids:
        if pid not in present:
            decisions.append({'paper_id': pid, 'reason': 'query_planning_required' if not runs and not req.queries
                              else 'source_failed_not_observed' if failed else 'not_discovered',
                              'scope': 'bounded_queries_only'})
    canceled = stopped()
    if canceled:
        warnings.append('canceled_by_user')
    if not runs and not canceled:
        warnings.append('query_planning_required' if not req.queries else 'no_sources_configured')
    partial = ranking_failed or warnings or any(r['truncated'] or r['status'] in ('failed', 'partial') for r in runs)
    status = ('partial' if canceled else 'skipped' if not runs and not req.queries else
              'failed' if runs and all(r['status'] == 'failed' for r in runs) else
              'partial' if partial else 'completed')
    origins = {paper.paper_id: {paper.paper_id, *(source['paper_id'] for source in paper.sources)}
               for paper in candidates}
    pilots = {}
    for query, hits in query_pilots.items():
        questions = dict.fromkeys(qid for row in recommendations
                                  if (row.get('applicability_review') or {}).get('applicability') != 'reference'
                                  and pilot_ids[query] & origins[row['paper_id']] for qid in row['question_ids'])
        pilots[query] = {'hits': hits, 'direct_question_ids': list(questions)}
    result = DiscoveryResult(request_id=req.request_id, user_id=req.user_id, status=status,
        candidates=candidates, recommendations=recommendations, source_runs=runs, warnings=warnings,
        decisions=decisions, audit={'policy': POLICY, 'input_fingerprint': fingerprint,
            'model_calls': getattr(ranker, 'model_calls', None) if ranker else 0,
            'paid_model_calls': getattr(ranker, 'paid_model_calls', None) if ranker else 0,
            'ranking_policy': getattr(ranker, 'policy', 'unavailable'),
            'ranking_audit': ranking_audit,
            'query_pilots': pilots,
            'official_source_plan': getattr(sources, 'official_plan', None),
            'journal_source_plan': journal_plan,
            'web_fallback_runs': web_fallback_runs,
            'review_errors': [*review_stage_errors, *(row['review_error'] for row in ranking_audit if row.get('review_error'))],
            'embedding_runs': getattr(ranker, 'embedding_runs', []),
            'fallback_audit': {'calls': list(web_fallback.audit), 'used': dict(web_fallback.used)} if web_fallback else None,
            'publication_window': [x.isoformat() for x in bounds] if bounds else None})
    if store and not canceled:
        store.put('discovery_result_v1', cache_id, result.model_dump(mode='json'), immutable=True)
        store.audit(req.request_id, {'event': 'discovery_result_v1', 'user_id': req.user_id,
                                    'cache_id': cache_id, 'status': status})
    return result
