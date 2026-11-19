"""Bounded metadata discovery; existing HTTP retry/rate coordination is reused."""
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime, timedelta
from urllib.parse import quote

import httpx

from .models import PaperRecord
from .sources import PublicSourceUnavailable, Sources, normalize_doi, paper_title_text

NS = {'a': 'http://www.w3.org/2005/Atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/'}


class DiscoverySources(Sources):
    """Bounded search lanes; historical recall never widens a daily publication window."""

    def __init__(self, max_results=30, transport=None, *, max_pages=2, concept_categories=(), official_plan=None, journal_plan=None):
        if not 1 <= max_results <= 100 or not 1 <= max_pages <= 3:
            raise ValueError('max_results=1..100 and max_pages=1..3 required')
        super().__init__(max_results=max_results, transport=transport)
        self.max_pages = max_pages
        self.concept_categories = tuple(concept_categories)
        self.rate_limited_until = {}
        self.policy = {'adapter': 'discovery-sources-v4-concept-phrases', 'page_size': max_results, 'max_pages': max_pages,
                       'concept_categories': self.concept_categories}
        self.configure_official_plan(official_plan)
        self.configure_journal_plan(journal_plan)

    def configure_journal_plan(self, plan):
        from copy import deepcopy

        from .discovery_journals import JOURNALS
        if plan is not None and (plan.get('schema_version') != '1'
                or plan['feeds'] != {name: value[0] for name, value in JOURNALS.items()}
                or plan['source_names'] != ['journal:'+name for name in JOURNALS]
                or not 1 <= plan['detail_limit_per_journal'] <= 6 or len(plan['queries']) > 24
                or plan['max_http_attempts'] != 21
                or plan.get('max_pubmed_http_attempts', 0) != (6 if plan.get('pubmed_exact') else 0)):
            raise ValueError('Invalid journal source plan')
        for cached in getattr(self, '_journal_cache', {}).values():
            for paper in cached['papers']:
                self.fulltext_candidates.pop(paper.url, None)
                self.official_evidence.pop(paper.url, None)
        self.journal_plan = deepcopy(plan)
        self._journal_cache = {}
        self.journal_doi_leads = ()
        self.policy['journal_plan'] = deepcopy(plan)

    def journal_text(self, url, byte_limit, *, media, params=None):
        attempt = {'endpoint': url, 'byte_limit': byte_limit}
        if params:
            attempt['params'] = params
        self.request_log.append(attempt)
        with self.http.stream('GET', url, params=params, follow_redirects=False) as response:
            attempt['http_status'] = response.status_code
            response.raise_for_status()
            mime = response.headers.get('content-type', '').lower()
            if response.status_code != 200 or not any(t in mime for t in media):
                raise RuntimeError('Journal response type/status unverified')
            parts, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > byte_limit:
                    raise RuntimeError('Journal response byte limit')
                parts.append(chunk)
            return b''.join(parts).decode('utf-8', errors='replace')

    def configure_official_plan(self, plan):
        from copy import deepcopy

        from .discovery_official_plan import CATALOG
        if plan is not None:
            allowed = {row[0]: row[3] for row in CATALOG}
            if (plan.get('schema_version') != '1' or not 1 <= plan['detail_limit_per_volume'] <= 5
                    or plan['index_byte_limit'] != 20_000_000 or plan['detail_byte_limit'] != 1_000_000
                    or len(plan['queries']) > 24 or len(plan['source_names']) > 9
                    or plan['source_names'] != [v['source'] for v in plan['volumes']]
                    or len(set(plan['source_names'])) != len(plan['source_names'])
                    or plan['max_http_attempts'] != len(plan['source_names'])*6
                    or any(allowed.get(v['source']) != v['index_url'] for v in plan['volumes'])):
                raise ValueError('Invalid bounded official source plan')
        self.official_plan = deepcopy(plan)
        self._official_cache = {}
        self.fulltext_candidates = {}
        self.official_evidence = {}
        self.policy['official_plan'] = deepcopy(plan)

    def official_html(self, url, byte_limit):
        """One bounded streamed attempt, no redirect or hidden per-query retries."""
        self.request_log.append({'endpoint': url, 'byte_limit': byte_limit})
        with self.http.stream('GET', url, follow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200 or 'text/html' not in response.headers.get('content-type', '').lower():
                raise ValueError('Official HTML status/type unverified')
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > byte_limit:
                    raise ValueError('OfficialHTMLByteLimit')
                chunks.append(chunk)
            return b''.join(chunks).decode('utf-8', errors='replace')

    def collect(self, source, query, since, until):
        self.request_log, self.source_errors, self.lane_runs = [], [], []
        self.journal_recovery = None
        if not hasattr(self, 'fulltext_candidates'):
            self.fulltext_candidates = {}
        if source.startswith('journal:'):
            from .discovery_journals import collect_journal
            return collect_journal(self, source, query, since, until)
        if source not in ('arxiv', 'crossref') and not source.startswith(('official-rss:', 'pmlr:', 'neurips:')):
            raise ValueError('Source not enabled by discovery policy')
        if source.startswith(('official-rss:', 'pmlr:', 'neurips:')):
            from .discovery_proceedings import collect_volume
            from .models import SourceResult
            result = SourceResult(source=source, query=query, since=since, until=until, status='failed')
            try:
                result.papers, result.truncated = collect_volume(self, source, query)
                result.status = 'partial' if result.truncated else 'ok' if result.papers else 'empty'
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                self.source_errors.append(type(exc).__name__)
            if self.source_errors:
                result.error = ';'.join(self.source_errors)
                result.status = 'partial' if result.papers else 'failed'
            return result
        result = super().collect(source, query, since, until)
        if self.source_errors:
            result.error = ';'.join(self.source_errors)
            result.status = 'partial' if result.papers else 'failed'
        return result

    def prefetch_independent(self, names, queries, since, until, should_stop):
        """Overlap independent providers while keeping each provider's pacing and audit."""
        names = tuple(dict.fromkeys(name for name in names if name in ('arxiv', 'crossref')))
        if len(names) < 2 or not queries or should_stop():
            return {}

        def collect_lane(name):
            lane = copy(self)
            lane.rate_limited_until = self.rate_limited_until.copy()
            lane.source_stop_requested = should_stop
            collected = {}
            for query in queries:
                if should_stop():
                    break
                result = lane.collect(name, query, since, until)
                collected[name, query] = (result, lane.request_log, lane.source_errors, lane.lane_runs)
            return collected, lane.rate_limited_until

        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(collect_lane, name) for name in names]
            results = {}
            for task in tasks:
                lane_results, cooldowns = task.result()
                results.update(lane_results)
                for key, value in cooldowns.items():
                    self.rate_limited_until[key] = max(value, self.rate_limited_until.get(key, value))
        return results

    def get(self, url, **kwargs):
        from urllib.parse import urlsplit
        host = urlsplit(url).hostname
        if time.monotonic() < self.rate_limited_until.get(host, 0):
            raise RuntimeError('SourceRateLimitedCooldown')
        if hasattr(self, 'request_log'):
            self.request_log.append({'endpoint': url, 'params': kwargs.get('params', {})})
        try:
            return super().get(url, **kwargs)
        except PublicSourceUnavailable as exc:
            if exc.status_code == 429:
                self.rate_limited_until[host] = time.monotonic()+300
            raise

    def _arxiv_papers(self, expression, since, until, *, start_offset=0, sort_by=None, sort_order='descending'):
        # Existing parser supplies original Atom published, version and DOI. Replace
        # only the legacy query date field through an explicit endpoint request.
        start = (datetime.fromisoformat(since) - timedelta(days=1)).strftime('%Y%m%d')
        end = (datetime.fromisoformat(until) + timedelta(days=1)).strftime('%Y%m%d')
        query = expression if since == '1900-01-01' else f'({expression}) AND submittedDate:[{start}0000 TO {end}2359]'
        params = {'search_query': query,
                  'start': start_offset, 'max_results': self.max_results,
                  'sortBy': sort_by or getattr(self, 'sort_by', 'relevance'), 'sortOrder': sort_order}
        content = self.get('https://export.arxiv.org/api/query', params=params).content
        return self.parse_arxiv(content)

    def parse_arxiv(self, content):
        root = ET.fromstring(content)
        ns = {**NS, 'x': 'http://arxiv.org/schemas/atom'}
        papers = []
        for entry in root.findall('a:entry', ns):
            url = entry.findtext('a:id', '', ns)
            if '/abs/' not in url:
                raise ValueError('arXiv error entry')
            raw = url.split('/abs/')[-1]
            match = re.fullmatch(r'(.+)v(\d+)', raw)
            pid, version = (match[1], 'v'+match[2]) if match else (raw, 'unversioned')
            papers.append(PaperRecord(paper_id='arxiv:'+pid, version=version,
                title=' '.join(entry.findtext('a:title', '', ns).split()),
                abstract=' '.join(entry.findtext('a:summary', '', ns).split()),
                authors=[a.findtext('a:name', '', ns) for a in entry.findall('a:author', ns)],
                doi=normalize_doi(entry.findtext('x:doi', None, ns)),
                published=entry.findtext('a:published', '', ns),
                updated=entry.findtext('a:updated', None, ns), sources=['arxiv'], url=url,
                is_preprint=True))
        return papers, int(root.findtext('o:totalResults', str(len(papers)), ns))

    def arxiv(self, query, since, until):
        if re.search(r'\b(?:all|ti|abs|au|cat):', query):
            expressions = [query]
        else:
            terms = re.findall(r'[A-Za-z0-9-]+', query)
            if not terms:
                raise ValueError('English query terms required; no implicit translation')
            expressions = [' AND '.join('all:'+x for x in terms)]
            if getattr(self, 'sort_by', 'relevance') == 'relevance':
                expressions.insert(0, 'ti:"'+' '.join(terms)+'"')
        lanes = [(expression, getattr(self, 'sort_by', 'relevance'), 'descending') for expression in expressions]
        if getattr(self, 'discovery_mode', None) == 'direction_bootstrap':
            # A separate oldest-first lane reduces recency/relevance bias. It is a
            # retrieval lane, not a claim that the earliest paper is representative.
            if self.concept_categories:
                from .discovery_queries import concept_expression
                try:
                    lanes.append((concept_expression(query, self.concept_categories), 'relevance', 'descending'))
                except ValueError as exc:
                    self.lane_runs.append({'query': query, 'status': 'skipped', 'pages': 0, 'count': 0,
                        'error': 'concept_planning:'+str(exc), 'truncated': False})
            else:
                lanes.append((expressions[-1], 'submittedDate', 'ascending'))
        papers, truncated = [], False
        for expression, sort_by, order in lanes:
            lane = {'query': expression, 'sort_by': sort_by, 'sort_order': order,
                    'pages': 0, 'count': 0, 'total': None, 'truncated': False, 'error': None}
            for page in range(self.max_pages):
                try:
                    batch, total = self._arxiv_papers(expression, since, until,
                        start_offset=page*self.max_results, sort_by=sort_by, sort_order=order)
                    papers.extend(batch)
                    lane.update(pages=page+1, count=lane['count']+len(batch), total=total)
                    lane['truncated'] = lane['count'] < total
                    if not lane['truncated'] or not batch:
                        break
                except (httpx.HTTPError, ValueError, RuntimeError, ET.ParseError, KeyError, TypeError) as exc:
                    lane['error'] = type(exc).__name__
                    self.source_errors.append('arxiv_expression:'+type(exc).__name__)
                    break
            self.lane_runs.append(lane)
            truncated |= lane['truncated']
        return papers, truncated

    def resolve(self, kind, value):
        if kind == 'arxiv':
            value = re.sub(r'v\d+$', '', value)
            if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})', value):
                raise ValueError('Invalid arXiv identifier')
            content = self.get('https://export.arxiv.org/api/query',
                               params={'id_list': value, 'max_results': 1}).content
            papers, _ = self.parse_arxiv(content)
            return papers
        if kind != 'doi':
            raise ValueError('Only declared DOI/arXiv identifiers are resolvable')
        doi = normalize_doi(value)
        if not doi or not re.fullmatch(r'10\.\d{4,9}/\S+', doi):
            raise ValueError('Invalid DOI')
        data = self.get('https://api.crossref.org/works/'+quote(doi, safe='')).json()['message']
        return [self.crossref_item(data)]

    @staticmethod
    def crossref_item(item):
        from .journals import clean
        from .models import PaperRecord
        dates = []
        for field in ('published', 'published-online', 'published-print'):
            for parts in item.get(field, {}).get('date-parts', []):
                if len(parts) == 3:
                    dates.append(datetime(*parts).date().isoformat())
        doi = normalize_doi(item['DOI'])
        return PaperRecord(paper_id='doi:'+doi, version='metadata',
                           title=clean(paper_title_text(' '.join(item.get('title', [])))),
                           abstract=clean(item.get('abstract', '')),
                           authors=[clean(' '.join(filter(None, [a.get('given'), a.get('family')])))
                                    for a in item.get('author', [])],
                           doi=doi, published=min(dates) if dates else '',
                           sources=['crossref:'+(' '.join(item.get('container-title', [])))],
                           url='https://doi.org/'+doi)

    def crossref(self, query, since, until):
        papers, truncated = [], False
        for kind in ('journal-article', 'proceedings-article'):
            try:
                msg = self.get('https://api.crossref.org/works', params={
                    'query.bibliographic': query, 'rows': self.max_results,
                    'filter': f'from-pub-date:{since},until-pub-date:{until},type:{kind}'}).json()['message']
                truncated |= int(msg['total-results']) > len(msg['items'])
                for index, item in enumerate(msg['items']):
                    try:
                        papers.append(self.crossref_item(item))
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        # A malformed deposited record must not discard its valid neighbours.
                        self.source_errors.append(f'{kind}:record:{index}:{type(exc).__name__}')
            except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
                self.source_errors.append(kind+':'+type(exc).__name__)
        return papers, truncated


def publication_basis(paper):
    if 'arxiv' in paper.sources:
        return 'arxiv_first_submission'
    if any(s.startswith('crossref:') for s in paper.sources):
        return 'earliest_publisher_publication_date_day_precision'
    if any(s.startswith(('official-rss:', 'pmlr:', 'neurips:')) for s in paper.sources):
        return 'official_citation_publication_date_day_precision'
    if any(s.startswith('journal-detail:') for s in paper.sources):
        return 'official_journal_first_publication_explicit_precision'
    if any(s.startswith('pubmed-exact:') for s in paper.sources):
        return 'pubmed_electronic_publication_day_precision'
    if 'deepseek-original-page' in paper.sources:
        return 'provider_original_page_date_day_precision'
    # RSS pubDate can describe feed publication; legacy adapters lose the field origin.
    return None
