"""Explicitly budgeted DeepSeek Search/Fetch fallback; no chat model or implicit retries.

Verified API: https://api-docs.deepseek.com/guides/search/ (2026-11-04).
Search snippets/summaries/dates are discovery leads, never original abstracts or
publication dates. This adapter retains raw page evidence for boundary validation.
"""
import copy
import json
import math
import re
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

UNIT_USD = 0.00075
MAX_WEB_SEARCHES = 64
MAX_WEB_FETCHES = 128
SEARCH_SITES = frozenset({'arxiv.org', 'proceedings.neurips.cc', 'proceedings.mlr.press', 'www.roboticsproceedings.org'})
PUBLISHER_HOSTS = frozenset({'nature.com', 'www.nature.com', 'science.org', 'www.science.org',
    'cell.com', 'www.cell.com', 'mdpi.com', 'www.mdpi.com'})
PUBLISHER_FEED_QUERIES = {'science.org': 'af=R', 'cell.com': 'rss=yes'}
SEARCH_SITES = SEARCH_SITES | PUBLISHER_HOSTS
OFFICIAL_HOSTS = frozenset({'arxiv.org', 'export.arxiv.org', 'proceedings.mlr.press',
    'proceedings.neurips.cc', 'www.roboticsproceedings.org', 'roboticsproceedings.org',
    'proceedings.iclr.cc', 'icml.cc', 'openreview.net', 'ojs.aaai.org', 'openaccess.thecvf.com',
    'ieeexplore.ieee.org', 'doi.org'}) | PUBLISHER_HOSTS


def canonical_page(url):
    """Canonicalize recognized arXiv mirrors and publisher article links.

    PDF itself is not sent to a Markdown page fetcher. Preserve the search URL in
    the lead, and retain this separately as the actual fetched document URL.
    """
    try:
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443):
            return url
        if parsed.hostname in {'arxiv.org', 'export.arxiv.org', 'browse-export.arxiv.org', 'ar5iv.labs.arxiv.org'}:
            match = re.fullmatch(r'/(?:pdf|abs|html)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/?', parsed.path)
            if match:
                return 'https://arxiv.org/abs/'+match[1]
        if parsed.hostname in PUBLISHER_HOSTS:
            host = parsed.hostname.removeprefix('www.')
            if not parsed.query or parsed.query == PUBLISHER_FEED_QUERIES.get(host):
                clean = urlunsplit(('https', 'www.'+host, parsed.path, '', ''))
                if paper_page_url(clean):
                    return clean
        return url
    except ValueError:
        return url


def official_url(url):
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == 'https' and parsed.hostname in OFFICIAL_HOSTS
                and not parsed.username and not parsed.password and parsed.port in (None, 443))
    except ValueError:
        return False


def paper_page_url(url):
    if not official_url(url):
        return False
    parsed = urlsplit(url)
    host, path = parsed.hostname, parsed.path
    if host in PUBLISHER_HOSTS:
        if parsed.query or parsed.fragment:
            return False
        if host in {'mdpi.com', 'www.mdpi.com'}:
            return bool(re.fullmatch(r'/\d{4}-\d{4}/\d+/\d+/\d+(?:/notes)?', path))
        if host in {'nature.com', 'www.nature.com'}:
            return bool(re.fullmatch(r'/articles/(?:[sd]\d{5}-[A-Za-z0-9-]+|nature\d+)', path))
        from .discovery_journals import journal_article_url
        return journal_article_url(url) is not None
    if host in {'arxiv.org', 'export.arxiv.org'}:
        return bool(re.fullmatch(r'/(?:abs|html)/\d{4}\.\d{4,5}(?:v\d+)?/?', path))
    if host == 'openreview.net':
        return path == '/forum' and bool(re.fullmatch(r'[A-Za-z0-9_-]{1,100}', parse_qs(parsed.query).get('id', [''])[0]))
    patterns = {
        'proceedings.mlr.press': r'/v\d+/[^/]+\.html',
        'proceedings.neurips.cc': r'/paper(?:_files)?/paper/\d{4}/hash/[^/]+[Aa]bstract[^/]*\.html',
        'proceedings.iclr.cc': r'/paper_files/paper/\d{4}/hash/[^/]+[Aa]bstract[^/]*\.html',
        'www.roboticsproceedings.org': r'/rss\d+/p\d+\.html',
        'roboticsproceedings.org': r'/rss\d+/p\d+\.html',
        'ieeexplore.ieee.org': r'/(?:abstract/)?document/\d+(?:/[^/]*)?',
        'doi.org': r'/10\.\d{4,9}/.+',
        'ojs.aaai.org': r'/index\.php/[^/]+/article/view/\d+',
        'openaccess.thecvf.com': r'/content[^/]*/.+\.html',
        'icml.cc': r'/virtual/\d{4}/poster/\d+',
    }
    return bool(re.fullmatch(patterns.get(host, r'(?!)'), path))


def _publisher_scope(url, site):
    if site not in PUBLISHER_HOSTS:
        return True
    parsed = urlsplit(url)
    host = site.removeprefix('www.')
    if parsed.hostname not in {host, 'www.'+host}:
        return False
    return host != 'nature.com' or bool(re.fullmatch(r'/articles/(?:s41586-[A-Za-z0-9-]+|nature\d+)', parsed.path))


class WebBudgetExceeded(RuntimeError):
    pass


class DeepSeekWeb:
    def __init__(self, api_key, *, enabled=False, search_limit=6, fetch_limit=12,
                 max_usd=.02, transport=None, recall_variant=False, site='', audit_callback=None, source_scoped=False):
        if not api_key or not enabled:
            raise ValueError('Explicit enabled=True and API key required for paid web tools')
        if (not 0 <= search_limit <= MAX_WEB_SEARCHES or not 0 <= fetch_limit <= MAX_WEB_FETCHES
                or not math.isfinite(max_usd) or max_usd <= 0):
            raise ValueError('Explicit bounded web-tool limits required')
        self.limits = {'search': search_limit, 'fetch': fetch_limit}
        self.used = {'search': 0, 'fetch': 0}
        self.max_usd, self.audit = max_usd, []
        self.audit_callback, self._collections = audit_callback, {}
        self.should_stop = None
        if site and site not in SEARCH_SITES:
            raise ValueError('Only explicit supported site restrictions may be configured')
        self.recall_variant, self.site = recall_variant, site
        self.source_scoped = source_scoped
        self.policy = {'provider': 'deepseek-web-v5-canonical-publisher-links', 'limits': self.limits, 'max_usd': max_usd,
                       'unit_reservation_usd': UNIT_USD, 'response_byte_limit': 2_000_000,
                       'source_scoped': source_scoped, 'site_mapping_policy': 'official-single-site-v1',
                       'query_policy': 'recall-variant-v1' if recall_variant else 'literal-v1', 'site': site}
        self.http = httpx.Client(base_url='https://api.deepseek.com', timeout=30,
            transport=transport, headers={'Authorization': 'Bearer '+api_key}, follow_redirects=False)

    def close(self):
        self.http.close()

    def _call(self, tool, body):
        if self.used[tool] >= self.limits[tool] or (sum(self.used.values())+1)*UNIT_USD > self.max_usd:
            raise WebBudgetExceeded('Web tool budget exhausted; no request sent')
        self.used[tool] += 1  # Charge reservation for EVERY attempt, even failures.
        entry = {'tool': tool, 'at': datetime.now(UTC).isoformat(), 'status': 'failed',
                 'reserved_cost_usd': UNIT_USD, 'attempt_id': len(self.audit)+1, 'call_index': len(self.audit)+1,
                 'used': dict(self.used), 'reserved_upper_cost_usd': round(sum(self.used.values())*UNIT_USD, 6)}
        self.audit.append(entry)
        if self.audit_callback:
            self.audit_callback({**copy.deepcopy(entry), 'phase': 'before', 'status': 'reserved'})
        try:
            with self.http.stream('POST', '/v1/tools/'+tool, json=body) as response:
                entry['http_status'] = response.status_code
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 2_000_000:
                        raise ValueError('Web response byte limit exceeded')
                    chunks.append(chunk)
                def unique_pairs(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result: raise ValueError('Duplicate JSON response field')
                        result[key] = value
                    return result
                def invalid_constant(_): raise ValueError('Nonfinite JSON response')
                def finite_float(value):
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError('Nonfinite JSON response number')
                    return number
                payload = json.loads(b''.join(chunks), object_pairs_hook=unique_pairs,
                                     parse_constant=invalid_constant, parse_float=finite_float)
            if tool == 'search' and (not isinstance(payload, dict) or not isinstance(payload.get('results'), list)):
                raise ValueError('Invalid Search response')
            if tool == 'search':
                if len(payload['results']) > body['count']:
                    raise ValueError('Search result count exceeds requested bound')
                for lead in payload['results']:
                    if (not isinstance(lead, dict) or not isinstance(lead.get('url'), str)
                            or len(lead['url']) > 8192
                            or any(key in lead and (not isinstance(lead[key], str) or len(lead[key]) > 200_000)
                                   for key in ('title', 'summary', 'snippet', 'content'))):
                        raise ValueError('Invalid bounded Search result')
            if tool == 'fetch' and (not isinstance(payload, dict)
                                   or not isinstance(payload.get('content'), str)
                                   or not isinstance(payload.get('title'), str)):
                raise ValueError('Invalid Fetch response')
            entry['status'] = 'completed'
            entry['nonempty'] = bool(payload['results'] if tool == 'search' else payload['content'])
            entry['billable_success'] = entry['nonempty']
            return payload
        except (httpx.HTTPError, ValueError, RecursionError) as exc:
            entry['error_type'] = type(exc).__name__
            raise
        finally:
            entry.setdefault('billable_success', None if entry.get('http_status') == 200 else False)
            if self.audit_callback:
                self.audit_callback({**copy.deepcopy(entry), 'phase': 'after'})

    def search(self, query, *, count=10):
        if not isinstance(query, str) or not query.strip() or len(query) > 128 or not 1 <= count <= 50:
            raise ValueError('Search query/count outside official API bounds')
        return self._call('search', {'query': query, 'count': count, 'need_content': False})

    def fetch(self, url):
        if not isinstance(url, str) or len(url) > 8192 or not paper_page_url(url):
            raise ValueError('Only explicit official scholarly HTTPS hosts may be fetched')
        return self._call('fetch', {'url': url})

    def fetch_pubmed_citation(self, pmid):
        """Fetch the official citation page for an already verified PMID."""
        if not isinstance(pmid, str) or not re.fullmatch(r'[1-9][0-9]{0,11}', pmid):
            raise ValueError('Invalid PubMed citation identifier')
        return self._call('fetch', {'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'})

    def fetch_wechat_article(self, url):
        """Fetch a canonical media article through the shared Fetch budget."""
        if not isinstance(url, str) or len(url) > 8192:
            raise ValueError('Canonical WeChat article URL required')
        from .wechat_discovery import canonical_wechat_article

        if canonical_wechat_article(url) != url:
            raise ValueError('Canonical WeChat article URL required')
        return self._call('fetch', {'url': url})

    def collect(self, query, *, max_fetches=2, site=None):
        if not 0 <= max_fetches <= 5:
            raise ValueError('max_fetches=0..5 required')
        effective_site = self.site if site is None else site
        if effective_site and effective_site not in SEARCH_SITES:
            raise ValueError('Unsupported official search site')
        key = json.dumps([query, max_fetches, effective_site, self.policy], sort_keys=True)
        if key not in self._collections:
            try:
                self._collections[key] = self._collect(query, max_fetches=max_fetches, site=effective_site)
            except RuntimeError as exc:
                self._collections[key] = exc
                raise
        if isinstance(self._collections[key], RuntimeError):
            raise self._collections[key]
        return copy.deepcopy(self._collections[key])

    def _collect(self, query, *, max_fetches, site):
        result = {'schema_version': '1', 'provider': 'deepseek_search_fetch', 'query': query,
                  'original_query': query, 'site': site,
                  'status': 'failed', 'leads': [], 'documents': [], 'warnings': [], 'fetch_limit_truncated': False,
                  'original_abstracts_verified': False, 'publication_dates_verified': False}
        try:
            from .discovery_queries import recall_variant
            plan = recall_variant(query) if self.recall_variant else None
            planned = plan['planned_query'] if plan else query
            effective_query = planned+' site:'+site if site else planned
            result.update(query_plan=plan, effective_query=effective_query)
            response = self.search(effective_query)
        except WebBudgetExceeded:
            result['warnings'].append('budget_insufficient')
            return result
        except (httpx.HTTPError, ValueError, RecursionError) as exc:
            result['warnings'].append('search_failed:'+type(exc).__name__)
            return result
        seen = set()
        for lead in response['results']:
            if not isinstance(lead, dict) or not isinstance(lead.get('url'), str):
                result['warnings'].append('invalid_search_lead')
                continue
            if lead['url'] in seen:
                continue
            seen.add(lead['url'])
            page_url = canonical_page(lead['url'])
            result['leads'].append({**lead, 'page_url': page_url,
                                    'evidence_scope': 'search_result_not_original_paper',
                                    'official_host': official_url(page_url),
                                    'paper_page': paper_page_url(page_url) and _publisher_scope(page_url, site)})
        unique_pages = {}
        for lead in result['leads']:
            if lead['paper_page']:
                unique_pages.setdefault(lead['page_url'], lead)
        result['candidate_pages'] = len(unique_pages)
        result['fetch_limit_truncated'] = len(unique_pages) > max_fetches
        for lead in list(unique_pages.values())[:max_fetches]:
            if self.should_stop and self.should_stop():
                result['warnings'].append('canceled_by_user')
                break
            try:
                page = self.fetch(lead['page_url'])
                result['documents'].append({'document_id': lead['page_url'], 'url': lead['page_url'],
                    'discovery_url': lead['url'],
                    'fetched_at': datetime.now(UTC).isoformat(), 'title': page['title'],
                    'content': page['content'], 'scope': 'provider_fetched_original_page_markdown',
                    'status': 'completed' if page['content'] else 'unavailable'})
            except WebBudgetExceeded:
                result['warnings'].append('budget_insufficient')
                break
            except (httpx.HTTPError, ValueError, RecursionError) as exc:
                result['warnings'].append('fetch_failed:'+type(exc).__name__)
        result['status'] = 'partial' if result['warnings'] else 'completed'
        if result['leads'] and not any(d['content'] for d in result['documents']):
            result['warnings'].append('original_page_not_retrieved')
            result['status'] = 'partial'
        result['zero_matches'] = not result['leads']
        return result


def original_page_record(document):
    """Extract only explicit fields from a fetched original page, never search summaries.

    Title-only pages remain candidates with unknown abstract/date/version. Day-only
    dates retain that precision and cannot pass strict Beijing daily selection.
    """
    from .models import PaperRecord
    from .sources import normalize_doi
    url, title, content = document['url'], document['title'].strip(), document['content']
    if not paper_page_url(url) or document.get('status') != 'completed' or not content.strip() or not title:
        return None
    arxiv = re.search(r'^https://(?:export\.)?arxiv\.org/(?:abs|html)/(\d{4}\.\d{4,5})(v\d+)?(?:[/?#]|$)', url)
    if arxiv:
        title = re.sub(r'^\[\d{4}\.\d{4,5}(?:v\d+)?\]\s*', '', title)
    version = arxiv[2] if arxiv and arxiv[2] else None
    if arxiv and not version:
        header_version = re.search(r'\barXiv:\s*'+re.escape(arxiv[1])+r'(v\d+)\b', content[:4000])
        version = header_version[1] if header_version else None
    abstracts = re.findall(r'(?:^|\n)#{1,6}\s+Abstract:?\s*\n(.*?)(?=\n#{1,6}\s|'
        r'\n(?:Published in|Date of Publication|DOI|References|Citations):|\Z)', content, re.DOTALL | re.IGNORECASE)
    abstract = max((a.strip() for a in abstracts if 'Show More' not in a), key=len, default='')
    if not abstract:
        arxiv_abstract = re.search(r'(?:^|\n)\s*Abstract:\s*(.*?)(?=\n\s*(?:Comments:|Subjects:|Cite as:|Journal reference:|DOI:|Submission history)|\n#{1,6}\s|\Z)',
                                  content, re.DOTALL)
        if arxiv_abstract:
            abstract = arxiv_abstract[1].strip()
    metadata = content.split('\n## References')[0]
    doi_match = re.search(r'\bDOI:\s*(?:https://doi\.org/)?(10\.\d{4,9}/[^\s)\]]+)', metadata, re.IGNORECASE)
    doi = normalize_doi(doi_match[1]) if doi_match else None
    if urlsplit(url).hostname in PUBLISHER_HOSTS:
        # Provider title/summary alone cannot establish original-page identity.
        host, path = urlsplit(url).hostname, urlsplit(url).path
        if path.endswith('/notes') or path.startswith('/articles/d41586-') or re.match(
                r'(?:author correction|correction|retraction|editorial|news|commentary)\b', title, re.IGNORECASE):
            return None
        headings = [line.lstrip('#').strip().removeprefix('Title:').strip() for line in content.splitlines()
                    if re.match(r'^#{1,2}\s+', line)]
        if not doi or not any(' '.join(h.casefold().split()) == ' '.join(title.casefold().split()) for h in headings):
            return None
        if 'nature.com' in host and doi != '10.1038/'+path.split('/')[-1]:
            return None
        if 'science.org' in host and doi != '10.1126/'+path.split('/10.1126/')[-1]:
            return None
        if 'cell.com' in host and not doi.startswith('10.1016/j.cell.'):
            return None
        if 'mdpi.com' in host and not doi.startswith('10.3390/'):
            return None
    pid = 'arxiv:'+arxiv[1] if arxiv else 'doi:'+doi if doi else 'official:'+url
    def day(text):
        for fmt in ('%d %b %Y', '%d %B %Y', '%d-%b-%Y', '%d-%B-%Y'):
            try:
                return datetime.strptime(text.replace(',', ''), fmt).date().isoformat()
            except ValueError:
                continue
        return ''

    published, updated = '', None
    date_match = re.search(r'\b(?:Submitted on|Date of Publication:)\s*(\d{1,2}[ -][A-Za-z]+[ ,\-]+\d{4})', metadata)
    if date_match:
        published = day(date_match[1])
    revised = re.search(r'\blast revised\s+(\d{1,2}[ -][A-Za-z]+[ ,\-]+\d{4})', metadata, re.IGNORECASE)
    if revised:
        updated = day(revised[1]) or None
    elif version == 'v1':
        updated = published or None
    return PaperRecord(paper_id=pid, version=version or 'unversioned',
        title=title, abstract=abstract, authors=[], doi=doi, published=published, updated=updated,
        sources=['deepseek-original-page'], url=url, is_preprint=bool(arxiv))


def fallback_after_failure(source_result, web, *, max_fetches=2, partial_failure=False):
    """Healthy emptiness never triggers paid web discovery. Caller must persist audit."""
    if source_result.status != 'failed' and not (source_result.status == 'partial' and partial_failure):
        return {'schema_version': '1', 'status': 'skipped', 'reason': 'primary_has_no_source_failure'}
    if getattr(web, 'source_scoped', False):
        source = source_result.source
        site = ('proceedings.neurips.cc' if source.startswith('neurips:') else
                'proceedings.mlr.press' if source.startswith('pmlr:') else
                'www.roboticsproceedings.org' if source.startswith('official-rss:') else
                'arxiv.org' if source in ('arxiv', 'crossref') else
                {'journal:Nature': 'www.nature.com', 'journal:Science Robotics': 'www.science.org',
                 'journal:Cell': 'www.cell.com', 'publisher:MDPI': 'www.mdpi.com'}.get(source))
        result = web.collect(source_result.query, max_fetches=max_fetches, site=site)
    else:
        result = web.collect(source_result.query, max_fetches=max_fetches)
    return {**result, 'primary_failure': {'source': source_result.source, 'status': source_result.status,
        'error': source_result.error, 'since': source_result.since, 'until': source_result.until},
        'audit': {'calls': list(web.audit), 'reserved_upper_cost_usd': round(sum(web.used.values())*UNIT_USD, 6)}}
