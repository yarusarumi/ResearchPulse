"""Request-scoped journal feed leads -> strictly identified official articles.

No legacy RSS dates, feed summaries, Crossref substitutions or PDF versions.
"""
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit

import httpx

from .discovery_contract import RecommendationRequest
from .discovery_pubmed_exact import known_dois, recover_exact
from .journals import clean
from .models import PaperRecord, SourceResult
from .sources import normalize_doi, paper_title_key, paper_title_text

JOURNAL_POLICY = 'official-journal-detail-v2'
JOURNALS = {
    'Nature': ('https://www.nature.com/nature.rss', 'www.nature.com'),
    'Science Robotics': ('https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=scirobotics', 'www.science.org'),
    'Cell': ('https://www.cell.com/cell/current.rss', 'www.cell.com'),
}
RESEARCH_TYPES = {'article', 'research article', 'research-article', 'letter', 'resource', 'report'}


class JournalMetadataUnavailable(ValueError):
    """Missing publisher identity is a source/parse failure, not healthy emptiness."""


def plan_journal_sources(request, *, pubmed_exact=False):
    req = RecommendationRequest.model_validate(request)
    return {'schema_version': '1', 'policy': JOURNAL_POLICY, 'request_id': req.request_id,
        'user_id': req.user_id, 'as_of': req.as_of.isoformat(), 'mode': req.mode,
        'queries': req.queries[:24], 'source_names': ['journal:'+name for name in JOURNALS],
        'feeds': {name: value[0] for name, value in JOURNALS.items()}, 'detail_limit_per_journal': 6,
        'max_http_attempts': 21, 'coverage': 'unknown',
        'pubmed_exact': pubmed_exact, 'max_pubmed_http_attempts': 6 if pubmed_exact else 0,
        'warnings': ['journal_recent_feed_only_not_historical_coverage', 'journal_daily_requires_original_publication_timezone']}


def journal_article_url(url, name=None):
    try:
        p = urlsplit(url)
        if p.scheme != 'https' or p.username or p.password or p.port not in (None, 443) or p.fragment:
            return None
        host = p.hostname
        allowed = {'Nature': ({'nature.com', 'www.nature.com'}, r'/articles/[A-Za-z0-9-]+'),
            'Science Robotics': ({'science.org', 'www.science.org'}, r'/doi/(?:abs/|full/)?10\.1126/scirobotics\.[A-Za-z0-9]+'),
            'Cell': ({'cell.com', 'www.cell.com'}, r'/cell/(?:fulltext|abstract)/S0092-8674\(\d{2}\)\d{5}-[A-Za-z0-9]')}
        for key, (hosts, pattern) in allowed.items():
            if (name is None or name == key) and host in hosts and re.fullmatch(pattern, p.path):
                if p.query and p.query not in ('af=R', 'rss=yes'):
                    return None
                return urlunsplit(('https', JOURNALS[key][1], p.path, '', ''))
    except ValueError:
        pass
    return None


class JournalPage(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.meta, self.abstracts = {}, []
        self.depth, self.capture, self.parts = 0, None, []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            key = (attrs.get('name') or attrs.get('property') or '').casefold()
            self.meta.setdefault(key, []).append(attrs.get('content', ''))
        if tag in ('div', 'section', 'p'):
            self.depth += 1
            marker = attrs.get('id', '')+' '+attrs.get('class', '')
            if self.capture is None and re.search(r'(?:^|[\s_-])abstract(?:$|[\s_-])', marker, re.IGNORECASE):
                self.capture, self.parts = self.depth, []
        if tag == 'br' and self.capture is not None:
            self.parts.append(' ')

    def handle_data(self, data):
        if self.capture is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ('div', 'section', 'p'):
            if self.depth == self.capture:
                self.abstracts.append(' '.join(''.join(self.parts).split()))
                self.capture = None
            self.depth -= 1


def _one(meta, keys):
    values = {v.strip() for key in keys for v in meta.get(key, []) if v.strip()}
    if len(values) > 1:
        raise ValueError('Conflicting journal identity metadata')
    return next(iter(values), '')


def first_publication(meta):
    """Select earliest original online/print date before any window filtering."""
    dates = []
    for key in ('citation_online_date', 'citation_publication_date', 'prism.publicationdate', 'datepublished'):
        for raw in meta.get(key, []):
            value = raw.strip().replace('/', '-')
            try:
                if len(value) == 10:
                    dt = datetime.strptime(value, '%Y-%m-%d')
                else:
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        continue
                dates.append((dt.date(), 0 if len(value) == 10 else 1, value, key))
            except ValueError:
                continue
    if not dates:
        return '', []
    aware = sorted((row for row in dates if row[1]), key=lambda row: datetime.fromisoformat(row[2]).astimezone(UTC))
    days = sorted((row for row in dates if not row[1]), key=lambda row: row[0])
    first = days[0] if days and (not aware or days[0][0] <= min(
        datetime.fromisoformat(aware[0][2]).date(), datetime.fromisoformat(aware[0][2]).astimezone(UTC).date())) else aware[0]
    return first[2], [{'field': key, 'value': value} for _, _, value, key in dates]


def _pdf_allowed(pdf, landing, doi):
    p, l = urlsplit(pdf), urlsplit(landing)
    if p.scheme != 'https' or p.netloc != l.netloc or p.query or p.fragment:
        return False
    if l.netloc == 'www.nature.com':
        return p.path == l.path+'.pdf'
    if l.netloc == 'www.science.org':
        return p.path == '/doi/pdf/'+doi or p.path == '/doi/epdf/'+doi
    return l.netloc == 'www.cell.com' and p.path == l.path.replace('/fulltext/', '/pdf/').replace('/abstract/', '/pdf/')+'.pdf'


def parse_journal_detail(html, url, name, *, expected_doi=None, expected_title=None):
    if journal_article_url(url, name) != url:
        raise ValueError('Journal article URL not verified')
    page = JournalPage(html)
    title = _one(page.meta, ('citation_title',))
    doi = normalize_doi(_one(page.meta, ('citation_doi',)))
    journal = _one(page.meta, ('citation_journal_title',))
    category = _one(page.meta, ('citation_article_type', 'dc.type', 'article:type')).casefold()
    if (not title or not re.fullmatch(r'10\.\d{4,9}/\S+', doi or '')
            or journal.casefold() != name.casefold()):
        raise JournalMetadataUnavailable('Journal DOI/title/name unverified; possible access challenge')
    if expected_doi and doi != normalize_doi(expected_doi) or expected_title and paper_title_key(title) != paper_title_key(expected_title):
        raise ValueError('Journal article identity mismatch')
    if category not in RESEARCH_TYPES or re.match(r'(?:author correction|correction|retraction|editorial|news|commentary)\b', title, re.IGNORECASE):
        raise ValueError('Research article classification unverified')
    host = urlsplit(url).hostname
    if (host == 'www.nature.com' and doi != '10.1038/'+urlsplit(url).path.split('/')[-1]
            or host == 'www.science.org' and doi != '10.1126/'+urlsplit(url).path.split('/10.1126/')[-1]
            or host == 'www.cell.com' and not doi.startswith('10.1016/j.cell.')):
        raise ValueError('Journal DOI and canonical article path mismatch')
    published, dates = first_publication(page.meta)
    abstracts = list({a for a in page.abstracts if a})
    if len(abstracts) > 1:
        raise ValueError('Conflicting original abstracts')
    abstract = abstracts[0] if abstracts else ''
    pdfs = [p for p in page.meta.get('citation_pdf_url', []) if _pdf_allowed(p, url, doi)]
    evidence = {'title': title, 'doi': doi, 'journal': journal, 'article_type': category,
        'url': url, 'publication_evidence': dates, 'publication_value': published or None,
        'abstract_scope': 'official_detail_abstract' if abstract else 'unknown',
        'declared_pdf_urls': page.meta.get('citation_pdf_url', []), 'pdf_urls': pdfs, 'document_version': None}
    return PaperRecord(paper_id='doi:'+doi, version='landing-metadata', title=title, doi=doi,
        authors=page.meta.get('citation_author', []), abstract=abstract, published=published,
        sources=['journal-detail:'+name], url=url), evidence


def journal_document_locator(paper):
    selected = [s for s in paper.sources if s.get('selected_document')]
    if len(selected) != 1 or not selected[0]['source'].startswith('journal-detail:'):
        return None
    source = selected[0]
    name = source['source'].split(':', 1)[1]
    url = source.get('url', '')
    if not journal_article_url(url, name) or url not in paper.original_links or source.get('paper_id') != paper.paper_id:
        return None
    return {'landing_url': url, 'pdf_candidates': list(paper.fulltext_links), 'paper_id': paper.paper_id,
            'title': paper.title, 'doi': paper.doi, 'document_version': None}


def verify_journal_document(paper, html, url):
    locator = journal_document_locator(paper)
    if not locator or locator['landing_url'] != url:
        raise ValueError('Journal selected landing mismatch')
    name = next(s['source'].split(':', 1)[1] for s in paper.sources if s.get('selected_document'))
    _, evidence = parse_journal_detail(html, url, name, expected_doi=paper.doi, expected_title=paper.title)
    if len(evidence['pdf_urls']) != 1 or len(evidence['declared_pdf_urls']) != 1:
        raise ValueError('Single official PDF not verified')
    return {**evidence, 'landing_url': url, 'pdf_url': evidence['pdf_urls'][0],
            'scope': 'journal_official_metadata_identity_only_not_pdf_content'}


def feed_leads(xml, name):
    root, leads = ET.fromstring(xml), []
    for node in root.iter():
        if node.tag.split('}')[-1] not in ('item', 'entry'):
            continue
        fields = {child.tag.split('}')[-1]: ''.join(child.itertext()).strip() for child in node}
        url = journal_article_url(fields.get('link', ''), name)
        if url:
            leads.append({'url': url, 'title': clean(paper_title_text(fields.get('title', ''))),
                          'doi': normalize_doi(fields.get('doi', fields.get('identifier', '')).removeprefix('doi:')),
                          'type': fields.get('type', fields.get('section', ''))})
    return list({lead['url']: lead for lead in leads}.values())


def _source_error(url, exc):
    label = f'{url}:{type(exc).__name__}'
    if isinstance(exc, httpx.HTTPStatusError):
        return f'{label}:{exc.response.status_code}'
    if isinstance(exc, RuntimeError):
        return f'{label}:{exc}'
    return label


def collect_journal(adapter, source, query, since, until):
    name = source.removeprefix('journal:')
    plan = getattr(adapter, 'journal_plan', None)
    result = SourceResult(source=source, query=query, since=since, until=until, status='failed')
    if not plan or name not in JOURNALS or query not in plan['queries']:
        result.error = 'JournalPlanNotConfigured'
        return result
    cached = adapter._journal_cache.get(name)
    cache_hit = cached is not None
    if cached is None:
        cached = {'papers': [], 'errors': [], 'omitted': 0, 'matched': 0, 'feed_count': 0, 'exclusions': [],
                  'pubmed': None, 'recovered': []}
        failed_dois = []
        adapter._journal_cache[name] = cached
        try:
            xml = adapter.journal_text(JOURNALS[name][0], 1_000_000, media=('xml', 'rss', 'atom'))
            leads = feed_leads(xml, name)
            cached['feed_count'] = len(leads)
            ranked = []
            for lead in leads:
                # Only retrieval filtering: detail is the mandatory classification authority.
                if name == 'Nature' and urlsplit(lead['url']).path.startswith('/articles/d41586-'):
                    cached['omitted'] += 1
                    cached['exclusions'].append({'url': lead['url'], 'reason': 'nature_nonresearch_article_family'})
                    continue
                if lead['type'] and lead['type'].casefold() not in RESEARCH_TYPES:
                    cached['omitted'] += 1
                    continue
                score = max((sum(t in lead['title'].casefold() for t in re.findall(r'[a-z0-9]+', q.casefold())) for q in plan['queries']), default=0)
                if score:
                    ranked.append((score, lead))
            ranked.sort(key=lambda row: (-row[0], row[1]['url']))
            cached['matched'] = len(ranked)
            for _, lead in ranked[:plan['detail_limit_per_journal']]:
                try:
                    html = adapter.journal_text(lead['url'], 2_000_000, media=('text/html',))
                    paper, evidence = parse_journal_detail(html, lead['url'], name, expected_doi=lead['doi'], expected_title=lead['title'])
                    cached['papers'].append(paper)
                    adapter.fulltext_candidates[paper.url] = evidence['pdf_urls']
                    adapter.official_evidence[paper.url] = evidence
                except (JournalMetadataUnavailable, httpx.HTTPError, RuntimeError) as exc:
                    cached['errors'].append(_source_error(lead['url'], exc))
                    failed_dois.extend(known_dois((lead['doi'] or '', lead['url']), name))
                except ValueError as exc:
                    cached['omitted'] += 1  # unverified metadata != transport failure
                    cached['exclusions'].append({'url': lead['url'], 'reason': str(exc)})
        except (httpx.HTTPError, ValueError, RuntimeError, ET.ParseError) as exc:
            cached['errors'].append(_source_error(JOURNALS[name][0], exc))
        if cached['errors'] and plan.get('pubmed_exact'):
            seeds = known_dois((*plan['queries'], *adapter.journal_doi_leads), name)
            recovered, cached['pubmed'] = recover_exact(adapter, name, list(dict.fromkeys(failed_dois + seeds)))
            cached['recovered'] = recovered
    adapter.source_errors.extend(cached['errors'])
    adapter.lane_runs.append({'source': source, 'feed_count': cached['feed_count'],
        'matched_titles': cached['matched'], 'details_verified': len(cached['papers']),
        'metadata_omitted': cached['omitted'], 'metadata_exclusions': cached['exclusions'],
        'scope': 'bounded_recent_feed_only', 'coverage_complete': False, 'cache_hit': cache_hit})
    terms = re.findall(r'[a-z0-9]+', query.casefold())
    result.papers = [p for p in cached['papers'] if any(t in p.title.casefold() for t in terms)]
    if cached['pubmed']:
        query_dois = set(known_dois((query,), name))
        recovered = [p for p in cached['recovered'] if p.doi in query_dois
                     or any(t in p.title.casefold() for t in terms)]
        adapter.journal_recovery = (recovered, cached['pubmed'], cache_hit)
    result.truncated = True  # finite recent feed never proves all publication coverage
    result.status = 'partial' if not cached['errors'] or result.papers else 'failed'
    result.error = ';'.join(cached['errors']) or None
    return result
