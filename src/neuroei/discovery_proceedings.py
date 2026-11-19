"""Explicit official volume channels to fill pinned PaperFlow venue gaps.

One configured volume, title filtering, <=5 detail pages. No inferred publication
times, accepted-status inference, full-text fetching or all-years coverage claim.
"""
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from .journals import clean
from .models import PaperRecord
from .sources import normalize_doi


def official_document_locator(paper):
    selected = [s for s in paper.sources if s.get('selected_document')]
    if len(selected) != 1 or not selected[0]['source'].startswith(('official-rss:', 'pmlr:', 'neurips:')):
        return None
    url = selected[0].get('url', '')
    if not _landing(url) or url not in paper.original_links or selected[0].get('paper_id') != paper.paper_id:
        return None
    return {'landing_url': url, 'pdf_candidates': list(paper.fulltext_links),
            'paper_id': paper.paper_id, 'title': paper.title, 'doi': paper.doi, 'document_version': None}


def _landing(url):
    p = urlsplit(url)
    return (p.scheme == 'https' and not p.query and not p.fragment and (
        p.netloc == 'www.roboticsproceedings.org' and re.fullmatch(r'/rss\d+/p\d+\.html', p.path)
        or p.netloc == 'proceedings.mlr.press' and re.fullmatch(r'/v\d+/[A-Za-z0-9_-]+\.html', p.path)
        or p.netloc == 'proceedings.neurips.cc' and re.fullmatch(r'/paper_files/paper/\d{4}/hash/[A-Za-z0-9_-]+\.html', p.path)))


def _pdf(url, landing):
    p, l = urlsplit(url), urlsplit(landing)
    if p.scheme != 'https' or p.query or p.fragment or not p.path.endswith('.pdf'):
        return False
    if l.netloc == 'www.roboticsproceedings.org':
        return p.netloc == l.netloc and p.path == l.path.removesuffix('.html')+'.pdf'
    if l.netloc == 'proceedings.mlr.press':
        volume, slug = l.path.strip('/').split('/')
        return ((p.netloc == l.netloc and p.path in (f'/{volume}/{slug[:-5]}/{slug[:-5]}.pdf', f'/{volume}/{slug[:-5]}.pdf')) or
                (p.netloc == 'raw.githubusercontent.com' and p.path == f'/mlresearch/{volume}/main/assets/{slug[:-5]}/{slug[:-5]}.pdf'))
    return p.netloc == l.netloc and p.path.startswith(l.path.split('/hash/')[0]+'/file/') and p.path.split('/')[-1].split('-')[0] == l.path.split('/')[-1].split('-')[0]


def verify_official_document(paper, html, url):
    locator = official_document_locator(paper)
    if not locator or url != locator['landing_url']:
        raise ValueError('Official selected landing mismatch')
    kind = 'official-rss' if 'roboticsproceedings' in url else 'pmlr' if 'mlr.press' in url else 'neurips'
    parsed, pdfs, evidence = parse_detail(html, url, kind)
    if (' '.join(parsed.title.casefold().split()) != ' '.join(paper.title.casefold().split())
            or paper.doi is not None and parsed.doi != paper.doi):
        raise ValueError('Official DOI/title identity mismatch')
    verified = [pdf for pdf in pdfs if _pdf(pdf, url)]
    if len(verified) != 1 or len(evidence['declared_pdf_urls']) != 1:
        raise ValueError('Official PDF candidate missing, conflicting or unsupported')
    return {**evidence, 'pdf_url': verified[0], 'landing_url': url,
            'document_version': None, 'scope': 'official_metadata_identity_only_not_pdf_content'}


class LandingPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta, self.links = {}, []
        self.href, self.text = None, []
        self.in_title, self.title_parts, self.last_title = False, [], ''

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name', '').startswith('citation_'):
            self.meta.setdefault(attrs['name'], []).append(attrs.get('content', ''))
        if tag == 'p' and attrs.get('class') == 'title':
            self.in_title, self.title_parts = True, []
        if tag == 'a' and attrs.get('href'):
            self.href, self.text = attrs['href'], []

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.href:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_title:
            self.last_title = clean(' '.join(self.title_parts))
            self.in_title = False
        if tag == 'a' and self.href:
            label = clean(' '.join(self.text))
            self.links.append((self.href, self.last_title if label == 'abs' else label))
            self.href = None


class AbstractText(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.parts = []
        self.feed(text)

    def handle_data(self, text):
        self.parts.append(text)


def parse_detail(text, url, kind):
    detail = LandingPage()
    detail.feed(text)
    fields = detail.meta
    title = fields.get('citation_title', [''])[0]
    if not title:
        raise ValueError('Missing citation title')
    if len({' '.join(v.casefold().split()) for v in fields.get('citation_title', [])}) != 1:
        raise ValueError('Conflicting official titles')
    if len({normalize_doi(v) for v in fields.get('citation_doi', [])}) > 1:
        raise ValueError('Conflicting official DOI metadata')
    doi_match = re.search(r'DOI\s*=\s*\{([^}]+)\}', text, re.IGNORECASE)
    doi = normalize_doi(fields.get('citation_doi', [doi_match[1] if doi_match else None])[0])
    patterns = [r'<div[^>]*(?:id|class)=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</div>',
                r'<p[^>]*class="paper-abstract"[^>]*>(.*?)</p>',
                r'<b>Abstract:</b>\s*(.*?)\s*(?:<b>Download:|<h\d|<b>Bibtex:)']
    fragment = next((m[1] for pattern in patterns
                     if (m := re.search(pattern, text, re.DOTALL | re.IGNORECASE))), '')
    abstract = ' '.join(''.join(AbstractText(fragment).parts).split())
    published = fields.get('citation_publication_date', [''])[0].replace('/', '-')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', published):
        published = ''  # year-only metadata is not a publication day.
    venue = fields.get('citation_conference_title', fields.get('citation_journal_title', ['']))[0]
    pid = 'doi:'+doi if doi else 'official:'+url
    paper = PaperRecord(paper_id=pid, version='landing-metadata', title=title,
        abstract=abstract, authors=fields.get('citation_author', []), doi=doi, published=published,
        sources=[kind+':'+venue], url=url)
    pdfs = [urljoin(url, link).replace('http://www.roboticsproceedings.org/', 'https://www.roboticsproceedings.org/', 1)
            for link in fields.get('citation_pdf_url', [])]
    declared = list(pdfs)
    pdfs = [pdf for pdf in pdfs if _pdf(pdf, url)]
    return paper, pdfs, {'title': title, 'doi': doi, 'url': url,
        'publication_field': 'citation_publication_date' if published else None,
        'publication_value': published or None, 'document_version': None,
        'pdf_urls': pdfs, 'declared_pdf_urls': declared, 'abstract_present': bool(abstract)}


def _terms(query):
    return set(re.findall(r'[a-z0-9]+', query.casefold()))


def _hits(title, query, planned):
    terms = _terms(query)
    hits = sum(bool(re.search(r'\b'+re.escape(t)+r'\b', title.casefold())) for t in terms)
    return hits if hits >= (min(2, len(terms)) if planned else len(terms)) and terms else 0


def collect_volume(adapter, source, query):
    kind, volume = source.split(':')
    if not re.fullmatch(r'\d{4}' if kind == 'neurips' else r'\d{1,3}', volume):
        raise ValueError('Explicit official volume number required')
    if kind == 'neurips' and not 1987 <= int(volume) <= 2100:
        raise ValueError('Explicit valid NeurIPS proceedings year required')
    base = (f'https://www.roboticsproceedings.org/rss{volume}/' if kind == 'official-rss'
            else f'https://proceedings.neurips.cc/paper_files/paper/{volume}/' if kind == 'neurips'
            else f'https://proceedings.mlr.press/v{volume}/')
    plan = getattr(adapter, 'official_plan', None)
    if plan:
        entry = next((v for v in plan['volumes'] if v['source'] == source), None)
        if entry is None or query not in plan['queries']:
            raise ValueError('Source/query outside configured request plan')
        index_url = entry['index_url']
    else:
        index_url = base.rstrip('/') if kind == 'neurips' else base
    if not hasattr(adapter, '_official_cache'):
        adapter._official_cache = {}
    key = source if plan else (source, query)
    cached = adapter._official_cache.get(key)
    cache_hit = cached is not None
    if cached is None:
        cached = {'papers': [], 'errors': [], 'truncated': False, 'index_entries': 0, 'matched_titles': 0}
        adapter._official_cache[key] = cached  # failed attempts also bounded and cached.
        try:
            index = adapter.official_html(index_url, 20_000_000) if plan else adapter.get(index_url).text
            if len(index.encode()) > 20_000_000:
                raise ValueError('Volume index size limit')
            parsed = LandingPage()
            parsed.feed(index)
            links = {}
            for href, title in parsed.links:
                url = urljoin(base, href)
                if url.startswith(base) and url.endswith('.html') and title:
                    links.setdefault(url, title)
            cached['index_entries'] = len(links)
            queries = plan['queries'] if plan else [query]
            ranked = [(max((_hits(title, q, bool(plan)) for q in queries), default=0), url) for url, title in links.items()]
            ranked = sorted(((score, url) for score, url in ranked if score), key=lambda row: (-row[0], row[1]))
            cached['matched_titles'] = len(ranked)
            detail_limit = plan['detail_limit_per_volume'] if plan else 5
            cached['truncated'] = len(ranked) > detail_limit
            for _, url in ranked[:detail_limit]:
                try:
                    text = adapter.official_html(url, 1_000_000) if plan else adapter.get(url).text
                    if len(text.encode()) > 1_000_000:
                        raise ValueError('Landing metadata size limit')
                    paper, pdfs, evidence = parse_detail(text, url, kind)
                    cached['papers'].append(paper)
                    adapter.fulltext_candidates[paper.paper_id] = pdfs
                    adapter.fulltext_candidates[url] = pdfs
                    if hasattr(adapter, 'official_evidence'):
                        adapter.official_evidence[url] = evidence
                except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                    cached['errors'].append(url+':'+type(exc).__name__)
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            cached['errors'].append(index_url+':'+type(exc).__name__)
    adapter.source_errors.extend(cached['errors'])
    adapter.lane_runs.append({'source': source, 'query': query, 'index_url': index_url,
        'cache_hit': cache_hit, 'index_entries': cached['index_entries'],
        'matched_titles_all_queries': cached['matched_titles'], 'details_fetched': len(cached['papers']),
        'truncated': cached['truncated'], 'coverage_complete': False,
        'title_match_policy': 'at_least_two_query_tokens' if plan else 'all_query_tokens',
        'publication_dates': 'explicit_citation_field_only; no_conference_date_inference'})
    papers = [p for p in cached['papers'] if _hits(p.title, query, bool(plan))]
    return papers, cached['truncated']
