"""Narrow, offline MDPI original-VoR evidence adapter; no HTTP or guessed versions."""
import re
import unicodedata
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from .sources import normalize_doi

PUBLISHER_VERSION_POLICY = 'mdpi-original-vor-v1'


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.meta, self.rows = {}, []
        self.row = None
        self.cell = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            key = (attrs.get('name') or attrs.get('property') or '').casefold()
            self.meta.setdefault(key, set()).add(attrs.get('content', ''))
        if tag == 'tr':
            self.row = {'cells': [], 'links': []}
        if tag in ('td', 'th') and self.row is not None:
            self.cell = []
        if tag == 'a' and self.row is not None and attrs.get('href'):
            self.row['links'].append(attrs['href'])

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.row is not None and self.cell is not None:
            self.row['cells'].append(' '.join(' '.join(self.cell).split()))
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _title(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _official(url):
    parsed = urlsplit(url)
    return (parsed.scheme == 'https' and parsed.netloc == 'www.mdpi.com'
            and not parsed.query and not parsed.fragment)


def _stamp(cells):
    # Require the offset abbreviation in the same record; never infer site timezone.
    value = ' '.join(cells)
    matches = list(re.finditer(r'\b(\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]+ \d{4})[, ]+(\d{2}:\d{2}(?::\d{2})?)\s+(CEST|CET|UTC)\b', value))
    if len(matches) != 1:
        return None
    date, clock, zone = matches[0].groups()
    fmt = '%Y-%m-%d' if '-' in date else '%d %B %Y'
    # English month names explicitly mapped to avoid depending on host locale.
    if '-' not in date:
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        day, month, year = date.split()
        if month not in months:
            return None
        date, fmt = f'{year}-{months.index(month)+1:02}-{int(day):02}', '%Y-%m-%d'
    try:
        parsed = datetime.strptime(date+' '+clock, fmt+(' %H:%M:%S' if len(clock) == 8 else ' %H:%M'))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone(timedelta(hours={'CEST': 2, 'CET': 1, 'UTC': 0}[zone])))


def resolve_mdpi_original_vor(candidate, *, article_url, article_html, notes_url, notes_html, as_of):
    """Return (Candidate, audit). Unresolved returns the original unchanged object.

    Only an explicit table record labelled 'Version of Record' and containing an
    official /pdf-vor link is supported. Updated VoR /pdf is never substituted.
    Caller owns HTTP status/redirect checks and supplies successful HTML bodies.
    """
    audit = {'policy': PUBLISHER_VERSION_POLICY, 'paper_id': candidate.paper_id,
             'status': 'unresolved', 'reason': None, 'source_urls': [article_url, notes_url]}
    def unresolved(reason):
        audit['reason'] = reason
        return candidate, audit
    if not _official(article_url) or not _official(notes_url) or notes_url != article_url+'/notes':
        return unresolved('official_article_and_notes_urls_required')
    if not re.fullmatch(r'/\d{4}-\d{4}/\d+/\d+/\d+', urlsplit(article_url).path):
        return unresolved('unsupported_article_path')
    if as_of.tzinfo is None:
        return unresolved('aware_as_of_required')
    if not candidate.doi or candidate.paper_id != 'doi:'+normalize_doi(candidate.doi) or candidate.arxiv_id:
        return unresolved('doi_candidate_identity_required')
    article, notes = Page(article_html), Page(notes_html)
    dois = {normalize_doi(v) for v in article.meta.get('citation_doi', set())}
    titles = {_title(v) for v in article.meta.get('citation_title', set())}
    if dois != {normalize_doi(candidate.doi)} or titles != {_title(candidate.title)}:
        return unresolved('official_doi_or_title_mismatch')
    if (notes.meta.get('citation_doi') and {normalize_doi(v) for v in notes.meta['citation_doi']} != dois
            or notes.meta.get('citation_title') and {_title(v) for v in notes.meta['citation_title']} != titles):
        return unresolved('notes_identity_conflict')
    records = [row for row in notes.rows if any(cell.casefold() == 'version of record' for cell in row['cells'])]
    if len(records) != 1:
        return unresolved('original_vor_record_missing_or_ambiguous')
    record = records[0]
    urls = {urljoin(notes_url, link) for link in record['links']}
    expected = article_url+'/pdf-vor'
    if expected not in urls or not all(_official(url) and urlsplit(url).path.startswith(urlsplit(article_url).path+'/') for url in urls):
        return unresolved('original_vor_pdf_link_unverified')
    stamp = _stamp(record['cells'])
    if stamp is None or stamp > as_of:
        return unresolved('vor_time_missing_invalid_or_after_as_of')
    utc = stamp.astimezone(UTC).isoformat()
    version = 'mdpi:original-vor:'+utc
    source = {'source': 'publisher:mdpi-version-notes', 'paper_id': candidate.paper_id,
        'paper_version': version, 'version': version, 'url': notes_url,
        'fulltext_links': [expected], 'selected_document': True, 'updated_at': utc,
        'published_at': candidate.published_at, 'date_basis': candidate.date_basis,
        'publisher_version_label': 'Version of Record', 'document_time': utc,
        'publisher_version_policy': PUBLISHER_VERSION_POLICY,
        'evidence': {'article_url': article_url, 'doi_meta': 'citation_doi', 'title_meta': 'citation_title',
                     'doi': candidate.doi, 'title': candidate.title, 'row_cells': record['cells'],
                     'pdf_url': expected, 'time_offset': stamp.strftime('%z')}}
    sources = [{**item, 'selected_document': False} for item in candidate.sources]+[source]
    result = candidate.model_copy(update={'paper_version': version, 'fulltext_links': [expected],
        'versions': sorted(set(candidate.versions+[version])), 'sources': sources})
    audit.update(status='resolved', reason='original_vor_official_record', paper_version=version, evidence=source['evidence'])
    return result, audit


def verified_mdpi_pdf_url(paper):
    """Extract only internally consistent original-VoR evidence; no network trust proof.

    Caller must obtain sources through the verified official acquisition boundary.
    A self-authored sources dictionary cannot authenticate itself cryptographically.
    """
    selected = [s for s in paper.sources if s.get('selected_document')]
    if len(selected) != 1:
        return None
    source = selected[0]
    evidence = source.get('evidence', {})
    article_url = evidence.get('article_url', '')
    if (source.get('source') != 'publisher:mdpi-version-notes'
            or source.get('publisher_version_policy') != PUBLISHER_VERSION_POLICY
            or not _official(article_url)
            or not re.fullmatch(r'/\d{4}-\d{4}/\d+/\d+/\d+', urlsplit(article_url).path)
            or source.get('url') != article_url+'/notes'
            or not paper.doi or paper.paper_id != 'doi:'+normalize_doi(paper.doi)
            or source.get('paper_id') != paper.paper_id
            or normalize_doi(evidence.get('doi', '')) != normalize_doi(paper.doi)
            or _title(evidence.get('title', '')) != _title(paper.title)):
        return None
    cells = evidence.get('row_cells')
    if not isinstance(cells, list) or not all(isinstance(cell, str) for cell in cells):
        return None
    stamp = _stamp(cells)
    if stamp is None or not any(cell.casefold() == 'version of record' for cell in cells):
        return None
    utc = stamp.astimezone(UTC).isoformat()
    expected_version = 'mdpi:original-vor:'+utc
    expected_pdf = article_url+'/pdf-vor'
    if (paper.paper_version != expected_version or source.get('paper_version') != expected_version
            or source.get('version') != expected_version or source.get('document_time') != utc
            or source.get('publisher_version_label') != 'Version of Record'
            or expected_version not in paper.versions or paper.fulltext_links != [expected_pdf]
            or source.get('fulltext_links') != [expected_pdf] or evidence.get('pdf_url') != expected_pdf):
        return None
    return expected_pdf


class MdpiOriginalVorResolver:
    """Explicit transport injection; callable(paper, request) -> Candidate.

    article_urls maps exact DOI to a preconfigured official article URL. No DOI
    redirect discovery, arbitrary downloads, retries, models or default network.
    fetch(url) returns an HTTP-like response; it must not follow redirects.
    """
    def __init__(self, fetch, *, article_urls):
        self.fetch, self.article_urls = fetch, dict(article_urls)
        self.policy = {'resolver': PUBLISHER_VERSION_POLICY, 'article_urls': self.article_urls,
                       'max_requests_per_resolution': 2, 'version_choice': 'original_vor_only'}
        self.audit = []

    def __call__(self, paper, request):
        record = {'policy': PUBLISHER_VERSION_POLICY, 'request_id': request.request_id,
                  'user_id': request.user_id, 'paper_id': paper.paper_id, 'status': 'unresolved',
                  'http_runs': []}
        self.audit.append(record)
        if paper.paper_version is not None:
            record['reason'] = 'existing_version_preserved'
            return paper
        url = self.article_urls.get(paper.doi)
        if not url or not _official(url) or not re.fullmatch(r'/\d{4}-\d{4}/\d+/\d+/\d+', urlsplit(url).path):
            record['reason'] = 'official_article_not_configured'
            return paper
        pages = []
        for target in (url, url+'/notes'):
            try:
                response = self.fetch(target)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                record['http_runs'].append({'url': target, 'status_code': status})
                record.update(reason='official_access_denied' if status in (403, 429) else 'official_source_failed',
                              error_type=type(exc).__name__)
                return paper
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                record.update(reason='official_source_failed', error_type=type(exc).__name__)
                return paper
            record['http_runs'].append({'url': target, 'status_code': response.status_code})
            if response.status_code != 200:
                record['reason'] = 'official_access_denied' if response.status_code in (403, 429) else 'official_source_failed'
                return paper
            if str(response.url) != target or 'text/html' not in response.headers.get('content-type', '').lower():
                record['reason'] = 'official_response_url_or_type_unverified'
                return paper
            pages.append(response.text)
        result, resolution = resolve_mdpi_original_vor(paper, article_url=url, article_html=pages[0],
            notes_url=url+'/notes', notes_html=pages[1], as_of=request.as_of)
        record.update(resolution)
        return result
