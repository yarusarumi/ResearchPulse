"""Scope-checked local fulltexts; paid reading requires an explicit per-job call cap."""
import json
import re
from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import httpx

from . import ai_backend
from .config import LLMConfig
from .models import now
from .paper_analysis import analyze

ACQUIRED_PDF = re.compile(r'(?:publisher|pmc)-pdf-sha256:[0-9a-f]{64}')
PMC_PDF_MAX_BYTES = 128 * 1024 * 1024
MAX_READING_CALLS = 100
READING_OUTPUT_TOKENS = 65_536


class ConfiguredReader:
    def __init__(self, store, job, manifest, *, artifacts_path=None, embedding_model_path=None,
                 max_model_calls=0, fetch_fulltexts=False, fulltext_transport=None, open_access_fallback=False):
        if not 0 <= max_model_calls <= MAX_READING_CALLS:
            raise ValueError(f'Per-job reading call cap must be 0..{MAX_READING_CALLS}')
        self.store, self.job, self.manifest = store, job, manifest
        self.artifacts_path, self.embedding_model_path = artifacts_path, embedding_model_path
        self.fetch_fulltexts, self.fulltext_transport = fetch_fulltexts, fulltext_transport
        self.open_access_fallback = open_access_fallback
        self.client = None
        if max_model_calls:
            config = LLMConfig(max_calls_per_run=max_model_calls, request_attempts=1,
                               schema_repair_attempts=0, max_tokens=READING_OUTPUT_TOKENS, timeout_seconds=600)
            self.client = ai_backend.text_client(config, store, 'reading:' + job['job_id'],
                                                 backend=ai_backend.selected())

    def __call__(self, paper, request):
        if request['user_id'] != self.job['user_id']:
            raise PermissionError('Reader owner mismatch')
        entry = self.manifest.get(request['user_id'], {}).get(paper['paper_id'])
        path = None
        if entry:
            if (entry['paper_version'], entry.get('project_id'), entry.get('direction_id')) != (
                    paper['paper_version'], request['project_id'], request['direction_id']):
                raise PermissionError('Fulltext version or scope mismatch')
            path = Path(entry['path']).resolve(strict=True)
        acquisition = None
        if not entry and self.fetch_fulltexts:
            try:
                path = official_fulltext(self.store, paper, request, transport=self.fulltext_transport)
                acquisition = 'available' if path else 'unsupported'
            except (httpx.HTTPError, OSError, ValueError) as exc:
                acquisition = 'failed'
                self.store.audit(request['request_id'], {'event': 'fulltext_acquisition_failed',
                    'paper_id': paper['paper_id'], 'paper_version': paper['paper_version'],
                    'error_type': type(exc).__name__})
        cache_dir = self.store.root / 'analysis-cache'
        if attempt := self.job.get('reading_recovery_attempt'):
            # A claimed retry can replace a failed result without rewriting the original cache.
            cache_dir = cache_dir / 'reading-recovery' / sha256(attempt.encode()).hexdigest()
        result = analyze(paper, request, path=path, cache_dir=cache_dir,
                       client=self.client, artifacts_path=self.artifacts_path,
                       embedding_model_path=self.embedding_model_path)
        if acquisition:
            result['reading_scope']['fulltext_acquisition'] = acquisition
        return result

    def close(self):
        if self.client:
            self.client.close()


    def resolve_version(self, paper, request):
        """Try verified publisher evidence, then an eligible open repository copy."""
        from .pmc_open_pdf import supports_open_pdf

        if request['user_id'] != self.job['user_id']:
            raise PermissionError('Reader owner mismatch')
        if not self.fetch_fulltexts or paper.get('paper_version'):
            return paper
        if not self.open_access_fallback or not supports_open_pdf(paper):
            return self._resolve_publisher_version(paper, request)
        try:
            resolved = self._resolve_publisher_version(paper, request)
        except (httpx.HTTPError, ValueError) as exc:
            self.store.audit(request['request_id'], {'event': 'publisher_before_repository_failed',
                'paper_id': paper['paper_id'], 'error_type': type(exc).__name__,
                'http_status': exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None})
            resolved = resolve_pmc_version(self.store, paper, request, transport=self.fulltext_transport)
            if (resolved['paper_version'] is None and isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 403):
                resolved = resolve_mdpi_sensors_pdf(self.store, paper, request, transport=self.fulltext_transport)
            if resolved['paper_version'] is None:
                raise
            return resolved
        if resolved['paper_version'] is None:
            return resolve_pmc_version(self.store, paper, request, transport=self.fulltext_transport)
        return resolved

    def _resolve_publisher_version(self, paper, request):
        """One DOI redirect plus official publisher evidence; never follow arbitrary redirects."""
        from .discovery_contract import Candidate, RecommendationRequest
        from .discovery_publisher_version import MdpiOriginalVorResolver

        if any(str(source.get('source', '')).startswith(('official-rss:', 'pmlr:', 'neurips:', 'journal-detail:'))
               for source in paper.get('sources', [])):
            return resolve_proceedings_version(self.store, paper, request, transport=self.fulltext_transport)
        if crossref_aircc_pdf_url(paper):
            return resolve_crossref_aircc_pdf(self.store, paper, request, transport=self.fulltext_transport)
        if not re.fullmatch(r'doi:10\.3390/[a-z0-9.-]+', paper['paper_id']):
            return paper
        doi = paper.get('doi')
        if paper['paper_id'] != 'doi:'+str(doi):
            raise ValueError('DOI identity mismatch')
        article_pattern = r'https://www\.mdpi\.com/\d{4}-\d{4}/\d+/\d+/\d+'
        with httpx.Client(timeout=20, follow_redirects=False, transport=self.fulltext_transport) as client:
            urls = [url for url in paper['original_links'] if re.fullmatch(article_pattern, url)]
            if not urls:
                redirect = client.get('https://doi.org/'+quote(doi, safe='/'))
                self.store.audit(request['request_id'], {'event': 'publisher_doi_redirect',
                    'paper_id': paper['paper_id'], 'status_code': redirect.status_code})
                if redirect.status_code not in (301, 302, 303, 307, 308):
                    redirect.raise_for_status()
                    return paper
                urls = [redirect.headers.get('location', '')]
            if len(set(urls)) != 1 or not re.fullmatch(article_pattern, urls[0]):
                return paper
            responses = []
            def fetch(url):
                response = client.get(url)
                responses.append(response)
                if len(response.content) > 2_000_000:
                    raise ValueError('Publisher metadata exceeds size budget')
                return response
            resolver = MdpiOriginalVorResolver(fetch, article_urls={doi: urls[0]})
            result = resolver(Candidate.model_validate(paper), RecommendationRequest.model_validate(request))
            self.store.audit(request['request_id'], {'event': 'publisher_version_resolution', **resolver.audit[-1]})
            for response in responses:
                response.raise_for_status()
            return result.model_dump(mode='json')


def resolve_pmc_version(store, paper, request, *, transport=None):
    from .pmc_open_pdf import lookup_pdf

    with httpx.Client(timeout=20, follow_redirects=False, transport=transport) as client:
        match = lookup_pdf(client, paper)
    if match is None:
        return paper
    url, checksum, metadata_url, metadata = match
    data = download_official_pdf(url, transport=transport, max_bytes=PMC_PDF_MAX_BYTES)
    if md5(data, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError('PMC PDF bytes differ from repository metadata')
    version = 'pmc-pdf-sha256:'+sha256(data).hexdigest()
    receipt = {'source': 'pmc-open-access-pdf', 'paper_id': paper['paper_id'], 'paper_version': version,
               'url': url, 'metadata_url': metadata_url, 'repository_metadata': metadata, 'fetched_at': now()}
    updated = {**paper, 'paper_version': version, 'versions': sorted({*paper['versions'], version}),
               'fulltext_links': [url], 'sources': [*paper['sources'], receipt]}
    save_official_pdf(store, updated, request, url, data)
    return updated


def acquired_pmc_pdf_url(paper):
    from .pmc_open_pdf import verify_pdf_metadata

    receipts = [s for s in paper.get('sources', []) if s.get('source') == 'pmc-open-access-pdf']
    if len(receipts) != 1:
        return None
    receipt, = receipts
    url, _ = verify_pdf_metadata(paper, receipt['repository_metadata'])
    if (receipt['url'] != url or receipt['paper_id'] != paper['paper_id']
            or receipt['paper_version'] != paper['paper_version'] or paper['paper_version'] not in paper['versions']):
        return None
    return url


def official_document_helpers(paper):
    """Use each module's identity verifier at the single acquisition boundary."""
    if any(s.get('selected_document') and str(s.get('source', '')).startswith('journal-detail:')
           for s in paper.get('sources', [])):
        from .discovery_journals import journal_document_locator, verify_journal_document
        return journal_document_locator, verify_journal_document
    from .discovery_proceedings import official_document_locator, verify_official_document
    return official_document_locator, verify_official_document


def resolve_proceedings_version(store, paper, request, *, transport=None):
    """Recheck official landing identity, then identify the actual acquired PDF bytes."""
    from .discovery_contract import Candidate
    official_document_locator, verify_official_document = official_document_helpers(paper)
    candidate = Candidate.model_validate(paper)
    locator = official_document_locator(candidate)
    if not locator:
        return paper
    url = locator['landing_url']
    try:
        with (httpx.Client(timeout=25, follow_redirects=False, transport=transport) as client,
              client.stream('GET', url) as response):
            response.raise_for_status()
            if response.status_code != 200 or 'text/html' not in response.headers.get('content-type', '').lower():
                raise ValueError('Official landing response required')
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 1_000_000:
                    raise ValueError('Official landing exceeds size budget')
        proof = verify_official_document(candidate, body.decode('utf8'), url)
        data, transfer = download_proceedings_pdf(proof['pdf_url'], transport=transport)
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status != 429 and not 500 <= status < 600:
                raise
        cached = reuse_publisher_version(store, paper, request, url)
        if cached is None:
            raise
        return cached
    return save_publisher_version(store, paper, request, proof, data, transfer)



def reuse_publisher_version(store, paper, request, landing_url):
    """Find a verified PDF from this scope after a transient publisher failure."""
    scope = ('user_id', 'project_id', 'direction_id')
    rows = store.select('recommendation_event', (*scope, 'created_at'), include_id=True)
    matching = (row for row in rows if all(row[key] == request[key] for key in scope))
    for row in sorted(matching, key=lambda row: row['created_at'] or '', reverse=True):
        event = store.get('recommendation_event', row['id'])
        for prior in (event.get('discovery') or {}).get('candidates', []):
            if prior['paper_id'] != paper['paper_id'] or prior['doi'] != paper['doi']:
                continue
            if ' '.join(prior['title'].casefold().split()) != ' '.join(paper['title'].casefold().split()):
                continue
            version = prior['paper_version']
            if not re.fullmatch(r'publisher-pdf-sha256:[0-9a-f]{64}', version or ''):
                continue
            receipts = [source for source in prior['sources']
                        if source['source'] == 'official-pdf-acquisition' and source.get('url') == landing_url]
            if len(receipts) != 1 or not receipts[0]['fulltext_links']:
                continue
            links = receipts[0]['fulltext_links']
            if not set(links) <= set(paper['fulltext_links']):
                continue
            updated = {**paper, 'paper_version': version, 'versions': sorted({*paper['versions'], version}),
                       'sources': [*paper['sources'], receipts[0]], 'fulltext_links': links}
            try:
                path = official_fulltext(store, updated, request, cached_only=True)
            except (OSError, ValueError):
                continue
            if path is not None:
                store.audit(request['request_id'], {'event': 'reused_verified_publisher_pdf',
                    'paper_id': paper['paper_id'], 'paper_version': version, 'source_event_id': row['id']})
                return updated
    return None


def save_publisher_version(store, paper, request, proof, data, transfer):
    url = proof['landing_url']
    digest = sha256(data).hexdigest()
    version = 'publisher-pdf-sha256:'+digest
    receipt = {'source': 'official-pdf-acquisition', 'paper_id': paper['paper_id'], 'url': url,
               'paper_version': version, 'version': version, 'selected_document': False,
               'fulltext_links': [proof['pdf_url']], 'document_sha256': digest,
               'metadata_verification': proof, 'fetched_at': now(),
               'transfer': transfer,
               'scope': 'Official metadata identity and acquired bytes; scientific PDF contents not independently reviewed'}
    updated = {**paper, 'paper_version': version, 'versions': sorted({*paper['versions'], version}),
               'fulltext_links': [proof['pdf_url']], 'sources': [*paper['sources'], receipt]}
    save_official_pdf(store, updated, request, proof['pdf_url'], data)
    return updated


def mdpi_sensors_urls(paper):
    doi = paper.get('doi') or ''
    match = re.fullmatch(r'10\.3390/s([1-9]\d)(\d{2})(\d{4})', doi)
    if not match or paper['paper_id'] != 'doi:'+doi or paper.get('venue') != 'Sensors':
        return None
    volume, issue, article = map(int, match.groups())
    landing = f'https://www.mdpi.com/1424-8220/{volume}/{issue}/{article}'
    asset = f'sensors-{volume}-{article:05d}'
    return landing, f'https://pub.mdpi-res.com/sensors/{asset}/article_deploy/{asset}.pdf'


def crossref_aircc_pdf_url(paper):
    doi = paper.get('doi') or ''
    match = re.fullmatch(r'10\.5121/csit\.20[0-9]{2}\.(([0-9]{2})[0-9]{4})', doi)
    if match and paper['paper_id'] == 'doi:'+doi:
        name, volume = match.groups()
        return f'https://aircconline.com/csit/papers/vol{int(volume)}/csit{name}.pdf'
    return None


def resolve_crossref_aircc_pdf(store, paper, request, *, transport=None):
    doi = paper['doi']
    expected = crossref_aircc_pdf_url(paper)
    with httpx.Client(timeout=12, follow_redirects=False, transport=transport) as client:
        response = client.get('https://api.crossref.org/works/'+quote(doi, safe='/'))
        response.raise_for_status()
        metadata = response.json().get('message', {})
    if (str(metadata.get('DOI', '')).casefold() != doi or
            ((metadata.get('resource') or {}).get('primary') or {}).get('URL') != expected):
        raise ValueError('Crossref does not declare the expected AIRCC PDF')
    data = download_official_pdf(expected, transport=transport)
    verify_pdf_front_page(data, paper)
    proof = {'kind': 'crossref-aircc-pdf-front-page', 'landing_url': 'https://doi.org/'+doi,
             'pdf_url': expected, 'doi': doi, 'title': paper['title']}
    return save_publisher_version(store, paper, request, proof, data, {'url': expected})


def verify_pdf_front_page(data, paper):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        front = PdfReader(BytesIO(data)).pages[0].extract_text()
    except (PdfReadError, IndexError) as exc:
        raise ValueError('Publisher PDF front page unreadable') from exc

    def compact(text):
        return re.sub(r'[^\w]+', '', text.casefold())

    title = compact(paper['title'])
    if not title or title not in compact(front) or not re.search(
            r'(?<![\w.])'+re.escape(paper['doi'])+r'(?=$|[\s,;)]|\.(?:\s|$))', front, re.I):
        raise ValueError('Publisher PDF front-page identity mismatch')


def resolve_mdpi_sensors_pdf(store, paper, request, *, transport=None):
    urls = mdpi_sensors_urls(paper)
    if not urls:
        return paper
    landing, pdf = urls
    data = download_official_pdf(pdf, transport=transport)
    verify_pdf_front_page(data, paper)
    doi = paper['doi']
    proof = {'kind': 'mdpi-sensors-pdf-front-page', 'landing_url': landing, 'pdf_url': pdf,
             'doi': doi, 'title': paper['title']}
    return save_publisher_version(store, paper, request, proof, data, {'url': pdf})


def acquired_proceedings_pdf_url(paper):
    from .discovery_contract import Candidate
    official_document_locator, _ = official_document_helpers(paper)
    receipts = [s for s in paper.get('sources', []) if s.get('source') == 'official-pdf-acquisition']
    if len(receipts) != 1:
        return None
    receipt = receipts[0]
    proof = receipt.get('metadata_verification', {})
    url = proof.get('pdf_url')
    if proof.get('kind') == 'mdpi-sensors-pdf-front-page':
        urls = mdpi_sensors_urls(paper)
        locator = {'landing_url': urls[0], 'pdf_candidates': [urls[1]]} if urls else None
    elif proof.get('kind') == 'crossref-aircc-pdf-front-page':
        expected = crossref_aircc_pdf_url(paper)
        locator = {'landing_url': 'https://doi.org/'+paper['doi'], 'pdf_candidates': [expected]} if expected else None
    else:
        locator = official_document_locator(Candidate.model_validate(paper))
    if proof.get('kind') in ('mdpi-sensors-pdf-front-page', 'crossref-aircc-pdf-front-page') and (
            proof.get('doi') != paper['doi'] or proof.get('title') != paper['title']):
        return None
    if (not locator or receipt.get('paper_id') != paper['paper_id'] or receipt.get('url') != locator['landing_url']
            or proof.get('landing_url') != locator['landing_url'] or receipt.get('paper_version') != paper['paper_version']
            or paper['paper_version'] != 'publisher-pdf-sha256:'+str(receipt.get('document_sha256'))
            or paper['paper_version'] not in paper['versions'] or receipt.get('fulltext_links') != [url]
            or url not in [u.replace('http://', 'https://', 1) for u in locator['pdf_candidates']]):
        return None
    return url


def download_proceedings_pdf(url, *, transport=None):
    """A declared PMLR repository file may use GitHub's equivalent public raw API."""
    try:
        return download_official_pdf(url, transport=transport), {'url': url}
    except httpx.HTTPError as exc:
        match = re.fullmatch(r'https://raw\.githubusercontent\.com/mlresearch/(v\d+)/main/assets/([A-Za-z0-9_-]+)/\2\.pdf', url)
        if not match:
            raise
        volume, slug = match.groups()
        api_url = f'https://api.github.com/repos/mlresearch/{volume}/contents/assets/{slug}/{slug}.pdf?ref=main'
        transfer = {'url': api_url, 'canonical_pdf_url': url, 'primary_error_type': type(exc).__name__,
                    'identity_basis': 'same publisher-declared repository, ref and file path; acquired bytes identified separately'}
        data = download_official_pdf(api_url, transport=transport, accept='application/vnd.github.raw+json')
        return data, transfer


def official_fulltext(store, paper, request, *, transport=None, cached_only=False):
    """One bounded official versioned PDF request, scoped to the authorized request."""
    pid, version = paper['paper_id'], paper['paper_version']
    if re.fullmatch(r'arxiv:\d{4}\.\d{4,5}', pid) and re.fullmatch(r'v[1-9]\d*', version or ''):
        url = 'https://arxiv.org/pdf/'+pid[6:]+version
    elif pid.startswith('doi:10.3390/') and (version or '').startswith('mdpi:original-vor:'):
        from .discovery_contract import Candidate
        from .discovery_publisher_version import verified_mdpi_pdf_url
        url = verified_mdpi_pdf_url(Candidate.model_validate(paper))
    elif re.fullmatch(r'publisher-pdf-sha256:[0-9a-f]{64}', version or ''):
        url = acquired_proceedings_pdf_url(paper)
    elif re.fullmatch(r'pmc-pdf-sha256:[0-9a-f]{64}', version or ''):
        url = acquired_pmc_pdf_url(paper)
    else:
        return None
    if not url or url not in paper.get('fulltext_links', []):
        return None
    scope = {key: request[key] for key in ('user_id', 'project_id', 'direction_id')}
    identity = {**scope, 'paper_id': pid, 'paper_version': version, 'url': url}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = store.root/'authorized-fulltexts'/(key+'.pdf')
    try:
        saved = store.get('fulltext_document', key)
    except ValueError:
        saved = None
    if saved:
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError('Fulltext identity mismatch')
        with path.open('rb') as stream:
            if path.stat().st_size != saved['bytes'] or stream.read(5) != b'%PDF-':
                raise ValueError('Cached fulltext incomplete')
        if ACQUIRED_PDF.fullmatch(version) and sha256(path.read_bytes()).hexdigest() != version.split(':', 1)[1]:
            raise ValueError('Cached PDF content version mismatch')
        return path
    if cached_only:
        return None
    if version.startswith('publisher-pdf-sha256:'):
        data, _ = download_proceedings_pdf(url, transport=transport)
    elif version.startswith('pmc-pdf-sha256:'):
        data = download_official_pdf(url, transport=transport, max_bytes=PMC_PDF_MAX_BYTES)
    else:
        data = download_official_pdf(url, transport=transport)
    if ACQUIRED_PDF.fullmatch(version) and sha256(data).hexdigest() != version.split(':', 1)[1]:
        raise ValueError('Acquired PDF content version mismatch')
    return save_official_pdf(store, paper, request, url, data)


def download_official_pdf(url, *, transport=None, accept='application/pdf', max_bytes=20_000_000):
    """One bounded PDF response; redirects and incomplete bodies remain failures."""
    with (httpx.Client(timeout=30, follow_redirects=False, transport=transport) as client,
          client.stream('GET', url, headers={'Accept-Encoding': 'identity', 'Accept': accept}) as response):
        response.raise_for_status()  # No redirect can silently substitute an unversioned document.
        if response.status_code != 200:
            raise ValueError('Full PDF response required')
        expected = int(response.headers.get('content-length', '0'))
        if expected > max_bytes:
            raise ValueError('Fulltext exceeds size budget')
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError('Fulltext exceeds size budget')
    if not data.startswith(b'%PDF-') or (expected and len(data) != expected):
        raise ValueError('Incomplete or non-PDF fulltext')
    return bytes(data)


def save_official_pdf(store, paper, request, url, data):
    identity = {**{k: request[k] for k in ('user_id', 'project_id', 'direction_id')},
                'paper_id': paper['paper_id'], 'paper_version': paper['paper_version'], 'url': url}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = store.root/'authorized-fulltexts'/(key+'.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing fulltext differs from acquired version')
        saved = store.get('fulltext_document', key)
        if any(saved[k] != v for k, v in identity.items()) or saved['bytes'] != len(data):
            raise ValueError('Existing fulltext identity mismatch')
        return path
    with path.open('xb') as stream:
        stream.write(data)
    store.put('fulltext_document', key, {**identity, 'path': str(path), 'bytes': len(data),
                                       'fetched_at': now()}, immutable=True)
    return path
