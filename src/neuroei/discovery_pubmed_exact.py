"""Resolve already known journal DOIs after a direct source failure."""
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date

import httpx

from .models import PaperRecord
from .sources import normalize_doi, paper_title_key

BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
MAX_DOIS = 6
MAX_PMIDS = 12
MAX_INDEX_PAGE_FETCHES = 2
JOURNAL_LABELS = {'Nature': 'nature', 'Science Robotics': 'sci robot|science robotics', 'Cell': 'cell'}
IDENTITY = {
    'Nature': ('0410462', {'0028-0836', '1476-4687'}, r'10\.1038/s41586-[a-z0-9-]+'),
    'Science Robotics': ('101733136', {'2470-9476'}, r'10\.1126/scirobotics\.[a-z0-9]+'),
    'Cell': ('0413066', {'0092-8674', '1097-4172'}, r'10\.1016/j\.cell\.\d{4}\.\d{2}\.\d+'),
}
NON_RESEARCH = {'comment', 'editorial', 'letter', 'news', 'published erratum', 'retraction of publication',
                'retracted publication', 'expression of concern', 'newspaper article'}


def known_dois(values, name):
    pattern = IDENTITY[name][2]
    return list(dict.fromkeys(doi for value in values for doi in re.findall(
        r'(?<![a-z0-9])'+pattern+r'(?![a-z0-9.-])', value.casefold())))


def _text(node):
    return ' '.join(''.join(node.itertext()).split()) if node is not None else ''


def _day(node):
    try:
        return date(*(int(node.findtext(k, '')) for k in ('Year', 'Month', 'Day'))).isoformat()
    except ValueError:
        return None


def parse_exact_xml(xml, name, requested, expected_pmids):
    nlm_id, issns, _ = IDENTITY[name]
    root = ET.fromstring(xml)
    if root.tag != 'PubmedArticleSet' or root.find('ERROR') is not None:
        raise ValueError('PubMed XML rejected')
    accepted, rejected, duplicates = {}, [], set()
    for row in root.findall('PubmedArticle'):
        citation = row.find('MedlineCitation')
        if citation is None or (pmid := citation.findtext('PMID')) not in expected_pmids:
            continue
        dois = {normalize_doi(n.text) for n in row.findall('PubmedData/ArticleIdList/ArticleId') if n.get('IdType') == 'doi'}
        if len(dois) != 1 or (doi := next(iter(dois))) not in requested:
            rejected.append({'pmid': pmid, 'reason': 'doi_mismatch'})
            continue
        article = citation.find('Article')
        record_issns = set(filter(None, (citation.findtext('MedlineJournalInfo/ISSNLinking'),
                                         citation.findtext('Article/Journal/ISSN'))))
        if (article is None or citation.findtext('MedlineJournalInfo/NlmUniqueID') != nlm_id
                or not record_issns or not record_issns <= issns):
            rejected.append({'doi': doi, 'pmid': pmid, 'reason': 'journal_identity_mismatch'})
            continue
        types = {_text(n).casefold() for n in article.findall('PublicationTypeList/PublicationType')}
        title = _text(article.find('ArticleTitle'))
        if ('journal article' not in types or types.intersection(NON_RESEARCH) or not title
                or re.match(r'(?:author correction|correction|retraction|editorial|news|commentary)\b', title, re.IGNORECASE)):
            rejected.append({'doi': doi, 'pmid': pmid, 'reason': 'research_type_unverified'})
            continue
        if doi in accepted or doi in duplicates:
            accepted.pop(doi, None)
            duplicates.add(doi)
            rejected.append({'doi': doi, 'pmid': pmid, 'reason': 'multiple_matching_citations'})
            continue
        dates = [d for n in article.findall('ArticleDate') if n.get('DateType') == 'Electronic' and (d := _day(n))]
        abstract = '\n'.join(_text(n) for n in article.findall('Abstract/AbstractText')).strip()
        authors = [a.findtext('CollectiveName') or ' '.join(filter(None, (a.findtext('ForeName'), a.findtext('LastName'))))
                   for a in article.findall('AuthorList/Author')]
        accepted[doi] = PaperRecord(paper_id='doi:'+doi, version='indexed', title=title, abstract=abstract,
            authors=[a for a in authors if a], doi=doi, published=min(dates, default=''),
            sources=['pubmed-exact:'+name], url='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/')
    return list(accepted.values()), rejected


def _request(adapter, path, params, media, limit):
    now = time.monotonic()
    if now < adapter.rate_limited_until.get(BASE, 0):
        raise RuntimeError('PubMedRateLimitedCooldown')
    time.sleep(max(0, adapter.rate_limited_until.get('pubmed-spacing', 0) - now))
    adapter.rate_limited_until['pubmed-spacing'] = time.monotonic() + 0.35
    try:
        return adapter.journal_text(BASE+path, limit, media=media, params={'db': 'pubmed', **params})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            adapter.rate_limited_until[BASE] = time.monotonic()+300
        raise


def recover_index_abstract(paper, web):
    """Supplement a verified EFetch record with its citation-page abstract only."""
    from .discovery_web import WebBudgetExceeded

    name = paper.sources[0].removeprefix('pubmed-exact:')
    pmid = paper.url.rstrip('/').rsplit('/', 1)[-1]
    audit = {'doi': paper.doi, 'pmid': pmid, 'url': paper.url, 'status': 'failed'}
    try:
        page = web.fetch_pubmed_citation(pmid)
        section = re.search(r'^#{1,6}[ \t]+Abstract[ \t]*\r?\n(.*?)(?=^#{1,6}[ \t]|^\[PubMed Disclaimer\]|\Z)',
                            page['content'], re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not section or not section[1].strip():
            raise ValueError('PubMed page abstract unavailable')
        header = page['content'][:section.start()]
        dois = {normalize_doi(doi.rstrip('.')) for doi in re.findall(
            r'\bdoi:\s*(?:\*\*\s*|\[|https://doi\.org/)*(10\.\d{4,9}/[^\s*)\]]+)', header, re.IGNORECASE)}
        pmids = set(re.findall(r'\bPMID:\s*(?:\*\*\s*)?\[?([0-9]+)\b', header, re.IGNORECASE))
        journal = re.search(r'^[ \t]*(?:\[(?:Button:[ \t]*)?)?(?:'+JOURNAL_LABELS[name]+
                            r')\.?[ \t]*(?:\]|\d{4}\b)', header, re.IGNORECASE | re.MULTILINE)
        titles = {paper_title_key(title) for title in re.findall(r'^#[ \t]+(.+)$', header, re.MULTILINE)}
        if dois != {paper.doi} or pmids != {pmid} or not journal or paper_title_key(paper.title) not in titles:
            raise ValueError('PubMed page identity mismatch')
        audit['status'] = 'completed'
        return paper.model_copy(update={'abstract': section[1].strip(),
            'sources': [*paper.sources, 'pubmed-index-page']}), audit
    except (httpx.HTTPError, ValueError, RecursionError, WebBudgetExceeded) as exc:
        audit['error'] = str(exc) if type(exc) is ValueError else type(exc).__name__
        return paper, audit


def recover_exact(adapter, name, dois):
    requested = dois[:MAX_DOIS]
    audit = {'status': 'skipped', 'requested_dois': requested, 'skipped_dois': dois[MAX_DOIS:],
             'coverage_complete': False, 'matched_dois': [], 'rejected': []}
    if not requested:
        audit['reason'] = 'no_exact_doi_leads'
        return [], audit
    try:
        raw = _request(adapter, 'esearch.fcgi', {'term': ' OR '.join(f'"{d}"[aid]' for d in requested),
            'retmode': 'json', 'retmax': MAX_PMIDS}, ('json',), 256_000)
        search = json.loads(raw)['esearchresult']
        if not isinstance(search, dict) or search.get('errorlist'):
            raise ValueError('PubMed ESearch rejected')
        ids = search['idlist']
        count = int(search['count'])
        if (not isinstance(ids, list) or len(ids) > MAX_PMIDS or count < len(ids)
                or any(not isinstance(pmid, str) or not pmid.isdigit() for pmid in ids)):
            raise ValueError('PubMed ESearch ID list invalid')
        audit.update(search_count=count, search_truncated=count > len(ids))
        papers = []
        if ids:
            xml = _request(adapter, 'efetch.fcgi', {'id': ','.join(ids), 'retmode': 'xml'}, ('xml',), 2_000_000)
            papers, audit['rejected'] = parse_exact_xml(xml, name, set(requested), set(ids))
        audit['matched_dois'] = [paper.doi for paper in papers]
        audit['unverified_dois'] = [doi for doi in requested if doi not in audit['matched_dois']]
        audit['status'] = 'partial' if audit['unverified_dois'] or audit['search_truncated'] or audit['skipped_dois'] else 'completed'
        return papers, audit
    except (httpx.HTTPError, KeyError, ValueError, TypeError, RuntimeError, ET.ParseError) as exc:
        audit.update(status='failed', error=type(exc).__name__)
        if isinstance(exc, httpx.HTTPStatusError):
            audit['http_status'] = exc.response.status_code
        return [], audit
