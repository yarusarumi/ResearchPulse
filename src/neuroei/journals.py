"""Bounded official RSS metadata inputs; unavailable feeds remain visible in reports."""
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html import unescape

import httpx

from .models import PaperRecord, SourceResult
from .sources import Sources, normalize_doi, paper_title_text

# Reused official feed locations from the pinned PaperFlow journal source registry.
FEEDS = {
    'Nature': 'https://www.nature.com/nature.rss',
    'Nature Biotechnology': 'https://www.nature.com/nbt.rss',
    'Nature Methods': 'https://www.nature.com/nmeth.rss',
    'Nature Machine Intelligence': 'https://www.nature.com/natmachintell.rss',
    'Nature Computational Science': 'https://www.nature.com/natcomputsci.rss',
    'Nature Communications': 'https://www.nature.com/ncomms.rss',
    'Science': 'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science',
    'Science Advances': 'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=sciadv',
    'Science Robotics': 'https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=scirobotics',
    'Cell': 'https://www.cell.com/cell/rss',
}

# Crossref contains publisher-deposited metadata, not publisher full text or a live RSS mirror.
FALLBACK_ISSNS = {'Cell': '0092-8674', 'Science Robotics': '2470-9476'}


def source_error(exc):
    # Retain actionable status without copying response bodies, URLs, or headers.
    if isinstance(exc, httpx.HTTPStatusError):
        return f'HTTP {exc.response.status_code}'
    return type(exc).__name__


def crossref_metadata(adapter, name, since, until, limit=30):
    response = adapter.get(f'https://api.crossref.org/journals/{FALLBACK_ISSNS[name]}/works',
        params={'filter': f'from-pub-date:{since},until-pub-date:{until},type:journal-article',
                'rows': limit, 'sort': 'published', 'order': 'desc'})
    if len(response.content) > 5_000_000:
        raise ValueError('Metadata size limit exceeded')
    message = response.json()['message']
    items = message['items']
    if not isinstance(items, list) or int(message['total-results']) < len(items):
        raise ValueError('Invalid Crossref result counts')
    papers, omitted = [], False
    for item in items[:limit]:
        doi = normalize_doi(item.get('DOI'))
        title = clean(paper_title_text(' '.join(item.get('title', []))))
        journals = [clean(x).casefold() for x in item.get('container-title', [])]
        if (not doi or not re.fullmatch(r'10\.\d{4,9}/\S+', doi) or not title
                or (FALLBACK_ISSNS[name] not in item.get('ISSN', []) and name.casefold() not in journals)):
            omitted = True
            continue
        # Online and print dates may differ; never invent a day from only a year/month.
        days = []
        for key in ('published', 'published-online', 'published-print'):
            for parts in item.get(key, {}).get('date-parts', []):
                try:
                    day = date(*parts) if len(parts) == 3 else None
                except (TypeError, ValueError):
                    day = None
                if day and since <= day.isoformat() <= until:
                    days.append(day.isoformat())
        if not days:
            omitted = True
            continue
        abstract = clean(item.get('abstract', ''))
        published = min(days)
        authors = [clean(a.get('name') or ' '.join(filter(None, [a.get('given'), a.get('family')])) )
                   for a in item.get('author', [])]
        revision = hashlib.blake2s(repr((doi, title, abstract, published, authors)).encode(),
                                   digest_size=8).hexdigest()
        papers.append(PaperRecord(paper_id='doi:'+doi, version='metadata-'+revision,
            title=title, abstract=abstract, authors=[a for a in authors if a], doi=doi,
            published=published, sources=['crossref:'+name], url='https://doi.org/'+doi))
    return papers, omitted or int(message['total-results']) > len(items) or len(items) > limit


def clean(value):
    return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]*>', ' ', value or ''))).strip()


def parse_feed(content, name, since, until, limit=30):
    papers, limited, undated = _feed_items(content, name, since, until, limit)
    return papers, limited or bool(undated)


def _feed_items(content, name, since, until, limit=30):
    root = ET.fromstring(content)
    result, undated = [], 0
    since_day, until_day = date.fromisoformat(since), date.fromisoformat(until)
    for item in (node for node in root.iter() if node.tag.split('}')[-1] in ('item', 'entry')):
        fields = {}
        for child in item:
            fields.setdefault(child.tag.split('}')[-1], []).append(child)

        def text(*keys, fields=fields):
            for key in keys:
                for node in fields.get(key, []):
                    value = ''.join(node.itertext()).strip()
                    if value:
                        return value
            return ''

        title = clean(text('title'))
        links = fields.get('link', [])
        url = next((x.get('href') or x.text or '' for x in links if x.get('rel', 'alternate')=='alternate'), '')
        url = url.strip().replace('http://', 'https://', 1)
        if not title or not url.startswith('https://'):
            continue
        stamp = text('date', 'pubDate', 'published', 'updated')
        try:
            day = datetime.fromisoformat(stamp).date()
        except ValueError:
            try:
                day = parsedate_to_datetime(stamp).date()
            except (ValueError, TypeError):
                day = None
        if day is None:
            # Undated items cannot be confirmed inside the window; exclude and report incomplete coverage.
            undated += 1
            continue
        if not since_day <= day <= until_day:
            continue
        doi = normalize_doi(text('doi', 'identifier'))
        if doi and not doi.startswith('10.'):
            doi = None
        abstract = clean(text('description', 'summary', 'abstract', 'encoded'))
        identity = 'doi:'+doi if doi else 'url:'+hashlib.blake2s(url.encode(), digest_size=12).hexdigest()
        revision = hashlib.blake2s((title+abstract+stamp).encode(), digest_size=8).hexdigest()
        result.append(PaperRecord(paper_id=identity, version='metadata-'+revision, title=title,
            abstract=abstract, authors=[clean(text('creator', 'author'))] if text('creator', 'author') else [],
            doi=doi, published=day.isoformat(), sources=['journal:'+name], url=url))
    return result[:limit], len(result) > limit, undated


def collect_journal(name, since, until, transport=None):
    if date.fromisoformat(since) > date.fromisoformat(until):
        raise ValueError('Valid date range required')
    result = SourceResult(source='journal:'+name, query='official RSS metadata', since=since, until=until, status='failed')
    adapter = Sources(30, transport=transport)
    try:
        response = adapter.get(FEEDS[name])
        if len(response.content)>5_000_000:
            raise ValueError('Feed size limit exceeded')
        result.papers, limited, undated = _feed_items(response.content, name, since, until)
        result.truncated = limited or bool(undated)
        if undated:
            # Coverage note only: the feed itself was fetched, so no error/status body is implied.
            result.error = (f'RSS date coverage: {undated} item(s) without a parseable publication date '
                            'excluded as unconfirmed; window coverage incomplete')
        result.status = 'partial' if result.truncated else 'ok' if result.papers else 'empty'
    except (httpx.HTTPError, ET.ParseError, ValueError, KeyError, RuntimeError, TypeError, AttributeError) as exc:
        result.error = 'RSS: '+source_error(exc)
        if name in FALLBACK_ISSNS:
            result.query = 'official RSS; fallback Crossref publisher-deposited metadata (index may lag)'
            try:
                result.papers, result.truncated = crossref_metadata(adapter, name, since, until)
                # Even zero registry results do not establish complete journal coverage.
                result.status = 'partial'
                result.error += '; Crossref metadata only; RSS coverage unavailable; index may lag'
            except (httpx.HTTPError, ValueError, KeyError, RuntimeError, TypeError, AttributeError) as fallback_exc:
                result.error += '; Crossref: '+source_error(fallback_exc)
    finally:
        adapter.close()
    return result
