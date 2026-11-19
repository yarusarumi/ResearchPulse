"""Search fixed public WeChat accounts for a direction and verify article provenance."""

import re
from datetime import datetime
from unicodedata import normalize
from urllib.parse import parse_qs, urlencode, urlsplit
from zoneinfo import ZoneInfo

import httpx

from .media import related_identifiers
from .models import now
from .wechat_accounts import WECHAT_ACCOUNTS

MAX_QUERIES = 3
MAX_FETCHES_PER_ACCOUNT = 5
_LONG_FIELDS = {'__biz': r'[A-Za-z0-9+/=]{8,64}', 'mid': r'\d{1,16}', 'idx': r'[1-9]\d{0,2}', 'sn': r'[0-9a-f]{32}'}
_NON_RESEARCH = re.compile(r'^(?:招聘|招生|喜报|祝贺|通知|报名)\s*(?:[|丨｜：:\s]|$)')
_AUTHOR = re.compile(r'^\s*(?:作者|公众号)\s*[:：]\s*(.+?)\s*$', re.MULTILINE)
_PUBLIC_TERMS = {
    'associative learning': '关联学习', 'neural circuit': '神经环路', 'neural circuits': '神经环路',
    'basal ganglia': '基底神经节', 'motor control': '运动控制', 'embodied intelligence': '具身智能',
    'robot manipulation': '机器人操作', 'dexterous manipulation': '灵巧操作',
    'world model': '世界模型', 'world models': '世界模型', 'imitation learning': '模仿学习',
}
_SEARCH_EXPANSIONS = {'associative learning': '神经环路 关联学习'}


def canonical_wechat_article(url):
    """Rebuild only public article identities and discard unrelated URL parameters."""
    try:
        p = urlsplit(url)
        if (p.scheme not in ('http', 'https') or p.hostname != 'mp.weixin.qq.com'
                or p.username or p.password or p.port is not None):
            return None
        base = 'https://mp.weixin.qq.com'
        if re.fullmatch(r'/s/[A-Za-z0-9_-]{8,80}', p.path):
            return base+p.path
        if p.path != '/s':
            return None
        params = parse_qs(p.query, keep_blank_values=True, max_num_fields=24)
        if not all(len(params.get(key, [])) == 1 and re.fullmatch(pattern, params[key][0])
                   for key, pattern in _LONG_FIELDS.items()):
            return None
        return base+'/s?'+urlencode({key: params[key][0] for key in _LONG_FIELDS})
    except ValueError:
        return None


def _plain(value):
    return ' '.join(normalize('NFKC', value).replace('**', '').split()).casefold()


def _relevance(text, queries):
    text = _plain(text)
    return [q for q in queries if re.search(r'(?<![a-z0-9])'+re.escape(_plain(q))+r'(?![a-z0-9])', text)
            or (_plain(q) in _PUBLIC_TERMS and _PUBLIC_TERMS[_plain(q)] in text)]


def _page_metadata(text):
    head = text[:1500]
    heading = re.search(r'^#\s+(.+)$', head, re.MULTILINE)
    author = _AUTHOR.search(head[:heading.start()]) if heading else None
    return heading, author


def _article(page, account, url, queries, request):
    text = page['content']
    title = page['title'].strip()
    head = text[:1500]
    heading, author = _page_metadata(text)
    if (not author or _plain(author[1]) != _plain(account['name']) or not heading or not title
            or _plain(heading[1]) != _plain(title) or _NON_RESEARCH.match(title)):
        return None
    matched = _relevance(title+'\n'+text, queries)
    exclusions = request.get('constraints', {}).get('exclusions', [])
    if not matched or any(_plain(term) in _plain(title+'\n'+text) for term in exclusions if term.strip()):
        return None
    stamp = re.search(r'^\s*发布时间\s*[:：]\s*(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2}))?)\s*$', head, re.MULTILINE)
    published = None
    if stamp:
        try:
            published = datetime.fromisoformat(stamp[1]).isoformat() if len(stamp[1]) > 10 else datetime.fromisoformat(stamp[1]).date().isoformat()
        except ValueError:
            pass
    window = request.get('publication_window') if request['mode'] == 'direction_daily' else None
    in_window = None
    if window:
        value = datetime.fromisoformat(published) if published else None
        if value and value.tzinfo is not None:
            in_window = datetime.fromisoformat(window['since']) <= value < datetime.fromisoformat(window['until'])
    # Multiple body references may be citations to prior work. Only hand off an unambiguous link.
    ids = related_identifiers(text)
    return {'account': account['name'], 'title': title[:300], 'url': url, 'published_at': published,
            'in_daily_window': in_window, 'matched_queries': matched, 'identifiers': ids if len(ids) == 1 else [],
            'scope': 'verified_account_article_link'}


def discover_wechat_accounts(request, web, *, should_stop=None):
    """One job-owned provider client, bounded Search/Fetch, no copied article bodies."""
    from .discovery_web import WebBudgetExceeded

    if request['mode'] not in ('direction_bootstrap', 'direction_daily'):
        return [], []
    queries = list(dict.fromkeys(q.strip() for q in request['queries'] if q.strip()))[:MAX_QUERIES]
    if not queries:
        return [], []
    runs, articles, seen = [], [], {}
    day = datetime.fromisoformat(request['as_of']).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
    for account in WECHAT_ACCOUNTS:
        links, requests, errors, accepted, pending = {}, [], [], [], []
        for query in queries:
            if should_stop and should_stop():
                return runs, articles
            literal = _plain(query)
            term = _SEARCH_EXPANSIONS.get(literal, _PUBLIC_TERMS.get(literal, query))
            effective = f'"{account["name"]}" {term} site:mp.weixin.qq.com/s/'
            requests.append({'operation': 'search', 'query': effective})
            try:
                for lead in web.search(effective, count=20)['results']:
                    title = lead.get('title', '')
                    url = canonical_wechat_article(lead['url'])
                    if url and not _NON_RESEARCH.match(title):
                        links.setdefault(url, set()).update(_relevance(title, queries))
            except (httpx.HTTPError, ValueError, WebBudgetExceeded, RecursionError) as exc:
                errors.append(type(exc).__name__)
        # Public share links can be readable when the older query-string form returns a verification screen.
        ordered = sorted(links, key=lambda url: (-len(links[url]), '/s?' in url))
        # This is a declared source catalogue link, read afresh and checked against the current questions.
        known = canonical_wechat_article(account.get('wechat_example_url', ''))
        if known and known not in links:
            ordered.insert(0, known)
        for url in ordered[:MAX_FETCHES_PER_ACCOUNT]:
            if should_stop and should_stop():
                articles.extend(accepted)
                return runs, articles
            attempt = {'operation': 'fetch' if url not in seen else 'reuse_fetch', 'url': url}
            attempt['origin'] = 'public_search' if url in links else 'curated_public_link'
            requests.append(attempt)
            try:
                if url not in seen:
                    try:
                        seen[url] = web.fetch_wechat_article(url)
                    except (httpx.HTTPError, ValueError, WebBudgetExceeded, RecursionError) as exc:
                        seen[url] = exc
                if isinstance(seen[url], Exception):
                    raise seen[url]
                page = seen[url]
                if page['title'].strip() == '微信公众平台' and '环境异常' in page['content'][:180]:
                    attempt['outcome'] = 'access_challenge'
                    errors.append('wechat_access_challenge')
                    continue
                article = _article(page, account, url, queries, request)
                attempt['outcome'] = 'verified' if article else 'unverified_or_unrelated'
                if article:
                    article['discovery_origin'] = attempt['origin']
                    accepted.append(article)
                elif not _page_metadata(page['content'])[1] and _relevance(page['title'], queries):
                    exclusions = request.get('constraints', {}).get('exclusions', [])
                    if not any(_plain(term) in _plain(page['title']+'\n'+page['content']) for term in exclusions if term.strip()):
                        pending.append({'url': url, 'title': page['title'][:300], 'verification': 'account_unverified',
                                        'matched_queries': _relevance(page['title'], queries)})
            except (httpx.HTTPError, ValueError, WebBudgetExceeded, RecursionError) as exc:
                attempt['outcome'] = 'failed'
                errors.append(type(exc).__name__)
                if isinstance(exc, httpx.HTTPStatusError):
                    attempt['http_status'] = exc.response.status_code
        articles.extend(accepted)
        runs.append({'source': 'wechat:'+account['name'], 'query': {'terms': queries}, 'since': None, 'until': day,
            'status': 'partial' if accepted or not errors else 'failed', 'count': len(accepted),
            'truncated': len(ordered) > MAX_FETCHES_PER_ACCOUNT or len(request['queries']) > MAX_QUERIES,
            'error': ';'.join(dict.fromkeys(errors)) or None, 'fetched_at': now(), 'coverage_complete': False,
            'effective_requests': requests, 'search_lanes': pending})
    return runs, articles
