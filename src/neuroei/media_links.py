"""Deterministic cross-source link and keyword screening for social/video leads.

Declared DOI/arXiv identifiers are the only association basis; a shared title is
not a merge key, and neither the link nor the screening is a scientific claim.
No model calls and no network access happen here.
"""
import re

from .directions import load_direction
from .media import media_snapshot

POLICY = {'model_analysis_enabled': False, 'scientific_verification': False,
          'association_basis': 'declared_identifiers_only', 'title_similarity_never_merges': True,
          'screening': 'declared_phrase_match_not_model_ranking', 'generated_feishu_upload_enabled': False}

VERSION_SUFFIX = re.compile(r'v\d+$', re.IGNORECASE)
SPLIT = re.compile(r'[，,。;；、/|（）()\[\]【】]+')
TOKEN = re.compile(r'[A-Za-z][A-Za-z0-9+#.-]{2,}')

TIERS = {'excluded_by_declared_exclusion': '命中声明排除内容',
         'declared_term_match': '命中声明主题词（关键词筛选，不是模型相关性排序）',
         'no_declared_term_match': '未命中声明主题词（不代表不相关）'}


def _identifier_key(identifier):
    kind, value = identifier['kind'], identifier['value'].strip()
    return kind, value.lower() if kind == 'doi' else VERSION_SUFFIX.sub('', value)


def _paper_identifiers(paper):
    keys = set()
    paper_id = paper['paper_id']
    if paper_id.startswith('arxiv:'):
        keys.add(('arxiv', VERSION_SUFFIX.sub('', paper_id[6:])))
    elif paper_id.startswith('doi:'):
        keys.add(('doi', paper_id[4:].lower()))
    if paper.get('doi'):
        keys.add(('doi', paper['doi'].lower()))
    return keys


def associate(items, papers):
    """Link leads to candidate-pool papers only through declared identifiers."""
    lookup = {}
    for paper in papers:
        for key in _paper_identifiers(paper):
            lookup.setdefault(key, []).append(paper)
    research, unlinked = {}, []
    for item in items:
        declared = item.get('related_identifiers') or []
        matched = {}
        for identifier in declared:
            key = _identifier_key(identifier)
            for paper in lookup.get(key, []):
                matched.setdefault((paper['paper_id'], paper['version']), (paper, set()))[1].add(key)
        if not declared:
            reason = 'no_declared_identifier'
        elif not matched:
            reason = 'declared_identifier_not_in_current_candidate_pool'
        else:
            reason = None
        if reason:
            unlinked.append({'item_id': item['item_id'], 'version_id': item['version_id'],
                             'title': item['title'], 'platform': item['platform'],
                             'content_scope': item['content_scope'], 'reason': reason})
            continue
        for (paper_id, version), (paper, keys) in matched.items():
            entry = research.setdefault((paper_id, version), {
                'paper_id': paper_id, 'version': version, 'title': paper['title'],
                'doi': paper.get('doi'), 'basis': 'declared_identifier', 'media': []})
            entry['media'].append({'item_id': item['item_id'], 'version_id': item['version_id'],
                'title': item['title'], 'platform': item['platform'], 'author': item['author'],
                'content_scope': item['content_scope'], 'published_at': item['published_at'],
                'url': item['url'],
                'matched_identifiers': [{'kind': k, 'value': v} for k, v in sorted(keys)]})
    research_list = sorted(research.values(), key=lambda entry: (-len(entry['media']), entry['paper_id'],
                                                                 entry['version']))
    unlinked.sort(key=lambda entry: (entry['platform'], entry['item_id'], entry['version_id']))
    return research_list, unlinked


def declared_terms(direction):
    """Explicit phrases and tokens from the user's own settings; nothing is inferred."""
    terms, seen = [], set()
    for text in [direction['title'], *direction['questions'], *direction['public_queries']]:
        for part in SPLIT.split(text):
            part = part.strip()
            if len(part) >= 2:
                key = part.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(part)
        for token in TOKEN.findall(text):
            if token.lower() not in seen:
                seen.add(token.lower())
                terms.append(token)
    return terms


def screen_item(item, terms, exclusions):
    """Phrase matching only: deterministic pre-filter, not relevance ranking."""
    body = '\n'.join([item['title'], item['description'], *[p['text'] for p in item['paragraphs']],
                      *[s['text'] for s in item['segments']]]).lower()
    excluded = [phrase for phrase in exclusions if len(phrase) >= 2 and phrase.lower() in body]
    hit = [term for term in terms if term.lower() in body]
    if excluded:
        tier = 'excluded_by_declared_exclusion'
    elif hit:
        tier = 'declared_term_match'
    else:
        tier = 'no_declared_term_match'
    return {'item_id': item['item_id'], 'version_id': item['version_id'], 'title': item['title'],
            'platform': item['platform'], 'content_scope': item['content_scope'], 'tier': tier,
            'matched_terms': hit[:6], 'matched_exclusions': excluded[:3],
            'published_at': item['published_at'],
            'version_discovered_at': item['version_discovered_at']}


def latest_direction_pool(store, user, direction_id):
    pools = [p for p in store.list('pool')
             if p.get('direction_id') == direction_id and p.get('user_id') == user]
    return max(pools, key=lambda p: (p.get('until', ''), p.get('pool_id', ''))) if pools else None


def analyze_media(store, user, direction_id, *, pool_id=None, report_date=None):
    direction = load_direction(store, user, direction_id)
    if pool_id is not None:
        pool = store.get('pool', pool_id)
        if 'direction_id' in pool and (pool['direction_id'] != direction['direction_id']
                                       or pool.get('user_id') != direction['user_id']):
            raise ValueError('Candidate pool belongs to another direction')
    else:
        pool = latest_direction_pool(store, direction['user_id'], direction['direction_id'])
    snapshot = media_snapshot(store, user, direction_id, report_date=report_date)
    terms = declared_terms(direction)
    screened = [screen_item(item, terms, direction['exclusions']) for item in snapshot['items']]
    research, unlinked = associate(snapshot['items'], (pool or {}).get('papers', []))
    return {'user_id': direction['user_id'], 'direction_id': direction['direction_id'],
            'captured_at': snapshot['captured_at'], 'report_date': report_date,
            'pool_id': (pool or {}).get('pool_id'), 'pool_available': pool is not None,
            'declared_terms': terms[:20], 'items': screened, 'research': research, 'unlinked': unlinked,
            'total': snapshot['total'],
            'counts': {'items': len(screened), 'linked': sum(len(entry['media']) for entry in research),
                       'unlinked': len(unlinked),
                       'excluded': sum(1 for i in screened if i['tier'] == 'excluded_by_declared_exclusion'),
                       'term_match': sum(1 for i in screened if i['tier'] == 'declared_term_match'),
                       'no_term_match': sum(1 for i in screened if i['tier'] == 'no_declared_term_match')},
            'policy': POLICY}


def _plain(value):
    return re.sub(r'([\\`*_{}\[\]<>#|])', r'\\\1', ' '.join(str(value).split()))


SCOPE = {'metadata_only': '仅标题／简介', 'article_text': '使用者提供的文章文本',
         'transcript': '使用者提供的带时间字幕'}
REASONS = {'no_declared_identifier': '线索内未出现已登记的 DOI／arXiv 标识',
           'declared_identifier_not_in_current_candidate_pool': '声明的标识未出现在本次候选池'}


def render_media_links(analysis):
    counts = analysis['counts']
    content = '\n\n## 线索与原始研究的声明式关联\n\n'
    content += ('只按线索中已登记的 DOI／arXiv 标识关联候选池论文；标题相似不合并，关联不代表线索支持论文结论，'
                '也不代表论文经过核实。关键词筛选只使用使用者声明的主题词，是检索前过滤，不是科研相关性排序。\n\n')
    if not analysis['pool_available']:
        content += '本次没有可对照的候选池，因此只登记线索，不判断是否属于已有研究。\n\n'
    content += (f"本次简报的 {counts['items']} 条线索：关联候选池论文 {counts['linked']} 条，"
                f"未关联 {counts['unlinked']} 条；命中声明主题词 {counts['term_match']} 条，"
                f"未命中 {counts['no_term_match']} 条，命中排除内容 {counts['excluded']} 条。\n\n")
    if analysis['research']:
        content += '### 已关联到候选池论文的线索\n\n'
        for entry in analysis['research']:
            content += '#### ' + _plain(entry['title']) + '\n\n'
            content += '研究条目：' + _plain(entry['paper_id']) + '（' + _plain(entry['version']) + '）'
            content += '；依据：线索内声明标识\n\n' if entry['basis'] == 'declared_identifier' else '\n\n'
            for media in entry['media']:
                identifiers = '、'.join(_plain(i['kind'] + ':' + i['value']) for i in media['matched_identifiers'])
                content += ('- ' + _plain(media['platform']) + '／' + _plain(media['author'] or '作者未注明')
                            + '《' + _plain(media['title']) + '》；读取范围：' + SCOPE[media['content_scope']]
                            + '；关联标识：' + identifiers + '\n')
            content += '\n'
    screened = [item for item in analysis['items'] if item['tier'] != 'no_declared_term_match']
    if screened:
        content += '### 线索的主题词筛查\n\n'
        for item in screened:
            content += ('- ' + _plain(item['title']) + '：' + TIERS[item['tier']] + '；命中：'
                        + '、'.join(_plain(term) for term in (item['matched_terms'] or item['matched_exclusions']))
                        + '\n')
        content += '\n'
    if analysis['unlinked']:
        content += '### 未关联候选池论文的线索\n\n'
        content += '未关联只表示没有可核对的声明标识或该标识不在本次候选池，不代表没有原始研究或没有价值。\n\n'
        for item in analysis['unlinked']:
            content += ('- ' + _plain(item['title']) + '（' + _plain(item['platform']) + '，'
                        + SCOPE[item['content_scope']] + '）：' + REASONS[item['reason']] + '\n')
    if analysis['total'] > counts['items']:
        content += '\n本简报只核对日期排序前 10 条线索，其余可在消息源列表查看。\n'
    return content
