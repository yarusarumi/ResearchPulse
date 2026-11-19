"""Versioned social/video leads. Supplied text is never promoted to paper evidence."""
import hashlib
import json
import re
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

from .directions import load_direction, optional
from .execution import project_lock
from .media_sources import MediaInput, fetch_wechat_article, fetch_youtube_feed, parse_timed_transcript
from .models import now

ZONE = ZoneInfo('Asia/Shanghai')
POLICY = {'automatic_collection_enabled': False, 'model_analysis_enabled': False,
          'generated_feishu_upload_enabled': False, 'source_claims_scientifically_verified': False}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def _identity(item):
    url = urlsplit(item['url'])
    # Explicit platform IDs may associate revisions; titles never serve as merge keys.
    if item['platform'] == 'youtube':
        video = url.path.strip('/') if url.hostname == 'youtu.be' else parse_qs(url.query).get('v', [None])[0]
        if video and re.fullmatch(r'[A-Za-z0-9_-]{11}', video):
            return 'youtube:' + video
    return item['url']


def related_identifiers(text):
    """Declared links, not verified scientific equivalence or paper publication dates."""
    identifiers = set()
    for doi in re.findall(r'(?i)\b10\.\d{4,9}/[^\s<>"\]]+', text):
        identifiers.add(('doi', doi.rstrip('.,;:，。；：）)')))
    for arxiv in re.findall(r'(?i)(?:arxiv:|arxiv\.org/(?:abs|html|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)', text):
        identifiers.add(('arxiv', arxiv))
    return [{'kind': kind, 'value': value.lower() if kind == 'doi' else value}
            for kind, value in sorted(identifiers)]


def save_media(store, payload, *, origin='user_supplied'):
    if origin not in ('user_supplied', 'operator_supplied', 'public_youtube_feed', 'public_wechat_article',
                      'browser_supplied'):
        raise ValueError('Unknown media origin')
    payload = dict(payload)
    if 'transcript_text' in payload:
        transcript = payload.pop('transcript_text')
        if payload.get('content_scope') != 'transcript' or payload.get('segments'):
            raise ValueError('Provide exactly one timed transcript')
        payload['segments'] = [s.model_dump() for s in parse_timed_transcript(transcript)]
    item = MediaInput.model_validate(payload).model_dump(mode='json')
    load_direction(store, item['user_id'], item['direction_id'])
    item_id = 'media-' + _digest([item['user_id'], item['direction_id'], item['platform'], _identity(item)])
    version_id = item_id + ':' + _digest([item, origin])
    with project_lock(store.root, item_id):
        pointer = optional(store, 'media_current', item_id)
        prior = store.get('media_version', pointer['version_id']) if pointer else None
        # Refreshing a feed cannot downgrade an explicitly supplied article/transcript.
        if prior and origin == 'public_youtube_feed' and prior['content_scope'] != 'metadata_only':
            return {**prior, 'reused': True, 'retained_supplied_content': True}
        body = optional(store, 'media_version', version_id)
        if body is None:
            paragraphs = [{'locator': f'paragraph:{i+1}', 'text': text.strip()}
                          for i, text in enumerate(re.split(r'\n\s*\n', item['text'])) if text.strip()]
            body = {**item, 'item_id': item_id, 'version_id': version_id, 'content_origin': origin,
                    'discovered_at': prior['discovered_at'] if prior else now(), 'version_discovered_at': now(),
                    'paragraphs': paragraphs, 'paragraph_count': len(paragraphs),
                    'segment_count': len(item['segments']), 'scientific_acceptance': False,
                    'related_identifiers': related_identifiers('\n'.join([item['url'], item['description'],
                        item['text'], *[s['text'] for s in item['segments']]]))}
            store.put('media_version', version_id, body, immutable=True)
        store.put('media_current', item_id, {'version_id': version_id, 'user_id': item['user_id'],
                                           'direction_id': item['direction_id']})
        return {**body, 'reused': prior is not None and prior['version_id'] == version_id}


def load_media(store, user, direction_id, version_id):
    direction = load_direction(store, user, direction_id)
    body = store.get('media_version', version_id)
    if (body['user_id'], body['direction_id']) != (direction['user_id'], direction['direction_id']):
        raise ValueError('Media owner or direction mismatch')
    return body


def media_state(store, user, direction_id):
    direction = load_direction(store, user, direction_id)
    items = [store.get('media_version', p['version_id']) for p in store.list('media_current')
             if (p['user_id'], p['direction_id']) == (direction['user_id'], direction['direction_id'])]
    # Feed ingestion order is not publication order. Unknown publication dates
    # use the saved-version time without inventing a publication date.
    items.sort(key=lambda i: (datetime.fromisoformat(i['published_at'] or i['version_discovered_at']),
                              i['version_id']), reverse=True)
    fields = ('item_id', 'version_id', 'title', 'url', 'platform', 'kind', 'author', 'published_at',
              'discovered_at', 'version_discovered_at', 'content_scope', 'content_origin', 'paragraph_count',
              'segment_count', 'related_identifiers')
    collections = [c for c in store.list('media_collection')
                   if (c['user_id'], c['direction_id']) == (direction['user_id'], direction['direction_id'])]
    collections.sort(key=lambda c: c['collected_at'], reverse=True)
    return {'user_id': direction['user_id'], 'direction_id': direction['direction_id'],
            'items': [{k: i[k] for k in fields} for i in items[:100]], 'total': len(items),
            'collections': collections[:20], 'policy': POLICY}


def import_browser_article(store, payload):
    """Import an explicit browser capture, never claim independent verification."""
    if set(payload) != {'user_id', 'direction_id', 'capture'}:
        raise ValueError('Explicit owner, direction and capture required')
    capture = payload['capture']
    if not isinstance(capture, dict) or set(capture) != {
            'format', 'status', 'error', 'url', 'title', 'author', 'published_text', 'text'}:
        raise ValueError('请粘贴公众号正文书签复制的完整数据')
    if capture['format'] != 'neuroei-wechat-v1' or not all(isinstance(v, str) for v in capture.values()):
        raise ValueError('公众号正文数据格式不正确')
    if capture['status'] != 'ok' or capture['error']:
        raise ValueError('未成功读取正文，不能导入；请先打开可阅读的公众号文章')
    if not capture['text'].strip() or not capture['title'].strip():
        raise ValueError('没有读取到标题或正文，请先在浏览器打开可阅读的文章')
    published = None
    date = re.fullmatch(r'\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})\s*', capture['published_text'])
    if date:
        published = datetime(*map(int, date.groups()), tzinfo=ZONE).isoformat()
    return save_media(store, {
        'user_id': payload['user_id'], 'direction_id': payload['direction_id'],
        'platform': 'wechat', 'kind': 'article', 'url': capture['url'],
        'title': capture['title'], 'author': capture['author'], 'published_at': published,
        'description': '浏览器提供的页面文本；未核实完整性，未读取图片或视频。页面日期：'
                       + (capture['published_text'] or '未知'),
        'content_scope': 'article_text', 'text': capture['text'],
    }, origin='browser_supplied')


def collect_media_article(store, payload, *, transport=None):
    if set(payload) != {'user_id', 'direction_id', 'url'}:
        raise ValueError('Explicit owner, direction and article URL required')
    direction = load_direction(store, payload['user_id'], payload['direction_id'])
    result = fetch_wechat_article(payload['url'], transport=transport)
    inputs = [MediaInput.model_validate({**item, 'user_id': direction['user_id'],
              'direction_id': direction['direction_id']}).model_dump(mode='json') for item in result['items']]
    versions = [save_media(store, item, origin='public_wechat_article')['version_id'] for item in inputs]
    body = {'collection_id': 'mediacollect-' + uuid4().hex, 'user_id': direction['user_id'],
            'direction_id': direction['direction_id'], 'url': payload['url'], 'collected_at': now(),
            'status': result['status'], 'error': result['error'], 'truncated': result['truncated'],
            'count': len(versions), 'item_versions': versions, 'policy': POLICY,
            'coverage': 'single_article_text_no_images_or_video'}
    store.put('media_collection', body['collection_id'], body, immutable=True)
    return body


def collect_media_channel(store, payload, *, transport=None):
    if set(payload) != {'user_id', 'direction_id', 'channel_id'}:
        raise ValueError('Explicit owner, direction and channel required')
    if not all(isinstance(v, str) for v in payload.values()):
        raise TypeError('String owner, direction and channel required')
    direction = load_direction(store, payload['user_id'], payload['direction_id'])
    channel = payload['channel_id']
    if not re.fullmatch(r'UC[A-Za-z0-9_-]{22}', channel):
        raise ValueError('Canonical YouTube channel ID required')
    with project_lock(store.root, 'media-collect:' + direction['direction_id'] + ':' + channel):
        result = fetch_youtube_feed(channel, transport=transport)
        body = {'collection_id': 'mediacollect-' + uuid4().hex, 'user_id': direction['user_id'],
                'direction_id': direction['direction_id'], 'channel_id': channel, 'collected_at': now(),
                'status': result['status'], 'error': result['error'], 'truncated': result['truncated'],
                'count': 0, 'item_versions': [], 'policy': POLICY,
                'coverage': 'single_public_feed_snapshot_not_complete_channel_history'}
        # Validate the whole response before writing any imported items.
        inputs = [MediaInput.model_validate({**item, 'user_id': direction['user_id'],
                  'direction_id': direction['direction_id']}).model_dump(mode='json') for item in result['items']]
        for item in inputs:
            saved = save_media(store, item, origin='public_youtube_feed')
            body['item_versions'].append(saved['version_id'])
        body['count'] = len(inputs)
        store.put('media_collection', body['collection_id'], body, immutable=True)
        return body


def media_snapshot(store, user, direction_id, *, report_date=None):
    state = media_state(store, user, direction_id)
    items = []
    for info in state['items']:
        item = load_media(store, user, direction_id, info['version_id'])
        discovered = datetime.fromisoformat(item['version_discovered_at']).astimezone(ZONE).date().isoformat()
        # Never inject newly discovered material into a historical dated report.
        if report_date is None or discovered <= report_date:
            items.append(item)
    return {'user_id': state['user_id'], 'direction_id': state['direction_id'],
            'captured_at': now(), 'items': items[:10], 'total': len(items),
            'selection': 'publication_date_or_saved_time_when_unknown_not_model_relevance_ranking', 'policy': POLICY}


def _plain(value):
    return re.sub(r'([\\`*_{}\[\]<>#|])', r'\\\1', ' '.join(str(value).split()))


def _excerpt(value, limit):
    return _plain(value[:limit]) + ('…（展示截断，完整已收录文本见来源详情）' if len(value) > limit else '')


def render_media_digest(snapshot):
    content = '\n\n## 公众号与视频线索\n\n'
    content += ('这是已收录来源的可追溯摘录，按实际发布时间排列，未知时用保存时间排列；尚未做科研价值排序或模型分析。'
                '转述、经验和观点不作为已核实论文结论。发布时间未知时不推测，转载不算论文首次发表。'
                '简报每来源最多展示4段／4条字幕，完整已收录文本见来源详情。\n\n')
    if not snapshot['items']:
        return content + '此方向尚无符合本次日期范围的已收录线索。\n'
    labels = {'metadata_only': '仅标题／简介，未读正文或字幕',
              'article_text': '使用者提供的文章文本，未核实完整性',
              'transcript': '使用者提供的带时间字幕，未核实音画或转写准确性'}
    for item in snapshot['items']:
        content += '### ' + _plain(item['title']) + '\n\n'
        content += f"来源：{_plain(item['platform'])}／{_plain(item['author'] or '作者未注明')}\n\n"
        content += '原链接：' + item['url'].replace('(', '%28').replace(')', '%29') + '\n\n'
        content += '实际发布时间：' + _plain(item['published_at'] or '未知')
        content += '；发现时间：' + _plain(item['discovered_at']) + '\n\n'
        scope = labels[item['content_scope']]
        if item.get('content_origin') == 'browser_supplied':
            scope = '浏览器提供的正文，未核实完整性，未读取图片或视频'
        elif item.get('content_origin') == 'public_wechat_article':
            scope = '公开页面正文，未核实完整性，未读取图片或视频'
        content += '读取范围：' + scope + '。\n\n'
        content += '内容版本：' + item['version_id'] + '\n\n'
        if item['description']:
            content += '> 简介：' + _excerpt(item['description'], 1000) + '\n\n'
        for paragraph in item['paragraphs'][:4]:
            content += '> [' + paragraph['locator'] + '] ' + _excerpt(paragraph['text'], 1500) + '\n\n'
        for index, segment in enumerate(item['segments'][:4]):
            content += f"> [segment:{index+1} · {segment['start_seconds']:g}s–"
            content += (f"{segment['end_seconds']:g}s" if segment['end_seconds'] is not None else '结束未知')
            content += '] ' + _excerpt(segment['text'], 1500) + '\n\n'
        if item['related_identifiers']:
            content += '原内容出现的研究标识（仅线索，未验证支持关系）：' + '；'.join(
                _plain(i['kind'] + ':' + i['value']) for i in item['related_identifiers']) + '\n\n'
    if snapshot['total'] > len(snapshot['items']):
        content += '仅列日期排序前 10 个已收录版本，其余可在消息源列表查看。\n'
    return content
