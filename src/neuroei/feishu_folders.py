"""Read a user's explicitly authorized Feishu folder for project evidence."""
import json
import re
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from .feishu import MAX_DOCUMENT_IMAGES, Feishu
from .feishu_image_ocr import CODEX_SECTION, OCR_POLICY, OCR_SECTION, SOL_POLICY, SOL_SECTION
from .feishu_user_auth import FEISHU_TIMEOUT, FeishuAuthorizationRequired, authorization, current_grant_id, invalidate_grant
from .llm import ModelFailure
from .models import now

TZ = ZoneInfo('Asia/Shanghai')
TOKEN = re.compile(r'[A-Za-z0-9]{1,128}')
READ_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError, OSError)
IMAGE_WORKERS = 4
IMAGE_PARALLEL_MIN = 8
MAX_FOLDER_LIST_REQUESTS = 512
MAX_FOLDER_DOCUMENTS = 1_000
MAX_FOLDER_TEXT_CHARS = 4_000_000
MAX_FOLDER_ATTACHMENT_CHARS = 600_000
TEXT_ATTACHMENT_SUFFIXES = {'.md', '.txt', '.csv'}
DAILY_SCOPE_POLICY = 'feishu-daily-delta-v7'
MANUAL_SELECTION_POLICY = 'feishu-manual-lines-v1'
MAX_MANUAL_LINES = 400
MAX_MANUAL_CHARS = 20_000
MANUAL_PREVIEW_TTL = 60
MANUAL_PREVIEW_CACHE_SIZE = 4
_MANUAL_PREVIEWS = OrderedDict()
_MANUAL_PREVIEWS_LOCK = threading.Lock()


class FolderUnavailable(ValueError):
    """Safe error to expose to users of the shared source."""


class DocumentReadUnavailable(FolderUnavailable):
    """A provider failure confined to a single document request."""


def _document_unavailable(status):
    return status in (403, 404, 408, 429) or 500 <= status < 600


class ManualSelectionError(FolderUnavailable):
    """A user-correctable problem with a manually selected daily section."""


def folder_url(url):
    try:
        parsed = urlsplit(url)
        match = re.fullmatch(r'/drive/folder/([A-Za-z0-9]{1,128})/?', parsed.path)
        if (parsed.scheme == 'https' and parsed.hostname and parsed.hostname.endswith('.feishu.cn')
                and parsed.netloc == parsed.hostname and match):
            return f'https://{parsed.hostname}/drive/folder/{match[1]}', match[1]
    except ValueError:
        pass
    raise FolderUnavailable('请在飞书云空间打开日报所在的文件夹，并复制地址栏中含 /drive/folder/ 的链接。')


def invalidate_previews(user_id):
    with _MANUAL_PREVIEWS_LOCK:
        for key in list(_MANUAL_PREVIEWS):
            if key[0] == user_id:
                _MANUAL_PREVIEWS.pop(key, None)


def entry_date(name, default_year=None):
    match = re.search(r'(?<!\d)(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*[日号]?(?!\d)', name)
    if match:
        try:
            return date(*map(int, match.groups()))
        except ValueError:
            pass
    if default_year is not None:
        match = re.search(r'(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?(?!\d)', name)
        if match:
            try:
                return date(default_year, *map(int, match.groups()))
            except ValueError:
                pass
    return None


def section_date(section, *, day=None, title=''):
    """Use the nearest dated heading; a bare M.D needs the source title's explicit year."""
    year = day.year if day else None
    title_years = set(re.findall(r'(?<!\d)20\d{2}(?!\d)', title)) if year else set()
    short_dotted = title_years == {str(year)}
    for heading in reversed((section or '').split(' / ')):
        if dated := entry_date(heading, default_year=year):
            return dated
        if short_dotted and (match := re.fullmatch(r'(\d{1,2})\.(\d{1,2})', heading.strip())):
            try:
                return date(year, *map(int, match.groups()))
            except ValueError:
                continue
    return None


def daily_model_records(records, prior_sources, day, manifest, review=None):
    """Project a verified Feishu snapshot down to the material for daily analysis."""
    paths = {'feishu:' + row['token']: row['path'] for row in manifest}
    selected, unverified, deferred, skipped = [], [], {}, []
    for record in records:
        parent_source = record.source_id.split(':attachment:', 1)[0]
        if parent_source not in paths:
            selected.append(record)
            continue
        prior = prior_sources.get(record.source_id, {})
        previous = prior.get('record', {}).get('blocks', [])
        pending = set(prior.get('deferred_dated_blocks', []))
        incomplete = set(prior.get('pending_block_ids', []))
        pending &= {block.block_id for block in record.blocks}
        if previous and not (pending or incomplete) and prior.get('version') == record.content_version:
            continue
        old = {block['block_id']: block['text'].replace('\r\n', '\n') for block in previous
               if block['block_id'] not in pending}
        path = paths[parent_source]
        prefix = ' / '.join(path)
        edited_today = record.source_modified_at and record.source_modified_at.astimezone(TZ).date() > day
        changed = []
        ambiguous = []
        has_dated_sections = has_undated_material = False
        for block in record.blocks:
            heading = (block.section or '').removeprefix(prefix).removeprefix(' / ')
            block_day = section_date(heading, day=day, title=path[-1])
            has_dated_sections |= block_day is not None
            has_undated_material |= block_day is None and bool(block.text.strip())
            if block_day and block_day > day:
                deferred.setdefault(record.source_id, []).append(block.block_id)
                continue
            if (edited_today and block_day is not None and block_day != day
                    and block.block_id not in pending):
                continue
            if old.get(block.block_id) == block.text.replace('\r\n', '\n') and not (
                    block.block_id in incomplete and block_day == day):
                continue
            if (review and block_day is None and record.record_date != day
                    and block.block_id != 'text' and block.text.strip()):
                ambiguous.append(block)
                continue
            if old or block_day == day or block.block_id in pending or (
                    block_day is None and record.record_date == day):
                changed.append(block)
        if ambiguous:
            # Strip folder/file names from the model's heading hierarchy.
            review_blocks = {block.block_id: block.model_copy(update={
                'section': (block.section or '').removeprefix(prefix).removeprefix(' / ') or None,
            }) for block in record.blocks}
            review_record = record.model_copy(update={'blocks': list(review_blocks.values())})
            verified, unresolved = review(review_record, [review_blocks[b.block_id] for b in ambiguous], previous, day)
            selected_ids = {b.block_id for b in [*changed, *verified]}
            changed = [b for b in record.blocks if b.block_id in selected_ids]
            if unresolved:
                unverified.append(record.source_id)
                pending_ids = deferred.setdefault(record.source_id, [])
                pending_ids.extend(i for i in unresolved if i not in pending_ids)
            if not changed and not unresolved:
                skipped.append(record.source_id)
        if not changed:
            if ambiguous:
                continue
            if not old and not has_dated_sections:
                # A newly created document with no dated sections has no
                # discoverable delta; it is out of scope, not a hard failure.
                skipped.append(record.source_id)
            elif not old and not deferred.get(record.source_id):
                if ':attachment:' in record.source_id and not has_undated_material:
                    skipped.append(record.source_id)
                else:
                    unverified.append(record.source_id)
            continue
        if len(changed) == 1 and changed[0].block_id == 'text' and 'text' in old:
            # A cloud Markdown/text file is one block. Keep the exact changed line ranges
            # so the reader can still resolve each quote to its original position.
            block = changed[0]
            lines = block.text.splitlines(keepends=True)
            matcher = SequenceMatcher(None, old['text'].splitlines(), block.text.splitlines())
            excerpts = [''.join(lines[start:end]) for tag, _, _, start, end in matcher.get_opcodes()
                        if tag in ('insert', 'replace')]
            if any(text and block.text.count(text) != 1 for text in excerpts):
                unverified.append(record.source_id)
                continue
            selected.extend(record.model_copy(update={'blocks': [block.model_copy(update={'text': text})]})
                            for text in excerpts if text.strip())
        else:
            selected.append(record.model_copy(update={'blocks': changed}))
    return selected, unverified, deferred, skipped


class SharedFolder(Feishu):
    def __init__(self, store, account, link, *, transport=None, grant_id=None):
        self.store, self.account = store, account
        self.link, self.target = folder_url(link)
        self.root_url, self.root = self.link, self.target
        # Live browsing uses the current grant. A queued job supplies its frozen
        # generation; an empty generation can never silently adopt a new grant.
        self.grant_id = current_grant_id(store, account) if grant_id is None else grant_id
        super().__init__([], transport=transport)
        self.http.timeout = FEISHU_TIMEOUT

    def _authenticate(self):
        # Never fall back to the application's tenant identity.
        return authorization(self.store, self.account, expected_grant_id=self.grant_id, http=self.http)

    def _get(self, path, headers=None, **kwargs):
        try:
            response = self.http.get(path, headers=self._authenticate(), **kwargs)
        except FeishuAuthorizationRequired:
            raise
        except httpx.TransportError as exc:
            raise DocumentReadUnavailable('飞书连接暂时不可用。') from exc
        if response.status_code == 401:
            invalidate_grant(self.store, self.account['user_id'], self.grant_id)
            raise FeishuAuthorizationRequired('飞书授权已失效，请重新授权。')
        listing = path == '/drive/v1/files'
        if _document_unavailable(response.status_code) and not (listing and response.status_code == 403):
            raise DocumentReadUnavailable(f'飞书 HTTP {response.status_code}')
        try:
            body = response.json()
        except ValueError as exc:
            raise FolderUnavailable('飞书读取失败，请稍后重试。') from exc
        if listing and body.get('code') == 99991672:
            raise FolderUnavailable('飞书应用缺少云空间读取权限。请在飞书开发者后台开通权限、发布版本，再重新授权。')
        if listing and (response.status_code == 403 or body.get('code') == 1061004):
            raise DocumentReadUnavailable('飞书拒绝读取文件夹（403）。请用刚授权的飞书账号打开同一个链接，确认能看到文件清单。如果无法浏览，请让文件夹所有者给该账号开通查看权限。')
        if response.status_code == 200 and path.startswith('/docx/v1/documents/') and body.get('code') == 1770032:
            raise DocumentReadUnavailable('飞书拒绝读取此文档。')
        if response.status_code != 200 or body.get('code') != 0:
            raise FolderUnavailable('飞书拒绝了本次读取，请确认你本人仍有该文档的阅读与下载权限。')
        return body['data']

    def _download_image(self, token, headers=None):
        return super()._download_image(token, self._authenticate())

    def tree(self, headers=None):
        """Read only the user-selected tree; shortcuts cannot extend its boundary."""
        root = {'token': self.target, 'name': '飞书日报文件夹', 'type': 'folder', 'path': [],
                'parents': [], 'readable': False, 'url': self.link, 'modified_at': None}
        seen_tokens, children, requests = {self.target}, {}, 0

        def list_children(parent):
            nonlocal requests
            if parent['token'] in children:
                return children[parent['token']]
            rows = []
            page_token, seen = '', set()
            while True:
                requests += 1
                if requests > MAX_FOLDER_LIST_REQUESTS:
                    raise FolderUnavailable('目录超过单次浏览上限，请选择较小的文件夹。')
                params = {'folder_token': parent['token'], 'page_size': 50}
                if page_token:
                    params['page_token'] = page_token
                data = self._get('/drive/v1/files', params=params)
                for item in data['files']:
                    key, kind, name = item['token'], item['type'], item['name']
                    if item.get('parent_token') != parent['token'] or not TOKEN.fullmatch(key) or key in seen_tokens:
                        raise FolderUnavailable('飞书返回了不一致的目录结构，请重试。')
                    seen_tokens.add(key)
                    stamp = item.get('modified_time')
                    modified = datetime.fromtimestamp(int(stamp), UTC).isoformat() if stamp else None
                    suffix = PurePosixPath(name).suffix.lower()
                    row = {'token': key, 'name': name, 'type': kind, 'path': [*parent['path'], name],
                           'parents': [*parent['parents'], parent['token']], 'modified_at': modified,
                           'readable': kind == 'docx' or (kind == 'file' and suffix in ('.md', '.txt')),
                           'url': self.document_url(item.get('url', ''), key, kind)}
                    rows.append(row)
                if not data['has_more']:
                    break
                page_token = data.get('next_page_token', data.get('page_token'))
                if not page_token or page_token in seen:
                    raise FolderUnavailable('飞书未返回完整的文件清单，请重试。')
                seen.add(page_token)
            children[parent['token']] = rows
            return rows

        rows, pending = {self.target: root}, [root]
        while pending:
            for row in list_children(pending.pop()):
                rows[row['token']] = row
                if row['type'] == 'folder':
                    pending.append(row)
        return rows

    def document_url(self, url, key, kind):
        parsed = urlsplit(url)
        if parsed.scheme == 'https' and parsed.hostname and parsed.hostname.endswith('.feishu.cn') and (
                parsed.netloc == parsed.hostname and parsed.path in (f'/docx/{key}', f'/file/{key}', f'/drive/folder/{key}')):
            return f'https://{parsed.hostname}{parsed.path}'
        path = {'folder': 'drive/folder', 'docx': 'docx', 'file': 'file'}.get(kind)
        return f'https://{urlsplit(self.root_url).hostname}/{path}/{key}' if path else None

    def catalog(self):
        try:
            return self.tree()
        except (FolderUnavailable, FeishuAuthorizationRequired, PermissionError):
            raise
        except READ_ERRORS as exc:
            raise FolderUnavailable('飞书文件夹读取失败，请检查你的授权或稍后重试。') from exc

    def read_blocks(self, row, headers, *, read_images=None, image_day=None, image_coverage=None,
                    on_image_progress=None, include_images=False, only_image_ids=None, attachments=None,
                    should_stop=None):
        key, url = row['token'], row['url']
        if row['type'] == 'docx':
            snapshot = (self._read_document(key, headers, include_images=True)
                        if (read_images or include_images) else self._read_document(key, headers))
            blocks = [{**block, 'file': url,
                       'section': ' / '.join([*row['path'], *filter(None, [block['section']])])}
                      for block in snapshot['blocks']]
            if attachments is not None:
                for item in snapshot.get('attachments', []):
                    if (isinstance(item['name'], str) and PurePosixPath(item['name']).suffix.lower() in TEXT_ATTACHMENT_SUFFIXES
                            and isinstance(item['token'], str) and TOKEN.fullmatch(item['token'])):
                        attachments.append({**item, 'document_id': key, 'file': url, 'revision': snapshot['revision'],
                                            'section': ' / '.join([*row['path'], *filter(None, [item['section']])])})
            images = snapshot.get('image_blocks', [])
            if only_image_ids is not None:
                allowed = set(only_image_ids)
                images = [image for image in images if image['block_id'] in allowed]
            image_ids = {image['block_id'] for image in images}
            unread = Counter('equation' if 'equation' in item.get('element', []) else 'other'
                             for item in snapshot['unread_items'] if item['block_id'] not in image_ids)
            if include_images and not read_images:
                return blocks, unread, [
                    {**image, 'file': url, 'text': None,
                     'section': ' / '.join([*row['path'], *filter(None, [image['section']])])}
                    for image in images]
            text_ids = {block['block_id'] for block in blocks}
            recognized = []
            eligible = []
            for image in images:
                if image_day:
                    recorded = section_date(image['section'], day=image_day, title=row['name'])
                    recorded = recorded or entry_date(row['name'], default_year=image_day.year)
                else:
                    recorded = entry_date(image['section'] or '') or entry_date(row['name'])
                if image['block_id'] in text_ids or (image_day and recorded != image_day):
                    image_coverage['skipped'] += 1
                else:
                    eligible.append((image, recorded))
            reused, readings = read_images([image for image, _ in eligible],
                                           lambda token: self._download_image(token, headers)) if eligible else (0, ())
            done = reused
            if eligible and on_image_progress:
                on_image_progress(done, len(eligible), reused, row['name'])
            for (image, recorded), (text, section, from_cache) in zip(eligible, readings, strict=True):
                if should_stop and should_stop():
                    break
                image_coverage['attempted'] += 1
                if not from_cache:
                    done += 1
                    if on_image_progress and ((done - reused) % 5 == 0 or done == len(eligible)):
                        on_image_progress(done, len(eligible), reused, row['name'])
                if not text:
                    image_coverage['unread'] += 1
                    continue
                recognized.append((recorded, {'document_id': key, 'block_id': image['block_id'], 'revision': snapshot['revision'],
                    'text': text, 'file': url, 'section': ' / '.join([section, *row['path'],
                        *filter(None, [image['section'], image['caption'] if isinstance(image['caption'], str) else None])])}))
            if recognized:
                try:
                    same_version = str(self._get(f'/docx/v1/documents/{key}', headers)['document']['revision_id']) == snapshot['revision']
                except READ_ERRORS:
                    same_version = False
                if same_version:
                    image_coverage['recognized'] += len(recognized)
                    image_coverage['sol'] += sum(block['section'].startswith((SOL_SECTION + ' / ', CODEX_SECTION + ' / '))
                                                 for _, block in recognized)
                else:
                    image_coverage['unread'] += len(recognized)
                    recognized.clear()
            return blocks, unread, recognized
        text, revision = self.download_text(f'/drive/v1/files/{key}/download')
        return [{'document_id': key, 'block_id': 'text', 'text': text,
                 'revision': revision, 'file': url, 'section': row['name']}], Counter(), []

    def download_text(self, path):
        with self.http.stream('GET', path, headers=self._authenticate()) as response:
            if response.status_code == 401:
                invalidate_grant(self.store, self.account['user_id'], self.grant_id)
                raise FeishuAuthorizationRequired('飞书授权已失效，请重新授权。')
            response.raise_for_status()
            is_json = response.headers.get('content-type', '').startswith('application/json')
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > 400_000:
                    raise FolderUnavailable('飞书文本文件超过 400 KB。')
        if is_json:
            try:
                code = json.loads(data).get('code')
            except (ValueError, AttributeError):
                code = None
            if path.startswith('/drive/v1/files/') and code == 1061004:
                raise DocumentReadUnavailable('飞书拒绝下载此文档。')
            raise FolderUnavailable('飞书未返回文本文件。')
        text = data.decode('utf-8-sig')
        if not text.strip() or '\x00' in text:
            raise FolderUnavailable('飞书文本文件为空或不是 UTF-8 文本。')
        return text, sha256(data).hexdigest()


def browse(store, account, url):
    client = SharedFolder(store, account, url)
    try:
        rows = client.catalog()
        depth = len(rows[client.target]['path'])
        return {'url': client.link, 'entries': [
            {**row, 'path': row['path'][depth:], 'parents': row['parents'][depth:]}
            for key, row in rows.items() if key != client.target]}
    finally:
        client.close()


def selected_rows(rows, selected):
    if any(key not in rows or not (rows[key]['type'] == 'folder' or rows[key]['readable']) for key in selected):
        raise FolderUnavailable('日报文件或子目录已移出共享范围，或当前无法读取。')
    return sorted((r for r in rows.values() if r['readable'] and
                   (not selected or r['token'] in selected or set(r['parents']).intersection(selected))),
                  key=lambda row: row['token'])


def manual_row(store, account, binding, source_id):
    """Resolve one selected daily document without applying the automatic day window."""
    if not isinstance(binding, dict) or not binding.get('url'):
        raise ManualSelectionError('请先连接飞书日报来源。')
    token = source_id[7:] if isinstance(source_id, str) and source_id.startswith('feishu:') else ''
    if not TOKEN.fullmatch(token):
        raise ManualSelectionError('请选择要分析的日报文件。')
    client = SharedFolder(store, account, binding['url'])
    try:
        rows = selected_rows(client.catalog(), binding.get('selection') or [])
        row = next((r for r in rows if r['token'] == token and r['type'] in ('docx', 'file')), None)
        if row is None:
            raise ManualSelectionError('所选日报不在当前来源范围内，请重新选择。')
        return row
    finally:
        client.close()


def _manual_page(document, source_id, offset, limit):
    all_lines, images, name, revision = document
    total = len(all_lines)
    lines, page_chars, truncated = [], 0, False
    for line in all_lines[offset:]:
        if limit is not None and len(lines) >= limit:
            break
        if page_chars + len(line['text']) + 1 > 200_000:
            truncated = True
            break
        lines.append({**line, 'date': line['date'].isoformat() if line['date'] else None})
        page_chars += len(line['text']) + 1
    return {'policy': MANUAL_SELECTION_POLICY, 'source_id': source_id, 'name': name,
            'revision': revision, 'line_count': total, 'lines': lines,
            'images': images, 'image_count': len(images),
            'offset': offset, 'limit': limit,
            'has_more': offset + len(lines) < total,
            'truncated': truncated, 'max_lines': MAX_MANUAL_LINES,
            'max_chars': MAX_MANUAL_CHARS}


def manual_preview(store, account, binding, source_id='', offset=0, limit=None):
    """List selectable daily documents and, for one document, a page of numbered lines."""
    if not isinstance(binding, dict) or not binding.get('url'):
        raise ManualSelectionError('请先连接飞书日报来源。')
    try:
        offset = int(offset or 0)
        limit = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        raise ManualSelectionError('分页参数不正确。')
    if offset < 0 or (limit is not None and not 1 <= limit <= 500):
        raise ManualSelectionError('分页参数不正确。')
    grant_id = current_grant_id(store, account)
    binding_digest = sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache_key = (account.get('user_id') or account.get('username') or '',
                 grant_id, binding_digest, source_id) if source_id else None
    if cache_key:
        current = time.monotonic()
        with _MANUAL_PREVIEWS_LOCK:
            cached = _MANUAL_PREVIEWS.get(cache_key)
            if cached and current - cached[0] <= MANUAL_PREVIEW_TTL:
                _MANUAL_PREVIEWS.move_to_end(cache_key)
                return _manual_page(cached[1], source_id, offset, limit)
            if cached:
                _MANUAL_PREVIEWS.pop(cache_key, None)
    client = SharedFolder(store, account, binding['url'])
    try:
        rows = selected_rows(client.catalog(), binding.get('selection') or [])
        if not source_id:
            files = [{'source_id': 'feishu:' + row['token'], 'name': row['name'],
                      'modified_at': row['modified_at'], 'type': row['type']}
                     for row in rows if row['type'] in ('docx', 'file')]
            files.sort(key=lambda item: str(item.get('modified_at') or ''), reverse=True)
            return {'policy': MANUAL_SELECTION_POLICY, 'files': files}
        row = next((r for r in rows if 'feishu:' + r['token'] == source_id and r['type'] in ('docx', 'file')), None)
        if row is None:
            raise ManualSelectionError('所选日报不在当前来源范围内，请重新选择。')
        blocks, _unread, image_blocks = client.read_blocks(row, client._authenticate(), include_images=True)
        if not blocks:
            raise ManualSelectionError('所选日报没有可读取的正文。')
        line_counts, running = [], 0
        for block in blocks:
            length = len(block['text'].splitlines() or [''])
            line_counts.append((block.get('order'), length))
            running += length
        images = []
        for image in image_blocks or []:
            if image.get('order') is None:
                anchor = 1
            else:
                anchor = sum(length for order, length in line_counts
                             if order is not None and order < image['order']) or 1
            anchor = min(max(1, anchor), running)
            images.append({'block_id': image['block_id'], 'revision': image.get('revision'),
                           'after_line': anchor, 'caption': image.get('caption'),
                           'section': image.get('section') or ''})
        all_lines = []
        for block in blocks:
            for block_line, text in enumerate(block['text'].splitlines() or [''], start=1):
                all_lines.append({'n': len(all_lines) + 1, 'block_id': block['block_id'],
                                  'block_line': block_line, 'text': text,
                                  'section': block.get('section', ''),
                                  'date': (entry_date(block.get('section') or row['name']) or None)})
        document = (all_lines, images, row['name'], blocks[0]['revision'])
        with _MANUAL_PREVIEWS_LOCK:
            _MANUAL_PREVIEWS[cache_key] = (time.monotonic(), document)
            _MANUAL_PREVIEWS.move_to_end(cache_key)
            while len(_MANUAL_PREVIEWS) > MANUAL_PREVIEW_CACHE_SIZE:
                _MANUAL_PREVIEWS.popitem(last=False)
        return _manual_page(document, source_id, offset, limit)
    finally:
        client.close()



def read_folder_images(store, account, project_id, document_id, *, limit=MAX_DOCUMENT_IMAGES):
    """Explicit read-only sample from one selected native Docx in the current account scope."""
    from .research_workspace import project_for

    project = project_for(store, account, project_id)
    if project['owner_id'] != account['user_id']:
        raise PermissionError('请使用项目创建账号读取日报图片。')
    if not (binding := project.get('feishu')):
        raise FolderUnavailable('项目尚未绑定飞书日报目录。')
    client = SharedFolder(store, account, binding['url'])
    try:
        headers = client._authenticate()
        rows = selected_rows(client.tree(headers), binding['selection'])
        document = next((row for row in rows if row['token'] == document_id and row['type'] == 'docx'), None)
        if not document:
            raise PermissionError('只能读取当前项目已选日报中的图片。')
        sample = client._read_document_images(document_id, headers, limit=limit)
        if client.tree(headers).get(document_id) != document:
            raise FolderUnavailable('飞书文件在读取时发生变化，请重新检查。')
        return {**sample, 'file': document['url']}
    finally:
        client.close()


def freeze_folder(store, account, binding, as_of, mode):
    client = SharedFolder(store, account, binding['url'])
    try:
        rows = selected_rows(client.catalog(), binding['selection'])
    finally:
        client.close()
    clock = datetime.fromisoformat(as_of).astimezone(TZ)
    today = clock.date()
    yesterday = today - timedelta(days=1)
    records = []
    for row in rows:
        recorded = entry_date(row['name'], default_year=today.year)
        modified = datetime.fromisoformat(row['modified_at']).astimezone(TZ) if row['modified_at'] else None
        if (modified and modified > clock) or (recorded and recorded > today):
            continue
        if mode == 'daily' and not ((modified and (modified.date() == yesterday or
                (modified.date() == today and row['type'] == 'docx'))) or
                (recorded == yesterday and (modified is None or modified.date() < today))):
            continue
        records.append(row)
    if len(records) > MAX_FOLDER_DOCUMENTS:
        raise FolderUnavailable(f'本次符合条件的日报超过 {MAX_FOLDER_DOCUMENTS:,} 份，请选择较小的子目录或文件范围。')
    return records


def folder_batch(store, account, project_id, binding, manifest, as_of, *, image_reader=None, image_day=None,
                 on_image_progress=None, image_workers=None, only_image_ids=None, sol_image_reader=None,
                    should_stop=None, grant_id=None, include_attachments=True, completed_attachments=None,
                    attachments_only=False, sol_policy=SOL_POLICY, sol_section=SOL_SECTION):
    run = {'source_id': 'feishu-folder:' + folder_url(binding['url'])[1], 'status': 'read'}
    batch = {'records': [], 'source_runs': [run], 'warnings': []}
    completed_attachments = completed_attachments or {}
    if manifest is None:
        run.update(status='failed', error='feishu_folder_unavailable')
        return batch
    if not manifest:
        return batch
    client = None
    image_coverage = Counter()
    workers = image_workers if image_workers is not None else (2 if sol_image_reader else IMAGE_WORKERS)
    pool = ThreadPoolExecutor(max_workers=workers) if image_reader and workers > 1 else None

    def add_record(row, source_id, recorded, blocks):
        batch['records'].append({'user_id': account['user_id'], 'project_id': project_id,
            'source_id': source_id, 'record_date': recorded.isoformat() if recorded else None,
            'available_at': as_of, 'source_modified_at': row['modified_at'],
            'synced_at': now(), 'blocks': blocks})

    def cached_text(kind, key):
        rows = store.select(kind, ('text',), key=key)
        return rows[0]['text'] if rows else None

    def read_images(images, download):
        def recognize(token, cached_ocr):
            if should_stop and should_stop():
                return None, OCR_SECTION
            try:
                asset = download(token)
            except Exception:  # noqa: BLE001 - One bad image must not discard the readable batch.
                return None, OCR_SECTION
            if asset['status'] != 'downloaded':
                return None, OCR_SECTION
            data = asset['data']
            if sol_image_reader:
                try:
                    text = sol_image_reader(data)
                except (ModelFailure, OSError, ValueError):
                    text = None
                if text:
                    return text, sol_section
                if should_stop and should_stop():
                    return None, OCR_SECTION
            if cached_ocr is not None:
                return cached_ocr, OCR_SECTION
            try:
                return image_reader(data), OCR_SECTION
            except Exception:  # noqa: BLE001 - One bad image must not discard the readable batch.
                return None, OCR_SECTION

        def recognize_and_cache(key, sol_key, token, cached_ocr):
            text, section = recognize(token, cached_ocr)
            if text is not None and not store.read_only:
                if section == sol_section:
                    store.put('feishu_image_sol', sol_key,
                              {'user_id': account['user_id'], 'project_id': project_id, 'text': text})
                elif cached_ocr is None:
                    store.put('feishu_image_ocr', key,
                              {'user_id': account['user_id'], 'project_id': project_id, 'text': text})
            return text or '', section

        keys, pending, known = [], {}, {}
        for image in images:
            token = image['token']
            if not isinstance(token, str) or not TOKEN.fullmatch(token):
                keys.append(None)
                continue
            identity = (account['user_id'], project_id, image['document_id'], image['revision'], token)
            key = sha256(':'.join((*identity, OCR_POLICY)).encode()).hexdigest()
            sol_key = sha256(':'.join((*identity, sol_policy)).encode()).hexdigest()
            keys.append((key, sol_key))
            if key not in known and key not in pending:
                sol_text = cached_text('feishu_image_sol', sol_key) if sol_image_reader else None
                ocr_text = None if sol_text else cached_text('feishu_image_ocr', key)
                if sol_text:
                    known[key] = sol_text, sol_section
                elif sol_image_reader or ocr_text is None:
                    pending[key] = (key, sol_key, token, ocr_text)
                else:
                    known[key] = ocr_text, OCR_SECTION
        cached_keys = frozenset(known)

        def results_in_order():
            parallel = pool and len(pending) >= (2 if sol_image_reader else IMAGE_PARALLEL_MIN)
            results = {key: pool.submit(recognize_and_cache, *candidate)
                       for key, candidate in pending.items()} if parallel else {}
            for pair in keys:
                if pair is None:
                    yield '', OCR_SECTION, False
                    continue
                key, _ = pair
                if key not in known:
                    known[key] = results[key].result() if parallel else recognize_and_cache(*pending[key])
                yield *known[key], key in cached_keys

        return sum(pair is not None and pair[0] in cached_keys for pair in keys), results_in_order()

    try:
        client = SharedFolder(store, account, binding['url'], grant_id=grant_id)
        before = client.tree()
        if any(before.get(row['token']) != row for row in manifest):
            raise FolderUnavailable('飞书文件在排队后发生变化，请重新检查。')
        size = 0
        documents = []
        selected = sorted(manifest, key=lambda row: row['modified_at'] or '', reverse=True) if image_reader else manifest
        for row in selected:
            if should_stop and should_stop():
                break
            attachments = []
            try:
                blocks, unread, images = client.read_blocks(row, None, read_images=read_images if image_reader else None,
                                                           image_day=image_day, image_coverage=image_coverage,
                                                           on_image_progress=on_image_progress,
                                                           only_image_ids=only_image_ids,
                                                           attachments=attachments if include_attachments else None,
                                                           should_stop=should_stop)
            except (DocumentReadUnavailable, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and not _document_unavailable(exc.response.status_code):
                    raise
                reason = (f'飞书 HTTP {exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError) else
                          str(exc) if isinstance(exc, DocumentReadUnavailable) else '飞书连接暂时不可用。')
                batch.setdefault('document_failures', []).append({'token': row['token'], 'name': row['name'], 'reason': reason})
                continue
            if attachments_only:
                if attachments:
                    documents.append((row, [], unread, [], attachments))
                continue
            size += sum(len(block['text']) for block in blocks)
            if size > MAX_FOLDER_TEXT_CHARS:
                raise FolderUnavailable('本次日报正文超过分析上限，请缩小来源范围。')
            if not blocks and not images and not attachments:
                raise FolderUnavailable('飞书文档没有可读取的正文或图片文字。')
            source = 'feishu:' + row['token']
            if blocks:
                add_record(row, source, entry_date(row['name']), blocks)
            for day, block in images:
                add_record(row, source + ':image:' + block['block_id'], day, [block])
            documents.append((row, blocks, unread, images, attachments))
        missing_document = False
        attachment_budget = min(MAX_FOLDER_ATTACHMENT_CHARS, MAX_FOLDER_TEXT_CHARS - size)
        for row, blocks, unread, images, attachments in documents:
            if should_stop and should_stop():
                return batch
            source = 'feishu:' + row['token']
            pending = [item for item in attachments if completed_attachments.get(
                       source + ':attachment:' + item['block_id']) != (item['revision'], item['section'])]
            reused = len(attachments) - len(pending)
            unread['other'] -= reused
            supplementary = []
            attachment_chars = 0
            over_limit = 0
            for index, item in enumerate(pending):
                if should_stop and should_stop():
                    return batch
                if attachment_chars >= attachment_budget:
                    over_limit += len(pending) - index
                    break
                try:
                    body, _ = client.download_text(f'/drive/v1/medias/{item["token"]}/download')
                except FeishuAuthorizationRequired:
                    raise
                except READ_ERRORS as exc:
                    error = (f'飞书 HTTP {exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError)
                             else '附件不是 UTF-8 文本' if isinstance(exc, UnicodeDecodeError)
                             else str(exc) if isinstance(exc, FolderUnavailable) else '读取失败，请重试')
                    batch.setdefault('attachment_failures', []).append({'name': item['name'], 'reason': error})
                    unread['other'] -= 1
                    continue
                text = f'附件：{item["name"]}\n{body}'
                if attachment_chars + len(text) <= attachment_budget:
                    supplementary.append({field: item[field] for field in (
                        'document_id', 'block_id', 'revision', 'file', 'section', 'order')} | {'text': text})
                    attachment_chars += len(text)
                else:
                    over_limit += 1
            if supplementary:
                metadata = client._get(f'/docx/v1/documents/{row["token"]}')['document']
                if str(metadata['revision_id']) != supplementary[0]['revision']:
                    raise FolderUnavailable('飞书文档在读取附件时发生变化，请重新检查。')
                prefix = ' / '.join(row['path'])
                for block in supplementary:
                    heading = block['section'].removeprefix(prefix).removeprefix(' / ')
                    recorded = section_date(heading) or entry_date(row['name'])
                    add_record(row, source + ':attachment:' + block['block_id'], recorded, [block])
                attachment_budget -= attachment_chars
                unread['other'] -= len(supplementary)
                batch['warnings'].append(f'feishu_attachments_read:{row["token"]}:{len(supplementary)}')
            if over_limit:
                unread['other'] -= over_limit
                batch['warnings'].append(f'feishu_attachments_over_limit:{row["token"]}:{over_limit}')
            if unread['other']:
                batch['warnings'].append(f'feishu_nontext_blocks_not_read:{row["token"]}:{unread["other"]}')
            if unread['equation']:
                batch['warnings'].append(f'feishu_equations_not_read:{row["token"]}:{unread["equation"]}')
            if not blocks and not images and not (supplementary or reused or over_limit):
                missing_document = True
        if missing_document:
            run.update(status='partial' if batch['records'] else 'failed', error='feishu_folder_unavailable')
        if should_stop and should_stop():
            return batch
        after = client.tree()
        if any(after.get(row['token']) != row for row in manifest):
            raise FolderUnavailable('飞书文件在读取时发生变化，请重新检查。')
        if not missing_document and batch.get('document_failures'):
            run.update(status='partial' if batch['records'] else 'failed', error='feishu_documents_unavailable')
    except FeishuAuthorizationRequired:
        run.update(status='partial' if batch['records'] else 'failed', error='feishu_authorization_required')
    except READ_ERRORS:
        run.update(status='partial' if batch['records'] else 'failed', error='feishu_folder_unavailable')
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        if client:
            client.close()
    if image_coverage:
        counts = [image_coverage[key] for key in ('recognized', 'unread', 'skipped')]
        if sol_image_reader:
            sol = image_coverage['sol']
            coverage = ['feishu_image_transcripts', *counts, sol, counts[0] - sol]
        else:
            coverage = ['feishu_image_ocr', *counts]
        batch['warnings'].append(':'.join(map(str, coverage)))
    return batch
