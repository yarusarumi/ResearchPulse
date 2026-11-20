"""Progress is per document and starts with the matching saved image results."""
from contextlib import closing
from threading import Event

from neuroei import feishu_folders
from neuroei.store import Store


def test_image_progress_starts_from_cache_when_retried(tmp_path, monkeypatch):
    documents = [
        {'token': token, 'name': name, 'url': f'https://example.feishu.cn/docx/{token}',
         'type': 'docx', 'path': [], 'modified_at': modified}
        for token, name, modified in [('doc1', '日报一', '2026-11-19T08:00:00+08:00'),
                                     ('doc2', '日报二', '2026-11-18T08:00:00+08:00')]
    ]
    downloads = []
    calls = []
    revision = '1'
    pixels = {}

    class LocalFolder(feishu_folders.SharedFolder):
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def tree(self):
            return {row['token']: row for row in documents}

        def _read_document(self, key, headers, *, include_images=False):
            return {'revision': revision, 'unread_items': [], 'blocks': [
                {'block_id': key + 'text', 'section': '', 'text': '日报正文 ' + revision}], 'image_blocks': [
                {'token': key + suffix, 'document_id': key, 'revision': revision,
                 'block_id': key + suffix, 'section': '', 'caption': ''}
                for suffix in (('img1', 'img2') if key == 'doc1' else ('img1',))
            ]}

        def _download_image(self, token, headers):
            downloads.append(token)
            return {'status': 'downloaded', 'data': pixels.get(token, token.encode())}

        def _get(self, path, headers=None):
            return {'document': {'revision_id': revision}}

    monkeypatch.setattr(feishu_folders, 'SharedFolder', LocalFolder)
    with closing(Store(tmp_path)) as store:
        args = (store, {'user_id': 'researcher'}, 'project',
                {'url': 'https://example.feishu.cn/drive/folder/journals'})
        stamp = '2026-11-19T12:00:00+08:00'
        def transcribe(data):
            calls.append(data)
            return data.decode()

        readers = {'image_reader': bytes.decode, 'sol_image_reader': transcribe, 'image_workers': 1}
        feishu_folders.folder_batch(*args, documents[:1], stamp, only_image_ids=['doc1img1'], **readers)
        timeline = []
        feishu_folders.folder_batch(*args, documents, stamp, **readers,
                                    on_image_progress=lambda *entry: timeline.append(entry))

        assert downloads == ['doc1img1', 'doc1img2', 'doc2img1']
        assert timeline == [(1, 2, 1, '日报一'), (2, 2, 1, '日报一'),
                            (0, 1, 0, '日报二'), (1, 1, 0, '日报二')]

        revision = '2'
        pixels['doc1img2'] = b'replaced-image'
        timeline.clear()
        calls.clear()
        updated = feishu_folders.folder_batch(*args, documents[:1], stamp, **readers,
                                               on_image_progress=lambda *entry: timeline.append(entry))
        assert calls == [b'replaced-image']
        assert timeline == [(0, 2, 0, '日报一'), (2, 2, 1, '日报一')]
        images = [r['blocks'][0] for r in updated['records'] if ':image:' in r['source_id']]
        assert [(block['text'], block['revision']) for block in images] == [('doc1img1', '2'), ('replaced-image', '2')]
        before = len(downloads)
        feishu_folders.folder_batch(*args, documents[:1], stamp, **readers)
        assert len(downloads) == before and calls == [b'replaced-image']

        pixels.clear()
        second_cached = Event()
        put = store.put

        def watch_cache(kind, key, body, **kwargs):
            put(kind, key, body, **kwargs)
            if kind == 'feishu_image_sol' and body.get('text') == 'doc1img2':
                second_cached.set()

        def slow_first_image(data):
            if data == b'doc1img1':
                assert second_cached.wait(3)
            return data.decode()

        monkeypatch.setattr(store, 'put', watch_cache)
        other_project = (store, args[1], 'another-project', args[3])
        feishu_folders.folder_batch(*other_project, documents[:1], stamp, image_reader=str,
                                    sol_image_reader=slow_first_image, image_workers=4)
        assert second_cached.is_set()
