"""Progress is per document and starts with the matching saved image results."""
from contextlib import closing

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

    class LocalFolder(feishu_folders.SharedFolder):
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def tree(self):
            return {row['token']: row for row in documents}

        def _read_document(self, key, headers, *, include_images=False):
            return {'revision': '1', 'unread_items': [], 'blocks': [
                {'block_id': key + 'text', 'section': '', 'text': '日报正文'}], 'image_blocks': [
                {'token': key + suffix, 'document_id': key, 'revision': '1',
                 'block_id': key + suffix, 'section': '', 'caption': ''}
                for suffix in (('img1', 'img2') if key == 'doc1' else ('img1',))
            ]}

        def _download_image(self, token, headers):
            downloads.append(token)
            return {'status': 'downloaded', 'data': token}

        def _get(self, path, headers=None):
            return {'document': {'revision_id': '1'}}

    monkeypatch.setattr(feishu_folders, 'SharedFolder', LocalFolder)
    with closing(Store(tmp_path)) as store:
        args = (store, {'user_id': 'researcher'}, 'project',
                {'url': 'https://example.feishu.cn/drive/folder/journals'})
        stamp = '2026-11-19T12:00:00+08:00'
        readers = {'image_reader': str, 'sol_image_reader': str, 'image_workers': 1}
        feishu_folders.folder_batch(*args, documents[:1], stamp, only_image_ids=['doc1img1'], **readers)
        timeline = []
        feishu_folders.folder_batch(*args, documents, stamp, **readers,
                                    on_image_progress=lambda *entry: timeline.append(entry))

    assert downloads == ['doc1img1', 'doc1img2', 'doc2img1']
    assert timeline == [(1, 2, 1, '日报一'), (2, 2, 1, '日报一'),
                        (0, 1, 0, '日报二'), (1, 1, 0, '日报二')]
