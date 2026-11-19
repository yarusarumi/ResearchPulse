"""Browser release check for the guided setup, using an isolated Feishu fixture."""
import json
import os
import subprocess
import threading
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from neuroei import feishu_folders, feishu_user_auth
from neuroei.portal import make_server
from neuroei.store import Store
from test_feishu_user_auth import identity, member, oauth_env, tokens  # noqa: F401


@pytest.mark.skipif(os.environ.get('NEUROEI_BROWSER_TESTS') != '1', reason='Opt-in Playwright release gate')
@pytest.mark.parametrize('desktop', [False, True], ids=['hosted', 'loopback'])
def test_feishu_onboarding(oauth_env, monkeypatch, tmp_path, desktop):
    url = 'https://example.feishu.cn/drive/folder/journal'
    entries = [
        {'token': 'recent', 'type': 'docx', 'readable': True, 'path': ['近期日报'], 'parents': []},
        {'token': 'history', 'type': 'docx', 'readable': True, 'path': ['历史日报'], 'parents': []},
        {'token': 'shortcut', 'type': 'shortcut', 'readable': False, 'path': ['旧快捷方式'], 'parents': []},
    ]

    def browse(store, account, folder):
        feishu_user_auth.authorization(store, account)
        if folder != url:
            raise feishu_folders.FolderUnavailable('当前飞书账号无权读取此目录。')
        return {'url': folder, 'entries': entries}

    real_complete = feishu_user_auth.complete

    def complete(store, account, session, state, code, *, error=None):
        width = 375 if account['username'] == 'mobile-onboarding' else 1440

        def respond(request):
            if request.url.path.endswith('/oauth/token'):
                payload = json.loads(request.content)
                assert payload['client_id'] == f'cli_fixture_{width}'
                assert payload['client_secret'] == 'fixture-secret'
                assert payload['code'] == 'fixture-code'
                if desktop:
                    assert payload['redirect_uri'] == root + '/oauth/feishu/callback'
                return httpx.Response(200, json={'code': 0, **tokens(account['username'])})
            assert request.url.path.endswith('/user_info')
            return httpx.Response(200, json={'code': 0, 'data': identity(account['username'])})

        return real_complete(store, account, session, state, code, error=error, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(feishu_folders, 'browse', browse)
    monkeypatch.setattr(feishu_user_auth, 'complete', complete)
    with closing(Store(tmp_path / 'store')) as store:
        for username in ('desktop-onboarding', 'mobile-onboarding'):
            member(store, username)
        server = make_server(store, 0, legacy_execution_enabled=False)
        root = f'http://{"localhost" if desktop else "127.0.0.1"}:{server.server_port}'
        if desktop:
            # Use a free port for concurrent test runs; the application requires 8768.
            monkeypatch.setattr(feishu_user_auth, 'DESKTOP_ORIGIN', root)
            monkeypatch.setenv('NEUROEI_DESKTOP', '1')
            monkeypatch.setenv('NEUROEI_PUBLIC_ORIGIN', root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                ['node', str(Path(__file__).with_name('feishu_onboarding.cjs')), root,
                 'loopback' if desktop else 'hosted'],
                capture_output=True, encoding='utf-8', errors='replace', timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert [row['feishu'] for row in store.list('research_project')] == [
                {'url': url, 'selection': ['history', 'recent']},
                {'url': url, 'selection': ['history', 'recent']},
            ]
            assert not store.list('recommendation_job')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
