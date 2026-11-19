"""Desktop auth and model choices at the real local HTTP boundary."""
import os
import re
import subprocess
from pathlib import Path
from threading import Thread

import httpx
import pytest

from neuroei import desktop_models
from neuroei.desktop import Desktop
from neuroei.portal import make_server
from neuroei.store import Store


@pytest.fixture
def codex_portal(tmp_path, monkeypatch):
    desktop = Desktop(tmp_path)
    store = Store(desktop.store_path)
    desktop.initialize(store)
    launch_calls, saves = [], []

    def catalog(profile):
        assert profile == desktop.profile
        return {'installed': True, 'signed_in': bool(launch_calls), 'models': [
            {'id': 'example-standard', 'label': 'Standard fixture', 'efforts': ['low', 'medium'],
             'default_effort': 'low', 'is_default': True, 'supports_images': False},
            {'id': 'example-strong', 'label': 'Strong fixture', 'efforts': ['medium', 'high'],
             'default_effort': 'medium', 'is_default': False, 'supports_images': True},
        ] if launch_calls else []}

    def login(profile):
        assert profile == desktop.profile
        launch_calls.append(profile)
        return {'started': True, 'message': '请在 OpenAI 页面完成登录。'}

    def save(profile, data):
        assert profile == desktop.profile
        saves.append(data)
        return {**settings, **data}

    settings = {'backend': 'sol-high', 'codex_installed': True, 'deepseek_configured': False,
                'fallback_enabled': False, 'codex_model': '', 'codex_effort': ''}
    monkeypatch.setattr(desktop_models, 'codex_catalog', catalog, raising=False)
    monkeypatch.setattr(desktop_models, 'start_codex_login', login, raising=False)
    monkeypatch.setattr(desktop_models, 'save', save)
    monkeypatch.setattr(desktop, 'model_settings', lambda: settings)
    launch_urls = []
    monkeypatch.setattr('neuroei.desktop.webbrowser.open', launch_urls.append)

    server = make_server(store, 0, desktop=desktop, legacy_execution_enabled=False)
    desktop.host = f'localhost:{server.server_port}'
    desktop.origin = 'http://' + desktop.host
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        desktop.open_browser()
        yield desktop.origin, launch_urls[0], launch_calls, saves, store
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        store.close()


def test_codex_login_requires_local_session_and_csrf(codex_portal):
    origin, browser_url, calls, saves, store = codex_portal
    header = {'Origin': origin}
    login = '/api/desktop/codex/login'
    catalog = '/api/desktop/codex/models'
    with httpx.Client(base_url=origin, trust_env=False) as client:
        assert client.get(catalog).status_code == 401
        assert client.post(login, headers=header, json={}).status_code == 401
        session = client.post('/api/desktop/open', headers=header, json={'token': browser_url.split('#')[1]})
        assert session.status_code == 200
        page = client.get('/desktop/settings')
        assert page.status_code == 200
        header['X-Review-Token'] = re.search(r'<script nonce="([^"]+)"', page.text)[1]
        assert client.post(login, headers={'Origin': origin}, json={}).status_code == 403
        assert client.post(login, headers=header, json={'password': 'example'}).status_code == 400
        store.put('recommendation_job', 'test-running-job', {'status': 'processing'})
        assert client.post(login, headers=header, json={}).status_code == 409
        store.put('recommendation_job', 'test-running-job', {'status': 'completed'})
        assert not calls
        assert client.post(login, headers=header, json={}).json()['started']
        assert len(calls) == 1
        models = client.get(catalog).json()
        assert models['signed_in'] and [item['id'] for item in models['models']] == [
            'example-standard', 'example-strong']
        params = {'backend': 'sol-high', 'codex_model': 'example-strong', 'codex_effort': 'high'}
        assert client.post('/api/desktop/models', headers=header, json=params).json()['codex_model'] == 'example-strong'
        assert saves == [params]


@pytest.mark.skipif(os.environ.get('NEUROEI_BROWSER_TESTS') != '1', reason='opt-in desktop browser test')
def test_codex_login_and_model_picker_in_edge(codex_portal):
    _, browser_url, calls, saves, _ = codex_portal
    script = Path(__file__).with_suffix('.cjs')
    result = subprocess.run(['node', str(script), browser_url], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=70)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    assert [entry['codex_model'] for entry in saves] == ['example-strong', '']
    assert [entry['codex_effort'] for entry in saves] == ['high', '']
