"""Test the local browser entry without opening real project data or cloud services."""
import re
from threading import Thread

import httpx

from neuroei.desktop import Desktop
from neuroei.portal import make_server
from neuroei.store import Store


def test_desktop_bootstrap_and_model_settings(tmp_path, monkeypatch):
    desktop = Desktop(tmp_path)
    store = Store(desktop.store_path)
    desktop.initialize(store)
    server = make_server(store, 0, desktop=desktop, legacy_execution_enabled=False)
    desktop.host = f'localhost:{server.server_port}'
    desktop.origin = 'http://' + desktop.host
    browser_opens = []
    monkeypatch.setattr('neuroei.desktop.webbrowser.open', browser_opens.append)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == '127.0.0.1'
        assert desktop.existing_instance()
        with httpx.Client(base_url=desktop.origin, trust_env=False) as client:
            assert client.get('/desktop/settings').status_code == 303
            assert client.get('/api/desktop/models').status_code == 401
            assert client.get('/api/desktop/instance', headers={'Host': 'other.test'}).status_code == 403
            assert client.get('/account/register').status_code == 404
            desktop.open_browser()
            launch_token = browser_opens[0].split('#')[1]
            body = {'token': launch_token}
            endpoint = '/api/desktop/open'
            assert client.post(endpoint, json=body, headers={'Origin': 'https://other.test'}).status_code == 403
            origin = {'Origin': desktop.origin}
            opened = client.post(endpoint, json=body, headers=origin)
            assert opened.status_code == 200
            assert 'HttpOnly; SameSite=Lax' in opened.headers['Set-Cookie']
            assert client.post(endpoint, json=body, headers=origin).status_code == 401
            assert not desktop.open_file.exists()
            assert '模型设置' in client.get('/').text
            assert '绑定邮箱' not in client.get('/').text
            assert client.get('/api/me').json()['user']['role'] == 'member'
            assert client.get('/api/admin/accounts').status_code == 403
            page = client.get('/desktop/settings')
            assert page.status_code == 200
            csrf = re.search(r'<script nonce="([^"]+)"', page.text)[1]
            key = 'test-local-api-key'
            state = client.post('/api/desktop/models', headers={**origin, 'X-Review-Token': csrf},
                                json={'backend': 'deepseek', 'deepseek_api_key': key})
            assert state.status_code == 200 and state.json()['deepseek_configured']
            assert key not in state.text and key not in (tmp_path / 'model-settings.json').read_text(encoding='utf-8')
            capability = client.get('/api/recommendation-jobs').json()['worker']
            assert capability['desktop'] and capability['model_configured']
            action = client.post('/api/recommendation-job', headers={**origin, 'X-Review-Token': csrf},
                                 json={'action': 'progress', 'auto_email': True})
            assert action.status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        store.close()
