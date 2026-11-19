"""Test the local browser entry without opening real project data or cloud services."""
import re
from smtplib import SMTPAuthenticationError
from threading import Thread

import httpx

from neuroei.desktop import Desktop
from neuroei.portal import make_server
from neuroei.store import Store


def test_desktop_bootstrap_and_settings(tmp_path, monkeypatch):
    desktop = Desktop(tmp_path)
    store = Store(desktop.store_path)
    desktop.initialize(store)
    server = make_server(store, 0, desktop=desktop, legacy_execution_enabled=False)
    desktop.host = f'localhost:{server.server_port}'
    desktop.origin = 'http://' + desktop.host
    browser_opens = []
    monkeypatch.setattr('neuroei.desktop.webbrowser.open', browser_opens.append)
    messages = []
    credentials = []

    class TestSMTP:
        def __init__(self, host, port, *, timeout, context):
            assert (host, port) == ('smtp.example.invalid', 465)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def login(self, address, code):
            if code == 'invalid-code-fixture':
                raise SMTPAuthenticationError(535, b'private upstream detail')
            credentials.append((address, code))

        def send_message(self, message):
            messages.append(message)

    monkeypatch.setattr('neuroei.account_email.smtplib.SMTP_SSL', TestSMTP)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == '127.0.0.1'
        assert desktop.existing_instance()
        with httpx.Client(base_url=desktop.origin, trust_env=False) as client:
            assert client.get('/desktop/settings').status_code == 303
            assert client.get('/api/desktop/models').status_code == 401
            assert client.get('/desktop/mail').status_code == 303
            assert client.get('/api/desktop/mail').status_code == 401
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
            assert '邮件设置' in client.get('/').text
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
            mail_page = client.get('/desktop/mail')
            assert mail_page.status_code == 200 and '发送测试邮件' in mail_page.text
            csrf = re.search(r'<script nonce="([^"]+)"', mail_page.text)[1]
            headers = {**origin, 'X-Review-Token': csrf}
            address = 'my-inbox@example.invalid'
            code = 'synthetic-smtp-authorization-code'
            details = {'address': address, 'host': 'smtp.example.invalid', 'port': 465,
                       'security': 'ssl', 'authorization_code': code}
            assert client.post('/api/desktop/mail/test', json={}, headers=headers).status_code == 400
            assert client.post('/api/desktop/mail', json=details, headers=origin).status_code == 403
            state = client.post('/api/desktop/mail', json=details, headers=headers)
            assert state.status_code == 200 and state.json()['configured']
            assert 'authorization_code' not in state.json()
            assert code not in (tmp_path / 'mail-settings.json').read_text(encoding='utf-8')
            assert client.post('/api/desktop/mail/test', json={'address': 'other@example.invalid'},
                               headers=headers).status_code == 400
            assert client.post('/api/desktop/mail/test', json={}, headers=headers).json() == {'sent': True}
            assert credentials == [(address, code)]
            message, = messages
            assert message['From'] == message['To'] == address
            client.post('/api/desktop/mail', headers=headers, json={**details, 'authorization_code': 'invalid-code-fixture'}).raise_for_status()
            failed = client.post('/api/desktop/mail/test', json={}, headers=headers)
            assert failed.status_code == 400 and 'SMTP 登录失败' in failed.json()['error']
            assert 'private upstream detail' not in failed.text
            client.post('/api/desktop/mail', headers=headers, json={'remove': True}).raise_for_status()
            assert not client.get('/api/desktop/mail').json()['configured']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        store.close()
