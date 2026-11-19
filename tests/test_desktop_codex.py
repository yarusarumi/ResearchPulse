"""Verify the desktop Codex transport without logging in or using model credits."""
import subprocess
import sys

from neuroei import desktop_codex


def test_catalog_reads_active_account_and_all_visible_model_pages(tmp_path, monkeypatch):
    runner = tmp_path / 'codex.exe'
    runner.touch()
    fake = tmp_path / 'app_server.py'
    fake.write_text(
        """
import json, sys

def reply(request, result):
    print(json.dumps({'id': request['id'], 'result': result}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    params = request.get('params', {})
    if method == 'initialize':
        assert params['clientInfo']['name'] == 'neuroei-desktop'
        reply(request, {'userAgent': 'fixture'})
    elif method == 'initialized':
        assert 'id' not in request
    elif method == 'account/read':
        assert not params['refreshToken']
        reply(request, {'account': {'type': 'chatgpt', 'email': 'private@example.test'}})
    elif method == 'model/list':
        assert params['includeHidden'] is False
        if 'cursor' not in params:
            data = [{'id': 'sku', 'model': 'example-main', 'displayName': 'Example',
                     'isDefault': True, 'inputModalities': ['text', 'image'],
                     'supportedReasoningEfforts': [{'reasoningEffort': 'medium'},
                                                   {'reasoningEffort': 'xhigh'}],
                     'defaultReasoningEffort': 'xhigh'}]
            reply(request, {'data': data, 'nextCursor': 'page-2'})
        else:
            assert params['cursor'] == 'page-2'
            data = [{'model': 'example-secondary', 'inputModalities': ['text']},
                    {'model': 'unavailable', 'hidden': True},
                    {'model': 'audio-only', 'inputModalities': ['audio']},
                    {'model': '--option'}]
            reply(request, {'data': data, 'nextCursor': None})
""",
        encoding='utf-8',
    )
    launch = subprocess.Popen

    def fake_server(command, **kwargs):
        assert command[:4] == [str(runner), 'app-server', '--listen', 'stdio://']
        return launch([sys.executable, '-u', str(fake)], **kwargs)

    monkeypatch.setattr(desktop_codex.subprocess, 'Popen', fake_server)
    result = desktop_codex.catalog(runner)
    assert result['installed'] and result['signed_in']
    assert result['models'] == [
        {'id': 'example-main', 'label': 'Example', 'efforts': ['medium', 'xhigh'],
         'default_effort': 'xhigh', 'is_default': True, 'supports_images': True},
        {'id': 'example-secondary', 'label': 'example-secondary', 'efforts': [],
         'default_effort': None, 'is_default': False, 'supports_images': False},
    ]
    assert 'private@example.test' not in str(result)
    missing = desktop_codex.catalog(None)
    assert missing['models'] == [] and not missing['installed']


def test_browser_login_starts_only_one_official_subprocess(tmp_path, monkeypatch):
    runner = tmp_path / 'codex.exe'
    runner.touch()
    calls = []

    class Login:
        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == desktop_codex._LOGIN_TIMEOUT

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return Login()

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'private-api-key')
    monkeypatch.setattr(desktop_codex, '_login_process', None)
    monkeypatch.setattr(desktop_codex.subprocess, 'Popen', start)
    assert not desktop_codex.start_login(None)['started']
    assert desktop_codex.start_login(runner)['started']
    assert not desktop_codex.start_login(runner)['started']
    (command, options), = calls
    assert command == [str(runner), 'login']
    assert options['stdin'] == options['stdout'] == options['stderr'] == subprocess.DEVNULL
    assert 'DEEPSEEK_API_KEY' not in options['env']
