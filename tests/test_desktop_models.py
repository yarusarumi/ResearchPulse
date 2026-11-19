import json
import os
import subprocess
import sys

import httpx
import pytest

from neuroei import ai_backend, desktop_models
from neuroei.codex_model_client import CodexModelClient
from neuroei.config import LLMConfig
from neuroei.llm import ModelFailure
from neuroei.store import Store


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Requires real Windows DPAPI')


@pytest.fixture
def profile(tmp_path, monkeypatch):
    for name in ('NEUROEI_DESKTOP_PROFILE', 'DEEPSEEK_API_KEY', 'NEUROEI_AI_BACKEND',
                 'NEUROEI_AI_BACKEND_FILE', 'NEUROEI_AI_CODEX_RUNNER'):
        monkeypatch.setenv(name, 'ignored-server-setting')
    monkeypatch.setattr(desktop_models, 'local_codex_runner', lambda: None)
    profile = tmp_path / 'profile'
    desktop_models.prepare(profile)
    return profile


def test_deepseek_only_uses_dpapi_key_after_new_process(profile, tmp_path, monkeypatch):
    key = 'example-local-only-key'
    assert desktop_models.load(profile)['backend'] == 'sol-high'
    assert not ai_backend.deepseek_available()
    assert ai_backend.runner() is None
    saved = desktop_models.save(profile, {'deepseek_api_key': key})
    assert saved == {'backend': 'deepseek', 'codex_installed': False, 'deepseek_configured': True,
                     'fallback_enabled': False, 'codex_model': '', 'codex_effort': '', 'codex_supports_images': True,
                     'codex_fast': False}
    assert key not in (profile / 'model-settings.json').read_text(encoding='utf-8')
    assert key not in json.dumps(saved)
    assert 'DEEPSEEK_API_KEY' not in os.environ

    result = subprocess.run([sys.executable, '-c',
                             'from neuroei import ai_backend; print(ai_backend.selected(), ai_backend.deepseek_available())'],
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == 'deepseek True'

    def fake_get(url, *, headers, timeout, follow_redirects):
        assert url == 'https://api.deepseek.com/models'
        assert headers['Authorization'] == 'Bearer ' + key and not follow_redirects
        return httpx.Response(200, json={'data': [{'id': 'deepseek-flash'}]}, request=httpx.Request('GET', url))

    monkeypatch.setattr(httpx, 'get', fake_get)
    assert desktop_models.probe(profile, 'deepseek')['ready']

    def complete(request):
        assert request.headers['Authorization'] == 'Bearer ' + key
        return httpx.Response(200, json={'model': 'deepseek-flash', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'primary completed'}}]})

    client_class = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda *args, **kw: client_class(transport=httpx.MockTransport(complete)))
    store = Store(tmp_path / 'store')
    with ai_backend.task_selection(lambda: pytest.fail('DeepSeek primary must not report a fallback')):
        desktop_models.save(profile, {'remove_deepseek': True})
        assert ai_backend.selected() == 'deepseek' and ai_backend.deepseek_available()
        with_client = ai_backend.text_client(LLMConfig(), store, 'dp-primary', backend=ai_backend.selected())
        try:
            assert with_client.complete('question', system='testing') == 'primary completed'
        finally:
            with_client.close()
    assert ai_backend.selected() == 'sol-high' and not ai_backend.deepseek_available()
    store.close()


def test_stays_on_sol_until_next_job_and_only_falls_back_with_consent(profile, tmp_path, monkeypatch):
    executable = tmp_path / 'codex.exe'
    executable.touch()
    monkeypatch.setattr(desktop_models, 'local_codex_runner', lambda: executable)
    monkeypatch.setattr('neuroei.codex_model_client.codex_login_ready', lambda path: path == executable)
    assert desktop_models.probe(profile, 'sol-high')['ready']
    assert desktop_models.save(profile, {})['backend'] == 'sol-high'

    desktop_models.save(profile, {'deepseek_api_key': 'local-fallback-key'})
    assert ai_backend.selected() == 'sol-high' and ai_backend.deepseek_available()

    def quota_error(self, *args, **kwargs):
        self.calls += 1
        raise ModelFailure('quota reached', reason='model_quota_exhausted')

    def complete(request):
        assert request.headers['Authorization'] == 'Bearer local-fallback-key'
        return httpx.Response(200, json={'model': 'deepseek-flash', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'fallback completed'}}]})

    monkeypatch.setattr(CodexModelClient, 'complete', quota_error)
    client_class = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda *args, **kw: client_class(transport=httpx.MockTransport(complete)))
    store = Store(tmp_path / 'store')
    reports = []
    with ai_backend.task_selection(lambda: reports.append('fallback')):
        desktop_models.save(profile, {'fallback_enabled': True})
        client = ai_backend.text_client(LLMConfig(), store, 'sol-primary', backend=ai_backend.selected())
        assert isinstance(client, CodexModelClient) and client.config.model == 'codex-default'
        with pytest.raises(ModelFailure, match='quota reached'):
            client.complete('question', system='testing')
        client.close()

    with ai_backend.task_selection(lambda: reports.append('fallback')):
        client = ai_backend.text_client(LLMConfig(), store, 'consented', backend=ai_backend.selected())
        try:
            assert client.complete('question', system='testing') == 'fallback completed'
            assert client.fallback_used
        finally:
            client.close()
    assert reports == ['fallback']
    store.close()


def test_codex_catalog_choice_reaches_job_and_preserves_image_capability(profile, tmp_path, monkeypatch):
    def model_catalog(_):
        return {'models': [{'id': 'gpt-6.2-astra', 'efforts': ['medium', 'high'], 'is_default': True,
                            'default_effort': 'medium', 'supports_images': False, 'supports_fast': True}]}

    monkeypatch.setattr(desktop_models, 'codex_catalog', model_catalog)
    with pytest.raises(ValueError, match='未提供'):
        desktop_models.save(profile, {'codex_model': 'other', 'codex_effort': 'medium'})
    with pytest.raises(ValueError, match='不支持'):
        desktop_models.save(profile, {'codex_model': 'gpt-6.2-astra', 'codex_effort': 'ultra'})
    saved = desktop_models.save(profile, {'codex_model': 'gpt-6.2-astra', 'codex_effort': ''})
    assert saved['codex_effort'] == 'medium' and not saved['codex_supports_images']
    store = Store(tmp_path / 'store')
    try:
        with ai_backend.task_selection(lambda: None):
            desktop_models.save(profile, {'codex_model': '', 'codex_effort': ''})
            config = ai_backend.model_config(LLMConfig(), ai_backend.selected())
            assert config.model == 'gpt-6.2-astra' and config.reasoning_effort == 'medium'
            client = ai_backend.text_client(LLMConfig(), store, 'image-capability', backend=ai_backend.selected())
            assert not client.supports_images
            client.close()
        default = ai_backend.model_config(LLMConfig(), ai_backend.selected())
        assert default.model == 'codex-default' and default.reasoning_effort is None
        desktop_models.save(profile, {'codex_model': '', 'codex_effort': 'high', 'codex_fast': True})
        configured = ai_backend.model_config(LLMConfig(), ai_backend.selected())
        assert (configured.model, configured.reasoning_effort, configured.service_tier) == ('codex-default', 'high', 'fast')
    finally:
        store.close()


def test_codex_native_binary_resolved_from_a_custom_npm_prefix(tmp_path, monkeypatch):
    prefix = tmp_path / 'custom-node-prefix'
    package = prefix / 'node_modules' / '@openai' / 'codex-win32-x64'
    executable = package / 'vendor' / 'x86_64-pc-windows-msvc' / 'codex' / 'codex.exe'
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.delenv('LOCALAPPDATA', raising=False)
    monkeypatch.delenv('APPDATA', raising=False)
    monkeypatch.setattr('shutil.which', lambda name: str(prefix / 'codex.cmd') if name == 'codex.cmd' else None)
    assert desktop_models.local_codex_runner() == executable
