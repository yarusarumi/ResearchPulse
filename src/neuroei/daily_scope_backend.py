"""Opt-in, same-host Codex pilot for ambiguous daily date attribution only."""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from .llm import ModelUnavailable

OPTIONS = {'deepseek': 'DeepSeek', 'astra-medium': 'Astra · medium', 'astra-high': 'Astra · high'}
SCOPE = '仅用于飞书日报日期归属复核；项目分析、论文推荐与深读仍使用原有模型。'


def uses_personal_selection():
    return not os.environ.get('NEUROEI_AI_BACKEND_FILE')


def authorized(user_id):
    return uses_personal_selection() and user_id in {v.strip() for v in os.environ.get('NEUROEI_CODEX_SCOPE_USERS', '').split(',') if v.strip()}


def runner_environment():
    # Keep the host login location and OS runtime, never the service's API secrets.
    names = {'PATH', 'SYSTEMROOT', 'WINDIR', 'PATHEXT', 'COMSPEC', 'LOCALAPPDATA',
             'APPDATA', 'USERPROFILE', 'HOME', 'TEMP', 'TMP', 'LANG', 'LC_ALL',
             'CODEX_HOME', 'XDG_CONFIG_HOME', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
             'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'}
    return {key: value for key, value in os.environ.items() if key.upper() in names}


def runner_ready():
    value = os.environ.get('NEUROEI_CODEX_SCOPE_RUNNER', '')
    executable = Path(value)
    if not value or not executable.is_absolute() or not executable.is_file():
        return False
    try:
        result = subprocess.run([str(executable), 'login', 'status'], capture_output=True, check=False,
                                text=True, encoding='utf-8', env=runner_environment(), timeout=5)
        return result.returncode == 0 and 'Logged in using ChatGPT' in result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired):
        return False


def selected(store, user_id):
    try:
        return store.get('daily_scope_preference', user_id)['backend']
    except ValueError:
        return 'deepseek'


def settings(store, user_id):
    if not authorized(user_id):
        return None
    return {'selected': selected(store, user_id), 'codex_available': runner_ready(),
            'options': OPTIONS, 'scope': SCOPE}


def save_preference(store, user_id, body):
    if not authorized(user_id):
        raise PermissionError('此账号未启用日报日期复核模型试用。')
    if set(body) != {'backend'} or body['backend'] not in OPTIONS:
        raise ValueError('请选择日报日期复核模型。')
    if body['backend'] != 'deepseek' and not runner_ready():
        raise ValueError('当前服务器尚未配置可用的 Codex 登录，未更改选择。')
    store.put('daily_scope_preference', user_id, {'backend': body['backend']})
    return settings(store, user_id)


class CodexScopeClient:
    def __init__(self, store, user_id, job_id, backend, should_stop=None):
        if backend not in ('astra-medium', 'astra-high') or not authorized(user_id) or not runner_ready():
            raise ModelUnavailable('scope_backend_unavailable', 'Selected daily-scope backend is unavailable')
        self.store, self.user_id, self.job_id = store, user_id, job_id
        self.backend, self.should_stop, self.calls = backend, should_stop, 0

    def structured(self, payload, schema, instruction):
        if not authorized(self.user_id):
            raise ModelUnavailable('scope_backend_unavailable', 'Daily-scope authorization was removed')
        if self.should_stop and self.should_stop():
            raise ModelUnavailable('canceled_by_user', 'Daily-scope review canceled')
        self.calls += 1
        started = time.monotonic()
        audit = {'user_id': self.user_id, 'job_id': self.job_id, 'backend': self.backend,
                 'scope': 'ambiguous_daily_date_attribution', 'model': 'gpt-6-astra',
                 'reasoning_effort': self.backend.split('-')[1], 'auth': 'same_host_chatgpt',
                 'payload': payload, 'instruction': instruction, 'status': 'failed'}
        try:
            with tempfile.TemporaryDirectory(prefix='neuroei-codex-scope-') as directory:
                schema_path = Path(directory) / 'review.json'
                schema_path.write_text(json.dumps(schema.model_json_schema()), encoding='utf-8')
                result = subprocess.run([os.environ['NEUROEI_CODEX_SCOPE_RUNNER'], 'exec', '--ephemeral',
                    '--ignore-user-config', '--skip-git-repo-check', '--sandbox', 'read-only',
                    '--disable', 'shell_tool', '-c', 'web_search="disabled"', '--cd', directory,
                    '--model', 'gpt-6-astra', '-c', f'model_reasoning_effort="{audit["reasoning_effort"]}"',
                    '--output-schema', str(schema_path), '--json', '-'],
                    input=instruction + '\n仅返回结构化判断，不调用工具。\n' + payload,
                    capture_output=True, check=False, text=True, encoding='utf-8', env=runner_environment(), timeout=120)
            if self.should_stop and self.should_stop():
                raise ModelUnavailable('canceled_by_user', 'Daily-scope review canceled')
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            completed = [e for e in events if e.get('type') == 'turn.completed']
            messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                        and e.get('item', {}).get('type') == 'agent_message']
            if result.returncode or len(completed) != 1 or not messages:
                raise ValueError('Incomplete Codex result')
            answer = schema.model_validate_json(messages[-1])
            audit.update(status='completed', result=answer.model_dump(mode='json'), usage=completed[0].get('usage'))
            return answer
        except ModelUnavailable:
            raise
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
            raise ModelUnavailable('scope_backend_unavailable', 'Codex date review unavailable; no backend substitution') from None
        finally:
            audit['duration_seconds'] = round(time.monotonic() - started, 3)
            self.store.put('daily_scope_model_call', self.job_id + ':' + uuid4().hex, audit)


def scope_client(store, user_id, job_id, backend, deepseek, should_stop=None):
    if backend == 'deepseek':
        return deepseek
    return CodexScopeClient(store, user_id, job_id, backend, should_stop)
