"""Read the local Codex catalog and start its official browser sign-in."""
import json
import re
import subprocess
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic

from .daily_scope_backend import runner_environment

MODEL_ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}')
EFFORT_PATTERN = re.compile(r'[a-z][a-z0-9_-]{0,23}')
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
_CATALOG_TIMEOUT = 15
_LOGIN_TIMEOUT = 600
_LOGIN_LOCK = Lock()
_login_process = None


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _read_messages(stream, results):
    try:
        for line in stream:
            try:
                value = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            if isinstance(value, dict):
                results.put(value)
    finally:
        results.put(None)


def _rows(data):
    models = {}
    for row in data:
        if not isinstance(row, dict) or row.get('hidden'):
            continue
        model = row.get('model') or row.get('id')
        modalities = row.get('inputModalities', ['text', 'image'])
        if (not isinstance(model, str) or not MODEL_ID_PATTERN.fullmatch(model)
                or not isinstance(modalities, list) or 'text' not in modalities):
            continue
        efforts = []
        for item in row.get('supportedReasoningEfforts') or ():
            effort = item.get('reasoningEffort') if isinstance(item, dict) else None
            if isinstance(effort, str) and EFFORT_PATTERN.fullmatch(effort) and effort not in efforts:
                efforts.append(effort)
        default_effort = row.get('defaultReasoningEffort')
        if default_effort not in efforts:
            default_effort = efforts[0] if efforts else None
        label = row.get('displayName')
        tiers = row.get('serviceTiers') or ()
        fast = 'fast' in (row.get('additionalSpeedTiers') or ()) or any(
            isinstance(tier, dict) and tier.get('id') in ('priority', 'fast') for tier in tiers)
        models[model] = {'id': model, 'label': label.strip()[:120] if isinstance(label, str) and label.strip() else model,
                         'efforts': efforts, 'default_effort': default_effort, 'is_default': row.get('isDefault') is True,
                         'supports_images': 'image' in modalities, 'supports_fast': fast}
    return list(models.values())


def _quota(result):
    buckets = result.get('rateLimitsByLimitId')
    codex = buckets.get('codex') if isinstance(buckets, dict) else None
    snapshot = codex if isinstance(codex, dict) else result.get('rateLimits')
    if not isinstance(snapshot, dict):
        return {}
    windows = {}
    for key in ('primary', 'secondary'):
        window = snapshot.get(key)
        if not isinstance(window, dict):
            continue
        name = {300: 'five_hour', 10080: 'weekly'}.get(window.get('windowDurationMins'))
        used = window.get('usedPercent')
        if name and type(used) in (int, float) and 0 <= used <= 100:
            reset = window.get('resetsAt')
            windows[name] = {'remaining_percent': round(100 - used, 1),
                             'resets_at': reset if type(reset) is int and reset > 0 else None}
    return windows


def _read_catalog(runner):
    arguments = [str(runner), 'app-server', '--listen', 'stdio://', '-c', 'model_provider="openai"']
    with subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding='utf-8', errors='replace', env=runner_environment(),
                          creationflags=_NO_WINDOW) as process:
        messages = Queue()
        reader = Thread(target=_read_messages, args=(process.stdout, messages), daemon=True)
        reader.start()
        deadline = monotonic() + _CATALOG_TIMEOUT
        request_id = 0

        def send(method, params=None):
            nonlocal request_id
            message = {'method': method}
            if params is not None:
                request_id += 1
                message.update(id=request_id, params=params)
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()
            if params is None:
                return None
            while monotonic() < deadline:
                response = messages.get(timeout=max(0, deadline - monotonic()))
                if response is None:
                    break
                if response.get('id') == request_id:
                    result = response.get('result')
                    if isinstance(result, dict):
                        return result
                    break
            raise ValueError('Codex app-server 请求未完成。')

        try:
            send('initialize', {'clientInfo': {'name': 'neuroei-desktop', 'title': 'ResearchPulse', 'version': '0.1.0'}})
            send('initialized')
            current = send('account/read', {'refreshToken': False})
            account = current.get('account')
            if not isinstance(account, dict) or account.get('type') != 'chatgpt':
                return {'signed_in': False}
            label = next((value.strip()[:254] for key in ('displayName', 'name', 'email')
                          if isinstance(value := account.get(key), str) and value.strip()), '')
            data = []
            params = {'limit': 100, 'includeHidden': False}
            state = {'signed_in': True, 'account_label': label}
            try:
                while True:
                    page = send('model/list', params)
                    rows = page.get('data')
                    if not isinstance(rows, list):
                        raise ValueError('Codex 模型列表格式无效。')
                    data.extend(rows)
                    cursor = page.get('nextCursor')
                    if not cursor:
                        state['models'] = _rows(data)
                        break
                    if not isinstance(cursor, str) or cursor == params.get('cursor'):
                        raise ValueError('Codex 模型列表分页无效。')
                    params['cursor'] = cursor
            except (Empty, OSError, ValueError):
                pass
            try:
                state['quota'] = _quota(send('account/rateLimits/read', {}))
            except (Empty, OSError, ValueError):
                pass
            return state
        finally:
            _stop(process)
            reader.join(timeout=3)


def catalog(runner: Path | None) -> dict:
    """Return the local account label, model choices and Codex quota windows."""
    installed = runner is not None and runner.is_absolute() and runner.is_file()
    state = {'installed': installed, 'signed_in': False, 'account_label': '', 'models': [], 'quota': {}}
    if not installed:
        return {**state, 'message': '请先安装 Codex CLI。'}
    try:
        state.update(_read_catalog(runner))
    except (Empty, OSError, ValueError):
        from .codex_model_client import codex_login_ready

        if codex_login_ready(runner):
            return {**state, 'signed_in': True, 'message': '已登录 Codex，但暂时无法读取模型列表，请重试。'}
        return {**state, 'message': '暂时无法确认 Codex 登录，请检查网络或更新 Codex CLI。'}
    message = ('已读取当前账号的 Codex 模型。' if state['models'] else
               '当前账号未返回可选模型。' if state['signed_in'] else '请先用 ChatGPT 账号登录 Codex。')
    return {**state, 'message': message}


def _watch_login(process):
    try:
        process.wait(timeout=_LOGIN_TIMEOUT)
    except subprocess.TimeoutExpired:
        _stop(process)


def start_login(runner: Path | None) -> dict:
    """Delegate sign-in to Codex. Its local callback and browser own all credentials."""
    global _login_process

    if runner is None or not runner.is_absolute() or not runner.is_file():
        return {'started': False, 'message': '请先安装 Codex CLI。'}
    with _LOGIN_LOCK:
        if _login_process is not None and _login_process.poll() is None:
            return {'started': False, 'message': 'Codex 登录已在进行中，请在浏览器完成。'}
        try:
            _login_process = subprocess.Popen(
                [str(runner), 'login'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=runner_environment(), creationflags=_NO_WINDOW,
            )
        except OSError:
            return {'started': False, 'message': '无法启动 Codex 登录，请检查 Codex CLI。'}
        Thread(target=_watch_login, args=(_login_process,), daemon=True).start()
    return {'started': True, 'message': '请在浏览器打开的 OpenAI 页面完成登录，然后回来刷新模型列表。'}
