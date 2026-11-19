"""Per-user Windows credentials and model settings for the desktop workspace."""

import base64
import ctypes
import json
import os
import shutil
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import httpx


PROFILE_ENV = 'NEUROEI_DESKTOP_PROFILE'
BACKENDS = ('sol-high', 'deepseek')
_FILE_NAME = 'model-settings.json'
_SETTINGS_LOCK = Lock()


class _DataBlob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.c_void_p)]


def _dpapi(value: bytes, *, protect: bool) -> bytes:
    if os.name != 'nt':
        raise ValueError('本机凭据存储仅支持 Windows。')
    buffer = ctypes.create_string_buffer(value)
    source = _DataBlob(len(value), ctypes.cast(buffer, ctypes.c_void_p))
    target = _DataBlob()
    library = ctypes.WinDLL('crypt32')
    action = library.CryptProtectData if protect else library.CryptUnprotectData
    action.argtypes = [ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
    action.restype = wintypes.BOOL
    if not action(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ValueError('Windows 无法处理本机凭据，请在当前账户下重新配置。')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        free = ctypes.WinDLL('kernel32').LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(target.data)


def protect_secret(value: str) -> str:
    return base64.b64encode(_dpapi(value.encode('utf-8'), protect=True)).decode('ascii')


def unprotect_secret(envelope: str) -> str:
    try:
        return _dpapi(base64.b64decode(envelope, validate=True), protect=False).decode('utf-8')
    except ValueError:
        raise ValueError('Windows 无法读取本机凭据，请在当前账户下重新配置。') from None


def prepare(profile: Path) -> None:
    """Select a local profile and keep inherited server model settings out of desktop mode."""
    profile = Path(profile).resolve()
    profile.mkdir(parents=True, exist_ok=True)
    os.environ[PROFILE_ENV] = str(profile)
    for name in ('DEEPSEEK_API_KEY', 'NEUROEI_AI_BACKEND', 'NEUROEI_AI_BACKEND_FILE', 'NEUROEI_AI_CODEX_RUNNER'):
        os.environ.pop(name, None)


def active_profile() -> Path | None:
    value = os.environ.get(PROFILE_ENV)
    return Path(value) if value else None


def local_codex_runner() -> Path | None:
    if executable := shutil.which('codex.exe'):
        return Path(executable).resolve()
    if local := os.environ.get('LOCALAPPDATA'):
        desktop = Path(local) / 'OpenAI' / 'Codex' / 'bin'
        if executables := list(desktop.glob('*/codex.exe')):
            return max(executables, key=lambda path: path.stat().st_mtime).resolve()
    shim = shutil.which('codex.cmd')
    appdata = os.environ.get('APPDATA')
    npm_locations = [Path(shim).parent] if shim else []
    if appdata:
        npm_locations.append(Path(appdata) / 'npm')
    # Read through the global npm prefix without invoking the command shim.
    for location in npm_locations:
        packages = location / 'node_modules' / '@openai'
        for base in (packages, packages / 'codex' / 'node_modules' / '@openai'):
            for path in base.glob('codex-win32-*/vendor/*/codex/codex.exe'):
                return path.resolve()
    return None


@dataclass(frozen=True)
class RuntimeModels:
    backend: str
    codex_runner: Path | None
    fallback_enabled: bool
    _deepseek_envelope: str = field(repr=False)
    codex_model: str = ''
    codex_effort: str = ''
    codex_supports_images: bool = True
    codex_fast: bool = False

    def deepseek_key(self) -> str | None:
        return unprotect_secret(self._deepseek_envelope) if self._deepseek_envelope else None

    def public_settings(self) -> dict:
        return {'backend': self.backend, 'codex_installed': self.codex_runner is not None,
                'deepseek_configured': bool(self._deepseek_envelope), 'fallback_enabled': self.fallback_enabled,
                'codex_model': self.codex_model, 'codex_effort': self.codex_effort,
                'codex_supports_images': self.codex_supports_images, 'codex_fast': self.codex_fast}

    def codex_choice(self):
        return self.codex_model, self.codex_effort, self.codex_supports_images, self.codex_fast


def _read(profile: Path) -> dict:
    try:
        settings = json.loads((profile / _FILE_NAME).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except (UnicodeError, ValueError):
        raise ValueError('本机模型设置无法读取。') from None
    if (not isinstance(settings, dict) or set(settings) - {'backend', 'fallback_enabled', 'deepseek_envelope',
                                                          'codex_model', 'codex_effort', 'codex_supports_images', 'codex_fast'}
            or settings.get('backend') not in BACKENDS or type(settings.get('fallback_enabled')) is not bool
            or not isinstance(settings.get('deepseek_envelope'), str)
            or not isinstance(settings.get('codex_model', ''), str)
            or not isinstance(settings.get('codex_effort', ''), str)
            or type(settings.get('codex_supports_images', True)) is not bool
            or type(settings.get('codex_fast', False)) is not bool):
        raise ValueError('本机模型设置无法读取。')
    return settings


def runtime(profile: Path) -> RuntimeModels:
    with _SETTINGS_LOCK:
        settings = _read(profile)
    runner = local_codex_runner()
    envelope = settings.get('deepseek_envelope', '')
    backend = settings.get('backend', 'deepseek' if envelope and not runner else 'sol-high')
    return RuntimeModels(backend, runner, bool(envelope and settings.get('fallback_enabled')), envelope,
                         settings.get('codex_model', ''), settings.get('codex_effort', ''),
                         settings.get('codex_supports_images', True), settings.get('codex_fast', False))


def load(profile: Path) -> dict:
    return runtime(profile).public_settings()


def codex_catalog(profile: Path):
    from .desktop_codex import catalog

    return catalog(runtime(profile).codex_runner)


def start_codex_login(profile: Path):
    from .desktop_codex import start_login

    return start_login(runtime(profile).codex_runner)


def _write(profile: Path, settings: dict) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=profile, delete=False) as output:
        json.dump(settings, output)
        candidate = Path(output.name)
    try:
        candidate.replace(profile / _FILE_NAME)
    finally:
        candidate.unlink(missing_ok=True)


def _codex_model(profile, model):
    models = codex_catalog(profile).get('models', [])
    return next((row for row in models if (row['id'] == model if model else row.get('is_default'))), None)


def save(profile: Path, payload: dict) -> dict:
    """Apply only the submitted preferences. Empty secret input preserves the saved credential."""
    if (not isinstance(payload, dict) or set(payload) - {'backend', 'deepseek_api_key', 'remove_deepseek',
                                                         'fallback_enabled', 'codex_model', 'codex_effort', 'codex_fast'}):
        raise ValueError('本机模型设置无效。')
    for flag in ('remove_deepseek', 'fallback_enabled', 'codex_fast'):
        if flag in payload and type(payload[flag]) is not bool:
            raise ValueError('本机模型设置无效。')
    new_key = payload.get('deepseek_api_key', '')
    if not isinstance(new_key, str):
        raise ValueError('请输入有效的 DeepSeek API Key。')
    new_key = new_key.strip()
    remove_key = payload.get('remove_deepseek', False)
    if (remove_key and new_key) or len(new_key) > 512 or any(not 33 <= ord(c) <= 126 for c in new_key):
        raise ValueError('请输入有效的 DeepSeek API Key。')
    choice = None
    if set(payload) & {'codex_model', 'codex_effort', 'codex_fast'}:
        if set(payload) & {'codex_model', 'codex_effort'} != {'codex_model', 'codex_effort'}:
            raise ValueError('请选择 Codex 模型和推理强度。')
        model, effort = payload['codex_model'], payload['codex_effort']
        fast = payload.get('codex_fast', False)
        if not isinstance(model, str) or len(model) > 128 or not isinstance(effort, str) or len(effort) > 24:
            raise ValueError('请选择有效的 Codex 模型和推理强度。')
        if model or effort or fast:
            available = _codex_model(profile, model)
            if not available:
                raise ValueError('当前 Codex 登录未提供此模型，请刷新模型列表后重新选择。')
            if effort and effort not in available['efforts']:
                raise ValueError('此模型不支持所选推理强度。')
            if fast and not available.get('supports_fast'):
                raise ValueError('当前 Codex 账号和模型不支持 Fast。')
            default_effort = available['default_effort'] if model else ''
            choice = model, effort or default_effort or '', available['supports_images'], fast
        else:
            choice = '', '', True, False
    with _SETTINGS_LOCK:
        previous = _read(profile)
        runner = local_codex_runner()
        has_key = not remove_key and bool(new_key or previous.get('deepseek_envelope'))
        default = 'deepseek' if has_key and not runner else 'sol-high'
        current = 'sol-high' if remove_key else previous.get('backend', default)
        backend = payload.get('backend', current)
        fallback = payload.get('fallback_enabled', previous.get('fallback_enabled', False))
        if backend not in BACKENDS:
            raise ValueError('请选择 Codex 或 DeepSeek。')
        if (backend == 'deepseek' or payload.get('fallback_enabled')) and not has_key:
            raise ValueError('请先配置 DeepSeek API Key。')
        envelope = '' if remove_key else protect_secret(new_key) if new_key else previous.get('deepseek_envelope', '')
        if choice is None:
            choice = (previous.get('codex_model', ''), previous.get('codex_effort', ''),
                      previous.get('codex_supports_images', True), previous.get('codex_fast', False))
        settings = RuntimeModels(backend, runner, bool(has_key and fallback), envelope, *choice)
        _write(profile, {'backend': backend, 'fallback_enabled': settings.fallback_enabled, 'deepseek_envelope': envelope,
                         'codex_model': settings.codex_model, 'codex_effort': settings.codex_effort,
                         'codex_supports_images': settings.codex_supports_images, 'codex_fast': settings.codex_fast})
        return settings.public_settings()


def probe(profile: Path, backend: str) -> dict:
    """Verify a local login or API key without sending documents or running inference."""
    if backend not in BACKENDS:
        raise ValueError('请选择 Codex 或 DeepSeek。')
    settings = runtime(profile)
    messages = {
        'ready': '连接正常。',
        'codex_missing': '未找到 Codex CLI。请安装后重试。',
        'codex_login_required': '未确认 Codex 登录。请先完成 Codex CLI 登录。',
        'codex_model_unavailable': '当前 Codex 登录未提供已保存的模型，请刷新模型列表后重新选择。',
        'deepseek_missing': '请先保存 DeepSeek API Key。',
        'deepseek_key_invalid': 'DeepSeek 拒绝了当前 API Key。',
        'deepseek_model_unavailable': '当前 DeepSeek 账号的模型列表未提供 deepseek-flash。',
        'deepseek_unreachable': '暂时无法验证 DeepSeek，请检查网络后重试。',
    }
    if backend == 'sol-high':
        from .codex_model_client import codex_login_ready

        if not settings.codex_runner:
            reason = 'codex_missing'
        else:
            reason = 'ready' if codex_login_ready(settings.codex_runner) else 'codex_login_required'
            if reason == 'ready' and (settings.codex_model or settings.codex_effort or settings.codex_fast):
                current = _codex_model(profile, settings.codex_model)
                if (not current or (settings.codex_effort and settings.codex_effort not in current['efforts'])
                        or (settings.codex_fast and not current.get('supports_fast'))):
                    reason = 'codex_model_unavailable'
    else:
        reason = _probe_deepseek(settings.deepseek_key())
    return {'backend': backend, 'ready': reason == 'ready', 'reason': reason, 'message': messages[reason]}


def _probe_deepseek(key: str | None) -> str:
    if not key:
        return 'deepseek_missing'
    try:
        response = httpx.get('https://api.deepseek.com/models', headers={'Authorization': 'Bearer ' + key},
                             timeout=8, follow_redirects=False)
        if response.status_code in (401, 403):
            return 'deepseek_key_invalid'
        response.raise_for_status()
        models = response.json()['data']
        return 'ready' if any(row['id'] == 'deepseek-flash' for row in models) else 'deepseek_model_unavailable'
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return 'deepseek_unreachable'
