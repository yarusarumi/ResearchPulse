"""Resolve the server model setting and construct only the requested backend."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

from . import desktop_models
from .llm import ModelFailure, ModelUnavailable, provider_key

OPTIONS = {'sol-high': 'GPT-6.2 Sol · 高', 'astra-medium': 'GPT-6 Astra · 中',
           'astra-high': 'GPT-6 Astra · 高', 'deepseek': 'DeepSeek'}
DEFAULT = 'deepseek'
_SOL_FALLBACK_REASONS = frozenset({'model_balance_insufficient', 'model_quota_exhausted'})
FALLBACK_NOTICE = 'GPT-6.2 SOL 额度已用尽，本次任务改用 DeepSeek。'
_task = ContextVar('neuroei_model_task', default=None)


@contextmanager
def task_selection(on_fallback):
    desktop = _desktop_runtime()
    token = _task.set({'backend': desktop.backend if desktop else selected(), 'desktop': desktop,
                       'quota': None, 'reported': False, 'report': on_fallback})
    try:
        yield
    finally:
        _task.reset(token)


class SolFallbackClient:
    """Keep SOL as the primary client and use DeepSeek only after a quota signal."""

    def __init__(self, primary, config, store, run_id, *, key_provider=None):
        self.primary = primary
        self.config = primary.config
        self._deepseek_config = config
        self._key_provider = key_provider
        self.store = store
        self.run_id = run_id
        self.fallback = None
        self._lock = RLock()
        self._task = _task.get()
        self._quota = None
        self._primary_quota_attempt = False
        self.fallback_notice = None

    @property
    def fallback_used(self):
        return self.fallback is not None

    @property
    def backend(self):
        return 'deepseek' if self.fallback_used else self.primary.backend

    @property
    def supports_images(self):
        return not self.fallback_used and getattr(self.primary, 'supports_images', True)

    @property
    def calls(self):
        return self.primary.calls + (self.fallback.calls if self.fallback else 0)

    @property
    def held(self):
        return self.primary.held

    @property
    def last_model(self):
        return self.fallback.last_model if self.fallback else self.primary.last_model

    @property
    def last_usage(self):
        return self.fallback.last_usage if self.fallback else self.primary.last_usage

    @property
    def last_finish_reason(self):
        return self.fallback.last_finish_reason if self.fallback else self.primary.last_finish_reason

    @property
    def should_stop(self):
        return self.primary.should_stop

    @should_stop.setter
    def should_stop(self, value):
        self.primary.should_stop = value
        if self.fallback:
            self.fallback.should_stop = value

    def hold(self, count):
        self.primary.hold(count)
        if self.fallback:
            self.fallback.hold(count)

    def release(self, count):
        self.primary.release(count)
        if self.fallback:
            self.fallback.release(count)

    def remaining(self):
        return self.fallback.remaining() if self.fallback else self.primary.remaining()

    def _deepseek(self, reason):
        if self.fallback is None:
            from .llm import FlashClient
            # A rejected quota request did no work. Replace that attempt while
            # keeping prior successful calls and held capacity in the budget.
            budget = self._deepseek_config.max_calls_per_run - self.primary.calls + self._primary_quota_attempt
            if budget <= self.primary.held:
                raise ModelUnavailable('model_budget_insufficient', 'No remaining fallback model calls')
            key = provider_key(self._deepseek_config.api_key_env, self._key_provider)
            if not key:
                raise ModelUnavailable('model_not_configured', 'DeepSeek fallback key is unavailable')
            options = {'key_provider': lambda: key} if self._key_provider else {}
            self.fallback = FlashClient(self._deepseek_config.model_copy(update={'max_calls_per_run': budget}),
                                        self.store, self.run_id, **options)
            self.fallback.should_stop = self.primary.should_stop
            if self.primary.held:
                self.fallback.hold(self.primary.held)
            self.fallback_notice = ('Codex 额度已用尽，本次任务改用 DeepSeek。'
                                    if self._key_provider is not None else FALLBACK_NOTICE)
            self.store.audit(self.run_id, {
                'event': 'model_fallback', 'status': 'selected',
                'from_model': self.primary.config.model, 'to_model': 'deepseek-flash',
                'from_backend': self.primary.backend, 'reason': reason,
            })
            if self._task and not self._task['reported']:
                self._task['report']()
                self._task['reported'] = True
        return self.fallback

    def _call(self, method, *args, images=(), **kwargs):
        with self._lock:
            reason = self._task['quota'] if self._task else self._quota
            if images and (self.fallback_used or reason):
                raise ModelUnavailable('model_quota_exhausted', 'SOL image quota unavailable')
            if self.fallback_used or (reason and not images):
                return getattr(self._deepseek(reason), method)(*args, **kwargs)
            if images:
                kwargs['images'] = images
            before = self.primary.calls
            try:
                return getattr(self.primary, method)(*args, **kwargs)
            except ModelFailure as error:
                if error.reason not in _SOL_FALLBACK_REASONS:
                    raise
                if self._task:
                    self._task['quota'] = error.reason
                self._quota = error.reason
                self._primary_quota_attempt = self.primary.calls > before
                if images:
                    raise
                return getattr(self._deepseek(error.reason), method)(*args, **kwargs)

    def complete(self, prompt, *, system, max_tokens=None, json_mode=False,
                 messages=None, thinking_enabled=True):
        return self._call('complete', prompt, system=system, max_tokens=max_tokens,
                          json_mode=json_mode, messages=messages, thinking_enabled=thinking_enabled)

    def structured(self, prompt, schema, role, *, images=()):
        return self._call('structured', prompt, schema, role, images=images)

    def close(self):
        self.primary.close()
        if self.fallback:
            self.fallback.close()



def validate(backend):
    if not isinstance(backend, str) or backend not in OPTIONS:
        raise ModelUnavailable('model_backend_unavailable', 'Invalid AI backend')
    return backend


def _desktop_runtime():
    if state := _task.get():
        return state['desktop']
    profile = desktop_models.active_profile()
    return desktop_models.runtime(profile) if profile is not None else None


def selected():
    """Read the current selection once at the request boundary."""
    if state := _task.get():
        return state['backend']
    if desktop := _desktop_runtime():
        return desktop.backend
    if value := os.environ.get('NEUROEI_AI_BACKEND_FILE'):
        path = Path(value)
        if not path.is_absolute():
            raise ModelUnavailable('model_backend_unavailable', 'AI backend file must use an absolute path')
        try:
            return validate(path.read_text(encoding='ascii').strip())
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            raise ModelUnavailable('model_backend_unavailable', 'AI backend file could not be read') from None
    return validate(os.environ.get('NEUROEI_AI_BACKEND', DEFAULT).strip())


def set_selected(backend):
    if backend not in ('sol-high', 'deepseek'):
        raise ValueError('Unsupported primary model')
    if backend == 'deepseek' and not os.environ.get('DEEPSEEK_API_KEY'):
        raise ModelUnavailable('model_not_configured', 'DeepSeek key is unavailable')
    path = Path(os.environ.get('NEUROEI_AI_BACKEND_FILE', ''))
    if not path.is_absolute():
        raise ModelUnavailable('model_backend_unavailable', 'AI backend file is not configured')
    with NamedTemporaryFile(mode='w', encoding='ascii', dir=path.parent, delete=False) as temporary:
        temporary.write(backend + '\n')
    candidate = Path(temporary.name)
    try:
        candidate.replace(path)
    finally:
        candidate.unlink(missing_ok=True)


def runner():
    if desktop := _desktop_runtime():
        return desktop.codex_runner
    configured = os.environ.get('NEUROEI_AI_CODEX_RUNNER')
    if configured:
        path = Path(configured)
        return path if path.is_absolute() and path.is_file() else None
    directory = Path.home() / '.config' / 'neuroei'
    server_runner = directory / 'codex-runner'
    temporary_relay = directory / 'codex-temporary-relay'
    return next((path for path in (server_runner, temporary_relay) if path.is_file()), None)


def runner_ready():
    from .codex_model_client import codex_login_ready

    path = runner()
    return path is not None and codex_login_ready(path)


def deepseek_key():
    if desktop := _desktop_runtime():
        try:
            return desktop.deepseek_key()
        except ValueError:
            return None
    return os.environ.get('DEEPSEEK_API_KEY')


def deepseek_available():
    return bool(deepseek_key())


def model_config(config, backend):
    if validate(backend) == 'deepseek':
        return config
    from .codex_model_client import config_for_backend

    desktop = _desktop_runtime()
    return config_for_backend(config, backend, choice=desktop.codex_choice() if desktop and backend == 'sol-high' else None)


def text_client(config, store, run_id, *, backend):
    desktop = _desktop_runtime()
    options = {'key_provider': desktop.deepseek_key} if desktop else {}
    if validate(backend) == 'deepseek':
        from .llm import FlashClient

        return FlashClient(config, store, run_id, **options)
    from .codex_model_client import CodexModelClient

    codex_options = {'choice': desktop.codex_choice()} if desktop and backend == 'sol-high' else {}
    primary = CodexModelClient(config, store, run_id, backend=backend, runner=runner(), **codex_options)
    if backend == 'sol-high' and (not desktop or desktop.fallback_enabled):
        return SolFallbackClient(primary, config, store, run_id, **options)
    return primary
