from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

from neuroei import ai_backend
from neuroei.config import LLMConfig
from neuroei.llm import ModelFailure


def test_sol_uses_server_runner_before_temporary_relay(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROEI_AI_BACKEND', 'sol-high')
    monkeypatch.delenv('NEUROEI_AI_BACKEND_FILE', raising=False)
    monkeypatch.delenv('NEUROEI_AI_CODEX_RUNNER', raising=False)
    monkeypatch.setattr(ai_backend.Path, 'home', staticmethod(lambda: tmp_path))
    root = tmp_path / '.config' / 'neuroei'
    root.mkdir(parents=True)
    relay = root / 'codex-temporary-relay'
    relay.touch()
    assert ai_backend.runner() == relay
    server = root / 'codex-runner'
    server.touch()
    assert ai_backend.runner() == server
    config = ai_backend.model_config(LLMConfig(), ai_backend.selected())
    assert (config.model, config.reasoning_effort) == ('gpt-6.2-sol', 'high')


def test_sol_fallback_is_only_used_after_quota(monkeypatch):
    from neuroei.llm import ModelFailure

    class Client:
        config = LLMConfig()
        backend = 'sol-high'
        calls = 0
        held = 0
        last_model = 'gpt-6.2-sol'
        last_usage = {}
        last_finish_reason = None
        should_stop = None

        def complete(self, *args, **kwargs):
            self.calls += 1
            raise ModelFailure('quota', reason='model_balance_insufficient')

        def close(self):
            pass

        def hold(self, count):
            self.held += count

        def release(self, count):
            self.held -= count

        def remaining(self):
            return 2

    class DeepSeek:
        config = LLMConfig()
        calls = 0
        held = 0
        last_model = 'deepseek-flash'
        last_usage = {}
        last_finish_reason = 'stop'
        should_stop = None

        def complete(self, *args, **kwargs):
            self.calls += 1
            return '{}'

        def close(self):
            pass

        def hold(self, count):
            self.held += count

        def release(self, count):
            self.held -= count

        def remaining(self):
            return 2

    deepseek = DeepSeek()
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    class Store:
        def audit(self, *args):
            pass
    monkeypatch.setattr('neuroei.llm.FlashClient', lambda *args: deepseek)
    reported = []
    monkeypatch.setenv('NEUROEI_AI_BACKEND', 'sol-high')
    monkeypatch.delenv('NEUROEI_AI_BACKEND_FILE', raising=False)
    with ai_backend.task_selection(lambda: reported.append(True)):
        wrapped = ai_backend.SolFallbackClient(Client(), LLMConfig(), Store(), 'fallback-test')
        assert wrapped.complete('prompt', system='role') == '{}'
        assert wrapped.complete('next', system='role') == '{}'
        next_stage = ai_backend.SolFallbackClient(Client(), LLMConfig(), Store(), 'fallback-later-stage')
        assert next_stage.complete('later stage', system='role') == '{}'
        assert wrapped.primary.calls == 1 and next_stage.primary.calls == 0
        assert wrapped.fallback_used and wrapped.last_model == 'deepseek-flash'
        assert reported == [True]


def test_model_switch_applies_to_next_task_without_changing_active_task(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROEI_AI_BACKEND', 'sol-high')
    monkeypatch.setenv('NEUROEI_AI_BACKEND_FILE', str(tmp_path / 'model-choice'))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    with ai_backend.task_selection(lambda: None):
        ai_backend.set_selected('deepseek')
        assert ai_backend.selected() == 'sol-high'
    assert ai_backend.selected() == 'deepseek'
    ai_backend.set_selected('sol-high')


def test_journal_images_use_four_real_codex_slots_and_slow_after_rate_limit(tmp_path, monkeypatch):
    from neuroei import codex_model_client

    config = LLMConfig(max_concurrency=4, request_attempts=1, max_calls_per_run=20)
    monkeypatch.setattr(ai_backend, '_IMAGE_RATE_LIMIT_COOLDOWN', 0)
    monkeypatch.setattr(codex_model_client, 'codex_login_ready', lambda _: True)
    barrier = Barrier(4)
    reduced = Barrier(2)
    limiter_seen = Event()
    lock = Lock()
    active = peak = rejected = 0

    def run(self, prompt, schema, images):
        nonlocal active, peak, rejected
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if not limiter_seen.is_set():
                barrier.wait(timeout=3)
                if images[0] == b'rate-limit':
                    rejected += 1
                    limiter_seen.set()
                    raise ModelFailure('temporary', reason='model_rate_limited')
            if images[0] in {b'4', b'5', b'6', b'7'}:
                reduced.wait(timeout=3)
            return '{"transcription":"ok"}', {}
        finally:
            with lock:
                active -= 1

    class Audit:
        def audit(self, *args):
            pass

    monkeypatch.setattr(codex_model_client.CodexModelClient, '_run', run)
    primary = codex_model_client.CodexModelClient(config, Audit(), 'vision', runner=tmp_path / 'fake.exe')
    client = ai_backend.JournalImageClient(primary)
    # Use structured image transport with the same real semaphore as the app.
    from neuroei.feishu_image_ocr import ImageTranscript

    def outcome(image):
        return client.structured('image', ImageTranscript, 'role', images=(image,)).transcription

    try:
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(outcome, [b'rate-limit', b'1', b'2', b'3'])) == ['ok'] * 4
            assert peak == 4 and rejected == 1 and primary.calls == 5
            peak = 0
            assert list(pool.map(outcome, [b'4', b'5', b'6', b'7'])) == ['ok'] * 4
            assert peak == 2
    finally:
        client.close()


def test_journal_quota_is_latched_for_waiting_images_and_text_fallback(monkeypatch):
    class Primary:
        config = LLMConfig(max_concurrency=1)
        supports_images = True
        calls = 0

        def structured(self, *args, **kwargs):
            self.calls += 1
            raise ModelFailure('fixture quota', reason='model_quota_exhausted')

    monkeypatch.setenv('NEUROEI_AI_BACKEND', 'sol-high')
    monkeypatch.delenv('NEUROEI_DESKTOP_PROFILE', raising=False)
    monkeypatch.delenv('NEUROEI_AI_BACKEND_FILE', raising=False)
    with ai_backend.task_selection(lambda: None):
        client = ai_backend.JournalImageClient(Primary())
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(client.structured, 'test', None, 'role', images=(b'img',)) for _ in range(8)]
            for future in futures:
                with pytest.raises(ModelFailure) as error:
                    future.result()
                assert error.value.reason == 'model_quota_exhausted'
        assert client.primary.calls == 1
        assert ai_backend._task.get()['quota'] == 'model_quota_exhausted'
    assert ai_backend.selected() == 'sol-high'
