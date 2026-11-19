from neuroei import ai_backend
from neuroei.config import LLMConfig


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
    assert ai_backend.selected() == 'sol-high'
