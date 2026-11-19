import json
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from neuroei import codex_model_client as codex
from neuroei.config import LLMConfig
from neuroei.llm import ModelFailure, ModelUnavailable


class Record(BaseModel):
    title: str
    source: int


class FakeStore:
    def __init__(self):
        self.rows = []

    def audit(self, run_id, row):
        self.rows.append((run_id, row.copy()))


@pytest.mark.parametrize(('message', 'expected'), [
    ("You've hit your usage limit", 'model_quota_exhausted'),
    ('insufficient_quota', 'model_quota_exhausted'),
    ('rate limit temporarily exceeded', 'model_service_unavailable'),
    ('maximum output tokens exceeded', 'output_truncated'),
])
def test_quota_is_distinct_from_transient_or_request_limits(message, expected):
    output = json.dumps({'type': 'turn.failed', 'error': {'message': message}}).encode()
    with pytest.raises(ModelFailure) as error:
        codex._read_turn(output, 1)
    assert error.value.reason == expected


def turn_output(content, usage=None):
    return "\n".join(json.dumps(row) for row in (
        {"type": "item.completed", "item": {"type": "reasoning", "text": "private"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": content}},
        {"type": "turn.completed", "usage": usage or {}},
    )).encode("utf-8")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    runner = tmp_path / "codex"
    runner.touch()
    calls = []

    def login(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, b"Logged in using ChatGPT\n", b"")

    monkeypatch.setattr(codex.subprocess, "run", login)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-passed")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-passed")
    return runner, calls


def test_structured_uses_chosen_model_and_sandbox_and_audits_usage(setup, monkeypatch):
    runner, logins = setup

    class Process:
        returncode = 0

        def __init__(self, command, **options):
            assert command[0] == str(runner)
            assert command[1:4] == ["-a", "never", "exec"]
            assert command[command.index("--model") + 1] == "gpt-6-astra"
            assert 'model_reasoning_effort="medium"' in command
            assert command[command.index("--sandbox") + 1] == "read-only"
            assert command[command.index("--disable") + 1] == "shell_tool"
            assert 'web_search="disabled"' in command
            assert "--ephemeral" in command
            assert "--ignore-user-config" in command
            assert not ({"OPENAI_API_KEY", "DEEPSEEK_API_KEY"} & options["env"].keys())
            schema = Path(command[command.index("--output-schema") + 1])
            assert json.loads(schema.read_text(encoding="utf-8")) == {
                **Record.model_json_schema(), "additionalProperties": False,
            }
            image = Path(command[command.index("--image") + 1])
            assert image.read_bytes() == b"example-png"
            assert image.parent == schema.parent == options["cwd"]
            payload = options["stdin"].read().decode("utf-8")
            assert "Do not invent facts" in payload and "日报正文" in payload

        def communicate(self, timeout):
            return turn_output('{"title":"可靠标题","source":2}',
                               {"input_tokens": 45, "cached_input_tokens": 12, "output_tokens": 8}), None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(codex.subprocess, "Popen", Process)
    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(), store, "first", backend="astra-medium", runner=runner)
    result = client.structured("日报正文", Record, "Analyze daily report", images=[b"example-png"])
    assert result == Record(title="可靠标题", source=2)
    assert client.config.model_dump(mode="json")["api_key_env"] is None
    assert client.calls == 1 and client.last_finish_reason == "stop"
    assert client.last_usage == {"input_tokens": 45, "cached_input_tokens": 12, "output_tokens": 8,
                                 "prompt_tokens": 45, "completion_tokens": 8, "total_tokens": 53}
    assert logins[0][0] == [str(runner), "login", "status"]
    assert "OPENAI_API_KEY" not in logins[0][1]["env"]
    run_id, audit = store.rows[0]
    assert run_id == "first"
    assert audit["request_model"] == "gpt-6-astra" and audit["reasoning_effort"] == "medium"
    assert audit["status"] == "ok" and audit["auth"] == "same_host_chatgpt"
    assert "日报正文" not in json.dumps(audit, ensure_ascii=False)
    assert "可靠标题" not in json.dumps(audit, ensure_ascii=False)


def test_complete_preserves_messages_and_rejects_non_object_json(setup, monkeypatch):
    runner, _ = setup
    calls = []

    def answer(prompt, schema, images):
        calls.append((prompt, schema, images))
        return ("[]" if len(calls) == 2 else '{"result":"ok"}'), {}

    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(), store, "json", runner=runner)
    monkeypatch.setattr(client, "_run", answer)
    transcript = [{"role": "system", "content": "Ground the answer"},
                  {"role": "user", "content": "Question"},
                  {"role": "assistant", "content": "Earlier answer"},
                  {"role": "user", "content": "Follow-up"}]
    assert client.complete("ignored", system="paperqa", messages=transcript,
                           max_tokens=1000, json_mode=True, thinking_enabled=False) == '{"result":"ok"}'
    rendered, schema, images = calls[0]
    assert json.loads(rendered.split("\n\n", 1)[1]) == transcript
    assert schema is None and not images
    audit = store.rows[0][1]
    assert audit["request_model"] == "gpt-6.2-sol" and audit["reasoning_effort"] == "high"
    assert audit["max_tokens"] == 1000 and audit["thinking_requested"] is False
    assert audit["output_limit_enforcement"] == "prompt_guidance"
    with pytest.raises(ModelFailure) as error:
        client.complete("prompt", system="role", json_mode=True)
    assert error.value.reason == "model_invalid_output"
    assert store.rows[-1][1]["status"] == "failed"


def test_schema_validation_can_retry_without_echoing_wrong_output(setup, monkeypatch):
    runner, _ = setup
    prompts = []
    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(schema_repair_attempts=1), store, "repair",
                                    backend="astra-high", runner=runner)

    def response(prompt, schema, images):
        prompts.append(prompt)
        return ('{"title":"invalid-output"}' if len(prompts) == 1 else '{"title":"ok","source":1}'), {}

    monkeypatch.setattr(client, "_run", response)
    assert client.structured("data", Record, "role") == Record(title="ok", source=1)
    assert "Previous response failed schema validation" in prompts[1]
    assert "invalid-output" not in prompts[1]
    assert [row["status"] for _, row in store.rows] == ["failed", "ok"]
    assert client.calls == 2 and client.last_model == "gpt-6-astra"


def test_understanding_schema_requires_nested_default_fields():
    from neuroei.project_understanding import UnderstandingExtraction

    original = UnderstandingExtraction.model_json_schema()
    fixed = codex._strict_output_schema(original)
    need = fixed["$defs"]["Need"]
    assert need["required"] == list(need["properties"])
    assert "change_item_ids" not in original["$defs"]["Need"]["required"]
    assert need["additionalProperties"] is False
    assert "default" not in need["properties"]["method_needed"]


def test_dynamic_map_still_validates_json_locally(setup, monkeypatch):
    class DynamicRecord(BaseModel):
        ratings: dict[str, int]

    runner, _ = setup
    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(schema_repair_attempts=0), store, "dynamic", runner=runner)

    def respond(prompt, schema, images):
        assert schema is None
        assert "Output JSON Schema" in prompt and "additionalProperties" in prompt
        return '{"ratings":{"specific-key":"invalid"}}', {}

    monkeypatch.setattr(client, "_run", respond)
    with pytest.raises(ModelFailure) as failure:
        client.structured("data", DynamicRecord, "report")
    assert failure.value.reason == "model_invalid_output"
    assert store.rows[0][1]["output_format"] == "validated_json"


def test_no_chatgpt_login_never_launches_or_audits_paid_request(setup, monkeypatch):
    runner, _ = setup
    monkeypatch.setattr(codex.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, b"Logged in using an API key.\n", b""))
    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(), store, "auth", runner=runner)

    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not start for API-key authentication")

    monkeypatch.setattr(codex.subprocess, "Popen", forbidden)
    with pytest.raises(ModelUnavailable) as error:
        client.complete("data", system="role")
    assert error.value.reason == "model_not_configured"
    assert client.calls == 0 and not store.rows


def test_hold_release_and_close(setup, monkeypatch):
    runner, _ = setup
    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(max_calls_per_run=2), store, "held", runner=runner)
    monkeypatch.setattr(client, "_run", lambda *args: ("answer", {}))
    client.hold(2)
    assert client.remaining() == 0
    with pytest.raises(ModelUnavailable) as error:
        client.complete("data", system="role")
    assert error.value.reason == "model_budget_insufficient"
    client.release(1)
    assert client.remaining() == 1
    assert client.complete("data", system="role") == "answer"
    assert client.calls == 1 and client.remaining() == 0
    client.close()
    with pytest.raises(ModelUnavailable) as error:
        client.complete("data", system="role")
    assert error.value.reason == "canceled_by_user"
    assert len(store.rows) == 1


@pytest.mark.parametrize("failure", ["canceled_by_user", "model_timeout"])
def test_inflight_cancel_or_timeout_stops_subprocess(setup, monkeypatch, failure):
    runner, _ = setup
    killed = []
    stopped = []
    moments = iter(range(5))
    monkeypatch.setattr(codex.time, "monotonic", lambda: next(moments))

    class Process:
        def __init__(self, command, **kwargs):
            if failure == "canceled_by_user":
                stopped.append(True)

        def communicate(self, timeout):
            assert killed

        def poll(self):
            return None

        def kill(self):
            killed.append(True)

    store = FakeStore()
    client = codex.CodexModelClient(LLMConfig(request_attempts=1, timeout_seconds=1),
                                    store, "stopped", runner=runner)
    client.should_stop = lambda: bool(stopped)
    monkeypatch.setattr(codex.subprocess, "Popen", Process)
    with pytest.raises(ModelFailure) as error:
        client.complete("data", system="role")
    assert error.value.reason == failure
    assert killed and store.rows[0][1]["reason"] == failure


def test_failed_turn_does_not_return_a_prior_message_or_leak_failure_text():
    failed = turn_output("preliminary").splitlines()[:-1]
    failed.append(b'{"type":"turn.failed","error":{"message":"private context_length_exceeded"}}')
    with pytest.raises(ModelFailure) as error:
        codex._read_turn(b"\n".join(failed), 1)
    assert error.value.reason == "model_input_too_large"
    assert "private" not in str(error.value)
