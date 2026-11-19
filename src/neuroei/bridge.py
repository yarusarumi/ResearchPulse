"""Per-run loopback compatibility bridge: PaperQA's internal LLM calls share FlashClient audit/budget."""
import json
import os
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer


@contextmanager
def paperqa_bridge(client):
    token = uuid.uuid4().hex

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.reply(401, {"error": {"message": "Unauthorized bridge request"}})
                return
            try:
                if self.path not in ("/v1/chat/completions", "/chat/completions"):
                    raise ValueError("Unsupported path")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2_000_000:
                    raise ValueError("Request size limit")
                payload = json.loads(self.rfile.read(length))
                if payload.get("model") not in ("deepseek-flash", "openai/deepseek-flash"):
                    raise ValueError("Disallowed model")
                if payload.get("stream") or payload.get("tools"):
                    raise ValueError("Streaming/tools are outside the verified reader protocol")
                messages = payload["messages"]
                if not messages or any(m.get("role") not in ("user", "assistant", "system") or
                                       not isinstance(m.get("content"), str) for m in messages):
                    raise ValueError("Only text conversation messages supported")
                text = client.complete("", system="PaperQA internal call", messages=messages,
                    max_tokens=payload.get("max_tokens", client.config.max_tokens),
                    json_mode=payload.get("response_format", {}).get("type") == "json_object")
                self.reply(200, {"id": "bridge-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": 0, "model": client.last_model, "usage": client.last_usage,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}]})
            except Exception:
                # No provider error body or document text leaves the local error handler.
                self.reply(400, {"error": {"message": "Audited Flash bridge rejected/failed request"}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    # Serialize PQA calls, preserving accurate usage mapping without shared-response races.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    old = {key: os.environ.get(key) for key in ("NEUROEI_PQA_BRIDGE_URL", "NEUROEI_PQA_BRIDGE_KEY")}
    os.environ["NEUROEI_PQA_BRIDGE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
    os.environ["NEUROEI_PQA_BRIDGE_KEY"] = token
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
