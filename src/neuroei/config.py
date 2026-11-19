import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def code_revision():
    """Release deployments carry an explicit marker; never infer a historical run's version."""
    if value := os.environ.get('NEUROEI_CODE_REVISION'):
        return value
    marker = Path(__file__).resolve().parents[2] / 'REVISION'
    try:
        value = marker.read_text(encoding='ascii').strip()
    except (OSError, UnicodeError):
        return 'unrecorded-working-tree'
    return value if value and all(c in '0123456789abcdef' for c in value) else 'unrecorded-working-tree'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(StrictModel):
    base_url: Literal["https://api.deepseek.com"] = "https://api.deepseek.com"
    model: Literal["deepseek-flash"] = "deepseek-flash"
    api_key_env: Literal["DEEPSEEK_API_KEY"] = "DEEPSEEK_API_KEY"
    thinking: Literal["enabled"] = "enabled"
    reasoning_effort: Literal["high"] = "high"
    request_attempts: int = Field(3, ge=1, le=3)
    schema_repair_attempts: int = Field(2, ge=0, le=2)
    max_concurrency: int = Field(2, ge=1, le=2)
    max_calls_per_run: int = Field(20, ge=1, le=100)
    max_tokens: int = Field(8192, ge=256, le=393216)
    timeout_seconds: int = Field(120, ge=1, le=1800)


class RuntimeConfig(StrictModel):
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    daily_deep_read_limit: int = Field(5, ge=1, le=5)
    publication_mode: Literal["draft"] = "draft"
    feedback_learning_enabled: Literal[False] = False
    reader_backend: Literal["ssh", "server-local"] = "ssh"
    data_dir: str = "data"


class FeishuConfig(StrictModel):
    # User instruction: Agent-generated artifacts must never be uploaded to Feishu.
    generated_upload_enabled: Literal[False] = False
    allowed_document_ids: list[str] = Field(default_factory=list)
    allowed_wiki_tokens: list[str] = Field(default_factory=list)


class SourcesConfig(StrictModel):
    enabled: list[Literal["arxiv", "pubmed", "biorxiv", "crossref"]] = ["arxiv", "pubmed", "biorxiv"]
    journals: list[str] = Field(default_factory=list)
    max_results: int = Field(30, ge=1, le=100)


class Config(StrictModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)

    @property
    def reader_backend(self) -> str:
        value = os.environ.get("NEUROEI_READER_BACKEND", self.runtime.reader_backend)
        if value not in ("ssh", "server-local"):
            raise ValueError("Explicit ssh or server-local reader backend required")
        return value

    @property
    def data_dir(self) -> Path:
        value = os.environ.get("NEUROEI_DATA_DIR", self.runtime.data_dir)
        if os.name != "nt" and ":/" in value:
            raise ValueError("Set NEUROEI_DATA_DIR to an absolute Linux server path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("Data directory must be absolute")
        return path.resolve()


def load_config(path: str | Path = "config.toml") -> Config:
    with Path(path).open("rb") as stream:
        return Config.model_validate(tomllib.load(stream))
