"""Optional reader adapters. Heavy imports remain inside explicitly invoked server functions."""
from importlib.metadata import version
import json
import hashlib
from pathlib import Path

from .models import EvidenceRecord
from .table_quality import table_quality


def pqa_chunk_key(name, text):
    """Page-range names are display labels, not unique chunk identifiers."""
    return "chunk-" + hashlib.blake2s((name + "\0" + text).encode("utf-8"), digest_size=8).hexdigest()


def compatibility_status(data_dir):
    """Read persisted pilot metadata without importing any heavy models or starting work."""
    result = {"protocol": "not_validated", "scientific_quality": "not_accepted", "bge_m3": "not_validated"}
    paths = sorted((Path(data_dir) / "reader-compatibility").glob("reader-*/result.json"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return result
    try:
        record = json.loads(paths[0].read_text(encoding="utf-8"))
        if (record["status"] == "completed" and len(record["papers"]) >= 3
                and all(p["status"] == "completed" for p in record["papers"])
                and record["embedding"]["shape"] == [3, 1024] and record["embedding"]["device"] == "cpu"):
            result.update(protocol="three-paper-pilot-passed", bge_m3="real-cpu-smoke-passed",
                          run_id=record["run_id"], sample_count=len(record["papers"]))
    except (KeyError, TypeError, ValueError, OSError):
        result["protocol"] = "invalid_pilot_metadata"
    return result


def docling_evidence(path: str | Path, paper_id: str, paper_version: str, *, model_artifacts_path=None, page_range=None):
    from paperqa_docling import parse_pdf_to_pages

    target = Path(path).resolve(strict=True)
    options = {
        "accelerator_options": {"device": "cpu", "num_threads": 2},
        "do_ocr": False,
    }
    if model_artifacts_path:
        options["artifacts_path"] = str(Path(model_artifacts_path).resolve(strict=True))
    parsed = parse_pdf_to_pages(target, parse_media=True, page_range=page_range, custom_pipeline_options=options)
    evidence, media_status, tables = [], [], []
    for page_key, content in parsed.content.items():
        # Official reader keys are physical 1-based PDF pages, not model-generated page guesses.
        page = int(page_key)
        text, media = content if isinstance(content, (tuple, list)) else (content, [])
        if text.strip():
            evidence.append(EvidenceRecord(evidence_id=f"{paper_id}:{paper_version}:page:{page}",
                paper_id=paper_id, paper_version=paper_version, file=str(target), page=page,
                block_id=f"page:{page}", text=text, parser_version="docling-" + version("docling")))
        for item in media:
            if item.info.get("type") == "table" and item.text:
                quality = table_quality(item.text)
                tables.append({"page": page, "index": item.index, **quality})
                evidence.append(EvidenceRecord(evidence_id=f"{paper_id}:{paper_version}:page:{page}:table:{item.index}",
                    paper_id=paper_id, paper_version=paper_version, file=str(target), page=page,
                    block_id=f"page:{page}:table:{item.index}", section="table", text=item.text,
                    parser_version="docling-" + version("docling"), quality_flags=quality["quality_flags"]))
            else:
                media_status.append({"page": page, "index": item.index, "type": item.info.get("type"),
                                     "status": "image_not_semantically_verified"})
    if not evidence:
        raise ValueError("Docling returned no located text evidence")
    return {"evidence": [e.model_dump() for e in evidence], "media_status": media_status, "table_quality": tables,
            "file": str(target), "parser": version("paper-qa-docling"),
            "ocr": False, "requested_page_range": page_range, "model_artifacts_path": options.get("artifacts_path"),
            "limitations": ["scanned text without OCR is not covered", "figures require human inspection"]}


def bge_cpu(model_path: str):
    """Load an explicitly provisioned local model only; never fallback to hash/cloud embeddings."""
    from sentence_transformers import SentenceTransformer

    path = Path(model_path).resolve(strict=True)
    model = SentenceTransformer(str(path), device="cpu", local_files_only=True, trust_remote_code=False)
    return model


def paperflow_profile(profile, *, topic_weights):
    if not topic_weights:
        raise ValueError('Explicit source-linked topic mapping required; sentences are not topic keys')
    return {"user_id": profile.project_id, "version": profile.version,
            "core_directions": dict(topic_weights), "topic_weights": dict(topic_weights),
            "author_heat": {}, "institution_heat": {}, "interest_vector": [],
            "reading_history": [], "methodology_preferences": {},
            "must_read": {"authors": [], "institutions": [], "keywords": []},
            "research_source_version": profile.version, "feedback_learning_enabled": False}


class PaperFlowFlashProvider:
    """Implement the pinned upstream provider protocol through the single audited client.

    Full reading/coordinator paths have additional clients and are not enabled by this provider alone.
    """
    name = "deepseek"
    model = "deepseek-flash"

    def __init__(self, client):
        self.client = client

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024):
        from paperflow.providers.llm import LLMResponse

        text = self.client.complete(prompt, system=system or "PaperFlow adapter", max_tokens=max_tokens)
        return LLMResponse(text=text, model=self.model, provider=self.name)

    def stream_generate(self, prompt, **kwargs):
        # Buffering deliberately shares the same payload checks/audit path.
        yield self.generate(prompt, **kwargs).text


def paperqa_settings(model_path: str, *, model_artifacts_path=None):
    """Explicit configuration for every known PQA model field; runtime audit still required."""
    from paperqa import Settings

    # A local OpenAI-compatible bridge can route every PQA request through FlashClient.
    # This function requires an explicitly started audit bridge, no direct supplier fallback.
    import os
    bridge = os.environ.get("NEUROEI_PQA_BRIDGE_URL")
    if not bridge or not bridge.startswith("http://127.0.0.1:"):
        raise ValueError("PaperQA requires an active local audited Flash bridge")
    route = {"model_list": [{"model_name": "openai/deepseek-flash", "litellm_params": {
        "model": "openai/deepseek-flash", "api_base": bridge, "api_key": os.environ["NEUROEI_PQA_BRIDGE_KEY"],
        "max_retries": 0}}], "router_kwargs": {"num_retries": 0}}
    pipeline_options = {"accelerator_options": {"device": "cpu", "num_threads": 2}, "do_ocr": False}
    if model_artifacts_path:
        pipeline_options["artifacts_path"] = str(Path(model_artifacts_path).resolve(strict=True))
    return Settings(llm="openai/deepseek-flash", llm_config=route,
        summary_llm="openai/deepseek-flash", summary_llm_config=route,
        embedding="st-" + str(Path(model_path).resolve(strict=True)),
        embedding_config={"device": "cpu", "batch_size": 4}, verbosity=0,
        answer={"evidence_k": 3, "answer_max_sources": 3, "max_answer_attempts": 1,
                "max_concurrent_requests": 1, "answer_length": "about 150 words"},
        agent={"agent_llm": "openai/deepseek-flash", "agent_llm_config": route},
        parsing={"use_doc_details": False, "multimodal": False,
                 "parse_pdf": "paperqa_docling.parse_pdf_to_pages",
                 "enrichment_llm": "openai/deepseek-flash", "enrichment_llm_config": route,
                 "reader_config": {"chunk_chars": 5000, "overlap": 250,
                     "custom_pipeline_options": pipeline_options}})
