"""Bounded, append-only project updates from explicitly authorized stored Feishu snapshots."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from typing import Literal

from pydantic import Field

from .config import StrictModel
from .execution import project_lock
from .feishu import PARSER_POLICY
from .models import ProjectFact, ProjectProfile, SourceBlock, now
from .projects import load_profile, recent_source_blocks

POLICY = "sourced-append-only-v1"


class SourceReview(StrictModel):
    block_id: str
    decision: Literal["supports_existing", "new_fact", "unrelated", "needs_context"]
    fact_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class IncrementalExtraction(StrictModel):
    additions: list[ProjectFact] = Field(max_length=10)
    source_review: list[SourceReview] = Field(max_length=100)
    unresolved: list[str] = Field(default_factory=list, max_length=20)


def input_update_summary(store, profile, source_results):
    """Match a prior extraction only to this exact profile and successfully read snapshot set."""
    if not source_results or any(source.get("status") != "read" for source in source_results):
        return None
    keys = {source.get("snapshot_key") for source in source_results}
    if None in keys:
        return None
    matching = [update for update in store.list("project_input_update")
                if update["project_id"] == profile.project_id and update["base_version"] == profile.version
                and {source["snapshot_key"] for source in update["source_snapshots"]} == keys]
    if not matching:
        return None
    update = max(matching, key=lambda item: item["created_at"])
    return {"update_key": update["update_key"], "status": update["status"],
            "candidate_version": update["candidate_version"],
            "candidate_blocks": update["candidate_blocks"],
            "decisions": dict(Counter(item["decision"] for item in update["extraction"]["source_review"])),
            "scientific_acceptance": False, "active_profile_changed": False}


def _normalized(text):
    return re.sub(r"[\s\u200b-\u200d\u2060\ufeff]+", "", unicodedata.normalize("NFKC", text))


def update_inputs(store, client, project, feishu_config, snapshot_keys):
    """Classify new source text; preserve active facts, and stage genuine additions separately."""
    with project_lock(store.root, project):
        return _update_inputs(store, client, project, feishu_config, sorted(set(snapshot_keys)))


def _update_inputs(store, client, project, feishu_config, snapshot_keys):
    if not snapshot_keys:
        raise ValueError("Explicit stored snapshot keys required")
    baseline = load_profile(store, project)
    known_ids = {block.block_id: block for block in baseline.blocks}
    incoming = {}
    sources = []
    for key in snapshot_keys:
        snapshot = store.get("feishu_snapshot", key)
        token = snapshot.get("wiki_node_token")
        if token not in feishu_config.allowed_wiki_tokens:
            raise ValueError("Stored wiki node is outside the explicit allowlist")
        expected = f"{token}:{snapshot['revision']}:{snapshot['content_digest']}:{PARSER_POLICY}"
        if (key != expected or snapshot.get("parser_policy") != PARSER_POLICY
                or snapshot.get("scope") != "explicit_node_only_no_sibling_crawl"):
            raise ValueError("Stored snapshot identity/context policy is not authorized")
        for item in snapshot["blocks"]:
            block = SourceBlock.model_validate(item)
            if (block.document_id, block.revision) != (snapshot["document_id"], snapshot["revision"]):
                raise ValueError("Stored snapshot block document/revision mismatch")
            if block.block_id in known_ids and block != known_ids[block.block_id]:
                raise ValueError("Referenced source changed; full profile review required")
            if block.block_id in incoming and incoming[block.block_id] != block:
                raise ValueError("Conflicting source block identities")
            incoming[block.block_id] = block
        sources.append({"snapshot_key": key, "title": snapshot.get("wiki_title", ""),
                        "document_id": snapshot["document_id"], "wiki_url": snapshot.get("wiki_url")})
    recent = [SourceBlock.model_validate(b) for b in recent_source_blocks(list(incoming.values()))]
    existing_text = _normalized("\n".join(b.text for b in baseline.blocks))
    qualifying = [b for b in recent if len(_normalized(b.text)) >= 20]
    candidates = [b for b in qualifying if _normalized(b.text) not in existing_text]
    if len(candidates) > 100 or sum(len(b.text) for b in candidates) > 8000:
        raise ValueError("Incremental input exceeds bounded budget; narrow the explicit source set")
    identity = {"policy": POLICY, "project": project, "base_version": baseline.version,
                "snapshot_keys": snapshot_keys, "candidates": [b.model_dump() for b in candidates]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    key = f"{project}:{baseline.version}:{digest}"
    try:
        return {**store.get("project_input_update", key), "reused": True}
    except ValueError as exc:
        if not str(exc).startswith("Missing project_input_update:"):
            raise
    by_id = {b.block_id: b for b in candidates}
    try:
        draft = store.get("project_input_update_draft", key)
    except ValueError as exc:
        if not str(exc).startswith("Missing project_input_update_draft:"):
            raise
        draft = None
    if draft:
        extraction = IncrementalExtraction.model_validate(draft["extraction"])
    elif candidates:
        extraction = client.structured(json.dumps({"existing_facts": [f.model_dump() for f in baseline.facts],
            "sources": sources, "candidate_blocks": [b.model_dump() for b in candidates],
            "instruction": "逐块分类新来源：supports_existing须引用已有fact_ids；new_fact才可提出新增项；"
            "其他项目标unrelated，缺失表头/数值/条件的碎片标needs_context，不从孤立表格片段推断实验结果。"
            "未匹配旧文本只是机械差异，不等于新事实。重复进展可支持已有目标，不重复创建目标或问题。"
            "文献Agent开发/PaperFlow/PaperQA实施计划属于另一个项目，不混入当前BNN/机械臂研究。"
            "保留所有原有事实与ID，不删除或重写；每个candidate block必须恰好分类一次。"
            "新增项source_blocks仅可用分类为new_fact的块；source_date必须取其章节实际日期，"
            "计划/拟/将要训练或测试只能作为goal/question，不能写成已完成的observation。"
            "历史日志不代表实时实验状态；有日期的项标historical/current_candidate，仍待负责人审阅。"
            "不足以新增时additions为空是正确结果。输出JSON。"}, ensure_ascii=False),
            IncrementalExtraction, "项目增量理解")
    else:
        extraction = IncrementalExtraction(additions=[], source_review=[])
    reviews = {item.block_id: item for item in extraction.source_review}
    if len(reviews) != len(extraction.source_review) or set(reviews) != set(by_id):
        raise ValueError("Incremental review must classify every candidate exactly once")
    fact_ids = {f.id for f in baseline.facts}
    for review in reviews.values():
        if review.decision == "supports_existing":
            if not review.fact_ids or not set(review.fact_ids) <= fact_ids:
                raise ValueError("Support review must reference existing project facts")
        elif review.fact_ids:
            raise ValueError("Only support reviews may link existing fact IDs")
    eligible = {bid for bid, review in reviews.items() if review.decision == "new_fact"}
    cited = set()
    additions = []
    for fact in extraction.additions:
        if not set(fact.source_blocks) <= eligible:
            raise ValueError("New project fact references unapproved incremental source blocks")
        cited.update(fact.source_blocks)
        dates = {date(*map(int, match)).isoformat() for bid in fact.source_blocks
                 for match in re.findall(r"(202\d)[.\-/](\d{1,2})[.\-/](\d{1,2})", by_id[bid].section or "")}
        if fact.source_date not in dates or fact.temporal_status == "undated":
            raise ValueError("New project fact requires a matching source date and temporal status")
        if fact.kind == "observation" and all(re.search(r"计划|拟|将|会抽时间", by_id[bid].text)
                                              for bid in fact.source_blocks):
            raise ValueError("Planned work cannot become an observed result")
        additions.append(fact.model_copy(update={"id": f"delta-{digest}-{fact.id}"}))
    if eligible != cited:
        raise ValueError("Every new-fact source must be cited by an addition")
    if draft is None:
        # Retain validated model output and its timestamp if staging is interrupted later.
        draft = {"created_at": now(), "extraction": extraction.model_dump()}
        store.put("project_input_update_draft", key, draft, immutable=True)
    candidate = None
    if additions:
        candidate = ProjectProfile(project_id=project, version=digest + "-delta-v1", created_at=draft["created_at"],
            facts=[*baseline.facts, *additions],
            blocks=[*baseline.blocks, *(block for bid, block in by_id.items() if bid in cited and bid not in known_ids)],
            unresolved=list(dict.fromkeys([*baseline.unresolved, *extraction.unresolved])))
        profile_key = f"{project}:{candidate.version}"
        store.put("snapshot", profile_key, {**identity, "base_profile": f"{project}:{baseline.version}",
            "blocks": [b.model_dump() for b in candidate.blocks]}, immutable=True)
        store.put("profile", profile_key, candidate.model_dump(), immutable=True)
        store.put("latest_profile", project, {"key": profile_key})
    result = {"update_key": key, "project_id": project, "base_version": baseline.version, "policy": POLICY,
        "created_at": draft["created_at"], "source_snapshots": sources, "recent_text_blocks": len(recent),
        "qualified_blocks": len(qualifying), "mechanically_matched_blocks": len(qualifying)-len(candidates),
        "candidate_blocks": len(candidates), "candidate_chars": sum(len(b.text) for b in candidates),
        "extraction": extraction.model_dump(), "candidate_version": candidate.version if candidate else None,
        "status": "pending_profile_review" if candidate else "no_new_project_facts",
        "active_profile_changed": False, "scientific_acceptance": False,
        "limitation": "机械差异及模型分类均需抽查；仅新增事实形成候选，支持既有事实的来源单独留痕，不重写旧档案。"}
    store.put("project_input_update", key, result, immutable=True)
    return {**result, "reused": False}
