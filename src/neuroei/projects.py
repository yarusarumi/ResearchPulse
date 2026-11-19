import hashlib
import json
import re
from datetime import date

from .models import ProfileExtraction, ProjectProfile, SourceBlock, now
from .store import Store


def recent_source_blocks(blocks, *, days=10):
    """Select distinct source dates across documents, without requiring Markdown line numbers."""
    if days < 1:
        raise ValueError("At least one source date is required")
    dated, md_dates = [], {}
    for block in blocks:
        source_date = None
        heading = re.match(r"^(#{1,6})\s", block.section or "")
        matches = re.findall(r"(202\d)[.\-/](\d{1,2})[.\-/](\d{1,2})", block.section or "")
        if matches:
            try:
                source_date = date(*map(int, matches[-1]))
            except ValueError:
                md_dates.pop(block.document_id, None)
                continue
            if heading and block.file and block.line_start:
                md_dates[block.document_id] = source_date, len(heading[1])
        elif heading and block.file and block.line_start and block.document_id in md_dates:
            previous_date, previous_depth = md_dates[block.document_id]
            if len(heading[1]) > previous_depth:
                source_date = previous_date
            else:
                md_dates.pop(block.document_id)
        if source_date is not None:
            dated.append((source_date, block))
    selected = set(sorted({source_date for source_date, _ in dated})[-days:])
    return [block.model_dump() for source_date, block in dated if source_date in selected]


def import_project(store: Store, client, project_id: str, snapshot: dict) -> ProjectProfile:
    blocks = [SourceBlock.model_validate(b) for b in snapshot["blocks"]]
    # Keep extraction policy in the profile identity; source snapshots and prior profiles remain immutable.
    source_version = hashlib.sha256(json.dumps([b.model_dump() for b in blocks], sort_keys=True,
                                              ensure_ascii=False).encode()).hexdigest()[:16]
    version = source_version + "-p2v3"
    key = f"{project_id}:{version}"
    existing = [p for p in store.list("profile") if p["project_id"] == project_id and p["version"] == version]
    if existing:
        return ProjectProfile.model_validate(existing[0])
    chunks, current, chars = [], [], 0
    for block in blocks:
        if current and chars + len(block.text) > 48000:
            chunks.append(current)
            current, chars = [], 0
        current.append(block)
        chars += len(block.text)
    if current:
        chunks.append(current)
    extractions = []
    instruction = ("提取研究目标、假设、已观察结果、问题、约束和排除范围。每项附准确的原文块ID；"
        "不要将计划当实验结论，不猜BNN缩写，不明确或矛盾内容写入unresolved。"
        "日志是历史记录，不代表服务器当前状态；记录source_date，未注明日期的‘正在跑’不算实时结果。"
        "只选重要事实，每批最多20项，不照抄所有超参数。基于原文提出的问题须标provenance=derived_question，"
        "不得派生实验结果；问题应覆盖10-15个真实研究痛点，不能凭空扩展目标。")
    for index, chunk in enumerate(chunks):
        cache_key = f"{project_id}:{source_version}:chunk:{index}"
        cached = [entry for entry in store.list("profile_chunk") if entry.get("cache_key") == cache_key]
        if cached:
            extraction = ProfileExtraction.model_validate(cached[0]["extraction"])
        else:
            extraction = client.structured(json.dumps({"blocks": [b.model_dump() for b in chunk],
                "instruction": instruction}, ensure_ascii=False), ProfileExtraction, "项目理解")
            allowed = {b.block_id for b in chunk}
            if any(not set(f.source_blocks) <= allowed for f in extraction.facts):
                raise ValueError("Extracted project fact references missing source block")
            for fact in extraction.facts:
                fact.id = f"c{index}-" + fact.id
            store.put("profile_chunk", cache_key, {"cache_key": cache_key,
                "extraction": extraction.model_dump()}, immutable=True)
        extractions.append(extraction)
    if len(extractions) == 1:
        extraction = extractions[0]
    else:
        extraction = client.structured(json.dumps({"chunk_extractions": [e.model_dump() for e in extractions],
            "latest_raw_source_blocks": recent_source_blocks(blocks),
            "instruction": instruction + "\n整合并去重；保留最新有日期的研究重点，同时保留关键历史条件。"
            "输出约10-15个问题、少量目标假设结果约束，所有source_blocks只能来自输入引用。"
            "最新原文优先于早期摘要，以所附原文的实际日期为准；不能遗漏最新研究方向，"
            "不要把早期排期当当前目标。"
            "temporal_status标注current_candidate/historical/undated，研究日志只是历史材料，current_candidate仍待确认。"
            "旧硬件故障、旧部署NOT READY、某次相对动作失败只能当historical观察，不能当当前约束/永久排除。"
            "不能把几个月前的running当当前运行。输出未确认档案用于负责人审阅。"}, ensure_ascii=False),
            ProfileExtraction, "项目理解整合")
    profile = ProjectProfile(project_id=project_id, version=version, facts=extraction.facts,
                             blocks=blocks, unresolved=extraction.unresolved)
    store.put("snapshot", key, snapshot, immutable=True)
    store.put("profile", key, profile.model_dump(), immutable=True)
    store.put("latest_profile", project_id, {"key": key})
    return profile


def load_profile(store: Store, project: str, *, approved=True) -> ProjectProfile:
    pointer = store.get("active_profile" if approved else "latest_profile", project)
    profile = ProjectProfile.model_validate(store.get("profile", pointer["key"]))
    if approved:
        approval = store.get("approval", pointer["key"])
        profile.approved_by, profile.approved_at = approval["reviewer"], approval["at"]
    return profile


def approve(store: Store, project: str, version: str, reviewer: str):
    profile = ProjectProfile.model_validate(store.get("profile", f"{project}:{version}"))
    if not reviewer.strip() or not any(f.kind == "question" for f in profile.facts):
        raise ValueError("A named reviewer and at least one sourced project question are required")
    # Approval is a separate immutable event; never mutate the extracted project facts.
    store.put("approval", f"{project}:{version}", {"reviewer": reviewer, "at": now()}, immutable=True)
    store.put("active_profile", project, {"key": f"{project}:{version}"})
