"""Mechanical, side-effect-free comparison of two project profiles.

This supports the P2 incremental-sync step: after source documents change, a
human re-confirms the profile before it becomes active. The comparison only
reports structural differences. It never claims that two texts are semantically
equivalent, never approves a profile and never activates one.
"""

from .models import ProjectFact, ProjectProfile

CRITICAL_FACT_KINDS = frozenset({"goal", "hypothesis", "observation", "constraint", "exclusion"})

LIMITATION = (
    "仅做机械对比，不判断语义等价，不自动审批，也不切换生效档案；"
    "事实按稳定 ID 对齐，ID 消失或出现按删除/新增保守处理，不按文本猜测重命名；"
    "同一事实的 source_blocks 只比较引用集合，顺序变化不算内容变化；"
    "文档只比较 document_id 与 revision 集合，忽略行号与块列表顺序；"
    "版本、创建时间和审批字段不参与比较。"
)


def _fact_content(fact: ProjectFact) -> tuple:
    # source_blocks are references, not ordered prose; the model validates them as a set.
    return (
        fact.text,
        fact.kind,
        tuple(sorted(fact.source_blocks)),
        fact.provenance,
        fact.source_date,
        fact.temporal_status,
    )


def _document_revisions(profile: ProjectProfile) -> dict[str, frozenset[str]]:
    revisions: dict[str, set[str]] = {}
    for block in profile.blocks:
        revisions.setdefault(block.document_id, set()).add(block.revision)
    return {document_id: frozenset(values) for document_id, values in revisions.items()}


def _is_critical(fact: ProjectFact | None) -> bool:
    return fact is not None and fact.kind in CRITICAL_FACT_KINDS


def compare_profiles(before: ProjectProfile, after: ProjectProfile) -> dict:
    """Return a mechanical diff of two profiles that must belong to one project."""
    if before.project_id != after.project_id:
        raise ValueError("Cannot compare project profiles from different projects")

    before_facts = {fact.id: fact for fact in before.facts}
    after_facts = {fact.id: fact for fact in after.facts}
    before_ids, after_ids = set(before_facts), set(after_facts)
    added_facts = sorted(after_ids - before_ids)
    removed_facts = sorted(before_ids - after_ids)
    changed_facts = sorted(
        fact_id
        for fact_id in before_ids & after_ids
        if _fact_content(before_facts[fact_id]) != _fact_content(after_facts[fact_id])
    )
    critical_fact_ids = sorted(
        fact_id
        for fact_id in [*added_facts, *removed_facts, *changed_facts]
        if _is_critical(before_facts.get(fact_id)) or _is_critical(after_facts.get(fact_id))
    )

    before_documents = _document_revisions(before)
    after_documents = _document_revisions(after)
    before_docs, after_docs = set(before_documents), set(after_documents)
    added_documents = sorted(after_docs - before_docs)
    removed_documents = sorted(before_docs - after_docs)
    changed_documents = sorted(
        document_id
        for document_id in before_docs & after_docs
        if before_documents[document_id] != after_documents[document_id]
    )

    # An ordered rewrite of the same unresolved notes is not new content.
    unresolved_changed = sorted(before.unresolved) != sorted(after.unresolved)
    has_changes = bool(
        added_facts
        or removed_facts
        or changed_facts
        or added_documents
        or removed_documents
        or changed_documents
        or unresolved_changed
    )
    return {
        "project_id": before.project_id,
        "from_version": before.version,
        "to_version": after.version,
        "has_changes": has_changes,
        # Policy for now: every content change needs re-confirmation. Critical facts
        # only mark the higher-risk subset; they do not change this decision.
        "requires_approval": has_changes,
        "facts": {"added": added_facts, "removed": removed_facts, "changed": changed_facts},
        "documents": {
            "added": added_documents,
            "removed": removed_documents,
            "changed": changed_documents,
        },
        "unresolved_changed": unresolved_changed,
        "critical_fact_ids": critical_fact_ids,
        "limitation": LIMITATION,
    }
