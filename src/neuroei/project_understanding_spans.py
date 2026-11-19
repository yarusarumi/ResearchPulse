"""Opt-in provider: model selects source IDs; program extracts canonical quotations."""
import json
import re

from pydantic import Field

from .config import StrictModel
from .feishu_image_ocr import is_image_transcript
from .llm import ModelFailure
from .project_understanding import (
    INSTRUCTION,
    MAX_UNDERSTANDING_INPUT_CHARS,
    FlashUnderstandingExtractor,
    StateItem,
    UnderstandingContextBudgetExceeded,
    UnderstandingExtraction,
    identity,
    model_payload,
)

SPAN_PROVIDER = "project-understanding-span-selection-v10"
# Keep persisted evidence IDs independent of changes to the model's selection protocol.
EVIDENCE_SPAN_NAMESPACE = "project-understanding-span-selection-v3"
MAX_SPAN_CHARS = 1200
MAX_SPANS = 1536
MAX_SOURCE_CHARS = 85_000
INPUT_ENVELOPE_CHARS = 2_000
PRIOR_CONTEXT_ROLLUP_AT = 100_000
PRIOR_CONTEXT_ROLLUP_TARGET = 50_000
PRIOR_CONTEXT_FIELDS = ("prior_items", "prior_citations", "prior_sources")


class SpanSelection(StrictModel):
    span_id: str = Field(min_length=1)


class SelectedStateItem(StateItem):
    citations: list[SpanSelection] = Field(min_length=1)


class SelectedExtraction(UnderstandingExtraction):
    items: list[SelectedStateItem]


def _model_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact_prior_context(payload):
    """Keep every prior claim and quotation, replacing repeated citation fields with indexes."""
    sources, citations, items = {}, {}, []
    for item in payload["prior_items"]:
        indexes = []
        for citation in item["citations"]:
            source = (citation["source_id"], citation["record_date"], citation["source_modified_at"])
            source_index = sources.setdefault(source, len(sources))
            ref = (source_index, citation["block_id"], citation["quote"])
            indexes.append(citations.setdefault(ref, len(citations)))
        items.append({**item, "citations": indexes})
    return {**payload, "prior_items": items, "prior_citations": list(citations), "prior_sources": list(sources)}


def _history_terms(text):
    words = re.findall(r"[a-z][a-z0-9_+-]+|[\u3400-\u9fff]+", text.casefold())
    return {word for word in words if word.isascii()} | {
        word[i:i + 2] for word in words if not word.isascii() for i in range(len(word) - 1)
    }


def _history_query(texts):
    # Keep the query itself bounded when planning a very long source document.
    terms = set()
    for i, text in enumerate(texts):
        if i >= 128:
            break
        terms.update(_history_terms(text[:400]))
    return terms


def _rollup_prior_context(indexed, original_size, query):
    """Keep every prior item and locator, eliding only lower-priority historical quotes."""
    refs = indexed["prior_citations"]
    size = original_size
    items = indexed["prior_items"]
    corrections = {target for item in items if item.get("relation") in ("corrects", "contradicts")
                   for target in item["target_ids"]}
    protected = {ref for item in items
                 if item.get("kind") in ("goal", "constraint", "exclusion") or item.get("state_status") == "contested"
                 or item.get("relation") in ("corrects", "contradicts") or item["id"] in corrections
                 for ref in item["citations"]}
    sources = indexed["prior_sources"]
    quotes = list(refs)
    archived = 0
    relevance = {}
    if query:
        for item in items:
            text = " ".join((item.get("text", ""), *item.get("conditions", [])))
            item_score = len(query & _history_terms(text))
            for ref in item["citations"]:
                relevance[ref] = max(item_score, relevance.get(ref, 0))
        for ref, (_, _, quote) in enumerate(refs):
            if quote and ref not in protected:
                relevance[ref] = max(len(query & _history_terms(quote)), relevance.get(ref, 0))
    relevant_archived = 0
    candidates = sorted((i for i in range(len(refs)) if i not in protected), key=lambda i: (
        relevance.get(i, 0), sources[refs[i][0]][1] or "", sources[refs[i][0]][2] or "", i))
    for i in candidates:
        if size <= PRIOR_CONTEXT_ROLLUP_TARGET:
            break
        source, block, quote = refs[i]
        savings = len(_model_json(quote)) - len("null")
        if savings > 0:
            quotes[i] = (source, block, None)
            size -= savings
            archived += 1
            relevant_archived += bool(relevance.get(i))
    return {**indexed, "prior_citations": quotes}, {"indexed_characters": original_size,
        "sent_characters": size, "archived_quotes": archived, "target_reached": size <= PRIOR_CONTEXT_ROLLUP_TARGET,
        "selection": "current_material_overlap_then_date", "protected_quotes": len(protected),
        "relevant_quotes_archived": relevant_archived}


def _prior_groups(items):
    """Keep corrections and contradictions with the historical items they qualify."""
    positions = {item["id"]: i for i, item in enumerate(items)}
    links = [set() for _ in items]
    for index, item in enumerate(items):
        if item.get("relation") in ("corrects", "contradicts"):
            for target in item.get("target_ids", ()):
                if target in positions:
                    other = positions[target]
                    links[index].add(other)
                    links[other].add(index)
    pending = set(range(len(items)))
    for index in range(len(items)):
        if index in pending:
            group, frontier = [], [index]
            pending.remove(index)
            while frontier:
                current = frontier.pop()
                group.append(current)
                unseen = links[current] & pending
                pending.difference_update(unseen)
                frontier.extend(unseen)
            yield group


def _select_prior_context(candidate, columns, rows, query, budget):
    """Fit complete historical rows into the space left over after the current evidence."""
    items, refs, sources = (candidate[field] for field in PRIOR_CONTEXT_FIELDS)
    ranks = []
    for index, item in enumerate(items):
        overlap = len(query & _history_terms(" ".join((item["text"], *item.get("conditions", []))))) if query else 0
        background = item.get("kind") in ("goal", "constraint", "exclusion")
        date = item.get("occurred_on") or max((sources[refs[ref][0]][1] or sources[refs[ref][0]][2] or ""
                                             for ref in item["citations"]), default="")
        ranks.append((overlap, background, date, index))
    groups = sorted(_prior_groups(items), key=lambda group: max(ranks[i] for i in group), reverse=True)
    reduced = {**candidate, "prior_item_columns": columns, "prior_items": [], "prior_citations": [],
               "prior_sources": [], "prior_items_omitted": len(items)}
    remaining = budget - len(_model_json(reduced))
    included, citation_ids, source_ids = set(), set(), set()
    for group in groups:
        new_refs = {ref for index in group for ref in items[index]["citations"]} - citation_ids
        new_sources = {refs[ref][0] for ref in new_refs} - source_ids
        # Renumbering is in original order, so numeric IDs only get shorter.
        cost = (sum(len(_model_json(rows[index])) + 1 for index in group)
                + sum(len(_model_json(refs[ref])) + 1 for ref in new_refs)
                + sum(len(_model_json(sources[source])) + 1 for source in new_sources))
        if cost <= remaining:
            remaining -= cost
            included.update(group)
            citation_ids.update(new_refs)
            source_ids.update(new_sources)
    ordered_refs, ordered_sources = sorted(citation_ids), sorted(source_ids)
    citation_index = {ref: i for i, ref in enumerate(ordered_refs)}
    source_index = {source: i for i, source in enumerate(ordered_sources)}
    citation_column = columns.index("citations")
    for index in sorted(included):
        row = rows[index].copy()
        row[citation_column] = [citation_index[ref] for ref in row[citation_column]]
        reduced["prior_items"].append(row)
    reduced["prior_citations"] = [(source_index[refs[i][0]], *refs[i][1:]) for i in ordered_refs]
    reduced["prior_sources"] = [sources[i] for i in ordered_sources]
    reduced["prior_items_omitted"] -= len(included)
    return _model_json(reduced), {"total_items": len(items), "sent_items": len(included),
        "omitted_items": reduced["prior_items_omitted"], "omitted_citations": len(refs) - len(citation_ids),
        "selection": "current_material_overlap_then_project_background_then_recency",
        "corrections_and_contradictions_grouped": True}


def _fit_model_input(projected, reserved=0, *, history_query=()):
    """Try successively smaller representations for both planning and actual requests."""
    plain = _model_json(projected)
    prior_size = len(_model_json({"prior_items": projected["prior_items"]}))
    if prior_size <= PRIOR_CONTEXT_ROLLUP_AT and len(plain) + reserved <= MAX_UNDERSTANDING_INPUT_CHARS:
        return plain, None, None
    indexed = _compact_prior_context(projected)
    indexed_size = len(_model_json({field: indexed[field] for field in PRIOR_CONTEXT_FIELDS}))
    candidate, encoding, audit = indexed, "indexed_citations", None
    if indexed_size > PRIOR_CONTEXT_ROLLUP_AT:
        rolled, rollup = _rollup_prior_context(indexed, indexed_size, set(history_query))
        if rollup["archived_quotes"]:
            candidate, encoding, audit = rolled, "indexed_citations_with_archived_quotes", rollup
    compact = _model_json(candidate)
    if len(compact) + reserved > MAX_UNDERSTANDING_INPUT_CHARS and candidate["prior_items"]:
        items = candidate["prior_items"]
        columns = list(dict.fromkeys(key for item in items for key in item))
        rows = [[item.get(key) for key in columns] for item in items]
        packed = _model_json({**candidate, "prior_item_columns": columns, "prior_items": rows})
        if len(packed) < len(compact):
            compact, encoding = packed, encoding + "_and_item_rows"
        # Planning needs room for at least one source piece plus its JSON envelope.
        minimum = min(reserved, INPUT_ENVELOPE_CHARS + 6 * MAX_SPAN_CHARS)
        if min(len(compact), len(plain)) + minimum > MAX_UNDERSTANDING_INPUT_CHARS:
            selected, selection_audit = _select_prior_context(
                candidate, columns, rows, set(history_query), MAX_UNDERSTANDING_INPUT_CHARS - reserved)
            return selected, "selected_historical_item_rows", {**(audit or {}), "item_selection": selection_audit}
    return (compact, encoding, audit) if len(compact) < len(plain) else (plain, None, None)


def _block_parts(text):
    """Yield exact intervals, including gaps and image names the model cannot cite."""
    cursor = previous = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        content = line.rstrip("\r\n")
        if not content.strip(" \t\r\n\u200b\ufeff"):
            continue
        if previous < start:
            yield previous, start, None
        end = start + len(content)
        if re.fullmatch(r"\S+\.(?:png|jpe?g|gif|webp)", content.strip(), flags=re.IGNORECASE):
            yield start, end, "image"
        else:
            for a in range(start, end, MAX_SPAN_CHARS):
                yield a, min(a + MAX_SPAN_CHARS, end), "text"
        previous = end
    if previous < len(text):
        yield previous, len(text), None


def _project_sources(records, originals):
    full = {(r.source_id, b.block_id): b for r in originals for b in r.blocks}
    versions = {r.source_id: r.content_version for r in originals if any(
        rec.source_id == r.source_id for rec in records)}
    spans, unread, journals = [], [], []
    for rec in records:
        sections = list(dict.fromkeys(b.section for b in rec.blocks))
        by_section = {section: i for i, section in enumerate(sections)}
        blocks = []
        for block in rec.blocks:
            full_block = full[(rec.source_id, block.block_id)]
            offset = full_block.text.find(block.text)
            if offset < 0 or full_block.text.find(block.text, offset + 1) >= 0:
                raise ValueError("Supplied excerpt has no unique position in its original block")
            parts = []
            version = versions[rec.source_id]
            for start, end, kind in _block_parts(full_block.text):
                a, b = max(offset, start), min(offset + len(block.text), end)
                if a >= b:
                    continue
                text = full_block.text[a:b]
                span_id = None
                if kind == "image":
                    unread.append({"source_id": rec.source_id, "block_id": block.block_id,
                                   "source_char_start": a, "text": text})
                if kind == "text":
                    span_id = f"s{len(spans) + 1:03d}"
                    spans.append({"span_id": span_id,
                        "evidence_span_id": "s" + identity([EVIDENCE_SPAN_NAMESPACE, rec.source_id,
                            version, block.block_id, a, b])[:12],
                        "source_id": rec.source_id, "block_id": block.block_id,
                        "source_version": version, "source_char_start": a,
                        "source_char_end": b, "offset_unit": "unicode_codepoints", "text": text})
                    if len(spans) > MAX_SPANS:
                        raise ValueError("Source span table exceeds bounded material budget")
                parts.append([span_id, text])
            blocks.append([by_section[block.section], parts])
        journals.append({**rec.model_dump(mode="json", exclude={"blocks", "user_id", "project_id"}),
                         "sections": sections, "blocks": blocks})
    if not spans or len({s["evidence_span_id"] for s in spans}) != len(spans):
        raise ValueError("Source span table is empty or contains ambiguous IDs")
    return spans, unread, journals


def build_span_table(records, originals):
    spans, unread, _ = _project_sources(records, originals)
    return spans, unread


def _plan_journal_chunks(records, source_budget):
    """Pack ordered blocks using the compact source representation's size and span count."""
    chunks = []
    def image_record(record):
        return all(is_image_transcript(block.section) for block in record.blocks)

    for record in sorted(records, key=image_record):
        blocks, sections = [], set()
        size = span_count = 0
        for block in record.blocks:
            parts = list(_block_parts(block.text))
            weights = [(len(_model_json(["s0000" if kind == "text" else None, block.text[a:b]])) + 1,
                        kind == "text") for a, b, kind in parts]
            section_chars = len(_model_json(block.section)) + 1
            block_overhead = 12  # Array delimiters and the section index.
            block_chars = block_overhead + sum(n for n, _ in weights)
            count = sum(n for _, n in weights)
            if block_chars + section_chars > source_budget or count > MAX_SPANS:
                # Split at existing piece boundaries so evidence intervals stay stable.
                start = cost = selected = 0
                if blocks:
                    chunks.append([record.model_copy(update={"blocks": blocks})])
                    blocks, sections, size, span_count = [], set(), 0, 0
                for (a, _, _), (n, is_span) in zip(parts, weights, strict=True):
                    if n + block_overhead + section_chars > source_budget:
                        raise UnderstandingContextBudgetExceeded("Source piece exceeds available model context budget")
                    if cost + n + block_overhead + section_chars > source_budget or selected + is_span > MAX_SPANS:
                        excerpt = block.model_copy(update={"text": block.text[start:a]})
                        chunks.append([record.model_copy(update={"blocks": [excerpt]})])
                        start = a
                        cost = selected = 0
                    cost += n
                    selected += is_span
                block = block.model_copy(update={"text": block.text[start:]})
                block_chars, count = block_overhead + cost, selected
            additional_section = section_chars if block.section not in sections else 0
            if blocks and (size + block_chars + additional_section > source_budget or span_count + count > MAX_SPANS):
                chunks.append([record.model_copy(update={"blocks": blocks})])
                blocks, sections, size, span_count = [], set(), 0, 0
                additional_section = section_chars
            blocks.append(block)
            sections.add(block.section)
            size += block_chars + additional_section
            span_count += count
        if blocks:
            chunks.append([record.model_copy(update={"blocks": blocks})])
    # A document may contain several small image transcripts. Share one request where
    # possible while keeping each picture's date, version and evidence identity.
    packed = []
    for chunk in chunks:
        if (packed and image_record(chunk[0]) and all(image_record(row) for row in packed[-1])
                and chunk[0].source_id not in {row.source_id for row in packed[-1]}):
            candidate = [*packed[-1], *chunk]
            if sum(1 for row in candidate for block in row.blocks
                   for _, _, kind in _block_parts(block.text) if kind == 'text') <= MAX_SPANS:
                _, _, journals = _project_sources(candidate, records)
                if len(_model_json(journals)) <= source_budget:
                    packed[-1] = candidate
                    continue
        packed.append(chunk)
    return packed


def canonical_from_selection(provider, table):
    by_id = {s["span_id"]: s for s in table}
    if len(by_id) != len(table):
        raise ValueError("Ambiguous source span IDs")
    raw = provider.model_dump(mode="json")
    items, conversions = [], []
    for item_index, item in enumerate(raw["items"]):
        citations = []
        for citation_index, selection in enumerate(item["citations"]):
            span = by_id.get(selection["span_id"])
            if span is None:
                raise ValueError("Unknown source span selection")
            citations.append({"source_id": span["source_id"], "block_id": span["block_id"], "quote": span["text"]})
            conversions.append({"item_id": item["id"], "item_index": item_index, "citation_index": citation_index, **selection,
                "evidence_span_id": span["evidence_span_id"],
                "source_id": span["source_id"], "block_id": span["block_id"],
                "source_version": span["source_version"],
                "quote_producer": "program_extracted_selected_span", "model_quote_supplied": False,
                "source_char_start": span["source_char_start"], "source_char_end": span["source_char_end"]})
        items.append({**item, "citations": citations})  # All claim text and other fields are untouched.
    extraction = UnderstandingExtraction.model_validate({**raw, "items": items})
    # Keep the selected occurrence attached when validation filters items or citations.
    for item_index, item in enumerate(extraction.items):
        for citation_index, citation in enumerate(item.citations):
            citation._provider_location = (item_index, citation_index)
    return extraction, conversions


class SpanSelectingUnderstandingExtractor(FlashUnderstandingExtractor):
    """Separate opt-in provider; the existing FlashUnderstandingExtractor is unchanged."""

    bounds_model_input = True

    def __init__(self, client):
        super().__init__(client)
        self.provider_audit = None
        self.model_journals = None

    @property
    def extraction_settings(self):
        return {**super().extraction_settings, "provider_protocol": SPAN_PROVIDER,
                "max_span_characters": MAX_SPAN_CHARS, "max_spans": MAX_SPANS,
                "max_source_characters": MAX_SOURCE_CHARS,
                "prior_context_rollup_at": PRIOR_CONTEXT_ROLLUP_AT,
                "prior_context_rollup_target": PRIOR_CONTEXT_ROLLUP_TARGET}

    def plan_chunks(self, records, prior_items, chunk_chars):
        prior = model_payload({"journals": [], "prior_items": prior_items})
        source_budget = MAX_SOURCE_CHARS
        if chunk_chars is not None:
            source_budget = min(source_budget, chunk_chars)
        query = _history_query(block.text for record in records for block in record.blocks)
        prior_chars = len(_fit_model_input(prior, reserved=source_budget + INPUT_ENVELOPE_CHARS,
                                           history_query=query)[0])
        source_budget = min(source_budget, MAX_UNDERSTANDING_INPUT_CHARS - prior_chars - INPUT_ENVELOPE_CHARS)
        if source_budget <= 0:
            raise UnderstandingContextBudgetExceeded("Existing project state exceeds the model context budget")
        return _plan_journal_chunks(records, source_budget)

    def prepare_sources(self, records, originals):
        images = any(is_image_transcript(block.section)
                     for record in records for block in record.blocks)
        self.provider_audit = {"provider_protocol": SPAN_PROVIDER, "status": "preparing",
                               "material_scope": "text_with_image_transcripts" if images else "text_only_images_not_read",
                               "raw_provider_output": None}
        try:
            table, unread, self.model_journals = _project_sources(records, originals)
        except ValueError:
            self.provider_audit["status"] = "source_table_rejected"
            raise
        self.provider_audit.update(status="prepared", span_table=table, unread_markers=unread)

    def citation_location(self, item_id, citation, record):
        """Preserve the chosen occurrence, never rediscover the first identical text."""
        audit = self.provider_audit or {}
        rows = [c for c in audit.get("conversions", []) if
                c["item_id"] == item_id and (c["item_index"], c["citation_index"]) == citation._provider_location]
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous selected citation location")
        selected = rows[0]
        spans = [s for s in audit.get("span_table", []) if s["span_id"] == selected["span_id"]]
        if len(spans) != 1:
            raise ValueError("Missing or ambiguous selected source span")
        span = spans[0]
        for name in ("evidence_span_id", "source_id", "block_id", "source_version", "source_char_start", "source_char_end"):
            if selected[name] != span[name]:
                raise ValueError("Selected citation provenance differs from source span")
        block = next((b for b in record.blocks if b.block_id == citation.block_id), None)
        start, end = span["source_char_start"], span["source_char_end"]
        if (span["source_id"] != record.source_id or span["source_id"] != citation.source_id or
                span["block_id"] != citation.block_id or span["source_version"] != record.content_version or
                block is None or type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(block.text) or
                block.text[start:end] != span["text"] or span["text"] != citation.quote):
            raise ValueError("Selected citation location does not match its exact source/version")
        return {"quote_char_start": start, "quote_char_end": end,
                "span_id": span["evidence_span_id"], "source_version": span["source_version"],
                "offset_unit": "unicode_codepoints", "quote_producer": "program_extracted_selected_span"}

    def extract(self, payload):
        if not self.provider_audit or self.provider_audit["status"] != "prepared":
            raise ValueError("Source preparation is required before span selection")
        table = self.provider_audit["span_table"]
        projected = {**model_payload({**payload, "journals": []}), "journals": self.model_journals}
        query = _history_query(text for journal in self.model_journals for _, parts in journal["blocks"]
                               for span_id, text in parts if span_id)
        model_input, encoding, rollup = _fit_model_input(projected, history_query=query)
        if encoding:
            self.provider_audit["prior_context_encoding"] = encoding
        if rollup:
            self.provider_audit["prior_context_rollup"] = rollup
        self.provider_audit["model_input_characters"] = len(model_input)
        if len(model_input) > MAX_UNDERSTANDING_INPUT_CHARS:
            self.provider_audit["status"] = "context_budget_exceeded"
            raise UnderstandingContextBudgetExceeded("Source span provider context budget exceeded")
        system = ("下面业务规则约束最终canonical结果。当前provider阶段只选择材料，不填写引文quote。\n" +
            INSTRUCTION + "\n本次仅按以下provider schema返回JSON。citations只写span_id，"
            "journals的每个block是[章节索引, 片段列表]，索引从0开始，对应同一日报的sections。"
            "每个片段是[短编号, 原文]，null编号的片段只供上下文，不可引用。"
            "若提供prior_citations，prior_items的citations为其中的零起点索引。"
            "若提供prior_item_columns，prior_items的每一行按该字段顺序展开；空值表示该字段为空或未提供。"
            "prior_citations每行是[prior_sources的零起点索引, 块ID, 逐字引文]；"
            "prior_sources每行是[来源ID, 记录日期, 来源修改时间]。这些是历史证据，不能选为本次引文。"
            "历史引文为null表示原文未送入本次分析，位置仍保留；不要推断被省略的原文内容。"
            "prior_items_omitted是因输入空间未送入的历史条目数；不能把缺少历史误判成首次发现或旧结论已撤回。"
            "对照已有背景，分别识别延续、已记录的修正与仍需验证的事项；仅在本次原文支持时才报告变化。"
            "project_context来自用户填写的说明或已分析的日报，仅作理解新日报的背景，不能作为本次引文或实验结果。"
            "先找到支持结论的原文，再逐字复制其编号（如s001）。source_id/block_id/版本/位置由程序绑定，禁止模型填写。"
            "禁止自填quote、改ID或合并不连续片段。"
            "多段证据分别选择，结论文字仍须有据；image.png等占位不代表读取图像。"
            "标为图片文字识别或 SOL 图片转写（待核对）的段落是自动转写，只能记录图中直接可见的内容。"
            "只依据这些段落的观察标为unconfirmed，不用于修正原有结论，不推断曲线、精确数值或图表含义。\n" +
            json.dumps(SelectedExtraction.model_json_schema(), ensure_ascii=False))
        response_options = {}
        for attempt in range(2):
            self.provider_audit["status"] = "request_started"
            try:
                content = self.client.complete(model_input, system=system, json_mode=True, **response_options)
            except ModelFailure as error:
                if (attempt or error.reason not in ("model_empty_content", "output_truncated")
                        or self.client.remaining() <= 0):
                    raise
                self.provider_audit.setdefault("provider_repair_attempts", []).append({
                    "rejection": error.reason, "finish_reason": getattr(self.client, "last_finish_reason", None)})
                system += "\n上次未交付完整结果。直接返回简洁、完整的 JSON，逐条使用本次材料的 span_id。"
                response_options["thinking_enabled"] = False
                continue
            finally:
                self.provider_audit["finish_reason"] = getattr(self.client, "last_finish_reason", None)
            self.provider_audit.update(raw_provider_output=content, status="response_received")
            self.provider_audit.pop("provider_object", None)
            rejection = "provider_schema_rejected"
            try:
                provider = SelectedExtraction.model_validate_json(content)
                self.provider_audit["provider_object"] = provider.model_dump(mode="json")
                rejection = "selection_rejected"
                canonical, conversions = canonical_from_selection(provider, table)
            except ValueError as error:
                self.provider_audit["status"] = rejection
                remaining = (self.client.remaining() if hasattr(self.client, "remaining") else
                             self.client.config.max_calls_per_run - self.client.calls)
                if attempt or remaining <= 0:
                    raise
                errors = [{"path": ".".join(map(str, row["loc"])), "type": row["type"]}
                          for row in getattr(error, "errors", lambda: [])()[:12]]
                self.provider_audit.setdefault("provider_repair_attempts", []).append({
                    "raw_provider_output": content, "rejection": rejection, "errors": errors})
                # Correct the response using the same source payload. Identity and quotations
                # still come only from the original span table after strict validation.
                system += ("\n上一份响应未通过结构或片段编号校验。请依据同一材料重新生成完整 JSON，"
                           "严格使用给定 schema 的字段和类别，不回显 project_context 或增加根字段；"
                           "citations 仅选当前材料列出的 span_id，不自填 quote 或来源身份。校验位置：" +
                           json.dumps(errors, ensure_ascii=False))
                continue
            self.provider_audit.update(status="canonical_converted", conversions=conversions)
            return canonical
