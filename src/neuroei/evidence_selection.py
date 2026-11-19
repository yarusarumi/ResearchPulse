"""Deterministic lexical selection with explicit coverage; no semantic retrieval claim."""
import re
from collections import Counter

from rank_bm25 import BM25Okapi

from .research import tokenize

# Deepest heading wins; tuple order only breaks ties when one heading matches two rules.
_SECTION_RULES = (
    ("limitations", re.compile(
        r"\b(limitations?|scope|failures?|confidence|uncertainty|not guarantee)\b", re.IGNORECASE)),
    ("settings", re.compile(
        r"\b(set.?up|settings|protocols?|statistical|implementation|training|hyperparameters?|metrics?)\b", re.IGNORECASE)),
    ("results", re.compile(
        r"\b(results?|experiments?|experimental|evaluations?|ablations?|comparisons?|success|performance|tables?)\b", re.IGNORECASE)),
    ("method", re.compile(
        r"\b(methods?|formulation|overview|mechanism|algorithms?|dynamics|approach)\b", re.IGNORECASE)),
)
_QUOTAS = (("method", 6), ("settings", 5), ("results", 6), ("limitations", 3))
_SETUP_QUOTA = 4
_REFERENCE_HEADING = re.compile(r"\b(references?|bibliography)\b", re.IGNORECASE)
_NON_BODY_HEADING = re.compile(
    r"\b(introduction|background|related work|motivation|abstract)\b", re.IGNORECASE)

# Experiment-setup prose can sit in unlabelled blocks; these signals make it selectable.
SETUP_SIGNALS = (
    ("random_seeds", re.compile(r"\brandom(?:ly)?\s+seed(?:s|ed|ing)?\b", re.IGNORECASE)),
    ("seed_count", re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:random\s+)?seeds?\b",
        re.IGNORECASE)),
    ("seeds", re.compile(r"\bseeds?\b", re.IGNORECASE)),
    ("training_runs", re.compile(r"\btraining\s+runs?\b", re.IGNORECASE)),
    ("trials", re.compile(r"\b(?:trials?|replicates?|replications?)\b", re.IGNORECASE)),
    ("standard_error", re.compile(r"\b(?:std\.?\s*err(?:or)?|standard\s+error)\b", re.IGNORECASE)),
    ("bootstrap", re.compile(r"\bbootstrap(?:ped|ping|s)?\b", re.IGNORECASE)),
    ("confidence_interval", re.compile(r"\bconfidence\s+intervals?\b", re.IGNORECASE)),
    ("ci", re.compile(r"\bCI\b", re.IGNORECASE)),
    ("training_protocol", re.compile(r"\bwe\s+train\b|\btrained\s+for\b", re.IGNORECASE)),
    ("dataset_size", re.compile(r"\b(?:dataset|data)\s+sizes?\b|\bdataset\s+of\b", re.IGNORECASE)),
    ("episodes", re.compile(r"\bepisodes?\b", re.IGNORECASE)),
    ("demonstrations", re.compile(r"\bdemonstrations?\b", re.IGNORECASE)),
    ("per_axis", re.compile(r"\bper\s+(?:model|modality|environment)s?\b", re.IGNORECASE)),
    ("zh_random_seed", re.compile(r"随机种子|种子")),
    ("zh_repeats", re.compile(r"独立重复|重复实验|重复训练|重复试验")),
    ("zh_standard_error", re.compile(r"标准误")),
    ("zh_confidence_interval", re.compile(r"置信区间")),
    ("zh_per_axis", re.compile(r"每(?:个|种|类)?(?:模型|模态|环境)")),
)
# Quantitative/error-bar and per-axis signals outrank generic setup wording.
_STRONG_SETUP_SIGNALS = frozenset((
    "random_seeds", "seed_count", "seeds", "trials", "standard_error", "bootstrap",
    "confidence_interval", "ci", "per_axis", "zh_random_seed", "zh_repeats",
    "zh_standard_error", "zh_confidence_interval", "zh_per_axis"))


def setup_signals(text):
    """Return matched setup-signal names in stable declaration order."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    return [name for name, pattern in SETUP_SIGNALS if pattern.search(normalized)]


def _block_headings(section_context, block_id):
    raw = (section_context.get(block_id) or {}).get("headings") or []
    headings = []
    for item in raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("title") or "")
        else:
            text = str(item)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            headings.append(text)
    return headings


def _strip_document_title(paths):
    """Drop the shared document title so its words cannot label every block."""
    non_empty = [path for path in paths if path]
    if not non_empty:
        return paths
    title, count = Counter(path[0] for path in non_empty).most_common(1)[0]
    shared = count * 5 >= len(non_empty) * 4
    if not shared or not any(len(path) > 1 and path[0] == title for path in non_empty):
        return paths
    return [path[1:] if path and path[0] == title else path for path in paths]


def _section_label(headings):
    for heading in reversed(headings):
        for label, pattern in _SECTION_RULES:
            if pattern.search(heading):
                return label
    return None


def _is_reference(record, headings):
    if any(_REFERENCE_HEADING.search(heading) for heading in headings):
        return True
    section = record.section or ""
    return bool(section and not section.startswith("#") and _REFERENCE_HEADING.search(section))


def _looks_tabular(record, headings, kind):
    if kind == "table" or " | " in record.text:
        return True
    if re.match(r"^[Tt]\d", record.block_id or ""):
        return True
    return any("table" in heading.lower() for heading in headings)


def _has_quantity(record):
    return bool(re.search(r"\d", record.text))


def _normalize_heading(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _related_to_focus(headings, focus_headings):
    """Same section or an adjacent subsections/parent of a focus heading."""
    for heading in headings:
        normalized = _normalize_heading(heading)
        if not normalized:
            continue
        for target in focus_headings:
            if normalized == target or normalized.startswith(target) or target.startswith(normalized):
                return True
    return False


def select_evidence(evidence, query, section_context=None, limit=24, *, focus_blocks=None):
    """Heading-first lexical selection; body words only fill blocks without a section label.

    ``focus_blocks`` seeds a small number of caller-required block ids (key setup/table
    notes) before the quotas run, and makes same/adjacent-section candidates outrank
    unrelated seed sentences. Without it the selection is unchanged.
    """
    context = section_context or {}
    eligible, paths, kinds = [], [], []
    for record in evidence:
        headings = _block_headings(context, record.block_id)
        if _is_reference(record, headings):
            continue
        eligible.append(record)
        paths.append(headings)
        kinds.append((context.get(record.block_id) or {}).get("kind"))
    if not eligible:
        raise ValueError("No body evidence available")
    paths = _strip_document_title(paths)
    labels = [_section_label(path) for path in paths]
    blocked = [bool(_NON_BODY_HEADING.search(" ".join(path))) for path in paths]
    scores = BM25Okapi([tokenize(record.text) for record in eligible]).get_scores(tokenize(query))
    order = sorted(range(len(eligible)), key=lambda i: (-scores[i], i))
    rank = {index: position for position, index in enumerate(order)}
    primary = {label: [] for label, _ in _QUOTAS}
    fallback = {label: [] for label, _ in _QUOTAS}
    for index in order:
        label = labels[index]
        if label:
            primary[label].append(index)
        elif not blocked[index]:
            for name, pattern in _SECTION_RULES:
                if pattern.search(eligible[index].text):
                    fallback[name].append(index)
    chosen, chosen_set, groups = [], set(), {}
    coverage = {label: 0 for label, _ in _QUOTAS}
    coverage["setup"] = 0
    coverage["lexical"] = 0
    requested_focus = [str(item) for item in focus_blocks or () if str(item or "")]
    focus_index = {str(record.block_id): index for index, record in enumerate(eligible)}
    focus_included, focus_missing = [], []
    for block_id in requested_focus:
        index = focus_index.get(block_id)
        if index is None:
            focus_missing.append(block_id)
            continue
        if index in chosen_set or len(chosen) >= limit:
            if index not in chosen_set:
                focus_missing.append(block_id)
            continue
        chosen.append(index)
        chosen_set.add(index)
        groups.setdefault(eligible[index].evidence_id, []).append("focus")
        focus_included.append(block_id)
    if requested_focus:
        coverage["focus"] = len(focus_included)
    focus_headings = {_normalize_heading(heading) for block_id in focus_included
                      for heading in paths[focus_index[block_id]]}
    focus_headings.discard("")

    def focused(index):
        return bool(focus_headings) and _related_to_focus(paths[index], focus_headings)

    def add(label, candidates, quota):
        added = 0
        for index in candidates:
            if added >= quota or len(chosen) >= limit:
                return
            if index in chosen_set:
                continue
            chosen.append(index)
            chosen_set.add(index)
            groups.setdefault(eligible[index].evidence_id, []).append(label)
            added += 1
            coverage[label] += 1

    def quantitative_first(candidates):
        def priority(index):
            if _looks_tabular(eligible[index], paths[index], kinds[index]):
                return 0
            return 1 if _has_quantity(eligible[index]) else 2
        return sorted(candidates, key=lambda i: (priority(i), rank[i]))

    for label, quota in _QUOTAS:
        if label == "results":
            add(label, quantitative_first(primary[label]) + quantitative_first(fallback[label]), quota)
        else:
            candidates = primary[label] + fallback[label]
            if focus_headings:
                candidates = sorted(candidates, key=lambda i: (0 if focused(i) else 1, rank[i]))
            add(label, candidates, quota)

    def setup_priority(index):
        names = setup_signals(eligible[index].text)
        strong = any(name in _STRONG_SETUP_SIGNALS for name in names)
        return (0 if focused(index) else 1, 0 if strong else 1, rank[index])

    setup_candidates = [index for index in range(len(eligible))
                        if index not in chosen_set and setup_signals(eligible[index].text)]
    add("setup", sorted(setup_candidates, key=setup_priority), _SETUP_QUOTA)
    for index in order:
        if len(chosen) >= limit:
            break
        if index in chosen_set:
            continue
        chosen.append(index)
        chosen_set.add(index)
        groups.setdefault(eligible[index].evidence_id, []).append("lexical")
        coverage["lexical"] += 1
    selected = [eligible[index] for index in sorted(chosen)]
    retrieval = {"strategy": "bm25-section-coverage-v3", "limit": limit, "query": query,
        "selected_ids": [record.evidence_id for record in selected], "selection_groups": groups,
        "coverage": coverage, "setup_blocks": coverage["setup"],
        "quotas": {label: quota for label, quota in _QUOTAS},
        "eligible_blocks": len(eligible), "available_blocks": len(evidence),
        "excluded_reference_blocks": len(evidence) - len(eligible),
        "limitations": "Lexical selection can miss evidence; quotas do not establish complete section coverage"}
    if requested_focus:
        retrieval["focus_requested"] = len(requested_focus)
        retrieval["focus_included"] = focus_included
        retrieval["focus_missing"] = focus_missing
    return selected, retrieval
