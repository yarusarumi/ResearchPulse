"""Explain quote migration failures without relaxing the analysis evidence gate."""
import re

from .html_reader import ArxivHTML


def presentation_text(text):
    # Only prose typography: do not normalize signs, decimals, equations or units.
    text = re.sub(r"(?<=[A-Za-z])[\u2010\u2011\u2013](?=[A-Za-z])", "-", text)
    text = re.sub(r"(?<=[A-Za-z)])\s+([.,;:!?])(?=\s|$)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_quote(quote, text):
    if not quote:
        return False
    for match in re.finditer(re.escape(quote), text):
        before, after = text[:match.start()], text[match.end():]
        if quote[0].isdigit() and before and before[-1] in "0123456789.-−":
            continue
        if quote[-1].isdigit() and after and after[0] in "0123456789.":
            continue
        if len(quote) > 1 and quote[-1] == "." and quote[-2].isdigit() and after[:1].isdigit():
            continue
        return True
    return False


def locate_quote(quote, records, *, original_records=(), expected_anchor=None):
    target = presentation_text(quote)
    result = {"status": "unresolved", "block_ids": [], "original_anchors": [],
              "original_text": None, "expected_anchor": expected_anchor,
              "source_anchor_found": False,
              "historical_anchor_mismatch": False, "semantic_verified": False}
    if not target:
        return result
    hits = [(anchor, text) for anchor, text in original_records if contains_quote(target, presentation_text(text))]
    original_by_anchor = dict(original_records)
    result["source_anchor_found"] = expected_anchor in original_by_anchor
    result["original_text"] = original_by_anchor.get(expected_anchor)
    result["original_anchors"] = [anchor for anchor, _ in hits]
    if hits:
        result["original_text"] = next((t for a, t in hits if a == expected_anchor), hits[0][1])
        result["historical_anchor_mismatch"] = bool(expected_anchor and expected_anchor not in result["original_anchors"])
    exact = [r["block_id"] for r in records if contains_quote(
             re.sub(r"\s+", " ", quote).strip(), re.sub(r"\s+", " ", r["text"]).strip())]
    if exact:
        return {**result, "status": "exact", "block_ids": exact}
    formatted = [r["block_id"] for r in records if contains_quote(target, presentation_text(r["text"]))]
    if formatted:
        return {**result, "status": "presentation", "block_ids": formatted}
    for i, first in enumerate(records):
        span = [first]
        for following in records[i + 1:i + 3]:
            previous_number = re.fullmatch(r"#/texts/(\d+)", span[-1]["block_id"])
            next_number = re.fullmatch(r"#/texts/(\d+)", following["block_id"])
            if (not previous_number or not next_number or
                    int(next_number[1]) != int(previous_number[1]) + 1 or
                    following.get("section") != first.get("section")):
                break
            span.append(following)
            if contains_quote(target, presentation_text(" ".join(r["text"] for r in span))):
                return {**result, "status": "adjacent", "block_ids": [r["block_id"] for r in span]}
    return {**result, "status": "original_only" if hits else "unresolved"}


def original_html_records(html):
    # Reuse existing located HTML parser for diagnostic comparison only, not table/fulltext fallback.
    parser = ArxivHTML()
    parser.feed(html)
    return parser.records
