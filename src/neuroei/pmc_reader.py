"""Europe PMC open-access JATS reader with strict identity, licence and body checks."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import httpx

from .models import EvidenceRecord
from .sources import normalize_doi

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MAX_BYTES = 5_000_000
PARSER_VERSION = "europepmc-oa-jats-v1"
_SUPPLEMENT_LIMIT = 20
_MATH_TAGS = {"math", "inline-formula", "disp-formula", "tex-math"}
_EXCLUDED = {"fig", "caption", "table-wrap", "table", "supplementary-material", "ref-list", "back"}


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _local(tag):
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _canonical_doi(value):
    if not value:
        return None
    candidate = normalize_doi(value)
    if not candidate or not re.fullmatch(r"10\.\d{4,9}/\S+", candidate):
        return None
    if re.search(r"[\"'\\<>\x00-\x1f\x7f]", candidate):
        return None
    return candidate


def _identity(paper):
    paper_id = paper.paper_id.strip()
    pmid = doi = None
    if re.fullmatch(r"pubmed:[0-9]+", paper_id):
        pmid = paper_id.split(":", 1)[1]
    elif paper_id[:4].lower() == "doi:":
        doi = _canonical_doi(paper_id[4:])
        if doi is None:
            raise ValueError("Unsupported DOI identity; canonical 10.<4-9 digits>/<suffix> required")
    else:
        raise ValueError("Unsupported paper identity; expected pubmed:<digits> or doi:<DOI>")
    field_doi = None
    if paper.doi:
        field_doi = _canonical_doi(paper.doi)
        if field_doi is None:
            raise ValueError("Unsupported DOI in paper.doi")
    if doi is not None and field_doi is not None and doi != field_doi:
        raise ValueError("Conflicting paper_id DOI and paper.doi")
    return pmid, doi or field_doi


def _bounded(client, url, params=None):
    with client.stream("GET", url, params=params) as response:
        if 300 <= response.status_code < 400 or response.is_redirect:
            raise ValueError("Europe PMC redirect refused")
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("Europe PMC response exceeds 5 MB limit")
            chunks.append(chunk)
        return b"".join(chunks), response.headers.get("content-type", "")


def _matches(records, pmid, doi):
    matches = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_pmid = str(record.get("pmid") or "").strip()
        record_doi = normalize_doi(record.get("doi")) or ""
        if pmid is not None and record_pmid != pmid:
            continue
        if doi is not None and record_doi != doi:
            continue
        if pmid is None and not record_doi:
            continue
        matches.append(record)
    return matches


def _resolve_record(client, pmid, doi):
    query = f"EXT_ID:{pmid} AND SRC:MED" if pmid is not None else f'DOI:"{doi}"'
    payload_bytes, _ = _bounded(client, BASE + "/search", params={
        "query": query, "format": "json", "resultType": "core", "pageSize": "5"})
    try:
        payload = json.loads(payload_bytes)
        results = payload["resultList"]["result"]
        if not isinstance(results, list):
            raise TypeError
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Invalid Europe PMC search JSON") from exc
    matches = _matches(results, pmid, doi)
    if not matches:
        if results:
            raise ValueError("Europe PMC result does not match the requested identity")
        raise ValueError("No exact Europe PMC record found")
    pmcids = {str(record.get("pmcid") or "").strip() for record in matches}
    valid = {pmcid for pmcid in pmcids if re.fullmatch(r"PMC[0-9]+", pmcid)}
    if len(valid) > 1:
        raise ValueError("Ambiguous Europe PMC records with different PMC identities")
    if not valid:
        raise ValueError("No open-access PMC full text for this record")
    pmcid = valid.pop()
    chosen = [record for record in matches if str(record.get("pmcid") or "").strip() == pmcid]
    open_records = [record for record in chosen if record.get("isOpenAccess") == "Y"]
    if not open_records:
        raise ValueError("Europe PMC record is not open access")
    return open_records[0], pmcid


def _strip_namespaces(root):
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        element.tag = _local(element.tag)
        if element.attrib:
            element.attrib = {_local(key): value for key, value in element.attrib.items()}


def _parse_jats(data):
    if re.search(rb"<!ENTITY", data, re.IGNORECASE):
        raise ValueError("XML ENTITY declarations are not accepted")
    if re.search(rb"<!DOCTYPE[^>]*\[", data, re.IGNORECASE):
        raise ValueError("XML internal DTD subset is not accepted")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError("Invalid JATS XML") from exc
    _strip_namespaces(root)
    if _local(root.tag) != "article":
        raise ValueError("Root element is not a JATS article")
    return root


def _inline_text(node, parts, flags):
    tag = _local(node.tag)
    if tag in _MATH_TAGS:
        flags.add("equation_not_transcribed")
        parts.append(" [equation not transcribed] ")
        return
    if tag in _EXCLUDED:
        return
    if node.text:
        parts.append(node.text)
    for child in node:
        _inline_text(child, parts, flags)
        if child.tail:
            parts.append(child.tail)


def _extract_paragraphs(body):
    records, used_ids = [], set()
    seen = 0

    def capture(paragraph, headings, ordinal):
        raw_id = (paragraph.get("id") or "").strip()
        if raw_id:
            if raw_id in used_ids:
                raise ValueError("Duplicate XML paragraph id")
            block_id = raw_id
        else:
            block_id = f"body-p{ordinal}"
            if block_id in used_ids:
                raise ValueError("Generated block id conflicts with XML id")
        used_ids.add(block_id)
        parts, flags = [], set()
        _inline_text(paragraph, parts, flags)
        text = _clean("".join(parts))
        if not text:
            return
        heading_path = " > ".join(headings)
        locator = f"#{block_id}" if raw_id else f"body paragraph {ordinal}"
        section = f"{heading_path} {locator}".strip()
        records.append({"block_id": block_id, "text": text, "section": section,
                        "headings": list(headings), "quality_flags": sorted(flags)})

    def walk(node, headings):
        nonlocal seen
        for child in node:
            tag = _local(child.tag)
            if tag in _EXCLUDED:
                continue
            if tag == "sec":
                title_element = next((item for item in child if _local(item.tag) == "title"), None)
                title = _clean("".join(title_element.itertext())) if title_element is not None else ""
                walk(child, headings + ([title] if title else []))
            elif tag == "p":
                seen += 1
                capture(child, headings, seen)
            else:
                walk(child, headings)

    walk(body, [])
    return records


def _supplementary_materials(root):
    items, seen = [], set()
    for element in root.iter():
        if _local(element.tag) != "supplementary-material":
            continue
        label_element = next((item for item in element if _local(item.tag) == "label"), None)
        label = _clean("".join(label_element.itertext())) if label_element is not None else ""
        label = label or (element.get("id") or "").strip() or f"supplementary-material {len(items) + 1}"
        description = _clean(" ".join(element.itertext()))[:1000]
        role = "peer_review" if re.search(r"\bpeer[\s-]*review\b", description, re.IGNORECASE) else "supplementary_material"
        references = []
        own_href = (element.get("href") or "").strip()
        if own_href:
            references.append(own_href)
        for child in element.iter():
            if child is element:
                continue
            href = (child.get("href") or "").strip()
            if href:
                references.append(href)
        for reference in references:
            key = (label, reference)
            if key in seen:
                continue
            if len(items) >= _SUPPLEMENT_LIMIT:
                return items
            seen.add(key)
            items.append({"label": label, "description": description, "role": role,
                          "reference": reference, "status": "not_fetched"})
    return items


_PMC_ALIASES = {"pmc", "pmcid", "pmcaid", "pmcaiid"}


def _normalize_pmc(kind, value):
    text = value.strip()
    if kind in {"pmc", "pmcid"}:
        match = re.fullmatch(r"(?:PMC)?([0-9]+)", text, re.IGNORECASE)
    else:
        match = re.fullmatch(r"([0-9]+)", text)
    return "PMC" + match.group(1) if match else None


def _xml_identities(article_meta, pmcid):
    pmc_values, pmid_values, doi_keys = [], [], {}
    for element in article_meta.findall("article-id"):
        kind = (element.get("pub-id-type") or "").strip().lower()
        value = _clean("".join(element.itertext()))
        if not kind or not value:
            continue
        if kind in _PMC_ALIASES:
            normalized = _normalize_pmc(kind, value)
            if normalized is None:
                raise ValueError("Invalid PMC identity in JATS XML")
            pmc_values.append(normalized)
        elif kind == "pmid":
            pmid_values.append(value)
        elif kind == "doi":
            canonical = _canonical_doi(value)
            doi_keys[canonical or value.lower()] = canonical
    pmc_set = set(pmc_values)
    if not pmc_set:
        raise ValueError("No PMC identity in JATS XML")
    if len(pmc_set) > 1:
        raise ValueError("Conflicting PMC identities in JATS XML")
    if pmc_set.pop() != pmcid:
        raise ValueError("XML PMC identity does not match Europe PMC record")
    if len(set(pmid_values)) > 1:
        raise ValueError("Conflicting PMID identities in JATS XML")
    if len(doi_keys) > 1:
        raise ValueError("Conflicting DOI identities in JATS XML")
    xml_pmid = pmid_values[0] if pmid_values else None
    return xml_pmid, next(iter(doi_keys.values()), None)


def _license_url(license_element):
    own_href = (license_element.get("href") or "").strip()
    if own_href:
        return own_href
    for child in license_element.iter():
        if child is license_element:
            continue
        href = (child.get("href") or "").strip()
        if re.fullmatch(r"https?://\S+", href, re.IGNORECASE):
            return href
    return None


def fetch_pmc_xml(paper, transport=None):
    pmid, doi = _identity(paper)
    with httpx.Client(timeout=30, follow_redirects=False, transport=transport) as client:
        record, pmcid = _resolve_record(client, pmid, doi)
        xml_url = f"{BASE}/{pmcid}/fullTextXML"
        xml_bytes, content_type = _bounded(client, xml_url)
        if content_type and "xml" not in content_type.lower():
            raise ValueError("Europe PMC full text is not XML")
        root = _parse_jats(xml_bytes)

    front = root.find("front")
    article_meta = front.find("article-meta") if front is not None else None
    if article_meta is None:
        raise ValueError("JATS article has no front/article-meta")
    xml_pmid, xml_doi = _xml_identities(article_meta, pmcid)
    record_pmid = str(record.get("pmid") or "").strip() or None
    record_doi = normalize_doi(record.get("doi")) or None
    if pmid is not None and xml_pmid != pmid:
        raise ValueError("XML PMID does not match input identity")
    if doi is not None and xml_doi != doi:
        raise ValueError("XML DOI does not match input identity")
    if record_pmid and xml_pmid and record_pmid != xml_pmid:
        raise ValueError("XML PMID does not match Europe PMC record")
    if record_doi and xml_doi and record_doi != xml_doi:
        raise ValueError("XML DOI does not match Europe PMC record")
    if pmid is not None and record_pmid and pmid != record_pmid:
        raise ValueError("Europe PMC PMID does not match input identity")
    if doi is not None and record_doi and doi != record_doi:
        raise ValueError("Europe PMC DOI does not match input identity")

    permissions = article_meta.find("permissions")
    license_element = permissions.find("license") if permissions is not None else None
    if license_element is None:
        raise ValueError("No permissions/license in JATS XML")
    license_text = _clean(" ".join(license_element.itertext()))
    if not license_text:
        raise ValueError("Empty licence in JATS XML")
    license_url = _license_url(license_element)
    license_type = (license_element.get("license-type") or "").strip() or None

    body = root.find("body")
    paragraphs = _extract_paragraphs(body) if body is not None else []
    if not paragraphs:
        raise ValueError("No body paragraphs in JATS XML; abstract-only text rejected")

    evidence = [EvidenceRecord(
        evidence_id=f"{paper.paper_id}:{paper.version}:pmc:{item['block_id']}",
        paper_id=paper.paper_id, paper_version=paper.version, file=xml_url,
        block_id=item["block_id"], page=None, section=item["section"], text=item["text"],
        parser_version=PARSER_VERSION, quality_flags=item["quality_flags"]) for item in paragraphs]
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("Duplicate PMC block identities")
    section_context = {item["block_id"]: {"headings": item["headings"], "kind": "paragraph"}
                       for item in paragraphs}
    revision = hashlib.blake2s("\n".join(item["text"] for item in paragraphs).encode("utf-8"),
                               digest_size=16).hexdigest()
    provenance = {"service": "europepmc", "pmcid": pmcid,
                  "pmid": xml_pmid or pmid or record_pmid, "doi": xml_doi or doi or record_doi,
                  "license": {"text": license_text, "url": license_url, "type": license_type},
                  "source_revision": revision, "xml_bytes": len(xml_bytes)}
    limitations = [
        "JATS XML body text only; figures, images and equations are not interpreted",
        "supplementary files are recorded only and were not fetched",
        "licence terms are taken as stated in the article XML, not independently verified",
        "no PDF page numbers and not a PaperQA/Docling compatibility acceptance",
        "located body paragraphs only; this is not evidence that every section was read",
    ]
    return {"paper": paper.model_copy(update={"fulltext_status": "available"}).model_dump(),
            "evidence": [item.model_dump() for item in evidence], "url": xml_url,
            "section_context": section_context,
            "supplementary_materials": _supplementary_materials(root),
            "provenance": provenance, "limitations": limitations}
