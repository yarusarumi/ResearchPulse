"""Acquire a bounded supplementary snapshot and keep download/parse states separate."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit


def acquire_supplement_text(payload, artifact_root, *, max_files=3, max_pages=40, transport=None):
    if type(max_files) is not int or not 1 <= max_files <= 3:
        raise ValueError("Supplement PDF limit must be an integer from 1 to 3")
    if type(max_pages) is not int or not 1 <= max_pages <= 40:
        raise ValueError("Supplement page limit must be an integer from 1 to 40")
    from .pmc_supplements import _preflight, fetch_supplement_archive
    from .supplement_pdf import parse_supplement_pdf

    # Cache reuse needs the same identity/URL/licence validation as a fresh request.
    paper, provenance, _ = _preflight(payload)
    if not isinstance(provenance.get("source_revision"), str) or not provenance["source_revision"]:
        raise ValueError("Supplement snapshot requires the original XML source revision")
    identity = (paper.paper_id, paper.version, provenance["pmcid"], provenance["source_revision"])
    snapshot = hashlib.blake2s(json.dumps([*identity, max_files, max_pages]).encode(), digest_size=12).hexdigest()
    folder = Path(artifact_root) / ("supplement-" + snapshot)
    manifest = folder / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if tuple(saved.get(key) for key in ("paper_id", "paper_version", "pmcid", "source_revision")) != identity:
            raise ValueError("Supplement snapshot identity changed; inspect the existing artifact")
        if not all(Path(item["file"]).is_file() for item in saved["files"] if item.get("file")):
            raise ValueError("Supplement snapshot is missing an original file")
        return {**saved, "reused_existing_snapshot": True}
    if folder.exists():
        raise ValueError("Incomplete supplement snapshot requires inspection; no automatic overwrite")

    archive = fetch_supplement_archive(payload, transport=transport)
    if archive["status"] != "downloaded_unparsed":
        return {"status": "unavailable", "reason": archive.get("reason", "archive_unavailable"),
                "evidence": [], "section_context": {}, "files": [],
                "limitations": ["Supplementary files are unavailable at Europe PMC; other access was not attempted"]}
    folder.mkdir(parents=True)
    archive_file = folder / "supplementary.zip"
    with archive_file.open("xb") as stream:
        stream.write(archive["archive_bytes"])
    result = {"paper_id": paper.paper_id, "paper_version": paper.version, "pmcid": identity[2],
              "source_revision": identity[3], "archive_url": archive["archive_url"],
              "archive_revision": archive["source_revision"],
              "archive_file": str(archive_file), "files": [], "evidence": [], "section_context": {},
              "max_files": max_files, "max_pages_per_file": max_pages,
              "limitations": ["Only XML-declared PDF supplements are parsed, at most 3 files and 40 pages per file",
                               "ZIP acquisition and PDF text extraction do not verify images, tables, equations or scientific claims",
                               "The XML reference manifest may be capped; unlisted archive members are not automatically analysed"]}
    attempted = 0
    declared_roles = {unquote(urlsplit(item.get("reference", "")).path.rsplit("/", 1)[-1]): item.get("role")
                      for item in payload.get("supplementary_materials", []) if isinstance(item, dict)}
    with zipfile.ZipFile(io.BytesIO(archive["archive_bytes"])) as zipped:
        for index, entry in enumerate(sorted(archive["files"], key=lambda item: item["name"]), 1):
            item = dict(entry)
            result["files"].append(item)
            if not entry["declared_in_xml"]:
                item["status"] = "not_parsed_undeclared"
                continue
            if declared_roles.get(entry["name"].rsplit("/", 1)[-1]) == "peer_review":
                item.update(status="excluded_peer_review", document_role="peer_review")
                continue
            if entry["kind"] != "pdf":
                item["status"] = "not_parsed_format"
                continue
            if attempted >= max_files:
                item["status"] = "not_parsed_file_limit"
                continue
            attempted += 1
            # Generated local names avoid interpreting any archive path as a filesystem path.
            target = folder / f"attachment-{index:03}.pdf"
            data = zipped.read(entry["name"])
            with target.open("xb") as stream:
                stream.write(data)
            item["file"] = str(target)
            try:
                parsed = parse_supplement_pdf(data, paper, source_file=str(target),
                                               member_name=entry["name"], max_pages=max_pages)
            except Exception as exc:  # noqa: BLE001 - one unreadable attachment must not erase the body draft
                item.update(status="parse_failed", error_type=type(exc).__name__)
                continue
            item.update(status="partial_text" if parsed["truncated"] else "parsed_text",
                        total_pages=parsed["total_pages"], pages_processed=parsed["pages_processed"],
                        evidence_blocks=len(parsed["evidence"]), empty_or_image_pages=parsed["empty_or_image_pages"],
                        issues=parsed["issues"], parser=parsed["provenance"],
                        document_role=parsed.get("document_role", "supplementary_material"))
            if item["document_role"] == "peer_review":
                item["status"] = "excluded_peer_review"
                continue
            result["evidence"].extend(parsed["evidence"])
            result["section_context"].update(parsed["section_context"])
            for limitation in parsed["limitations"]:
                if limitation not in result["limitations"]:
                    result["limitations"].append(limitation)
    declared = [item for item in result["files"] if item["declared_in_xml"] and item["status"] != "excluded_peer_review"]
    complete = declared and all(item["status"] == "parsed_text" for item in declared)
    result["status"] = "parsed_text" if complete and result["evidence"] else "partial" if result["evidence"] else "downloaded_unparsed"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {**result, "reused_existing_snapshot": False}
