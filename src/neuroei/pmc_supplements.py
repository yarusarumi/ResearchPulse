"""Europe PMC supplementary ZIP retrieval and in-memory manifest inspection."""
import hashlib
import io
import re
import stat
import zipfile
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import ValidationError

from .models import PaperRecord
from .pmc_reader import BASE
from .sources import normalize_doi

ARCHIVE_MAX_BYTES = 10_000_000
MEMBER_MAX_BYTES = 10_000_000
EXPANDED_MAX_BYTES = 40_000_000
MAX_MEMBERS = 100
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


def _preflight(payload):
    if not isinstance(payload, dict):
        raise TypeError("Payload must be the dict returned by fetch_pmc_xml")
    try:
        paper = PaperRecord.model_validate(payload.get("paper"))
    except (ValidationError, TypeError) as exc:
        raise ValueError("Payload paper is not a valid PaperRecord") from exc
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("service") != "europepmc":
        raise ValueError("Payload provenance is not a Europe PMC record")
    pmcid = provenance.get("pmcid")
    if not isinstance(pmcid, str) or not re.fullmatch(r"PMC[0-9]+", pmcid):
        raise ValueError("Payload provenance lacks a valid PMC identity")
    license_info = provenance.get("license")
    if (not isinstance(license_info, dict)
            or not isinstance(license_info.get("text"), str)
            or not license_info["text"].strip()):
        raise ValueError("Payload provenance lacks licence text")
    if payload.get("url") != f"{BASE}/{pmcid}/fullTextXML":
        raise ValueError("Payload URL is not the official Europe PMC fullTextXML URL")
    _check_identity(paper, provenance)
    return paper, provenance, pmcid


def _check_identity(paper, provenance):
    paper_id = paper.paper_id.strip()
    pubmed = re.fullmatch(r"pubmed:([0-9]+)", paper_id)
    if pubmed:
        if str(provenance.get("pmid") or "").strip() != pubmed.group(1):
            raise ValueError("Payload PMID does not match paper identity")
    elif paper_id[:4].lower() == "doi:":
        identity_doi = normalize_doi(paper_id[4:])
        if not identity_doi or identity_doi != normalize_doi(provenance.get("doi")):
            raise ValueError("Payload DOI does not match paper identity")
    else:
        raise ValueError("Payload paper identity is not pubmed:<digits> or doi:<DOI>")
    if paper.doi:
        field_doi = normalize_doi(paper.doi)
        if not field_doi or field_doi != normalize_doi(provenance.get("doi")):
            raise ValueError("Payload DOI does not match paper.doi")


def _expected_basenames(references):
    if not isinstance(references, list):
        raise TypeError("expected_references must be a list of strings")
    basenames = set()
    for reference in references:
        if not isinstance(reference, str):
            raise TypeError("expected_references must be a list of strings")
        path = urlsplit(reference.strip()).path
        basename = unquote(path.rsplit("/", 1)[-1])
        if basename:
            basenames.add(basename)
    return basenames


def _safe_name(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("ZIP member has an invalid name")
    if "\\" in name:
        raise ValueError("ZIP member name contains a backslash")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError("ZIP member name is absolute")
    clean = name.removesuffix("/")
    if not clean or any(part in ("", ".", "..") for part in clean.split("/")):
        raise ValueError("ZIP member name traverses directories")
    return clean


def _member_metadata(info):
    name = info.filename
    clean = _safe_name(name)
    if info.flag_bits & 0x1:
        raise ValueError("Encrypted ZIP members are not accepted")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError("Symlink ZIP members are not accepted")
    file_type = stat.S_IFMT(mode)
    if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError("Non-regular ZIP members are not accepted")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ValueError("Unsupported ZIP compression method")
    is_directory = info.is_dir() or name.endswith("/")
    if not is_directory and info.file_size > MEMBER_MAX_BYTES:
        raise ValueError("ZIP member exceeds the 10 MB expanded limit")
    return clean, name, is_directory, info.file_size, info.compress_size


def inspect_supplement_archive(data, expected_references):
    """Inspect ZIP metadata in memory; never extract, execute or touch the filesystem."""
    if not isinstance(data, bytes):
        raise TypeError("Archive data must be bytes")
    declared = _expected_basenames(expected_references)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ValueError("ZIP archive has more than 100 members")
            seen, files, total = set(), [], 0
            for info in infos:
                clean, name, is_directory, expanded, compressed = _member_metadata(info)
                key = clean.casefold()
                if key in seen:
                    raise ValueError("ZIP member names collide ignoring case")
                seen.add(key)
                if is_directory:
                    continue
                total += expanded
                if total > EXPANDED_MAX_BYTES:
                    raise ValueError("ZIP expanded size exceeds the 40 MB limit")
                basename = name.rsplit("/", 1)[-1]
                files.append({"name": name, "size": expanded, "compressed_size": compressed,
                              "kind": "pdf" if basename.casefold().endswith(".pdf") else "other",
                              "declared_in_xml": basename in declared,
                              "status": "downloaded_unparsed"})
            return files
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
        raise ValueError("Invalid or corrupt ZIP archive") from exc


def _unavailable():
    return {"status": "unavailable", "reason": "not_available_at_europepmc", "files": []}


def _xml_references(payload):
    materials = payload.get("supplementary_materials")
    references = []
    if isinstance(materials, list):
        for item in materials:
            if isinstance(item, dict) and isinstance(item.get("reference"), str):
                references.append(item["reference"])
    return references


def fetch_supplement_archive(payload, transport=None, *, max_bytes=ARCHIVE_MAX_BYTES):
    """Fetch the official Europe PMC supplementary ZIP; one GET, no redirects or retries."""
    if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= ARCHIVE_MAX_BYTES):
        raise ValueError("max_bytes must be an int between 1 and 10,000,000")
    paper, provenance, pmcid = _preflight(payload)
    archive_url = f"{BASE}/{pmcid}/supplementaryFiles?includeInlineImage=false"
    timeout = httpx.Timeout(60.0, connect=10.0)
    with (httpx.Client(timeout=timeout, follow_redirects=False, transport=transport) as client,
          client.stream("GET", archive_url) as response):
        if 300 <= response.status_code < 400 or response.is_redirect:
            raise ValueError("Europe PMC supplementary redirect refused")
        if response.status_code == 404:
            return _unavailable()
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Europe PMC supplementary archive exceeds max_bytes")
            chunks.append(chunk)
        data = b"".join(chunks)
    if data[:4] not in _ZIP_MAGIC:
        raise ValueError("Europe PMC supplementary response is not a ZIP archive")
    files = inspect_supplement_archive(data, _xml_references(payload))
    return {"status": "downloaded_unparsed",
            "paper_id": paper.paper_id,
            "paper_version": paper.version,
            "pmcid": pmcid,
            "archive_url": archive_url,
            "archive_size": len(data),
            "archive_bytes": data,
            "files": files,
            "source_revision": hashlib.blake2s(data, digest_size=16).hexdigest(),
            "license": dict(provenance["license"])}
