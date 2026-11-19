"""In-memory parsing of supplementary-material PDF text into located evidence.

This module only extracts text. It never fetches files, calls a model or draws
scientific conclusions. Extracted text stays bound to the input paper identity,
the caller-provided saved attachment path and physical 1-based PDF page numbers.
"""
import hashlib
import re
from io import BytesIO

from .models import EvidenceRecord, PaperRecord

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PAGE_CHARS = 100_000
MAX_PAGES = 40
PDF_HEADER = b"%PDF-"
PARSER_NAME = "pypdf"
PARSER_VERSION = "pypdf-supplement-v1"
PARSER_LIBRARY_VERSION = "6.10.0"  # pinned by the supplements extra
REPLACEMENT_CHAR = "\ufffd"
REPLACEMENT_FLAG = "replacement_characters"
LIMITATIONS = (
    "pypdf text extraction only; tables, equations, figures and page layout are not independently verified",
    "scanned or image-only pages are reported but not OCRed or interpreted",
    "pages are physical 1-based PDF pages, not journal page numbers or printed page labels",
    "the saved attachment path is recorded as given and is not opened, validated or turned into a URL here",
    "supplement text is not linked to main-text claims and is not a scientific acceptance",
)


def _text_field(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _member_key(member_name: str) -> str:
    """Stable short identity for one supplement member, independent of file name."""
    return hashlib.blake2s(member_name.encode("utf-8"), digest_size=8).hexdigest()


def parse_supplement_pdf(data: bytes, paper: PaperRecord, *, source_file: str,
                         member_name: str, max_pages: int = 40) -> dict:
    """Extract located page evidence from supplementary PDF bytes.

    ``data`` must be a ``%PDF`` byte string of at most 10 MiB. ``source_file`` is
    the real saved attachment path supplied by the caller; it is recorded, never
    opened or rewritten into a URL. ``member_name`` names the supplement inside
    the paper (for example a supplementary file title) and scopes stable block IDs.
    At most ``max_pages`` (1..40) physical pages are read, starting at page 1.
    Encrypted, malformed, page-less or text-less PDFs raise ``ValueError``; a
    missing optional ``pypdf`` dependency raises ``RuntimeError``.
    """
    source_file = _text_field(source_file, "source_file")
    member_name = _text_field(member_name, "member_name")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        # Uniform ValueError keeps rejected supplement input on one error path.
        raise ValueError("max_pages must be an integer between 1 and 40")  # noqa: TRY004
    if not 1 <= max_pages <= MAX_PAGES:
        raise ValueError("max_pages must be an integer between 1 and 40")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("data must be PDF bytes")  # noqa: TRY004
    raw = bytes(data)
    if not raw:
        raise ValueError("PDF data is empty")
    if len(raw) > MAX_PDF_BYTES:
        raise ValueError(f"PDF data exceeds {MAX_PDF_BYTES} bytes")
    if not raw.startswith(PDF_HEADER):
        raise ValueError("PDF data does not start with a %PDF header")

    try:
        # Optional pinned dependency: imported on first real use, not at module import.
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed; install the 'supplements' extra (pypdf==6.10.0)") from exc

    try:
        reader = PdfReader(BytesIO(raw))
    except Exception as exc:  # pypdf raises several types for malformed files
        raise ValueError(f"invalid PDF supplement: {exc}") from exc
    if reader.is_encrypted:
        raise ValueError("encrypted PDF supplements are not supported; decrypt before parsing")
    try:
        total_pages = len(reader.pages)
    except Exception as exc:
        raise ValueError(f"invalid PDF supplement: {exc}") from exc
    if total_pages < 1:
        raise ValueError("PDF supplement has no pages")

    processed = min(total_pages, max_pages)
    member_key = _member_key(member_name)
    evidence: list[EvidenceRecord] = []
    section_context: dict[str, dict[str, object]] = {}
    empty_or_image_pages: list[int] = []
    issues: list[str] = []
    skipped_text_pages = False
    document_role = "supplementary_material"
    if total_pages > processed:
        issues.append(f"processed first {processed} of {total_pages} pages (max_pages={max_pages})")

    for page_number in range(1, processed + 1):
        try:
            extracted = reader.pages[page_number - 1].extract_text()
        except Exception as exc:  # noqa: BLE001 - one damaged page must not hide the others
            issues.append(f"page {page_number}: text extraction failed ({type(exc).__name__})")
            empty_or_image_pages.append(page_number)
            continue
        text = extracted.strip() if isinstance(extracted, str) else ""
        if page_number == 1 and re.search(
                r"(?im)^\s*(?:transparent\s+)?peer[\s-]*review\s+file\s*$", text[:1000]):
            document_role = "peer_review"
        if not text:
            empty_or_image_pages.append(page_number)
            continue
        if len(text) > MAX_PAGE_CHARS:
            # Never pass a silently truncated page off as reliable evidence.
            issues.append(
                f"page {page_number}: text exceeds {MAX_PAGE_CHARS} characters and was skipped")
            skipped_text_pages = True
            continue
        block_id = f"supp:{member_key}:p{page_number}"
        flags = [REPLACEMENT_FLAG] if REPLACEMENT_CHAR in text else []
        evidence.append(EvidenceRecord(
            evidence_id=f"{paper.paper_id}:{paper.version}:{block_id}",
            paper_id=paper.paper_id, paper_version=paper.version, file=source_file,
            block_id=block_id, page=page_number,
            section="Supplementary material: " + member_name, text=text,
            parser_version=PARSER_VERSION, quality_flags=flags))
        section_context[block_id] = {"headings": ["Supplementary material", member_name],
                                     "kind": "paragraph"}

    if not evidence:
        raise ValueError("PDF supplement has no extractable text pages")

    return {
        "evidence": [item.model_dump() for item in evidence],
        "section_context": section_context,
        "total_pages": total_pages,
        "pages_processed": processed,
        "truncated": total_pages > processed or skipped_text_pages,
        "empty_or_image_pages": empty_or_image_pages,
        "issues": issues,
        "document_role": document_role,
        "provenance": {
            "member_name": member_name,
            "source_file": source_file,
            "paper_id": paper.paper_id,
            "paper_version": paper.version,
            "parser": PARSER_NAME,
            "parser_version": PARSER_LIBRARY_VERSION,
        },
        "limitations": list(LIMITATIONS),
    }
