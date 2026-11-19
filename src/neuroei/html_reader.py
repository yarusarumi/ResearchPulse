"""Located arXiv HTML text reader: explicit degraded path while PDF/semantic readers are unavailable."""
import re
import shutil
import subprocess
from html.parser import HTMLParser
from io import BytesIO

import httpx

from .models import EvidenceRecord


class ArxivHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.capture_depth = None
        self.parts = []
        self.anchor = ""
        self.records = []
        self.math_depth = None
        self.headings = []
        self.heading = None
        self.context = {}
        self.table_cells = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        void = tag in {"img", "br", "hr", "meta", "link", "input", "wbr", "source"}
        if not void:
            self.depth += 1
        classes = attrs.get("class", "").split()
        if re.fullmatch(r"h[1-6]", tag):
            self.heading = (int(tag[1]), [])
        if self.capture_depth is None and attrs.get("id") and (
            "ltx_para" in classes or "ltx_table" in classes):
            self.capture_depth = self.depth
            self.anchor = attrs["id"]
            self.parts = []
            self.context[self.anchor] = {"headings": [text for _, text in self.headings],
                                         "kind": "table" if "ltx_table" in classes else "paragraph"}
        if self.capture_depth is not None:
            if tag == "math":
                self.parts.append(" " + attrs.get("alttext", "[equation not transcribed]") + " ")
                self.math_depth = self.depth
            elif tag in ("td", "th"):
                self.table_cells[self.anchor] = self.table_cells.get(self.anchor, 0) + 1
                self.parts.append(" | ")
            elif tag in ("tr", "p", "br"):
                self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # Most self-closing markup on arXiv is MathML; let its enclosing math alttext stand.
        if tag in ("br", "img"):
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self.heading and tag == "h" + str(self.heading[0]):
            level, parts = self.heading
            self.headings = [(old_level, text) for old_level, text in self.headings if old_level < level]
            self.headings.append((level, re.sub(r"\s+", " ", "".join(parts)).strip()))
            self.heading = None
        if tag in {"img", "br", "hr", "meta", "link", "input", "wbr", "source"}:
            return
        if self.math_depth == self.depth and tag == "math":
            self.math_depth = None
        if self.capture_depth == self.depth:
            text = re.sub(r"[ \t]+", " ", "".join(self.parts)).strip()
            if text:
                self.records.append((self.anchor, text))
            self.capture_depth = None
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        if self.heading:
            self.heading[1].append(data)
        if self.capture_depth is not None and self.math_depth is None:
            self.parts.append(data)


def fetch_arxiv_pdf_text(paper, transport=None):
    """Read official PDF text in memory; no OCR, model calls or inferred table cells."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    url = "https://arxiv.org/pdf/" + paper.paper_id.split(":", 1)[1] + paper.version
    with (httpx.Client(timeout=45, follow_redirects=True, transport=transport) as client,
          client.stream("GET", url) as response):
        response.raise_for_status()
        if response.url.host != "arxiv.org":
            raise ValueError("No official arXiv PDF available")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > 20_000_000:
                raise ValueError("PDF exceeds text-reader size limit")
    if not data.startswith(b"%PDF-"):
        raise ValueError("Official response is not PDF")
    try:
        reader = PdfReader(BytesIO(data))
        evidence = []
        layout_pages = None
        if shutil.which('pdftotext'):
            try:
                converted = subprocess.run(['pdftotext', '-layout', '-enc', 'UTF-8', '-', '-'],
                    input=bytes(data), capture_output=True, timeout=45, check=True)
                layout_pages = converted.stdout.decode('utf-8').split('\f')
                if len(layout_pages) != len(reader.pages) + 1 or layout_pages[-1].strip():
                    layout_pages = None
            except (OSError, subprocess.SubprocessError, UnicodeError):
                layout_pages = None
        for number, page in enumerate(reader.pages, 1):
            text = layout_pages[number-1] if layout_pages is not None else page.extract_text(extraction_mode="layout") or ""
            if text.strip():
                evidence.append(EvidenceRecord(
                    evidence_id=f"{paper.paper_id}:{paper.version}:pdf:page:{number}",
                    paper_id=paper.paper_id, paper_version=paper.version, file=url,
                    page=number, block_id=f"page:{number}", section=f"PDF page {number}",
                    text=text, parser_version="poppler-page-layout-v1" if layout_pages is not None
                    else "pypdf-page-layout-v2").model_dump())
        if not evidence:
            raise ValueError("PDF has no extractable text; OCR not performed")
        return evidence, len(reader.pages)
    except PyPdfError as exc:
        raise ValueError("Official PDF could not be parsed") from exc


def fetch_arxiv_html(paper, transport=None):
    if not re.fullmatch(r"arxiv:\d{4}\.\d{4,5}", paper.paper_id) or not re.fullmatch(r"v\d+", paper.version):
        raise ValueError("Explicit modern arXiv ID and version required")
    url = "https://arxiv.org/html/" + paper.paper_id.split(":", 1)[1] + paper.version
    network_route = "environment"
    try:
        with httpx.Client(timeout=30, follow_redirects=True, transport=transport) as client:
            response = client.get(url)
    except (httpx.ConnectError, httpx.ProxyError):
        # A local proxy can fail the TLS handshake while this same public official URL
        # is reachable directly. Retry once with normal certificate validation; HTTP
        # denials (including 403) never enter this fallback. No global proxy changes.
        with httpx.Client(timeout=30, follow_redirects=True, transport=transport, trust_env=False) as client:
            response = client.get(url)
        network_route = "direct_tls_after_connection_failure"
    response.raise_for_status()
    if response.url.host != "arxiv.org" or not response.headers.get("content-type", "").startswith("text/html"):
        raise ValueError("No official arXiv HTML available")
    if len(response.content) > 5_000_000:
        raise ValueError("HTML document exceeds pilot size limit")
    parser = ArxivHTML()
    parser.feed(response.text)
    if len(parser.records) < 10:
        raise ValueError("Insufficient located HTML body; no abstract-only fallback")
    evidence = [EvidenceRecord(evidence_id=f"{paper.paper_id}:{paper.version}:html:{anchor}",
        paper_id=paper.paper_id, paper_version=paper.version, file=url,
        block_id=anchor, page=None, section="#" + anchor, text=text,
        parser_version="arxiv-html-located-v2") for anchor, text in parser.records]
    if len({e.evidence_id for e in evidence}) != len(evidence):
        raise ValueError("Duplicate HTML anchors")
    incomplete_tables = [anchor for anchor, context in parser.context.items()
                         if context["kind"] == "table" and not parser.table_cells.get(anchor)]
    limitations = ["HTML text only; figures not interpreted", "separate supplementary files not fetched",
                   "not PaperQA/Docling compatibility acceptance"]
    integrity = {"status": "not_independently_verified", "format": "html"}
    if incomplete_tables:
        integrity = {"status": "incomplete", "format": "html", "empty_tables": incomplete_tables}
        limitations.append("来源正文不完整：HTML 表格缺少表体，已解析块覆盖率不代表全文覆盖。")
        try:
            pdf_evidence, pages = fetch_arxiv_pdf_text(paper, transport)
        except (ImportError, ValueError, OSError, httpx.HTTPError) as exc:
            limitations.append("官方 PDF 补读未完成：" + type(exc).__name__ + "；保留不完整 HTML 供有限分析。")
        else:
            limitations[0] = "PDF text only; figures and table row/column alignment not verified"
            evidence = [EvidenceRecord.model_validate(row) for row in pdf_evidence]
            integrity.update(status="pdf_text_fallback", format="pdf", total_pages=pages,
                             extracted_pages=len(evidence))
            limitations.append(f"已改读官方 PDF：提取 {len(evidence)}/{pages} 页文本；未做 OCR、图像或表格行列核验。")
            parser.context = {e.block_id: {"headings": [e.section], "kind": "paragraph"} for e in evidence}
    return {"paper": paper.model_copy(update={"fulltext_status": "available"}).model_dump(),
            "evidence": [e.model_dump() for e in evidence], "url": url,
            "network_route": network_route,
            "section_context": parser.context, "source_integrity": integrity,
            "limitations": limitations}
