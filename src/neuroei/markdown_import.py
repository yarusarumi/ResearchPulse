"""Authorized Markdown export ingestion with immutable file/line provenance."""
import hashlib
import re
from pathlib import Path

from .models import SourceBlock, now


def snapshot_markdown(paths: list[Path]) -> dict:
    blocks, unread = [], []
    for supplied in paths:
        path = supplied.resolve(strict=True)
        if path.suffix.lower() != ".md":
            raise ValueError("Only explicitly supplied Markdown files are imported")
        text = path.read_text(encoding="utf-8-sig")
        revision = hashlib.sha256(text.encode()).hexdigest()[:16]
        document_id = "md:" + path.as_posix()
        lines = text.splitlines(keepends=True)
        start, pending, size, section, fence = 1, [], 0, "", False

        def flush(end):
            nonlocal pending, start, size
            body = "".join(pending).strip()
            if body:
                blocks.append(SourceBlock(document_id=document_id,
                    block_id=f"{path.name}:L{start}-L{end}", text=body, revision=revision,
                    file=str(path), line_start=start, line_end=end, section=section).model_dump())
            pending, size, start = [], 0, end + 1

        for number, line in enumerate(lines, 1):
            if not fence and re.match(r"^#{1,6}\s", line):
                flush(number - 1)
                section = line.strip()
                start = number
            pending.append(line)
            size += len(line)
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
            for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", line):
                unread.append({"file": str(path), "line": number, "target": target,
                               "status": "image_not_read_text_import_only"})
            if size >= 5000 and not fence and not line.strip():
                flush(number)
        flush(len(lines))
    if not blocks:
        raise ValueError("No text blocks in supplied files")
    return {"kind": "authorized_markdown_export", "fetched_at": now(), "blocks": blocks,
            "unread_items": unread, "scope": [str(p.resolve()) for p in paths]}
