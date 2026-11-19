import hashlib
import json
import os
import re
from urllib.parse import urlparse

import httpx

from .models import SourceBlock, now

PARSER_POLICY = "heading-context-v6"
MAX_IMAGE_BYTES = 5_000_000
MAX_DOCUMENT_IMAGES = 5
# Page, divider, grid, grid column, table cell, and view have no standalone content.
STRUCTURAL_BLOCK_TYPES = {1, 22, 24, 25, 32, 33}


def _element_text(element):
    if 'text_run' in element:
        return element['text_run'].get('content', '')
    equation = element.get('equation') or {}
    content = equation.get('content')
    if isinstance(content, str) and content.strip():
        return f'${content}$'
    return None


def _table_markdown(block, by_id):
    table = block.get('table') or {}
    prop = table.get('property') or {}
    height, width = prop.get('row_size'), prop.get('column_size')
    cells = table.get('cells') or block.get('children', [])
    if (type(height) is not int or type(width) is not int or height < 1 or width < 1
            or not isinstance(cells, list) or len(cells) != height * width
            or any(not isinstance(cell, str) for cell in cells)
            or set(cells) != set(block.get('children', []))
            or any(by_id[cell].get('block_type') != 32 for cell in cells)):
        return '', set()
    flat, covered = [], set()
    for cell in cells:
        pending = [cell]
        paragraphs = []
        while pending:
            child = by_id[pending.pop()]
            elements = [element for value in child.values() if isinstance(value, dict)
                        for element in value.get('elements', [])]
            if elements:
                parts = [_element_text(element) for element in elements]
                text = ''.join(part for part in parts if part is not None)
                paragraphs.extend(text.splitlines())
                if None not in parts:
                    covered.add(child['block_id'])
            pending.extend(reversed(child.get('children', [])))
        flat.append(' / '.join(part.strip().replace('|', r'\|') for part in paragraphs if part.strip()))
    if not any(flat):
        return '', set()
    rows = [flat[index:index + width] for index in range(0, len(flat), width)]
    if not prop.get('header_row') or height == 1 or not any(rows[0]):
        rows.insert(0, [f'第 {index + 1} 列' for index in range(width)])
    rows.insert(1, ['---'] * width)
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows), covered


def contextualize_snapshot(snapshot, *, include_images=False):
    """Reparse saved API blocks locally, retaining the original fetch and revision evidence."""
    raw = snapshot["raw_blocks"]
    by_id = {block["block_id"]: block for block in raw}
    if len(by_id) != len(raw):
        raise ValueError("Duplicate Feishu block IDs")
    ordered, seen, stack = [], set(), [snapshot["document_id"]]
    while stack:
        block_id = stack.pop()
        if block_id in seen or block_id not in by_id:
            raise ValueError("Feishu document tree has repeated or missing blocks")
        seen.add(block_id)
        block = by_id[block_id]
        ordered.append(block)
        stack.extend(reversed(block.get("children", [])))
    if len(seen) != len(raw):
        raise ValueError("Feishu document tree has unreachable blocks")

    blocks, unread, headings, images = [], [], [], []
    table_text_blocks = set()
    attachments = []
    for order, block in enumerate(ordered):
        if block['block_id'] in table_text_blocks:
            continue
        parts = []
        unread_before = len(unread)
        for value in block.values():
            if isinstance(value, dict):
                for element in value.get("elements", []):
                    part = _element_text(element)
                    if part is not None:
                        parts.append(part)
                    elif element:
                        unread.append({"block_id": block["block_id"], "element": list(element)})
        text = "".join(parts)
        if block.get('block_type') == 31:
            text, covered = _table_markdown(block, by_id)
            if text:
                parts = [text]
                table_text_blocks.update(covered)
            else:
                unread.append({'block_id': block['block_id'], 'reason': 'table_structure_unavailable'})
        level = next((int(key[7:]) for key in block if re.fullmatch(r"heading[1-9]", key)), None)
        if level is not None:
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, text))
        section = " / ".join(title for _, title in headings if title) or None
        if block.get('block_type') == 23 and block.get('file'):
            attachments.append({'block_id': block['block_id'], 'section': section, 'order': order,
                                'name': block['file'].get('name'), 'token': block['file'].get('token')})
        if include_images and block.get("block_type") == 27:
            image = block.get("image") or {}
            caption = image.get("caption") or {}
            images.append({"document_id": snapshot["document_id"], "revision": snapshot["revision"],
                           "block_id": block["block_id"], "section": section, "order": order,
                           "caption": caption.get("content") if isinstance(caption, dict) else None,
                           "token": image.get("token")})
        if parts:
            blocks.append(SourceBlock(document_id=snapshot["document_id"], block_id=block["block_id"],
                text=text, revision=snapshot["revision"], order=order,
                section=section).model_dump())
        elif len(unread) == unread_before and block.get("block_type") not in STRUCTURAL_BLOCK_TYPES:
            unread.append({"block_id": block["block_id"], "block_type": block.get("block_type")})
    if not blocks and not images and not attachments:
        raise ValueError("No readable text blocks; cannot extract project")
    result = {**snapshot, "blocks": blocks, "unread_items": unread, "attachments": attachments,
              "parser_policy": PARSER_POLICY, "block_order": "validated_document_tree_preorder"}
    if include_images:
        result["image_blocks"] = images
    return result


def image_type(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def save_snapshot(store, snapshot, *, directory=None):
    """Reuse unchanged immutable inputs; parsing policy changes get separate identities."""
    policy = snapshot.get("parser_policy")
    key = snapshot["wiki_node_token"] + ":" + snapshot["revision"] + ":" + snapshot["content_digest"]
    if policy:
        key += ":" + policy
    if not re.fullmatch(r"[A-Za-z0-9:-]+", key):
        raise ValueError("Invalid Feishu snapshot identity")
    try:
        previous = store.get("feishu_snapshot", key)
    except ValueError as exc:
        if not str(exc).startswith("Missing feishu_snapshot:"):
            raise
        previous = None
    reused = previous is not None
    if reused:
        if ({k: v for k, v in previous.items() if k != "fetched_at"}
                != {k: v for k, v in snapshot.items() if k != "fetched_at"}):
            raise ValueError("Feishu snapshot identity conflicts with stored content")
        snapshot = previous
    else:
        store.put("feishu_snapshot", key, snapshot, immutable=True)
    path = None
    if directory is not None:
        path = directory / (key.replace(":", "-") + ".json")
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("Feishu snapshot file conflicts with stored content")
        else:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(snapshot, stream, ensure_ascii=False, indent=2)
    return key, snapshot, reused, path


class Feishu:
    def __init__(self, allowed_document_ids: list[str], transport=None, *, allowed_wiki_tokens=None):
        self.allowed = set(allowed_document_ids)
        self.allowed_wiki = set(allowed_wiki_tokens or [])
        self.last_node = None
        self.http = httpx.Client(base_url="https://open.feishu.cn/open-apis", timeout=30,
                                 follow_redirects=False, transport=transport)

    def close(self):
        self.http.close()

    def read(self, document_id: str) -> dict:
        if document_id not in self.allowed:
            raise ValueError("Document is outside the explicit allowlist")
        return self._read_document(document_id, self._authenticate())

    def _authenticate(self):
        app, secret = os.environ.get("FEISHU_APP_ID"), os.environ.get("FEISHU_APP_SECRET")
        if not app or not secret:
            raise ValueError("FEISHU_APP_ID / FEISHU_APP_SECRET not configured")
        response = self.http.post("/auth/v3/tenant_access_token/internal",
                                   json={"app_id": app, "app_secret": secret})
        response.raise_for_status()
        token_data = response.json()
        if token_data.get("code") != 0:
            raise ValueError("Feishu authentication rejected")
        return {"Authorization": "Bearer " + token_data["tenant_access_token"]}

    def _get(self, path, headers, **kwargs):
        r = self.http.get(path, headers=headers, **kwargs)
        body = r.json()
        if r.status_code != 200 or body.get("code") != 0:
            raise ValueError(f"Feishu read failed: HTTP {r.status_code}, code {body.get('code')}, "
                             f"message {body.get('msg', '')[:600]}")
        return body["data"]

    def read_wiki(self, url_or_token):
        self.last_node = None
        token = url_or_token
        if "://" in token:
            url = urlparse(token)
            if url.scheme != "https" or url.hostname != "m13ru2b365j.feishu.cn":
                raise ValueError("Wiki host is outside the authorized tenant")
            match = re.fullmatch(r"/wiki/([A-Za-z0-9]+)", url.path)
            if not match:
                raise ValueError("Explicit wiki node URL required")
            token = match[1]
        if token not in self.allowed_wiki:
            raise ValueError("Wiki node is outside the explicit allowlist")
        headers = self._authenticate()
        node = self._get("/wiki/v2/spaces/get_node", headers, params={"token": token, "obj_type": "wiki"})["node"]
        self.last_node = node
        if node.get("node_token") != token or node.get("obj_type") != "docx":
            raise ValueError("Wiki node identity/type is not the authorized document")
        document_id = node.get("obj_token", "")
        if not re.fullmatch(r"[A-Za-z0-9]+", document_id):
            raise ValueError("Wiki returned an invalid document identity")
        snapshot = self._read_document(document_id, headers)
        snapshot.update(wiki_node_token=token, wiki_title=node.get("title", ""),
                        wiki_url="https://m13ru2b365j.feishu.cn/wiki/"+token,
                        space_id=node.get("space_id"), scope="explicit_node_only_no_sibling_crawl")
        return snapshot

    def _read_document(self, document_id, headers, *, include_images=False):
        def get(path, **kwargs):
            return self._get(path, headers, **kwargs)

        metadata = get(f"/docx/v1/documents/{document_id}")["document"]
        revision = str(metadata["revision_id"])
        raw, seen_tokens, page_token = [], set(), ""
        for _ in range(100):
            # Read the authorized current document. Exact revision selectors returned 403 for
            # the read-only wiki applications; the surrounding metadata reads detect edits.
            params = {"page_size": 500, "document_revision_id": -1}
            if page_token:
                params["page_token"] = page_token
            data = get(f"/docx/v1/documents/{document_id}/blocks", params=params)
            raw.extend(data.get("items", []))
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
            if not page_token or page_token in seen_tokens:
                raise ValueError("Feishu repeated/missing page token")
            seen_tokens.add(page_token)
        else:
            raise ValueError("Feishu page limit exceeded; document incomplete")
        # A concurrent edit must not silently become a mixed-revision profile.
        after = get(f"/docx/v1/documents/{document_id}")["document"]
        if str(after["revision_id"]) != revision:
            raise ValueError("Document changed during fetch; sync again")
        return contextualize_snapshot({"document_id": document_id, "revision": revision, "fetched_at": now(),
                "revision_read_mode": "latest_with_before_after_metadata_check",
                "raw_blocks": raw, "content_digest": hashlib.sha256(str(raw).encode()).hexdigest()[:16]},
                include_images=include_images)

    def _download_image(self, token, headers):
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9]{1,128}", token):
            return {"status": "invalid_token"}
        try:
            with self.http.stream("GET", f"/drive/v1/medias/{token}/download", headers=headers) as response:
                if response.status_code in (400, 401, 403):
                    result = {"status": "permission_denied" if response.status_code != 400 else "download_failed",
                              "http_status": response.status_code}
                    if response.headers.get("content-type", "").startswith("application/json"):
                        try:
                            body = json.loads(next(response.iter_bytes(chunk_size=4096), b""))
                        except ValueError:
                            body = None
                        if isinstance(body, dict) and type(code := body.get("code")) is int:
                            result["feishu_code"] = code
                            if code == 99991672:
                                result["status"] = "scope_missing"
                    return result
                if response.status_code == 429:
                    return {"status": "rate_limited", "http_status": 429}
                if response.status_code != 200:
                    return {"status": "download_failed", "http_status": response.status_code}
                length = response.headers.get("content-length", "")
                if length.isdecimal() and int(length) > MAX_IMAGE_BYTES:
                    return {"status": "size_limit"}
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=64_000):
                    if len(data) + len(chunk) > MAX_IMAGE_BYTES:
                        return {"status": "size_limit"}
                    data.extend(chunk)
        except httpx.HTTPError:
            return {"status": "download_failed"}
        mime = image_type(data)
        if not mime:
            return {"status": "unsupported_format"}
        return {"status": "downloaded", "media_type": mime, "size_bytes": len(data),
                "content_sha256": hashlib.sha256(data).hexdigest(), "data": bytes(data)}

    def _read_document_images(self, document_id, headers, *, limit=MAX_DOCUMENT_IMAGES):
        if not 1 <= limit <= MAX_DOCUMENT_IMAGES:
            raise ValueError("Invalid image sample size")
        snapshot = self._read_document(document_id, headers, include_images=True)
        downloaded, images = {}, []
        for index, block in enumerate(snapshot["image_blocks"]):
            locator = {key: value for key, value in block.items() if key != "token"}
            if index >= limit:
                result = {"status": "sample_limit"}
            elif (token := block["token"]) is None or not isinstance(token, str):
                result = {"status": "invalid_token"}
            else:
                if token not in downloaded:
                    downloaded[token] = self._download_image(token, headers)
                result = downloaded[token]
            images.append({**locator, **result})
        if images:
            after = self._get(f"/docx/v1/documents/{document_id}", headers)["document"]
            if str(after["revision_id"]) != snapshot["revision"]:
                raise ValueError("Document changed during image fetch; sync again")
        return {"document_id": document_id, "revision": snapshot["revision"], "images": images}


def inspect_inputs(config, store):
    """Sync explicit read-only nodes without silently changing the active project profile."""
    adapter=Feishu(config.allowed_document_ids,allowed_wiki_tokens=config.allowed_wiki_tokens)
    results=[]
    try:
        for token in config.allowed_wiki_tokens:
            try:
                snapshot=adapter.read_wiki(token)
                key,snapshot,reused,_=save_snapshot(store,snapshot)
                results.append({'wiki_node':token,'status':'read','title':snapshot['wiki_title'],
                    'snapshot_key':key,'text_blocks':len(snapshot['blocks']),'unread_items':len(snapshot['unread_items']),
                    'parser_policy':snapshot['parser_policy'],'snapshot_reused':reused,
                    'profile_update':'pending_explicit_import_review'})
            except Exception as exc:  # noqa: BLE001 - retain each authorized source's independent failure state
                results.append({'wiki_node':token,'status':'unavailable','error_type':type(exc).__name__,
                    'title':(adapter.last_node or {}).get('title'),
                    'detail':str(exc)[:200] if type(exc) is ValueError else 'Read unavailable'})
    finally:
        adapter.close()
    return results
