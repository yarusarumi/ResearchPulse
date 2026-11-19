"""Bounded media metadata and transcript parsing for user-provided sources.

This module fetches only explicitly supplied public WeChat article URLs and accepts pasted text or
transcripts, validates public media URLs, and reads one fixed YouTube metadata
feed endpoint. Captions are parsed from user-supplied VTT/SRT, never guessed
from prose, and no ASR, OCR, login, proxy, or token flow exists here.
"""

from __future__ import annotations

import html
import ipaddress
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from .config import StrictModel

__all__ = [
    "MediaInput",
    "MediaSegment",
    "fetch_youtube_feed",
    "parse_timed_transcript",
    "parse_youtube_feed",
]

_MAX_SEGMENT_TEXT = 12_000
_MAX_TITLE = 300
_MAX_AUTHOR = 300
_MAX_DESCRIPTION = 8_000
_MAX_INPUT_TEXT = 120_000
_MAX_SEGMENTS = 500
_MAX_TRANSCRIPT_CHARS = 120_000
_MAX_FEED_BYTES = 1_048_576
_MAX_FEED_ENTRIES = 30
_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id="
_TIMEOUT_SECONDS = 20.0

_PLATFORM_HOSTS: dict[str, frozenset[str] | None] = {
    "wechat": frozenset({"mp.weixin.qq.com"}),
    "wechat_channels": frozenset({"channels.weixin.qq.com"}),
    "youtube": frozenset({"www.youtube.com", "youtube.com", "youtu.be"}),
    "bilibili": frozenset({"www.bilibili.com", "bilibili.com", "b23.tv"}),
    "other": None,
}
_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_NUMERIC_HOST_RE = re.compile(r"[0-9.]+")

_ATOM = "{http://www.w3.org/2005/Atom}"
_YT = "{http://www.youtube.com/xml/schemas/2015}"
_MEDIA = "{http://search.yahoo.com/mrss/}"
_CHANNEL_RE = re.compile(r"UC[A-Za-z0-9_-]{22}")
_VIDEO_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FORBIDDEN_XML = (b"<!doctype", b"<!entity")

_VTT_HEADER_RE = re.compile(r"WEBVTT(?:[ \t].*)?")
_VTT_BLOCK_RE = re.compile(r"(?:NOTE|STYLE|REGION)(?:\s|$)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TIME_RE = re.compile(r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_TAG_RE = re.compile(
    r"</?(?:c|v|i|b|u|ruby|rt|lang|font)(?:\.[^\s<>]+)?(?:\s[^<>]*)?>", re.IGNORECASE
)
_CUE_TIME_TAG_RE = re.compile(r"<\d{1,3}:\d{2}(?::\d{2})?[.,]\d{3}>")


def _parse_iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("published_at must be an ISO 8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("published_at must include a timezone")
    return parsed


def _is_public_hostname(host: str) -> bool:
    if not host or len(host) > 253 or host.endswith("."):
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    if any(host.endswith(suffix) for suffix in _PRIVATE_HOST_SUFFIXES):
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    if _NUMERIC_HOST_RE.fullmatch(host):
        return False
    labels = host.split(".")
    if len(labels) < 2 or any(_LABEL_RE.fullmatch(label) is None for label in labels):
        return False
    return not labels[-1].isdigit()


def _validate_media_url(url: str, platform: str) -> None:
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("url must be a nonempty https URL without surrounding whitespace")
    if any(character in url for character in " \t\r\n"):
        raise ValueError("url must not contain whitespace")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValueError("url is malformed") from exc
    if parts.scheme != "https":
        raise ValueError("only https URLs are allowed")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("url must not contain userinfo")
    if parts.fragment:
        raise ValueError("url must not contain a fragment")
    if port not in (None, 443):
        raise ValueError("only the default https port is allowed")
    host = parts.hostname
    if host is None:
        raise ValueError("url must include a public hostname")
    host = host.lower()
    if not _is_public_hostname(host):
        raise ValueError("url host must be a public hostname")
    allowed = _PLATFORM_HOSTS[platform]
    if allowed is not None and host not in allowed:
        raise ValueError(f"url host is not allowed for platform {platform}")


class MediaSegment(StrictModel):
    text: str = Field(..., min_length=1, max_length=_MAX_SEGMENT_TEXT)
    start_seconds: float | None = None
    end_seconds: float | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _plain_text(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("segment text must not be blank")
        return value

    @field_validator("start_seconds", "end_seconds", mode="before")
    @classmethod
    def _bounded_seconds(cls, value):
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("seconds must be a finite nonnegative number")  # noqa: TRY004 - Pydantic validation
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("seconds must be a finite nonnegative number") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be a finite nonnegative number")
        return seconds

    @model_validator(mode="after")
    def _consistent_window(self):
        if self.end_seconds is not None:
            if self.start_seconds is None:
                raise ValueError("end_seconds requires start_seconds")
            if self.end_seconds < self.start_seconds:
                raise ValueError("end_seconds must not precede start_seconds")
        return self


class MediaInput(StrictModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    direction_id: UUID
    platform: Literal["wechat", "wechat_channels", "youtube", "bilibili", "other"]
    kind: Literal["article", "video"]
    url: str
    title: str = Field(..., min_length=1, max_length=_MAX_TITLE)
    author: str = Field("", max_length=_MAX_AUTHOR)
    published_at: str | None = None
    description: str = Field("", max_length=_MAX_DESCRIPTION)
    content_scope: Literal["metadata_only", "article_text", "transcript"]
    text: str = Field("", max_length=_MAX_INPUT_TEXT)
    segments: list[MediaSegment] = Field(default_factory=list, max_length=_MAX_SEGMENTS)

    @field_validator("user_id", "title", mode="before")
    @classmethod
    def _required_text(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("value must not be blank")
        return value

    @field_validator("author", mode="before")
    @classmethod
    def _optional_author(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("published_at", mode="before")
    @classmethod
    def _published_at(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("published_at must be an ISO 8601 string")  # noqa: TRY004 - Pydantic validation
        value = value.strip()
        if not value:
            return None
        _parse_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def _scope_contract(self):
        _validate_media_url(self.url, self.platform)
        has_text = bool(self.text.strip())
        if self.content_scope == "metadata_only":
            if has_text or self.segments:
                raise ValueError("metadata_only does not allow text or segments")
            return self
        if self.content_scope == "article_text":
            if self.kind != "article":
                raise ValueError("article_text requires kind='article'")
            if not has_text:
                raise ValueError("article_text requires nonempty text")
            if self.segments:
                raise ValueError("article_text does not allow segments")
            return self
        if self.kind != "video":
            raise ValueError("transcript requires kind='video'")
        if has_text:
            raise ValueError("transcript does not allow text")
        if not self.segments:
            raise ValueError("transcript requires segments")
        previous_start: float | None = None
        previous_end: float | None = None
        for segment in self.segments:
            if segment.start_seconds is None:
                raise ValueError("transcript segments require start_seconds")
            if previous_start is not None:
                if segment.start_seconds <= previous_start:
                    raise ValueError("transcript segment starts must increase")
                if previous_end is not None and segment.start_seconds < previous_end:
                    raise ValueError("transcript segments must not overlap")
            previous_start = segment.start_seconds
            previous_end = segment.end_seconds
        return self


def _parse_timestamp(token: str) -> float:
    match = _TIME_RE.fullmatch(token)
    if match is None:
        raise ValueError("invalid caption timestamp")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = match.group(4)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid caption timestamp")
    milliseconds = int(fraction.ljust(3, "0")[:3])
    total = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    if not math.isfinite(total):
        raise ValueError("invalid caption timestamp")
    return total


def _parse_timing_line(line: str) -> tuple[float, float]:
    left, arrow, right = line.partition("-->")
    if not arrow:
        raise ValueError("caption timing line is malformed")
    left_tokens = left.strip().split()
    right_tokens = right.strip().split()
    if len(left_tokens) != 1 or not right_tokens:
        raise ValueError("caption timing line is malformed")
    start = _parse_timestamp(left_tokens[0])
    end = _parse_timestamp(right_tokens[0])
    if end < start:
        raise ValueError("caption cue ends before it starts")
    return start, end


def _cue_text(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines:
        cleaned = _CUE_TIME_TAG_RE.sub(" ", line)
        cleaned = _CUE_TAG_RE.sub(" ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def _caption_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_timed_transcript(text: str) -> list[MediaSegment]:
    """Parse bounded WebVTT or SRT captions; prose is never guessed into cues."""
    if not isinstance(text, str):
        raise ValueError("transcript text must be a string")  # noqa: TRY004 - consistent parse error contract
    if _CONTROL_RE.search(text):
        raise ValueError("transcript contains binary control characters")
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) > _MAX_TRANSCRIPT_CHARS:
        raise ValueError("transcript is too long")
    if "-->" not in normalized:
        raise ValueError("text does not contain timed caption cues")
    lines = normalized.split("\n")
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        raise ValueError("transcript is empty")
    is_vtt = _VTT_HEADER_RE.fullmatch(lines[first].strip()) is not None
    body = lines[first:]
    if is_vtt:
        header_end = next((index for index, line in enumerate(body) if not line.strip()), None)
        if header_end is None:
            raise ValueError("WebVTT header has no cues")
        body = body[header_end + 1 :]
    segments: list[MediaSegment] = []
    for block in _caption_blocks(body):
        if is_vtt and _VTT_BLOCK_RE.match(block[0].strip()):
            continue
        timing_index = next((index for index, line in enumerate(block) if "-->" in line), None)
        if timing_index is None:
            raise ValueError("caption block without a timing line")
        if timing_index > 1:
            raise ValueError("caption block has an unexpected prefix")
        start, end = _parse_timing_line(block[timing_index])
        cue_lines = block[timing_index + 1 :]
        if any("-->" in line for line in cue_lines):
            raise ValueError("caption cue contains a nested timing line")
        cue_text = _cue_text(cue_lines)
        if not cue_text:
            continue
        if len(segments) >= _MAX_SEGMENTS:
            raise ValueError("too many caption cues")
        if segments:
            previous = segments[-1]
            if start <= previous.start_seconds:
                raise ValueError("caption cues are not increasing")
            if previous.end_seconds is not None and start < previous.end_seconds:
                raise ValueError("caption cues overlap")
        try:
            segments.append(MediaSegment(text=cue_text, start_seconds=start, end_seconds=end))
        except ValidationError as exc:
            raise ValueError("invalid caption cue") from exc
    if not segments:
        raise ValueError("no valid timed caption cues")
    return segments


def _find(element, namespace: str, name: str):
    found = element.find(namespace + name)
    if found is None:
        found = element.find(name)
    return found


def _findall(element, namespace: str, name: str) -> list[ET.Element]:
    found = element.findall(namespace + name)
    if not found:
        found = element.findall(name)
    return found


def _text(element, namespace: str, name: str) -> str:
    if element is None:
        return ""
    found = _find(element, namespace, name)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def _author_name(element) -> str:
    author = _find(element, _ATOM, "author")
    if author is None:
        return ""
    return " ".join(_text(author, _ATOM, "name").split())


def _channel_marker_from_uri(uri: str) -> str:
    if not uri:
        return ""
    path = urlsplit(uri).path.rstrip("/")
    if "/channel/" not in path:
        return ""
    return path.rsplit("/", 1)[-1]


def _channel_matches_request(marker: str, channel_id: str) -> bool:
    """YouTube may omit the leading ``UC`` in feed identity fields.

    The request channel id always keeps the full ``UC`` prefix. Real feeds have
    been observed with ``yt:channelId`` and ``id`` set to the 22-character core
    while the author URI and entries still use the full id. Both exact forms are
    accepted, but nothing shorter, longer, or different is.
    """
    return marker == channel_id or marker == channel_id[2:]


def _feed_identity_markers(root) -> list[str]:
    markers: list[str] = []
    channel = _text(root, _YT, "channelId")
    if channel:
        markers.append(channel)
    feed_id = _text(root, _ATOM, "id")
    if feed_id.startswith("yt:channel:"):
        markers.append(feed_id[len("yt:channel:") :])
    author_uri = _text(_find(root, _ATOM, "author"), _ATOM, "uri")
    channel_from_uri = _channel_marker_from_uri(author_uri)
    if channel_from_uri:
        markers.append(channel_from_uri)
    return markers


def _entry_video_id(entry) -> str:
    candidates: list[str] = []
    video_id = _text(entry, _YT, "videoId")
    if video_id:
        candidates.append(video_id)
    entry_id = _text(entry, _ATOM, "id")
    if entry_id.startswith("yt:video:"):
        candidates.append(entry_id[len("yt:video:") :])
    for link in _findall(entry, _ATOM, "link"):
        if (link.get("rel") or "alternate").lower() != "alternate":
            continue
        href = (link.get("href") or "").strip()
        if not href:
            continue
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https"):
            raise ValueError("entry link is not an http(s) URL")
        host = (parts.hostname or "").lower()
        if host == "youtu.be":
            candidate = parts.path.strip("/").split("/")[0]
        elif host in ("www.youtube.com", "youtube.com"):
            candidate = (parse_qs(parts.query).get("v") or [""])[0]
        else:
            raise ValueError("entry link host is not YouTube")
        if candidate:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("entry has no video identity")
    unique = set(candidates)
    if len(unique) != 1:
        raise ValueError("entry id, videoId, and url disagree")
    video = unique.pop()
    if _VIDEO_RE.fullmatch(video) is None:
        raise ValueError("entry video id is malformed")
    return video


def _feed_entry(entry, default_author: str) -> dict:
    video = _entry_video_id(entry)
    title = " ".join(_text(entry, _ATOM, "title").split())
    group = _find(entry, _MEDIA, "group")
    if not title and group is not None:
        title = " ".join(_text(group, _MEDIA, "title").split())
    if not title:
        raise ValueError("entry title is missing")
    author = _author_name(entry) or default_author
    description = _text(group, _MEDIA, "description")
    if not description:
        description = _text(entry, _MEDIA, "description")
    if not description:
        description = _text(entry, _ATOM, "summary")
    description = description.strip()
    published = _text(entry, _ATOM, "published")
    if published:
        _parse_iso_datetime(published)
    return {
        "platform": "youtube",
        "kind": "video",
        "url": f"https://www.youtube.com/watch?v={video}",
        "title": title[:_MAX_TITLE],
        "author": author[:_MAX_AUTHOR],
        "published_at": published or None,
        "description": description[:_MAX_DESCRIPTION],
        "content_scope": "metadata_only",
        "text": "",
        "segments": [],
    }


def _is_atom_feed(root) -> bool:
    return root.tag in (_ATOM + "feed", "feed")


def _parse_youtube_feed(content: bytes, channel_id: str) -> tuple[list[dict], bool]:
    if not isinstance(channel_id, str) or _CHANNEL_RE.fullmatch(channel_id) is None:
        raise ValueError("channel_id must match UC[A-Za-z0-9_-]{22}")
    if isinstance(content, bytearray):
        content = bytes(content)
    if not isinstance(content, bytes):
        raise ValueError("feed content must be bytes")  # noqa: TRY004 - consistent parse error contract
    if len(content) > _MAX_FEED_BYTES:
        raise ValueError("feed exceeds 1 MiB")
    lowered = content.lower()
    if any(marker in lowered for marker in _FORBIDDEN_XML):
        raise ValueError("DTD and entity declarations are not allowed")
    try:
        root = ET.fromstring(content)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("feed is not well-formed XML") from exc
    if not _is_atom_feed(root):
        raise ValueError("feed root is not an Atom feed")
    markers = _feed_identity_markers(root)
    if not markers:
        raise ValueError("feed has no channel identity")
    for marker in markers:
        if not _channel_matches_request(marker, channel_id):
            raise ValueError("feed channel does not match channel_id")
    entries = _findall(root, _ATOM, "entry")
    default_author = _author_name(root)
    items: list[dict] = []
    for entry in entries[:_MAX_FEED_ENTRIES]:
        entry_channel = _text(entry, _YT, "channelId")
        if entry_channel and not _channel_matches_request(entry_channel, channel_id):
            raise ValueError("entry channel does not match channel_id")
        items.append(_feed_entry(entry, default_author))
    return items, len(entries) > _MAX_FEED_ENTRIES


def parse_youtube_feed(content: bytes, channel_id: str) -> list[dict]:
    """Parse a YouTube Atom feed into bounded, metadata-only media dicts."""
    items, _ = _parse_youtube_feed(content, channel_id)
    return items


def fetch_youtube_feed(channel_id: str, *, transport=None) -> dict:
    """Fetch the fixed YouTube feed URL once and return a bounded result summary.

    Errors expose only an exception type name or HTTP status. Bodies over 1 MiB
    are never parsed; they fail with ``truncated=True``.
    """
    result: dict = {"status": "failed", "items": [], "error": None, "truncated": False}
    if not isinstance(channel_id, str) or _CHANNEL_RE.fullmatch(channel_id) is None:
        result["error"] = "ValueError"
        return result
    url = _FEED_URL + channel_id
    try:
        with httpx.Client(
            timeout=_TIMEOUT_SECONDS, follow_redirects=False, transport=transport
        ) as client, client.stream("GET", url) as response:
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}"
                return result
            chunks: list[bytes] = []
            received = 0
            oversized = False
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > _MAX_FEED_BYTES:
                    oversized = True
                    break
                chunks.append(chunk)
        if oversized:
            result["truncated"] = True
            result["error"] = "ValueError"
            return result
        items, capped = _parse_youtube_feed(b"".join(chunks), channel_id)
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        result["error"] = type(exc).__name__
        return result
    result.update(
        status="ok" if items else "empty",
        items=items,
        error=None,
        truncated=capped,
    )
    return result


class _WechatArticle(HTMLParser):
    """Read article text only inside WeChat's explicit article container."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.body_depth = None
        self.skip_depth = None
        self.parts = []
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('property', attrs.get('name', ''))] = attrs.get('content', '')
        if tag in ('br', 'hr', 'img', 'meta', 'link', 'input', 'source', 'wbr'):
            if self.body_depth is not None and tag == 'br':
                self.parts.append('\n')
            return
        self.depth += 1
        if attrs.get('id') == 'js_content':
            self.body_depth = self.depth
        if self.body_depth is not None and tag in ('script', 'style'):
            self.skip_depth = self.depth

    def handle_endtag(self, tag):
        if tag in ('br', 'hr', 'img', 'meta', 'link', 'input', 'source', 'wbr'):
            return
        if self.body_depth is not None and tag in ('p', 'section', 'div', 'h2', 'h3', 'li'):
            self.parts.append('\n\n')
        if self.depth == self.skip_depth:
            self.skip_depth = None
        if self.depth == self.body_depth:
            self.body_depth = None
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        if self.body_depth is not None and self.skip_depth is None:
            self.parts.append(data)


def fetch_wechat_article(url: str, *, transport=None) -> dict:
    """One bounded public request; a login/challenge page is never an article."""
    result = {'status': 'failed', 'items': [], 'error': None, 'truncated': False}
    try:
        _validate_media_url(url, 'wechat')
        if not re.fullmatch(r'/s(?:/[A-Za-z0-9_-]+)?', urlsplit(url).path):
            raise ValueError('Only public WeChat article URLs are supported')
        with (httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client,
              client.stream('GET', url) as response):
            if response.status_code in (301, 302, 303, 307, 308):
                result.update(status='blocked', error='需要微信验证或发生跳转，未读取正文')
                return result
            if response.status_code != 200:
                result['error'] = f'HTTP {response.status_code}'
                return result
            chunks, received = [], 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > _MAX_FEED_BYTES:
                    result.update(error='正文超过读取上限，未导入', truncated=True)
                    return result
                chunks.append(chunk)
        parser = _WechatArticle()
        parser.feed(b''.join(chunks).decode('utf-8', errors='replace'))
        text = '\n\n'.join(p.strip() for p in re.split(r'\n\s*\n', ''.join(parser.parts)) if p.strip())
        title = parser.meta.get('og:title', '').strip()
        if not title or not text:
            result.update(status='blocked', error='未取得可确认的文章正文，可能需要微信验证')
            return result
        if len(text) > _MAX_INPUT_TEXT:
            result.update(error='正文超过导入上限，未导入', truncated=True)
            return result
        result.update(status='ok', items=[{'platform': 'wechat', 'kind': 'article', 'url': url,
            'title': title, 'author': parser.meta.get('author', ''), 'published_at': None,
            'description': parser.meta.get('og:description', '')[:_MAX_DESCRIPTION],
            'content_scope': 'article_text', 'text': text, 'segments': []}])
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        result['error'] = type(exc).__name__
    return result
