"""Small public metadata adapters. Full text acquisition is a separate explicit step."""
import hashlib
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from datetime import date
from html import unescape
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import httpx

from .execution import WorkflowBusy, project_lock
from .models import PaperRecord, SourceResult

_ARXIV_THREAD_LOCK = Lock()

# Retry policy for transient public-metadata failures. Four attempts means three sleeps.
_RETRY_ATTEMPTS = 4
_RETRY_STATUS = (429, 500, 502, 503, 504)
_RETRY_BASE_DELAYS = (1.5, 3.0, 6.0)
# Deterministic jitter: keeps concurrent clients from retrying in lockstep while the
# sequence stays reproducible for tests (1.5s, 3.3s, 5.4s).
_RETRY_JITTER = (1.0, 1.1, 0.9)


def _retry_delay(attempt: int) -> float:
    index = min(attempt, len(_RETRY_BASE_DELAYS) - 1)
    return _RETRY_BASE_DELAYS[index] * _RETRY_JITTER[index % len(_RETRY_JITTER)]


class PublicSourceUnavailable(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__('Public metadata endpoint exhausted retries')


@contextmanager
def _arxiv_request_slot(root):
    """One active legacy API request per data directory, at least three seconds apart."""
    with _ARXIV_THREAD_LOCK:
        deadline = time.monotonic() + 90
        with ExitStack() as stack:
            while True:
                try:
                    stack.enter_context(project_lock(Path(root), 'public-source:arxiv'))
                    break
                except WorkflowBusy:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('arXiv request coordination timed out') from None
                    time.sleep(0.1)
            stamp = Path(root) / 'locks' / 'arxiv-last-request.time'
            previous = float(stamp.read_text()) if stamp.exists() else 0
            elapsed = max(0, time.time() - previous)
            time.sleep(max(0, 3 - elapsed))
            stamp.write_text(str(time.time()))
            yield


class UnsupportedSearchQuery(ValueError):
    """Query contains CJK text, has no searchable term, or uses unsupported syntax."""


class PubMedQueryRejected(ValueError):
    """PubMed esearch returned an errorlist."""


class PubMedQueryIgnored(ValueError):
    """PubMed dropped terms or reduced the query to date constraints."""


class NoQueryMatchedCandidates(ValueError):
    """Every returned candidate lacks the requested search terms."""


class PubMedFetchIncomplete(ValueError):
    """esearch returned IDs but efetch returned no article."""


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_QUERY_NOISE = frozenset({
    "all", "and", "or", "not", "ti", "abs", "au", "cat", "tiab", "mesh", "majr", "sh", "tw",
    "title", "abstract", "author", "authors", "journal", "field", "fields", "term", "terms",
    "doi", "pmid", "pubmed", "arxiv", "biorxiv", "preprint",
    "the", "a", "an", "of", "for", "in", "on", "to", "with", "by", "from", "vs", "via", "into",
    "using", "based", "is", "are", "as", "at", "be", "this", "that", "these", "those", "we", "our",
})
_DATE_FIELD = re.compile(
    r'(?:"[^"]*"|[0-9][0-9/.\-:]*)\s*\[\s*(?:date[^\]]*|dp|edat|pdat|pdt)\s*\]', re.IGNORECASE)


def _query_terms(query: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9]+", query):
        lowered = token.lower()
        if lowered in _QUERY_NOISE or (len(lowered) == 1 and not lowered.isdigit()):
            continue
        terms.append(lowered)
    return terms


# Generic modifiers sharpen a phrase but cost recall. They are dropped only when a
# query is widened as a last resort; noun-like terms stay so the source is not flooded.
_MODIFIER_TERMS = frozenset({
    # closed-loop / real-time style compounds, split apart by tokenisation
    "closed", "loop", "loops", "real", "realtime", "time", "end",
    # evaluation vocabulary
    "evaluation", "evaluations", "evaluate", "evaluated", "evaluating",
    "assessment", "assess",
    # adjectives
    "robotic", "autonomous", "automatic", "automated", "adaptive", "interactive",
    "novel", "new", "recent", "robust", "efficient", "effective", "improved",
    "enhanced", "advanced", "integrated", "general", "generic", "large", "small",
    # participles
    "driven", "aware", "enabled", "assisted", "oriented", "guided",
    # meta words that rarely appear verbatim in a title or abstract
    "approach", "approaches", "framework", "study", "studies", "analysis",
})


def _core_terms(query: str, limit: int) -> list[str]:
    """Up to *limit* noun-like terms, or no fallback when fewer than two survive."""
    core = [term for term in _query_terms(query) if term not in _MODIFIER_TERMS]
    return core[:limit] if len(core) >= 2 else []


def _ensure_supported_query(query: str) -> None:
    if _CJK.search(query):
        raise UnsupportedSearchQuery("CJK search terms cannot be sent to public sources")
    if not _query_terms(query):
        raise UnsupportedSearchQuery("No searchable Latin or numeric term in query")


# bioRxiv has no keyword search: terms are matched locally. Boolean grouping and field
# prefixes would be rewritten into a different AND-of-terms query, so reject them instead.
_BIORXIV_BOOLEAN = re.compile(r"\b(?:AND|ANDNOT|OR|NOT)\b")
_BIORXIV_FIELD = re.compile(
    r'(?<![A-Za-z0-9])(?:all|ti|abs|au|cat|tiab|mesh|majr|sh|tw|title|abstract|author|'
    r'authors|journal|doi|pmid)\s*:'
    r'|\[(?:all|ti|abs|au|cat|tiab|mesh|majr|sh|tw|title|abstract|author|authors|journal|'
    r'doi|pmid)\]', re.IGNORECASE)


def _ensure_local_phrase_query(query: str) -> None:
    if _BIORXIV_BOOLEAN.search(query) or _BIORXIV_FIELD.search(query):
        raise UnsupportedSearchQuery(
            "bioRxiv supports plain English term queries only; boolean and field syntax are unsupported")


def _biorxiv_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _stems(term: str) -> set[str]:
    stems = {term}
    if term.endswith("ies") and len(term) > 4:
        stems.add(term[:-3] + "y")
    elif term.endswith("es") and len(term) > 4:
        stems.add(term[:-2])
    elif term.endswith("s") and len(term) > 3:
        stems.add(term[:-1])
    elif term.endswith("y") and len(term) > 3:
        stems.add(term[:-1] + "ies")
    if term.endswith("ing") and len(term) > 5:
        stems.add(term[:-3])
    elif term.endswith("ed") and len(term) > 4:
        stems.add(term[:-2])
    return stems


def _contains_term(text: str, term: str) -> bool:
    for stem in _stems(term):
        if re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?:s|es|ed|ing|d|ly)?(?![a-z0-9])", text):
            return True
    return False


def _filter_candidates(papers: list[PaperRecord], query: str) -> tuple[list[PaperRecord], bool]:
    terms = _query_terms(query)
    if not terms:
        return papers, False
    kept = []
    for paper in papers:
        text = (paper.title + " " + paper.abstract).lower()
        if any(_contains_term(text, term) for term in terms):
            kept.append(paper)
    return kept, len(kept) < len(papers)


def _translation_has_terms(translation: str) -> bool:
    remaining = _DATE_FIELD.sub(" ", translation)
    remaining = re.sub(r"\b(?:AND|OR|NOT)\b", " ", remaining, flags=re.IGNORECASE)
    remaining = re.sub(r"[^A-Za-z0-9]+", " ", remaining)
    return bool(remaining.strip())


_TITLE_SCRIPT = re.compile(r"\s*<(sup|sub)>\s*([^<>]*?\S)\s*</\1>(?:\s*(?=[),.;:!?\]}]|$))?", re.IGNORECASE)


def paper_title_text(value: str | None) -> str:
    """Remove only inline script markup, including line breaks around its contents."""
    text = unescape(value or "")
    return _TITLE_SCRIPT.sub(lambda match: match[2], text)


def paper_title_key(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", paper_title_text(value)).casefold().split()).removesuffix(".")


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.IGNORECASE).lower()


def deduplicate(papers: list[PaperRecord]) -> list[PaperRecord]:
    # Keep versions separate. Same title alone is never a merge key.
    result: dict[tuple[str, str], PaperRecord] = {}
    aliases: dict[tuple[str, str], tuple[str, str]] = {}
    for paper in papers:
        paper = paper.model_copy(deep=True)
        paper.doi = normalize_doi(paper.doi)
        pid = (paper.paper_id, paper.version)
        doi_key = ("doi:" + paper.doi, paper.version) if paper.doi else pid
        key = aliases.get(pid, aliases.get(doi_key, pid))
        if key in result:
            result[key].sources = sorted(set(result[key].sources + paper.sources))
            result[key].doi = result[key].doi or paper.doi
        else:
            result[key] = paper
        aliases[pid] = aliases[doi_key] = key
    return list(result.values())


class Sources:
    def __init__(self, max_results=30, transport=None):
        self.max_results = max_results
        self.coordination_dir = None
        self._biorxiv_pages = {}
        self.http = httpx.Client(timeout=25, follow_redirects=True, transport=transport,
                                 headers={"User-Agent": "ResearchPulse-literature-agent/0.1 (research pilot)"})

    def close(self):
        self.http.close()

    def get(self, url, **kwargs):
        last_status = None
        for attempt in range(_RETRY_ATTEMPTS):
            if getattr(self, 'source_stop_requested', lambda: False)():
                raise RuntimeError('Source collection stopped')
            try:
                if self.coordination_dir and urlsplit(str(url)).hostname == 'export.arxiv.org':
                    with _arxiv_request_slot(self.coordination_dir):
                        request = {**kwargs, 'headers': {**kwargs.get('headers', {}), 'Connection': 'close'}}
                        response = self.http.get(url, **request)
                else:
                    response = self.http.get(url, **kwargs)
                last_status = response.status_code
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    return response
            except httpx.TransportError:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_retry_delay(attempt))
        raise PublicSourceUnavailable(last_status)

    def collect(self, source: str, query: str, since: str, until: str) -> SourceResult:
        start, end = date.fromisoformat(since), date.fromisoformat(until)
        if start > end:
            raise ValueError("Valid date range required")
        result = SourceResult(source=source, status="failed", query=query, since=since, until=until)
        try:
            _ensure_supported_query(query)
            papers, truncated = getattr(self, source)(query, since, until)
            result.papers = papers
            result.truncated = truncated
            result.status = "partial" if truncated else "ok" if papers else "empty"
            if source == 'crossref':
                result.error = '出版社登记元数据，索引可能滞后；仅覆盖已登记的期刊与会议论文，不代表全部顶会顶刊或全文'
        except (httpx.HTTPError, ET.ParseError, ValueError, KeyError, RuntimeError) as exc:
            # Never log API response body or authorization headers.
            result.error = (f'HTTP{exc.status_code}AfterRetries' if isinstance(exc, PublicSourceUnavailable)
                            else type(exc).__name__)
        return result

    def crossref(self, query, since, until):
        """Discover publisher-deposited journal and conference metadata, not full text."""
        from .journals import clean
        papers, incomplete = [], False
        for kind in ('journal-article', 'proceedings-article'):
            response = self.get('https://api.crossref.org/works', params={
                'query.bibliographic': query, 'rows': self.max_results,
                'filter': f'from-pub-date:{since},until-pub-date:{until},type:{kind}'})
            if len(response.content) > 5_000_000:
                raise ValueError('Metadata size limit exceeded')
            message = response.json()['message']
            items = message['items']
            incomplete |= int(message['total-results']) > len(items)
            for item in items:
                doi = normalize_doi(item.get('DOI'))
                title = clean(paper_title_text(' '.join(item.get('title', []))))
                # A month/year is insufficient evidence that a paper belongs to this week.
                dates = []
                for field in ('published', 'published-online', 'published-print'):
                    for parts in item.get(field, {}).get('date-parts', []):
                        try:
                            stamp = date(*parts).isoformat() if len(parts) == 3 else None
                        except (ValueError, TypeError):
                            stamp = None
                        if stamp and since <= stamp <= until:
                            dates.append(stamp)
                if (item.get('type') != kind or not title or not dates or not doi
                        or not re.fullmatch(r'10\.\d{4,9}/\S+', doi)):
                    incomplete = True
                    continue
                abstract = clean(item.get('abstract', ''))
                venue = clean(' '.join(item.get('container-title', [])))
                published = min(dates)
                revision = hashlib.blake2s(repr((doi, title, abstract, published)).encode(), digest_size=8).hexdigest()
                papers.append(PaperRecord(paper_id='doi:'+doi, version='metadata-'+revision,
                    title=title, abstract=abstract, doi=doi, published=published,
                    authors=[clean(a.get('name') or ' '.join(filter(None, [a.get('given'), a.get('family')])))
                             for a in item.get('author', [])],
                    sources=['crossref:'+(venue or kind)], url='https://doi.org/'+doi))
        papers, dropped = _filter_candidates(papers, query)
        return papers, incomplete or dropped

    def _arxiv_papers(self, expression, since, until):
        q = f'({expression}) AND lastUpdatedDate:[{since.replace("-", "")}0000 TO {until.replace("-", "")}2359]'
        root = ET.fromstring(self.get("https://export.arxiv.org/api/query", params={
            "search_query": q, "start": 0, "max_results": self.max_results,
            "sortBy": "lastUpdatedDate", "sortOrder": "descending"}).content)
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom",
              "o": "http://a9.com/-/spec/opensearch/1.1/"}
        papers = []
        for entry in root.findall("a:entry", ns):
            url = entry.findtext("a:id", "", ns)
            if "/abs/" not in url:
                raise ValueError("arXiv returned an error entry")
            raw = url.split("/abs/")[-1]
            match = re.fullmatch(r"(.+)v(\d+)", raw)
            pid, version = (match[1], "v" + match[2]) if match else (raw, "unversioned")
            papers.append(PaperRecord(paper_id="arxiv:" + pid, version=version,
                title=" ".join(entry.findtext("a:title", "", ns).split()),
                abstract=" ".join(entry.findtext("a:summary", "", ns).split()),
                authors=[a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
                doi=normalize_doi(entry.findtext("x:doi", None, ns)),
                published=entry.findtext("a:published", "", ns),
                updated=entry.findtext("a:updated", None, ns), sources=["arxiv"], url=url, is_preprint=True))
        total = int(root.findtext("o:totalResults", str(len(papers)), ns))
        return papers, total

    @staticmethod
    def _all_terms_expression(terms) -> str:
        return " AND ".join(f"all:{term}" for term in terms)

    def _arxiv_core_expression(self, query, terms) -> str:
        """Expression built from noun-like core terms, or "" when it would repeat."""
        if len(terms) <= 2:
            return ""
        core = _core_terms(query, 3)
        if not core:
            return ""
        expression = self._all_terms_expression(core)
        return "" if expression == self._all_terms_expression(terms) else expression

    def arxiv(self, query, since, until):
        # lastUpdatedDate includes revised manuscripts inside the rolling lookback.
        field_query = re.search(r"\b(all|ti|abs|au|cat):", query)
        expression = query if field_query else 'all:"' + query.replace('"', '') + '"'
        terms = _query_terms(query)
        core_expression = "" if field_query else self._arxiv_core_expression(query, terms)
        try:
            papers, total = self._arxiv_papers(expression, since, until)
            if not papers and total == 0 and not field_query and len(terms) > 1:
                # An exact phrase can be too narrow for a short window; retry requiring all
                # query terms so a source is not silently empty. Ranking/screening remains
                # responsible for precision; the report still labels the user-facing query.
                papers, total = self._arxiv_papers(
                    self._all_terms_expression(terms), since, until)
        except httpx.HTTPError as error:
            # A rejected or timed-out request is retried once with core terms, which keeps
            # a transient failure from hiding published work. The original error type is
            # preserved when the fallback fails too (no response body is ever inspected).
            if not core_expression:
                raise
            try:
                papers, total = self._arxiv_papers(core_expression, since, until)
            except httpx.HTTPError:
                raise error from None
        else:
            if not papers and total == 0 and core_expression:
                try:
                    papers, total = self._arxiv_papers(core_expression, since, until)
                except httpx.HTTPError:
                    papers, total = [], 0
        truncated = total > len(papers)
        if papers:
            papers, dropped = _filter_candidates(papers, query)
            if not papers:
                raise NoQueryMatchedCandidates("No arXiv candidate contains a requested term")
            truncated = truncated or dropped
        return papers, truncated

    def pubmed(self, query, since, until):
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        data = self.get(base + "esearch.fcgi", params={"db": "pubmed", "term": query,
            "retmode": "json", "retmax": self.max_results, "datetype": "edat",
            "mindate": since, "maxdate": until}).json()
        search = data.get("esearchresult") if isinstance(data, dict) else None
        if not isinstance(search, dict):
            raise PubMedQueryRejected("PubMed esearch response missing result")
        errors = search.get("errorlist")
        if errors and (not isinstance(errors, dict) or any(errors.values())):
            raise PubMedQueryRejected("PubMed query rejected")
        warnings = search.get("warninglist")
        if warnings:
            if not isinstance(warnings, dict):
                raise PubMedQueryIgnored("PubMed ignored part of the query")
            for key, value in warnings.items():
                if key == "outputmessages":
                    messages = value if isinstance(value, list) else [value]
                    # "No items found." is an informational empty-window signal, not an ignored query.
                    for message in messages:
                        if not isinstance(message, str) or message.strip().lower() != "no items found.":
                            raise PubMedQueryIgnored("PubMed ignored part of the query")
                elif value:
                    raise PubMedQueryIgnored("PubMed ignored part of the query")
        translation = search.get("querytranslation")
        if isinstance(translation, str) and not _translation_has_terms(translation):
            raise PubMedQueryIgnored("PubMed reduced the query to date constraints")
        ids = search.get("idlist") or []
        count = int(search.get("count") or len(ids))
        if not ids:
            return [], count > 0
        # Stay below NCBI's unauthenticated request rate, including the paired fetch.
        time.sleep(0.35)
        root = ET.fromstring(self.get(base + "efetch.fcgi", params={"db": "pubmed",
                               "id": ",".join(ids), "retmode": "xml"}).content)
        papers = []
        for item in root.findall("PubmedArticle"):
            citation = item.find("MedlineCitation")
            article = citation.find("Article")
            pmid = citation.findtext("PMID")
            doi = next((n.text for n in item.findall(".//ArticleId") if n.get("IdType") == "doi"), None)
            day = article.find("Journal/JournalIssue/PubDate")
            published = " ".join(day.itertext()) if day is not None else "unknown"
            papers.append(PaperRecord(paper_id="pubmed:" + pmid, version="indexed",
                title="".join(article.find("ArticleTitle").itertext()),
                abstract="\n".join("".join(n.itertext()) for n in article.findall("Abstract/AbstractText")),
                authors=[a.findtext("CollectiveName") or " ".join(filter(None,
                    [a.findtext("ForeName"), a.findtext("LastName")]))
                    for a in article.findall("AuthorList/Author")],
                doi=normalize_doi(doi), published=published, sources=["pubmed"],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
        if not papers:
            raise PubMedFetchIncomplete("PubMed fetch returned no article")
        effective_query = query
        if isinstance(translation, str):
            # Valid MeSH expansion can explain an acronym even when the acronym
            # itself is absent from the abstract. Dates and field names are not terms.
            expanded = re.sub(r'\[[^\]]+\]', ' ', _DATE_FIELD.sub(' ', translation))
            effective_query += ' ' + expanded
        papers, dropped = _filter_candidates(papers, effective_query)
        if not papers:
            raise NoQueryMatchedCandidates("No PubMed candidate contains a requested term")
        return papers, dropped or len(papers) < len(ids) or count > len(ids)

    @staticmethod
    def _biorxiv_paper(item) -> PaperRecord:
        doi = normalize_doi(item["doi"])
        return PaperRecord(paper_id="doi:" + doi, version="v" + item["version"],
            title=item["title"], abstract=item["abstract"],
            authors=item["authors"].split(";"), doi=doi, published=item["date"],
            updated=item["date"], sources=["biorxiv"],
            url="https://www.biorxiv.org/content/" + doi + "v" + item["version"], is_preprint=True)

    def biorxiv(self, query, since, until):
        # bioRxiv does not expose general keyword search: filter fetched metadata locally.
        # Every informative query term must match a whole word (or existing word form) in
        # title+abstract; arbitrary substrings such as "arm" in "pharmacology" must not.
        _ensure_local_phrase_query(query)
        terms = _query_terms(query)
        if not terms:
            raise UnsupportedSearchQuery("No searchable Latin or numeric term in query")
        start, end = date.fromisoformat(since), date.fromisoformat(until)
        # Bound page count explicitly, preserving partial coverage when the date window is larger.
        papers, scanned, cursor, total = [], [], 0, 1
        max_pages = 4
        for _ in range(max_pages):
            page_key = (since, until, cursor)
            data = self._biorxiv_pages.get(page_key)
            cached_page = data is not None
            if data is None:
                data = self.get(f"https://api.biorxiv.org/details/biorxiv/{since}/{until}/{cursor}/json").json()
            message = data["messages"][0]
            if message.get("status") not in ("ok", None):
                raise ValueError("bioRxiv returned error status")
            total = int(message.get("total", 0))
            entries = data["collection"]
            if not isinstance(entries, list):
                raise ValueError('bioRxiv collection is not a list')  # noqa: TRY004 - invalid remote payload
            for item in entries:
                # The version date is authoritative; dates embedded in old DOIs are not.
                published = _biorxiv_date(item.get("date"))
                if published is None or not start <= published <= end:
                    continue
                text = (item["title"] + " " + item["abstract"]).lower()
                scanned.append((item, text))
                if all(_contains_term(text, term) for term in terms):
                    papers.append(self._biorxiv_paper(item))
            self._biorxiv_pages[page_key] = data
            cursor += len(entries)
            if not entries or cursor >= total or len(papers) >= self.max_results:
                break
            if not cached_page:
                time.sleep(0.35)
        truncated = cursor < total or len(papers) > self.max_results
        if not papers and truncated:
            # The page cap can hide matches: widen to the first two core terms and re-filter
            # the pages already fetched. This never relaxes the query when anything matched.
            wide_terms = _core_terms(query, 2)
            if len(terms) > 2 and wide_terms:
                papers = [self._biorxiv_paper(item) for item, text in scanned
                          if all(_contains_term(text, term) for term in wide_terms)]
                truncated = True
        return papers[:self.max_results], truncated
