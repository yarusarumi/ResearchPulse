"""Optional one-hop bibliography discovery. No full-text analysis or model calls."""
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

import httpx

from .discovery_proceedings import LandingPage
from .journals import clean
from .models import PaperRecord

POLICY = 'bibliography-one-hop-v1'
ARXIV = r'\d{4}\.\d{4,5}'


def select_citing_documents(request, papers, rows):
    """Up to two parents/question, shared-question support before single-topic hits.

    This schedules a discovery hop, not recommendations or importance labels.
    Input ordering is the saved PaperFlow ordering; no paper IDs are preferred.
    """
    lookup = {p.paper_id: p for p in papers}
    eligible = [row for row in rows if len(set(row['question_ids'])) >= min(2, len(request.research_questions))]
    selected = []
    for question in sorted(request.research_questions, key=lambda q: -q.priority):
        added = 0
        for row in eligible:
            pid = row['paper_id']
            if added == 2 or len(selected) == 6:
                break
            if question.id not in row['question_ids'] or any(d['paper_id'] == pid for d in selected):
                continue
            paper = lookup[pid]
            url = ('https://arxiv.org/abs/'+paper.arxiv_id+paper.paper_version
                if paper.arxiv_id and re.fullmatch(ARXIV, paper.arxiv_id)
                and paper.paper_version and re.fullmatch(r'v\d+', paper.paper_version) else None)
            if url:
                selected.append({'paper_id': pid, 'url': url,
                    'query': 'semantic_parent_question:'+question.id,
                    'selection_question_id': question.id, 'question_ids': row['question_ids'],
                    'ranking_basis': row['ranking_basis']})
                added += 1
    return selected


class Bibliography(HTMLParser):
    """Only explicit LaTeXML bibliography items; body identifiers are ignored."""
    def __init__(self):
        super().__init__()
        self.items, self.current = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'li' and 'ltx_bibitem' in attrs.get('class', '').split():
            self.current = {'locator': attrs.get('id'), 'parts': [], 'links': []}
        if self.current is not None and tag == 'a' and attrs.get('href'):
            self.current['links'].append(attrs['href'])

    def handle_data(self, data):
        if self.current is not None:
            self.current['parts'].append(data)

    def handle_endtag(self, tag):
        if tag == 'li' and self.current is not None:
            item = self.current
            text = clean(' '.join(item['parts']))
            joined = text+' '+' '.join(item['links'])
            # Explicit arXiv URL/label/DOI, never bare body numbers or a guessed title.
            ids = re.findall(r'(?:arxiv(?:\.org/(?:abs|pdf)/|:\s*)|10\.48550/arxiv\.)('+ARXIV+r')(?:v\d+)?',
                             joined, re.IGNORECASE)
            self.items.extend({'kind': 'arxiv', 'value': pid, 'locator': item['locator'],
                               'reference_text': text} for pid in dict.fromkeys(ids))
            self.current = None


def arxiv_landing_record(text, identifier):
    """Read official citation metadata and UTC submission history, not bibliography text."""
    if not re.fullmatch(ARXIV+r'(?:v\d+)?', identifier):
        raise ValueError('Explicit arXiv identifier required')
    page = LandingPage()
    page.feed(text)
    fields = page.meta
    identity = fields.get('citation_arxiv_id', [''])[0]
    base = re.sub(r'v\d+$', '', identifier)
    if re.sub(r'v\d+$', '', identity) != base or not fields.get('citation_title'):
        raise ValueError('Official page identifier/title mismatch')
    history = re.search(r'class="submission-history"[^>]*>(.*?)</div>', text, re.DOTALL)
    dates = {}
    if history:
        plain = clean(history[1])
        for version, value in re.findall(r'\[(v\d+)\]\s*([A-Za-z]{3},\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+UTC)', plain):
            dates[version] = parsedate_to_datetime(value.replace('UTC', '+0000')).isoformat()
    requested = re.search(r'v\d+$', identifier)
    explicit = re.search(r'v\d+$', identity)
    selected = re.findall(r'<strong>\s*\[(v\d+)\]\s*</strong>', history[1]) if history else []
    version = selected[0] if len(selected) == 1 else explicit[0] if explicit else None
    if (requested and requested[0] != version) or (explicit and explicit[0] != version):
        raise ValueError('Selected document version mismatch')
    if version not in dates or 'v1' not in dates:
        raise ValueError('Selected version publication history unavailable')
    return PaperRecord(paper_id='arxiv:'+base, version=version,
        title=fields['citation_title'][0], abstract=fields.get('citation_abstract', [''])[0],
        authors=fields.get('citation_author', []), published=dates['v1'], updated=dates[version],
        sources=['arxiv', 'arxiv-official-landing'], url='https://arxiv.org/abs/'+base+version,
        is_preprint=True)


class CitationExpansion:
    """<=6 discovered documents, <=10 original metadata pages; no recursive hop.

    References only schedule discovery. Cross-document counts are not importance,
    relevance, calibrated probabilities, or grounds for automatically recommending.
    """
    def __init__(self, *, max_documents=6, max_resolutions=10, question_aliases=None, known_ids=()):
        if not 0 <= max_documents <= 6 or not 0 <= max_resolutions <= 10:
            raise ValueError('Citation expansion exceeds frozen one-hop bounds')
        self.policy = {'policy': POLICY, 'max_documents': max_documents,
                       'max_resolutions': max_resolutions, 'selection': 'input_order_unique_paper',
                       'ranking': 'distinct_citing_papers_desc_then_first_seen'}
        self.question_aliases, self.known_ids = question_aliases or {}, set(known_ids)
        self.policy.update(question_aliases=self.question_aliases, known_ids=sorted(self.known_ids))
        self.next_request = 0

    def get(self, sources, url):
        time.sleep(max(0, self.next_request-time.monotonic()))
        self.next_request = time.monotonic()+3
        return sources.get(url)

    def __call__(self, documents, sources):
        runs, discovered, seen, aggregated, unscheduled_documents = [], [], set(), {}, []
        for doc in documents:
            match = re.fullmatch(r'https://arxiv\.org/(?:abs|html)/('+ARXIV+r')(v\d+)?', doc['url'])
            if not match or match[1] in seen:
                continue
            seen.add(match[1])
            if len(seen) > self.policy['max_documents']:
                unscheduled_documents.append(doc)
                continue
            url = 'https://arxiv.org/html/'+match[1]+(match[2] or '')
            run = {'source': 'arxiv-bibliography', 'url': url, 'query': doc['query'],
                   'fetched_at': datetime.now(UTC).isoformat(), 'coverage_complete': False,
                   'status': 'failed', 'count': 0, 'truncated': False}
            try:
                response = self.get(sources, url)
                text = response.text
                if len(text) > 5_000_000:
                    raise ValueError('HTML size limit')
                identity = LandingPage()
                identity.feed(text)
                declared = identity.meta.get('citation_arxiv_id', [])
                if declared and declared[0] not in (match[1], match[1]+(match[2] or '')):
                    raise ValueError('Citing document identity/version mismatch')
                try:
                    effective = str(response.url)
                except RuntimeError:  # offline transport fixtures can omit Request
                    effective = None
                if effective and effective != url:
                    raise ValueError('Citing document URL changed; version identity unverified')
                run.update(requested_document_version=match[2] or None,
                           document_identity_declared=declared[0] if declared else None,
                           version_independently_verified=bool(match[2] and declared and declared[0] == match[1]+match[2]))
                parser = Bibliography()
                parser.feed(text)
                run.update(status='completed', count=len(parser.items), references=parser.items,
                           citing_paper_id='arxiv:'+match[1], citing_version=match[2] or None,
                           scope='references_only_no_fulltext_analysis')
                if not parser.items:
                    run['warning'] = 'no_explicit_arxiv_references'
                for item in parser.items:
                    group = aggregated.setdefault(item['value'], [])
                    if not any(origin['citing_paper_id'] == run['citing_paper_id'] for origin in group):
                        group.append({**item, 'citing_paper_id': run['citing_paper_id'],
                                      'citing_document': url, 'citing_version': match[2] or None,
                                      'fetched_at': run['fetched_at'], 'query': doc['query']})
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                run['error'] = type(exc).__name__
            runs.append(run)
        ranked = sorted(aggregated, key=lambda pid: -len(aggregated[pid]))
        cached = [pid for pid in ranked if 'arxiv:'+pid in self.known_ids]
        ranked = [pid for pid in ranked if 'arxiv:'+pid not in self.known_ids]
        if self.question_aliases:
            queues = [[pid for pid in ranked if any(alias.casefold() in origin['reference_text'].casefold()
                       for origin in aggregated[pid] for alias in aliases)]
                      for aliases in self.question_aliases.values()]
            scheduled = []
            while any(queues):
                for queue in queues:
                    while queue and queue[0] in scheduled:
                        queue.pop(0)
                    if queue:
                        scheduled.append(queue.pop(0))
            ranked = scheduled+[pid for pid in ranked if pid not in scheduled]
        for pid in ranked[:self.policy['max_resolutions']]:
            url = 'https://arxiv.org/abs/'+pid
            run = {'source': 'citation-original-metadata', 'url': url, 'query': pid,
                   'fetched_at': datetime.now(UTC).isoformat(), 'coverage_complete': False,
                   'status': 'failed', 'count': 0, 'truncated': False,
                   'citation_origins': aggregated[pid]}
            try:
                paper = arxiv_landing_record(self.get(sources, url).text, pid)
                discovered.append(paper)
                run.update(status='completed', count=1, paper_id=paper.paper_id)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                run['error'] = type(exc).__name__
            runs.append(run)
        return {'papers': discovered, 'source_runs': runs, 'policy': self.policy,
                'truncated': bool(unscheduled_documents or len(ranked) > self.policy['max_resolutions']),
                'unscheduled_documents': unscheduled_documents,
                'cached_identifiers_not_refetched': cached,
                'scheduled_identifiers': ranked[:self.policy['max_resolutions']],
                'unscheduled_identifiers': ranked[self.policy['max_resolutions']:],
                'warnings': ['citation_counts_are_local_discovery_signals_not_importance',
                             'arxiv_identifiers_only_no_doi_title_resolution']}
