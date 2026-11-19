"""Explicit opt-in all-concept evidence gates; never query rewriting or ranking."""
import re

from .discovery_aliases import GENERIC_WORDS


def validate_question_gates(request):
    gates = getattr(request, 'constraints', {}).get('question_evidence_gates', {})
    if not isinstance(gates, dict):
        raise TypeError('question_evidence_gates must map current question IDs')
    if not gates:
        return gates
    questions = {q.id: q for q in request.research_questions}
    query_map = request.constraints.get('query_question_ids')
    if len(questions) == 1 and query_map is None:
        query_map = {query: list(questions) for query in request.queries}
    if not isinstance(query_map, dict):
        raise TypeError('Gate query_question_ids must be an explicit mapping')
    for qid, gate in gates.items():
        if qid not in questions or not isinstance(gate, dict):
            raise ValueError('Unknown question gate')
        if set(gate) != {'source_refs', 'query_refs', 'all_of'}:
            raise ValueError('Gate fields must match frozen public shape')
        refs, queries, groups = (gate.get(k) for k in ('source_refs', 'query_refs', 'all_of'))
        if (not isinstance(refs, list) or not refs or len(refs) != len(set(refs))
                or not set(refs) <= set(questions[qid].source_refs)):
            raise ValueError('Gate source_refs must belong to current question')
        if (not isinstance(queries, list) or not queries or len(queries) != len(set(queries)) or any(
                query not in request.queries or not isinstance(query_map.get(query), list)
                or qid not in query_map[query] for query in queries)):
            raise ValueError('Gate queries need explicit current query-to-question mapping')
        if not isinstance(groups, list) or not 2 <= len(groups) <= 3:
            raise ValueError('Gate requires 2..3 necessary concept groups')
        used = set()
        for group in groups:
            if not isinstance(group, dict) or set(group) != {'any_of'}:
                raise ValueError('Concept group fields must match frozen public shape')
            phrases = group.get('any_of') if isinstance(group, dict) else None
            if not isinstance(phrases, list) or not 1 <= len(phrases) <= 4:
                raise ValueError('Concept group requires 1..4 literal phrases')
            current = set()
            for phrase in phrases:
                if not isinstance(phrase, str) or len(phrase) > 200:
                    raise ValueError('Invalid gate phrase')
                plain = ' '.join(phrase.split())
                if (not re.fullmatch(r'[A-Za-z](?:[A-Za-z0-9 -]*[A-Za-z0-9])?', plain)
                        or re.search(r'\b(?:AND|OR|NOT)\b', plain)
                        or plain.casefold() in GENERIC_WORDS or len(plain) < 3):
                    raise ValueError('Gate phrase must be explicit non-generic English concept')
                normalized = ' '.join(re.split(r'[\s\-‐‑–]+', plain.casefold()))
                if normalized in current:
                    raise ValueError('Duplicate literal within concept group')
                current.add(normalized)
            if current & used:
                raise ValueError('Different necessary groups cannot reuse the same literal')
            used.update(current)
    return gates


def gate_matches(title, abstract, request):
    matches = []
    for qid, gate in validate_question_gates(request).items():
        evidence = []
        for index, group in enumerate(gate['all_of']):
            found = None
            for phrase in group['any_of']:
                pieces = re.split(r'[\s\-‐‑–]+', phrase.strip())
                pattern = r'(?<![a-z0-9])'+r'[\s\-‐‑–]+'.join(re.escape(p) for p in pieces)+r'(?![a-z0-9])'
                for scope, text in (('title', title), ('abstract', abstract or '')):
                    hit = re.search(pattern, text, re.IGNORECASE)
                    if hit:
                        left = max(text.rfind(mark, 0, hit.start()) for mark in ('.', '!', '?', '\n'))+1
                        right = min([p+1 for mark in ('.', '!', '?', '\n')
                                     if (p := text.find(mark, hit.end())) >= 0] or [len(text)])
                        found = {'scope': scope, 'phrase': phrase, 'matched_text': hit[0],
                            'start': hit.start(), 'end': hit.end(), 'text': text[left:right],
                            'sentence_start': left, 'sentence_end': right, 'concept_group_index': index}
                        break
                if found:
                    break
            if found is None:
                break
            evidence.append(found)
        if len(evidence) == len(gate['all_of']):
            matches.append({'topic_id': 'explicit-gate:'+qid, 'fact_ids': [qid],
                'matched_aliases': [e['phrase'] for e in evidence], 'evidence': evidence,
                'source_refs': gate['source_refs'], 'query_refs': gate['query_refs'],
                'scope': 'question_concept_evidence_not_applicability'})
    return matches
