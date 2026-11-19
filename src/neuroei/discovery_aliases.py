"""Build literal query gates with explicit question provenance; no model calls."""
import re

from .discovery_contract import RecommendationRequest

GENERIC_WORDS = frozenset({'model', 'models', 'learning', 'control', 'robot', 'robots',
    'planning', 'policy', 'policies', 'world', 'dynamics', 'training', 'evaluation',
    'representation', 'representations', 'research', 'paper', 'papers', 'ai', 'ml',
    'rl', 'llm', 'data', 'method', 'methods', 'algorithm', 'algorithms'})


def literal_query_support(query, question_descriptions):
    """Preflight only: recognize entity-shaped literals, never certify relevance.

    Single tokens require letters plus digits or interior CamelCase, and an exact
    token mention in a mapped question. Case is ignored for the question match.
    No proper-name lookup, implicit synonym or broad-word exception is made.
    """
    phrase = ' '.join(query.split())
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 -]*', phrase)
            or re.search(r'\b(?:AND|OR|NOT)\b', phrase)):
        return {'supported': False, 'reason': 'plain_literal_query_required'}
    tokens = re.findall(r'[A-Za-z0-9]+', phrase)
    if len(tokens) >= 2:
        return {'supported': True, 'kind': 'literal_phrase', 'alias': phrase.lower()}
    entity_shape = (bool(re.search(r'[A-Za-z]', phrase) and re.search(r'\d', phrase))
                    or bool(re.search(r'[a-z][A-Z]', phrase)))
    if phrase.lower() in GENERIC_WORDS or not entity_shape or len(phrase) < 3:
        return {'supported': False, 'reason': 'single_generic_or_untyped_word'}
    pattern = r'(?<![A-Za-z0-9])'+re.escape(phrase)+r'(?![A-Za-z0-9])'
    if not all(re.search(pattern, description, re.IGNORECASE) for description in question_descriptions):
        return {'supported': False, 'reason': 'entity_not_in_each_mapped_question'}
    return {'supported': True, 'kind': 'explicit_entity_literal', 'alias': phrase.lower()}


def aliases_from_queries(request, query_question_ids=None):
    """Return {aliases, provenance, status} for dynamic question IDs.

    Single-question requests need no mapping. For multiple questions the upstream
    planner passes {exact_request_query: [question_ids]}, optionally stored under
    constraints.query_question_ids. Shared queries may explicitly name several
    questions. No positional guess or all-questions fanout is performed.

    Aliases are whole whitespace-normalized literal phrases, not generated
    synonyms or broad individual tokens. Evidence still has to occur in the
    candidate title/abstract. These gates never establish applicability.
    """
    req = RecommendationRequest.model_validate(request)
    ids = [q.id for q in req.research_questions]
    mapping = query_question_ids if query_question_ids is not None else req.constraints.get('query_question_ids')
    automatic = mapping is None and len(ids) == 1
    if automatic:
        mapping = {query: ids for query in req.queries}
    if mapping is None:
        return {'schema_version': '1', 'status': 'planning_required', 'aliases': {},
                'provenance': [], 'warnings': ['query_question_mapping_required']}
    if not isinstance(mapping, dict) or not set(mapping) <= set(req.queries):
        raise ValueError('Mapping keys must be exact queries from this request')
    aliases, provenance, warnings, checks = {qid: [] for qid in ids}, [], [], []
    for query, linked in mapping.items():
        if not isinstance(linked, list) or not linked or not set(linked) <= set(ids):
            raise ValueError('Each query needs valid current question IDs')
        support = literal_query_support(query, [q.description for q in req.research_questions if q.id in linked])
        checks.append({'query': query, 'question_ids': list(linked), **support})
        if not support['supported']:
            warnings.append('unsupported_literal_query:'+query)
            continue
        phrase = support['alias']
        for qid in dict.fromkeys(linked):
            if phrase not in aliases[qid]:
                aliases[qid].append(phrase)
            provenance.append({'request_id': req.request_id, 'user_id': req.user_id,
                'question_id': qid, 'source_refs': next(q.source_refs for q in req.research_questions if q.id == qid),
                'query': query, 'alias': phrase,
                'literal_kind': support['kind'],
                'mapping_basis': 'only_question' if automatic else 'explicit_query_question_mapping',
                'scope': 'literal_query_gate_not_applicability_review'})
    missing = [qid for qid in ids if not aliases[qid]]
    if missing:
        warnings.append('questions_without_aliases:'+','.join(missing))
    if any(len(values) > 20 for values in aliases.values()):
        raise ValueError('At most 20 aliases per question; narrow the explicit mapping')
    return {'schema_version': '1', 'status': 'planning_required' if missing else 'ready',
            'aliases': aliases, 'provenance': provenance, 'warnings': warnings, 'query_checks': checks}
