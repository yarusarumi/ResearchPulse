"""Validate an explicitly supplied abstract review without replacing PaperFlow order.

No model calls here. A review may come from a separately audited provider or a
human; provenance and exact abstract quotes are mandatory, not quality proof.
"""
import hashlib
import json

REVIEW_ISOLATION_POLICY = 'abstract-review-isolation-v1'


def isolate_review_evidence(request, papers, review):
    """Identity/provenance failures abort globally; bad evidence is paper-local.

    Keep the strict validator unchanged. Never repair quotes or turn evidence
    errors into an applicability judgment. Input review is never mutated.
    """
    if (review.get('schema_version') != '1' or not review.get('audit_ref')
            or review.get('request_id') != request.request_id or review.get('user_id') != request.user_id
            or review.get('request_fingerprint') != request_fingerprint(request)):
        raise ValueError('Review request mismatch')
    lookup = {p.paper_id: p for p in papers}
    questions = {q.id for q in request.research_questions}
    seen = set()
    for item in review['papers']:
        pid = item['paper_id']
        if pid not in lookup or pid in seen:
            raise ValueError('Unknown or duplicate review identity')
        seen.add(pid)
        paper = lookup[pid]
        if ('paper_version' not in item or item['paper_version'] != paper.paper_version
                or item.get('document_fingerprint') != document_fingerprint(paper)):
            raise ValueError('Review document version mismatch')
        if not set(item.get('question_ids', [])) <= questions:
            raise ValueError('Review question provenance invalid')
    valid, errors = [], list(review.get('paper_errors', []))
    if (len({error['paper_id'] for error in errors}) != len(errors)
            or any(error['paper_id'] not in lookup or error['paper_id'] in seen
                   or error.get('reason') != 'review_evidence_invalid' for error in errors)):
        raise ValueError('Unknown review error identity')
    for error in errors:
        paper = lookup[error['paper_id']]
        if ('paper_version' not in error or error['paper_version'] != paper.paper_version
                or error.get('document_fingerprint') != document_fingerprint(paper)):
            raise ValueError('Review error document version mismatch')
    for item in review['papers']:
        try:
            apply_abstract_review(request, papers, [], {**review, 'papers': [item]})
        except ValueError as exc:
            errors.append({'paper_id': item['paper_id'], 'reason': 'review_evidence_invalid',
                           'paper_version': item['paper_version'],
                           'document_fingerprint': item['document_fingerprint'],
                           'error_type': type(exc).__name__, 'message': str(exc)})
        else:
            valid.append(item)
    return {**review, 'papers': valid, 'paper_errors': errors,
            'status': 'partial' if errors else 'completed', 'isolation_policy': REVIEW_ISOLATION_POLICY}



def request_fingerprint(request):
    return hashlib.sha256(json.dumps(request.model_dump(mode='json'), sort_keys=True).encode()).hexdigest()


def document_fingerprint(paper):
    return hashlib.sha256(json.dumps([paper.paper_id, paper.paper_version, paper.title,
                                     paper.abstract], ensure_ascii=False).encode()).hexdigest()


def apply_abstract_review(request, papers, rows, review):
    if review.get('schema_version') != '1' or not review.get('audit_ref'):
        raise ValueError('Versioned review with audit provenance required')
    if (review.get('request_id') != request.request_id or review.get('user_id') != request.user_id
            or review.get('request_fingerprint') != request_fingerprint(request)):
        raise ValueError('Review request mismatch')
    lookup = {p.paper_id: p for p in papers}
    questions = {q.id: q for q in request.research_questions}
    routes = {r['id']: r for r in getattr(request, 'constraints', {}).get('coverage_routes', [])}
    items = {}
    for item in review['papers']:
        pid = item['paper_id']
        if pid not in lookup or pid in items:
            raise ValueError('Unknown or duplicate review identity')
        paper = lookup[pid]
        if ('paper_version' not in item or item['paper_version'] != paper.paper_version
                or item.get('document_fingerprint') != document_fingerprint(paper)):
            raise ValueError('Review document version mismatch')
        if item['applicability'] not in ('direct', 'reference', 'unrelated', 'insufficient'):
            raise ValueError('Invalid applicability')
        if not item.get('applicability_reason'):
            raise ValueError('Concrete applicability reason required')
        if item['applicability'] in ('direct', 'reference'):
            if not item.get('question_ids') or not set(item['question_ids']) <= questions.keys():
                raise ValueError('Review question provenance invalid')
            for field in ('method_quote', 'assumptions_quote'):
                if not isinstance(item.get(field), str) or len(item[field].strip()) < 8 or item[field] not in (paper.abstract or ''):
                    raise ValueError('Review evidence must be an exact original abstract quote')
            if not item.get('transfer_suggestion') or not item.get('condition_differences'):
                raise ValueError('Specific transfer and condition differences required')
            if item.get('experiment_quote') and item['experiment_quote'] not in (paper.abstract or ''):
                raise ValueError('Experiment quote not in original abstract')
            for route in item.get('method_routes', []):
                if route['id'] not in routes or not route.get('evidence_quote') or route['evidence_quote'] not in (paper.abstract or ''):
                    raise ValueError('Route provenance invalid')
        items[pid] = item
        for role in item.get('representative_roles', []):
            if role['role'] not in ('basic_method', 'methodological_progress'):
                raise ValueError('Unknown representative role')
            evidence = role['evidence']
            if (any(not isinstance(value, str) or not value.strip() for value in
                    (evidence.get('document_id'), evidence.get('source_ref'), role.get('reason')))
                    or not evidence.get('quote') or len(evidence['quote'].strip()) < 8
                    or evidence['quote'] not in evidence.get('source_text', '')):
                raise ValueError('Role requires attributable supplied evidence and explicit reason')
            if evidence.get('scope') == 'abstract' and (evidence['quote'] not in (paper.abstract or '')
                    or evidence.get('paper_version') != paper.paper_version):
                raise ValueError('Role abstract evidence belongs to another document version')
    output, decisions = [], {}
    for row in rows:
        pid = row['paper_id']
        item = items.get(pid)
        if item is None:
            decisions[pid] = 'applicability_review_missing'
            continue
        scope = item['applicability']
        if scope in ('unrelated', 'insufficient') or (request.mode == 'direction_bootstrap' and scope == 'reference'):
            decisions[pid] = 'applicability_'+scope
            continue
        output.append({**row, 'question_ids': item['question_ids'], 'basis': 'abstract',
            'reason': '基于原摘要；适用性：'+item['applicability_reason']+
                '；可借鉴建议（非论文结论）：'+item['transfer_suggestion']+
                '；条件差异/未核实项：'+item['condition_differences']+'；未读全文。',
            'applicability_review': {**item, 'audit_ref': review['audit_ref']},
            'representative_roles': item.get('representative_roles', []),
            'method_excerpt': {'scope': 'abstract', 'text': item['method_quote']},
            'experiment_excerpt': {'scope': 'abstract', 'text': item['experiment_quote']} if item.get('experiment_quote') else None,
            'method_routes': [{'id': r['id'], 'label': routes[r['id']]['label'],
                'source_ref': routes[r['id']]['source_ref'],
                'evidence': {'scope': 'abstract', 'text': r['evidence_quote']}}
                for r in item.get('method_routes', [])],
            'source_refs': {qid: questions[qid].source_refs for qid in item['question_ids']}})
    return output, decisions
