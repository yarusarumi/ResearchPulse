"""Evidence review of a bounded PaperFlow score shortlist."""
import hashlib
import json

from .discovery_gates import gate_matches, validate_question_gates
from .discovery_fulltext_review import POLICY as FULLTEXT_REVIEW_POLICY, evidence_matches_paper
from .discovery_review import (
    REVIEW_ISOLATION_POLICY,
    apply_abstract_review,
    isolate_review_evidence,
)
from .llm import ModelFailure


REVIEWED_SHORTLIST_POLICY = 'paperflow-reviewed-score-shortlist-v2'


class ReviewedPaperFlowRanker:
    """Opt-in alternative to literal admission and upstream push-count selection.

    The inner scorer still runs once. Its complete scores, threshold and score
    order are retained. A supplied human/provider review supplies attributable
    abstract evidence instead of requiring discovery queries to occur verbatim.
    This is a distinct selection policy, never a replay of the old policy.
    """
    def __init__(self, ranker, review, *, max_candidates=40, fulltext_review=None):
        if not 1 <= max_candidates <= 40:
            raise ValueError('Reviewed shortlist must be 1..40')
        self.ranker, self.review, self.max_candidates = ranker, review, max_candidates
        self.fulltext_review = fulltext_review
        if callable(review) and not getattr(review, 'policy', None):
            raise ValueError('Explicit review provider policy identity required')
        self.policy = {'adapter': REVIEWED_SHORTLIST_POLICY, 'ranker': ranker.policy,
                       'isolation_policy': REVIEW_ISOLATION_POLICY,
                       'max_candidates': max_candidates,
                       'review_fingerprint': hashlib.sha256(json.dumps(
                           review.policy if callable(review) else review, sort_keys=True).encode()).hexdigest()}
        if fulltext_review:
            self.policy['fulltext_rescue'] = FULLTEXT_REVIEW_POLICY
        self.decisions, self.ranking_audit = {}, []
        self.review_errors = []

    @property
    def model_calls(self):
        return (getattr(self.ranker, 'model_calls', 0)+getattr(self.review, 'model_calls', 0)
                +getattr(self.fulltext_review, 'paid_model_calls', 0))

    @property
    def paid_model_calls(self):
        return (getattr(self.ranker, 'paid_model_calls', 0)+getattr(self.review, 'paid_model_calls', 0)
                +getattr(self.fulltext_review, 'paid_model_calls', 0))

    @property
    def embedding_runs(self):
        return getattr(self.ranker, 'embedding_runs', [])

    def __call__(self, request, papers):
        self.decisions, self.ranking_audit = {}, []
        self.review_errors = []
        validate_question_gates(request)
        scorer_error = None
        try:
            literal_rows = self.ranker(request, papers)
        except (RuntimeError, OSError, ImportError) as exc:
            if not callable(self.review):
                raise
            scorer_error = type(exc).__name__
            literal_rows = []
        self.ranking_audit = [dict(row, pre_review_stage=row['stage'],
            original_judgment_scope='original literal/push pipeline; not reviewed admission')
            for row in getattr(self.ranker, 'ranking_audit', ())]
        lookup = {p.paper_id: p for p in papers}
        if scorer_error:
            self.review_errors = [{'stage': 'paperflow', 'reason': scorer_error}]
            self.ranking_audit = [{'paper_id': paper.paper_id, 'score': None, 'score_rank': None,
                                   'threshold': None, 'stage': 'abstract_review_fallback'}
                                  for paper in papers]
        elif (len(self.ranking_audit) != len(lookup)
                or {row['paper_id'] for row in self.ranking_audit} != set(lookup)
                or any(row['score'] is None or row['score_rank'] is None for row in self.ranking_audit)):
            raise ValueError('Complete recorded PaperFlow scores and ranks required')
        # Use recorded PaperFlow order, not an invented new relevance score.
        if scorer_error:
            ordered = [row for row in self.ranking_audit if lookup[row['paper_id']].abstract]
        else:
            ordered = sorted(self.ranking_audit, key=lambda row: row['score_rank'])
        shortlist = ordered[:self.max_candidates]
        shortlisted_ids = {row['paper_id'] for row in shortlist}
        seeds = [{'paper_id': row['paper_id'], 'question_ids': [], 'basis': 'abstract',
                  'ranking_basis': {'algorithm': ('bounded abstract review after PaperFlow error' if scorer_error
                                                   else 'pinned PaperFlow scored order with supplied abstract review'),
                    'score': row['score'], 'score_basis': None if scorer_error else 'paperflow_pre_pipeline_score',
                    'score_rank': row['score_rank'], 'calibrated_probability': False,
                    'review_policy': REVIEWED_SHORTLIST_POLICY}}
                 for row in shortlist if scorer_error or (row['score'] is not None
                 and (row['threshold'] is None or row['score'] >= row['threshold']))]
        review_papers = [lookup[row['paper_id']] for row in seeds]
        if not seeds:
            for entry in self.ranking_audit:
                reason = 'below_ranking_threshold' if entry['paper_id'] in shortlisted_ids else 'review_shortlist_not_selected'
                entry.update(stage=reason, review_policy=REVIEWED_SHORTLIST_POLICY,
                             review_threshold_passed=False if entry['paper_id'] in shortlisted_ids else None,
                             review_evidence_passed=None, review_audit_ref=None)
                self.decisions[entry['paper_id']] = reason
            return []
        # An explicitly provisioned provider is called once with this bounded pool;
        # retries and provider budgets belong to its audited caller, never this adapter.
        if callable(self.review):
            try:
                review = self.review(request, review_papers)
            except (ModelFailure, ValueError, OSError) as exc:
                if isinstance(exc, ModelFailure) and exc.reason == 'canceled_by_user':
                    raise
                self.review_errors.append({'stage': 'abstract_review', 'reason': type(exc).__name__})
                self.decisions = dict(getattr(self.ranker, 'decisions', {}))
                return literal_rows
        else:
            review = self.review
        review = isolate_review_evidence(request, review_papers, review)
        self.review_errors.extend(review['paper_errors'])
        rows, review_decisions = apply_abstract_review(request, review_papers, seeds, review)
        review_decisions.update({error['paper_id']: 'review_evidence_invalid' for error in review['paper_errors']})
        accepted = []
        gates = request.constraints.get('question_evidence_gates', {})
        for row in rows:
            paper = lookup[row['paper_id']]
            passed = {qid for match in gate_matches(paper.title, paper.abstract, request) for qid in match['fact_ids']}
            if any(qid in gates and qid not in passed for qid in row['question_ids']):
                review_decisions[paper.paper_id] = 'question_evidence_insufficient'
                continue
            quotes = [row['applicability_review'][key] for key in ('method_quote', 'assumptions_quote')]
            row['matched_evidence'] = [{'scope': 'abstract', 'text': quote, 'matched_text': quote,
                'start': paper.abstract.index(quote), 'end': paper.abstract.index(quote)+len(quote)} for quote in quotes]
            accepted.append(row)
        fulltext_status = {}
        if self.fulltext_review and not scorer_error:
            fulltext_status = self._rescue(request, seeds, lookup, accepted, review_decisions, gates)
        admitted = {row['paper_id'] for row in accepted}
        reviewed = {row['paper_id']: row for row in review['papers']}
        error_by_id = {error['paper_id']: error for error in review['paper_errors']}
        for entry in self.ranking_audit:
            pid = entry['paper_id']
            reason = ('fulltext_review_eligible' if fulltext_status.get(pid) == 'evidence_found' else
                      'reviewed_adapter_eligible' if pid in admitted else
                      review_decisions.get(pid, 'below_ranking_threshold' if pid in shortlisted_ids
                                           else 'review_shortlist_not_selected'))
            entry.update(stage=reason, review_policy=REVIEWED_SHORTLIST_POLICY,
                         review_audit_ref=review['audit_ref'],
                         review_threshold_passed=entry['score'] >= entry['threshold'] if entry['threshold'] is not None else True,
                         review_evidence_passed=(pid in admitted) if pid in reviewed or pid in review_decisions
                             and review_decisions[pid] == 'review_evidence_invalid' else None,
                         review_explicit_gate_passed=False if reason == 'question_evidence_insufficient' else
                             True if pid in admitted else None,
                         review_applicability=reviewed.get(pid, {}).get('applicability'),
                         review_error=error_by_id.get(pid),
                         review_score_basis=None if scorer_error else 'paperflow_pre_pipeline_score')
            if pid in fulltext_status:
                entry['fulltext_review_status'] = fulltext_status[pid]
            self.decisions[pid] = reason
        return accepted

    def _rescue(self, request, seeds, lookup, accepted, decisions, gates):
        questions = {q.id: q for q in request.research_questions}
        covered = {qid for row in accepted if row['applicability_review']['applicability'] == 'direct'
                   for qid in row['question_ids']}
        statuses = {}
        for seed in seeds:
            remaining = set(questions) - covered
            if not remaining or len(accepted) >= request.max_recommendations:
                break
            pid = seed['paper_id']
            if decisions.get(pid) not in ('applicability_insufficient', 'question_evidence_insufficient'):
                continue
            paper = lookup[pid]
            refs, status = self.fulltext_review(request, paper, remaining)
            if status == 'not_attempted':
                break
            statuses[pid] = status
            supported = {}
            for qid, rows in refs.items():
                if qid not in remaining or not evidence_matches_paper(paper, rows):
                    continue
                if qid in gates:
                    material = '\n'.join([paper.abstract or '', *(row['text'] for row in rows)])
                    if not any(qid in match['fact_ids'] for match in gate_matches(paper.title, material, request)):
                        continue
                supported[qid] = rows
            if not supported:
                if refs:
                    statuses[pid] = 'question_evidence_insufficient'
                continue
            qids = [qid for qid in questions if qid in supported]
            covered.update(qids)
            quotes = [row for qid in qids for row in supported[qid]]
            pages = '、'.join(str(n) for n in sorted({row['page_no'] for row in quotes}))
            accepted.append({**seed, 'basis': 'fulltext', 'question_ids': qids,
                'reason': f'摘要未提供足够细节。已核对指定版本 PDF 第 {pages} 页的相关原文，可用于回答所关联的研究问题。',
                'applicability_review': {'applicability': 'direct', 'scope': 'selected_pdf_pages',
                                        'audit_ref': self.fulltext_review.client.run_id,
                                        'paper_version': paper.paper_version},
                'source_refs': {qid: questions[qid].source_refs for qid in qids},
                'matched_evidence': quotes,
                'ranking_basis': {**seed['ranking_basis'], 'fulltext_review_policy': FULLTEXT_REVIEW_POLICY}})
        return statuses
