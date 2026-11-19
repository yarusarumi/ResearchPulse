"""Thin bridge to the already pinned upstream scorer. No model downloads or calls."""
import hashlib
import re

from .discovery_phrases import PHRASE_POLICY
from .models import PaperRecord, ProjectFact, ProjectProfile, SourceBlock
from .paperflow_adapter import InterestMapping, score_paperflow, topic_matches

RANKING_AUDIT_POLICY = 'paperflow-ranking-audit-v1'

_REVIEW_TITLE = re.compile(
    r'\b(?:survey|systematic review|literature review|review of|overview|tutorial|state of the art)\b'
    r'|综述|研究进展|文献回顾', re.IGNORECASE)


def review_like(title):
    """Title-level review/survey signal only; abstracts often say 'we review related work'."""
    return bool(title and _REVIEW_TITLE.search(title))


def abstract_evidence(abstract, routes):
    """Conservative extractive cues, not semantic proof or a literature quality score.

    Background/questions are never promoted to this paper's method by keyword alone.
    If author-attributed contribution cues are absent, the method stays unknown.
    """
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', abstract or '') if not s.rstrip().endswith('?')]
    contribution = re.compile(
        r'\b(?:we\s+(?:(?:further|also|can(?:\s+even)?)\s+)?(?:introduce|propose|present|develop|design|approach|explore|learn|train|show)'
        r'|our\s+(?:method|model|agent|approach)\s+(?:learns|uses|solves|combines)'
        r'|this\s+(?:paper|work)\s+(?:introduces|proposes|presents))\b', re.IGNORECASE)
    methods = [s for s in sentences if contribution.search(s)]
    def method_priority(sentence):
        return (bool(re.search(r'\b(?:algorithm|transformer|policy|agent|model|skills|dynamics)\b', sentence, re.IGNORECASE)),
                bool(re.search(r'\b(?:learns|processes|outputs|generat\w*|constrain\w*|optim\w*)\b', sentence, re.IGNORECASE)))
    method = max(methods, key=method_priority, default=None)
    experiments = []
    for sentence in sentences:
        if re.search(r'\b(?:prior|previous|existing|earlier)\s+(?:work|methods?|models?|studies|approaches)\b',
                     sentence, re.IGNORECASE):
            continue
        setting = re.search(r'\b(?:tasks|domains|environments|datasets|trajectories|benchmark|success|episodes|real-world|simulat\w*)\b',
                            sentence, re.IGNORECASE)
        evaluation = re.search(r'\b(?:evaluat\w*|benchmark\w*|experiment\w*|demonstrat(?:e|es|ed|ing)|outperform\w*)\b',
                               sentence, re.IGNORECASE)
        own_evaluation = (contribution.search(sentence) or re.search(r'\b(?:we|our)\b', sentence, re.IGNORECASE)
            or re.match(r'(?:Extensive\s+)?(?:Experiments|Evaluations|Empirical results)\b', sentence, re.IGNORECASE))
        if setting and own_evaluation and evaluation:
            experiments.append(sentence)
    experiment = max(experiments, key=lambda s: bool(re.search(r'\d|thousands|hundreds', s)), default=None)
    evidence_routes = []
    for route in routes:
        quote = next((s for s in methods if any(t.casefold() in s.casefold() for t in route['terms'])), None)
        if quote:
            evidence_routes.append({'id': route['id'], 'label': route['label'],
                'evidence': {'scope': 'abstract', 'text': quote}, 'source_ref': route['source_ref']})
    return method, experiment, evidence_routes


class PaperFlowVectors:
    """Consume explicitly provisioned semantic vectors, never hash/mock embeddings.

    vectors = {paper_id: vector}; query_vectors follow request question order.
    aliases = {question_id: [literal English phrases]}. Aliases are explicit
    relevance evidence gates, not a replacement ranking algorithm.
    """
    paid_model_calls = 0
    model_calls = 0

    def __init__(self, vectors, query_vectors, aliases, *, embedding_audit_ref):
        if not embedding_audit_ref:
            raise ValueError('Real embedding provenance required')
        self.vectors, self.query_vectors, self.aliases = vectors, query_vectors, aliases
        self.audit_ref = embedding_audit_ref
        self.policy = 'paperflow-pinned-vectors-v3:'+PHRASE_POLICY+':'+RANKING_AUDIT_POLICY+':'+hashlib.sha256(
            repr((vectors, query_vectors, aliases, embedding_audit_ref)).encode()).hexdigest()

    def __call__(self, request, papers):
        self.decisions = {}
        self.ranking_audit = []
        from .discovery_gates import gate_matches, validate_question_gates
        gates = validate_question_gates(request)
        if len(self.query_vectors) != len(request.research_questions):
            raise ValueError('Question vectors misaligned')
        blocks, facts, topics = [], [], []
        for index, question in enumerate(request.research_questions):
            block_id = 'question-'+str(index)
            blocks.append(SourceBlock(document_id=request.request_id, block_id=block_id,
                text=question.description, revision=request.trigger.version))
            facts.append(ProjectFact(id=question.id, text=question.description, kind='question',
                                     source_blocks=[block_id]))
            topics.append({'topic_id': 'neuroei-q'+str(index), 'label': question.description,
                           'fact_ids': [question.id], 'aliases': self.aliases[question.id],
                           'weight': question.priority/max(q.priority for q in request.research_questions)})
        profile = ProjectProfile(project_id=request.project_id or request.direction_id or request.request_id,
                                 version=request.request_id, blocks=blocks, facts=facts)
        mapping = InterestMapping(project_id=profile.project_id, profile_version=profile.version,
            mapping_version='request-v1', provenance='Explicit request questions; original source_refs in audit',
            topics=topics)
        legacy = [PaperRecord(paper_id=p.paper_id, version=p.paper_version or 'unversioned', title=p.title,
            abstract=p.abstract or '', authors=p.authors, doi=p.doi, published=p.published_at or '',
            sources=[s['source'] for s in p.sources], url=p.original_links[0] if p.original_links else '')
            for p in papers]
        result = score_paperflow(profile, legacy, [self.vectors[p.paper_id] for p in papers],
                                self.query_vectors, mapping)
        scored = {r['paper_id']: r for r in result['scored']}
        lookup = {p.paper_id: p for p in papers}
        threshold = result['metadata']['effective_weights']['threshold_maybe_interested']
        score_ranks = {row['paper_id']: index for index, row in enumerate(result['scored'], 1)}
        pipeline_ranks = {row['paper_id']: index for index, row in enumerate(result['upstream_pipeline'], 1)}
        for paper in papers:
            entry = scored.get(paper.paper_id, {})
            selected = paper.paper_id in pipeline_ranks
            self.ranking_audit.append({'paper_id': paper.paper_id, 'paper_version': paper.paper_version,
                'score': entry.get('score'), 'score_rank': score_ranks.get(paper.paper_id),
                'upstream_output_rank': pipeline_ranks.get(paper.paper_id),
                'upstream_score': None, 'threshold': threshold, 'threshold_passed': None,
                'question_evidence_passed': None,
                'stage': 'upstream_selected' if selected else 'upstream_not_selected',
                'upstream_exclusion_detail': None,
                'calibrated_probability': False, 'policy': RANKING_AUDIT_POLICY})
            if not selected:
                self.decisions[paper.paper_id] = 'upstream_not_selected'
        audit_by_id = {row['paper_id']: row for row in self.ranking_audit}
        # Upstream owns ranking/diversity. Our evidence gate prevents its minimum-count
        # policy from filling recommendations with unsupported items.
        rows = []
        for item in result['upstream_pipeline']:
            paper = lookup[item['paper_id']]
            audit = audit_by_id[paper.paper_id]
            matches = topic_matches(paper.title, paper.abstract or '', mapping)
            if gates:
                matches = [m for m in matches if not any(qid in gates for qid in m['fact_ids'])]
                matches.extend(gate_matches(paper.title, paper.abstract, request))
            audit.update(upstream_score=float(item['score']),
                         threshold_passed=threshold is None or item['score'] >= threshold,
                         question_evidence_passed=bool(matches))
            if not matches or (threshold is not None and item['score'] < threshold):
                self.decisions[paper.paper_id] = 'question_evidence_insufficient' if not matches else 'below_ranking_threshold'
                audit['stage'] = self.decisions[paper.paper_id]
                continue
            ids = sorted({qid for m in matches for qid in m['fact_ids']})
            evidence = [e for match in matches for e in match.get('evidence', [])]
            excerpts = sorted({e['matched_text'] for e in evidence})
            questions = '；'.join(q.description for q in request.research_questions if q.id in ids)
            method_sentence, experiment_sentence, method_routes = abstract_evidence(
                paper.abstract, request.constraints.get('coverage_routes', []))
            if request.mode == 'direction_bootstrap' and method_sentence is None:
                self.decisions[paper.paper_id] = 'method_evidence_insufficient'
                audit['stage'] = 'method_evidence_insufficient'
                continue
            audit['stage'] = 'adapter_eligible'
            rows.append({'paper_id': paper.paper_id, 'question_ids': ids,
                'review_like': review_like(paper.title),
                'reason': '标题或摘要原文提到：“'+'”、“'.join(excerpts)+'”，对应问题：'+questions+
                    ('；作者陈述的方法线索（原文）：“'+method_sentence+'”。适配建议：先核对该方法对当前问题的适用条件；'
                     if method_sentence else '；标题/摘要不足以确认本论文的具体方法贡献，需核查原文；')+
                    ('论文摘要中的实验范围原文：“'+experiment_sentence+'”；' if experiment_sentence else
                     '论文摘要未明确实验范围；')+'当前实验条件：'+
                    str(request.constraints.get('experimental_conditions', '未提供当前实验条件'))+
                    '；与论文实验的硬件、数据量、控制频率差异尚未核实。未读全文，不推断方法有效性。',
                'basis': 'abstract' if paper.abstract else 'title',
                'matched_evidence': evidence,
                'method_excerpt': {'scope': 'abstract', 'text': method_sentence} if method_sentence else None,
                'experiment_excerpt': {'scope': 'abstract', 'text': experiment_sentence} if experiment_sentence else None,
                'method_routes': method_routes,
                'ranking_basis': {'algorithm': 'pinned PaperFlow sort_and_categorize',
                                  'score': float(item['score']), 'calibrated_probability': False,
                                  'embedding_audit_ref': self.audit_ref,
                                  'question_gate_evidence': [m for m in matches if m.get('scope') == 'question_concept_evidence_not_applicability'],
                                  'topic_match_provenance': scored[paper.paper_id]['topic_match_provenance']},
                'source_refs': {q.id: q.source_refs for q in request.research_questions if q.id in ids}})
        return rows


class EmbeddingBudgetExceeded(RuntimeError):
    pass


class PaperFlowSemanticRanker:
    """Encode newly discovered metadata with an explicitly provisioned CPU model.

    Reuses PaperFlowVectors and upstream sorting. No model loading, downloads,
    provider discovery, paid API, or feedback is performed by this adapter.
    """
    paid_model_calls = 0

    def __init__(self, encoder, aliases, *, model_revision, embedding_audit_ref,
                 max_candidates=500, max_calls=2, max_input_characters=4_000_000):
        if str(encoder.device) != 'cpu' or not model_revision or not embedding_audit_ref:
            raise ValueError('Explicit CPU encoder identity and audit required')
        if (not 1 <= max_candidates <= 1200 or not 1 <= max_calls <= 2
                or not 1 <= max_input_characters <= 8_000_000):
            raise ValueError('Embedding bounds exceed module policy')
        self.encoder, self.aliases = encoder, aliases
        self.audit_ref, self.max_candidates, self.max_calls = embedding_audit_ref, max_candidates, max_calls
        self.max_input_characters = max_input_characters
        self.model_calls, self.decisions, self.embedding_runs = 0, {}, []
        self.policy = {'adapter': 'paperflow-cpu-semantic-v2-bounded-batches', 'phrase_policy': PHRASE_POLICY, 'model_revision': model_revision,
            'ranking_audit_policy': RANKING_AUDIT_POLICY,
            'aliases': aliases, 'max_candidates': max_candidates, 'max_calls': max_calls,
            'max_seq_length': encoder.max_seq_length, 'embedding_audit_ref': embedding_audit_ref,
            'max_input_characters': max_input_characters, 'encode_batch_size': 4, 'max_questions': 20}

    def preflight(self, request, papers):
        characters = sum(len(p.title)+1+len(p.abstract or '') for p in papers)+sum(
            len(q.description) for q in request.research_questions)
        reason = ('candidate_count_limit' if len(papers) > self.max_candidates else
                  'question_count_limit' if len(request.research_questions) > 20 else
                  'model_call_limit' if self.model_calls >= self.max_calls else
                  'input_character_limit' if characters > self.max_input_characters else None)
        return {'allowed': reason is None, 'reason': reason, 'candidate_count': len(papers),
            'question_count': len(request.research_questions), 'input_characters': characters,
            'max_candidates': self.max_candidates, 'max_input_characters': self.max_input_characters,
            'max_calls': self.max_calls, 'calls_used': self.model_calls,
            'max_questions': 20,
            'planned_batches': (len(papers)+len(request.research_questions)+3)//4}

    def __call__(self, request, papers):
        from .discovery_gates import validate_question_gates
        self.decisions, self.ranking_audit = {}, []
        validate_question_gates(request)
        budget = self.preflight(request, papers)
        if not budget['allowed']:
            self.embedding_runs.append({'request_id': request.request_id, 'user_id': request.user_id,
                'status': 'rejected_before_encode', 'budget': budget, 'audit_ref': self.audit_ref})
            raise EmbeddingBudgetExceeded(budget['reason']+'; no encode attempted')
        self.model_calls += 1
        texts = [p.title+'\n'+(p.abstract or '') for p in papers]+[q.description for q in request.research_questions]
        run_ref = self.audit_ref+':'+str(self.model_calls)
        run = {'audit_ref': run_ref, 'request_id': request.request_id,
            'user_id': request.user_id, 'input_manifest': [
                {'paper_id': p.paper_id, 'paper_version': p.paper_version} for p in papers],
            'status': 'started', 'budget': budget, 'batches': [],
            'max_seq_length': self.encoder.max_seq_length,
            'token_scope': 'encoder_max_seq_length_applies_no_manual_character_truncation'}
        self.embedding_runs.append(run)
        try:
            vectors = []
            for start in range(0, len(texts), 4):
                chunk = texts[start:start+4]
                batch = {'start': start, 'count': len(chunk), 'characters': sum(map(len, chunk)), 'status': 'started'}
                run['batches'].append(batch)
                values = self.encoder.encode(chunk, batch_size=4, normalize_embeddings=True,
                                             show_progress_bar=False).tolist()
                if len(values) != len(chunk):
                    raise ValueError('Encoder output not aligned with papers/questions')
                vectors.extend(values)
                batch['status'] = 'completed'
        except Exception as exc:
            run.update(status='failed', error_type=type(exc).__name__)
            if run['batches']:
                run['batches'][-1].update(status='failed', error_type=type(exc).__name__)
            raise
        run['status'] = 'completed'
        scorer = PaperFlowVectors({p.paper_id: vector for p, vector in zip(papers, vectors)},
            vectors[len(papers):], self.aliases, embedding_audit_ref=run_ref)
        rows = scorer(request, papers)
        self.decisions = scorer.decisions
        self.ranking_audit = getattr(scorer, 'ranking_audit', [])
        return rows
