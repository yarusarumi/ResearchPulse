"""Sourced topic hints for the pinned PaperFlow scorer; no facts or feedback are learned here."""
import copy
import importlib
import math
from pathlib import Path
from unittest.mock import patch

from pydantic import Field

from .config import StrictModel
from .discovery_phrases import PHRASE_POLICY, phrase_evidence
from .readers import paperflow_profile


class InterestTopic(StrictModel):
    topic_id: str = Field(pattern=r'^neuroei-[a-z0-9]+(?:-[a-z0-9]+)*$')
    label: str = Field(min_length=1)
    fact_ids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1, max_length=20)
    weight: float = Field(gt=0, le=1)


class InterestMapping(StrictModel):
    project_id: str
    profile_version: str
    mapping_version: str
    provenance: str
    topics: list[InterestTopic] = Field(min_length=1, max_length=20)


def validate_embedding_reuse(previous, manifest, profile, papers, query_fact_ids, pool_id):
    expected = {f'E{i:03}': {'paper_id': p.paper_id, 'version': p.version} for i, p in enumerate(papers, 1)}
    embedding = previous.get('embedding', {})
    if (previous.get('status') != 'completed' or previous.get('command') != 'paired-ranking-pilot'
            or previous.get('pool_id') != pool_id or previous.get('profile_version') != profile.version
            or previous.get('query_fact_ids') != query_fact_ids or previous.get('seed') != 20261023
            or manifest != expected or embedding.get('device') != 'cpu'
            or embedding.get('model_revision') != '5617a9f61b028005a4858fdac845db406aefb181'
            or embedding.get('shape') != [len(papers)+len(query_fact_ids), 1024]
            or embedding.get('reused_existing_real_vectors')):
        raise ValueError('Embedding cache identities/configuration differ from this paired experiment')


def validate_mapping(profile, mapping):
    mapping = InterestMapping.model_validate(mapping)
    if (mapping.project_id, mapping.profile_version) != (profile.project_id, profile.version):
        raise ValueError('Interest mapping project/profile version mismatch')
    facts = {f.id: f for f in profile.facts if f.kind in ('goal', 'question', 'hypothesis')}
    topic_ids = [t.topic_id for t in mapping.topics]
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError('Duplicate mapped topic IDs')
    for topic in mapping.topics:
        if not set(topic.fact_ids) <= facts.keys():
            raise ValueError('Topic refers to missing or non-interest project facts')
        if any(not alias.strip() or len(alias) > 100 for alias in topic.aliases):
            raise ValueError('Empty/overlong topic alias')
    return mapping


def mapped_profile(profile, mapping):
    mapping = validate_mapping(profile, mapping)
    mapped = paperflow_profile(profile, topic_weights={t.topic_id: t.weight for t in mapping.topics})
    # Sentence text is provenance, not an exact-match topic key.
    mapped['topic_weights'] = {t.topic_id: t.weight for t in mapping.topics}
    mapped['core_directions'] = dict(mapped['topic_weights'])
    mapped['interest_mapping_version'] = mapping.mapping_version
    return mapped


def topic_matches(title, abstract, mapping):
    matches = []
    for topic in mapping.topics:
        evidence = [hit for alias in topic.aliases
                    if (hit := phrase_evidence(title, abstract, alias)) is not None]
        if evidence:
            matches.append({'topic_id': topic.topic_id,
                            'matched_aliases': [e['phrase'] for e in evidence],
                            'fact_ids': topic.fact_ids, 'evidence': evidence})
    return matches


def score_paperflow(profile, papers, vectors, query_vectors, mapping):
    """Run pure upstream ranking with explicit resources, feature provenance and network denial."""
    import numpy as np

    mapping = validate_mapping(profile, mapping)
    if len(papers) != len(vectors) or not len(query_vectors):
        raise ValueError('Missing/misaligned PaperFlow embedding inputs')
    pv = np.asarray(vectors, dtype=float)
    qv = np.asarray(query_vectors, dtype=float)
    if (pv.ndim != 2 or qv.ndim != 2 or pv.shape[1] != qv.shape[1]
            or not np.isfinite(pv).all() or not np.isfinite(qv).all()
            or np.any(np.linalg.norm(pv, axis=1) == 0) or np.any(np.linalg.norm(qv, axis=1) == 0)):
        raise ValueError('Invalid real embedding vectors')
    centroid = qv.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if not math.isfinite(norm) or not norm:
        raise ValueError('Zero project interest vector')
    mapped = mapped_profile(profile, mapping)
    mapped['interest_vector'] = (centroid / norm).tolist()
    original = copy.deepcopy(mapped)

    def deny_network(*_args, **_kwargs):
        raise RuntimeError('Pure PaperFlow scoring attempted network access')

    # The upstream daily module's CONFIG_PATH points at deployments/config in its wheel.
    # Select the actual installed config package, without editing any third-party files.
    env = {'PYTHON_DOTENV_DISABLED': '1', 'PAPERFLOW_WIKI_INGEST': 'false',
           'PAPERFLOW_RELEVANCE_THRESHOLD': '', 'PAPERFLOW_DAILY_LIMIT': ''}
    with patch.dict('os.environ', env), patch('socket.socket.connect', deny_network), patch('socket.socket.connect_ex', deny_network):
        upstream = importlib.import_module('deployments.feishu.daily-push-agent.main')
        lexicon = importlib.import_module('config.direction_lexicon')
        config_dir = Path(lexicon.__file__).resolve().parent
        weights_file = config_dir / 'scoring_weights.yaml'
        if not weights_file.is_file():
            raise ValueError('Pinned PaperFlow scoring resource missing; no silent default fallback')
        previous_path = upstream.CONFIG_PATH
        with patch.object(upstream, 'CONFIG_PATH', str(config_dir)):
            weights = upstream.load_scoring_weights()
        prepared, scored = [], []
        for paper, vector in zip(papers, pv):
            matches = topic_matches(paper.title, paper.abstract, mapping)
            native_topics = upstream.canonicalize_direction_terms(
                upstream.extract_topics_from_text(paper.title, paper.abstract), keep_unknown=True)
            item = {'paper_id': paper.paper_id, 'title': paper.title, 'abstract': paper.abstract,
                    'authors': paper.authors, 'source': paper.sources[0], 'publish_date': paper.published,
                    'embedding': vector.tolist(), 'institution': '', 'categories': [],
                    'topics': sorted(set(native_topics + [m['topic_id'] for m in matches]))}
            item['quality_score'] = upstream.estimate_quality_score(item)
            score = upstream.calculate_paper_score(item, mapped, weights)
            scored.append({'paper_id': paper.paper_id, 'version': paper.version, 'score': score,
                'topic_matches': [m['topic_id'] for m in matches], 'topic_match_provenance': matches,
                'native_topics': native_topics, 'quality_prior': item['quality_score']})
            prepared.append(item)
        scored.sort(key=lambda r: (-r['score'], r['paper_id']))
        raw = upstream.sort_and_categorize(copy.deepcopy(prepared), copy.deepcopy(mapped), copy.deepcopy(weights))
        if original != mapped:
            raise ValueError('Fixed interest profile was mutated')
    return {'scored': scored,
        'upstream_pipeline': [{'paper_id': p.paper['paper_id'], 'score': p.score, 'category': p.category} for p in raw],
        'metadata': {'pinned_revision': '4a835e737490785b3ce51025c85b630cd224318c',
            'pipeline_file': upstream.__file__, 'scorer_file': upstream.profile_updater.__file__,
            'original_config_path': previous_path, 'config_path': str(config_dir), 'config_path_exists': True,
            'scoring_weights_file': str(weights_file), 'effective_weights': weights,
            'topic_match_count': sum(bool(r['topic_matches']) for r in scored),
            'phrase_policy': PHRASE_POLICY,
            'profile_mapping': mapping.model_dump(),
            'profile_policy': 'Fixed source-linked topic hints + real query-vector centroid; no learned feedback',
            'implementation': 'Lab topic adapter, not unchanged upstream cold-start',
            'network_guard': 'All socket connects denied during upstream import/scoring'}}
