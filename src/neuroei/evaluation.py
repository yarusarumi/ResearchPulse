"""Validate explicit human-review exports against their frozen candidate identities."""
from .ranking import relevance_metrics


def _review_object(value, message):
    if not isinstance(value, dict):
        raise ValueError(message)  # noqa: TRY004 - review contract requires ValueError for malformed fields
    return value


def validate_review(payload, manifest, profile_version):
    payload = _review_object(payload, 'Review payload must be an object')
    manifest = _review_object(manifest, 'Evaluation manifest must be an object')
    expected = _review_object(manifest.get('sample_mapping'), 'Manifest sample mapping must be an object')
    if not expected:
        raise ValueError('Review manifest has no samples')
    if type(manifest.get('pool_id')) is not str or not manifest['pool_id']:
        raise ValueError('Manifest pool ID must be a non-empty string')
    if type(profile_version) is not str or not profile_version:
        raise ValueError('Expected profile version must be a non-empty string')
    if payload.get('pool_id') != manifest['pool_id'] or payload.get('profile_version') != profile_version:
        raise ValueError('Review pool/profile version mismatch')
    reviewer = payload.get('reviewer')
    if payload.get('human_review_attested') is not True or type(reviewer) is not str or not reviewer.strip():
        raise ValueError('Explicit human reviewer/attestation required')
    rows = payload.get('reviews')
    if not isinstance(rows, list):
        raise ValueError('Review rows must be a list')  # noqa: TRY004
    ids = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get('sample_id')) is not str or not row['sample_id']:
            raise ValueError('Review rows need a non-empty string sample ID')
        ids.append(row['sample_id'])
    if len(set(ids)) != len(ids) or set(ids) != expected.keys():
        raise ValueError('Review must preserve every frozen sample ID exactly once')

    frozen = set()
    for identity in expected.values():
        if not isinstance(identity, dict):
            raise ValueError('Manifest sample identity must be an object')  # noqa: TRY004
        paper_id, version = identity.get('paper_id'), identity.get('version')
        if type(paper_id) is not str or not paper_id or type(version) is not str or not version:
            raise ValueError('Manifest sample identity needs paper/version strings')
        if paper_id in frozen:
            raise ValueError('Manifest assigns one paper ID to multiple samples')
        frozen.add(paper_id)

    labels = {}
    for row in rows:
        identity = expected[row['sample_id']]
        if (row.get('paper_id'), row.get('version')) != (identity['paper_id'], identity['version']):
            raise ValueError('Review sample paper/version mismatch')
        label = row.get('label_0_to_3')
        if label is not None and (type(label) is not int or not 0 <= label <= 3):
            raise ValueError('Human labels must be null or integer 0–3')
        labels[identity['paper_id']] = label
    return labels


def evaluate_benchmark(benchmark, review, manifest, benchmark_mapping, profile_version, *, k=5):
    """Recompute this fixed pool's metrics for a completed benchmark; never writes feedback."""
    labels = validate_review(review, manifest, profile_version)
    if type(k) is not int or k < 1:
        raise ValueError('Cutoff k must be a true integer of at least 1')
    benchmark = _review_object(benchmark, 'Benchmark result must be an object')
    if benchmark.get('status') != 'completed':
        raise ValueError('Benchmark is not completed')
    if benchmark.get('pool_id') != manifest['pool_id'] or benchmark.get('profile_version') != profile_version:
        raise ValueError('Benchmark pool/profile version mismatch')
    if not isinstance(benchmark_mapping, dict) or benchmark_mapping != manifest['sample_mapping']:
        raise ValueError('Benchmark candidate mapping differs from the reviewed manifest')
    methods = benchmark.get('methods')
    if not isinstance(methods, dict) or not methods:
        raise ValueError('Completed benchmark must contain methods')

    output_methods = {}
    for name, method in methods.items():
        if type(name) is not str or not name:
            raise ValueError('Benchmark method names must be non-empty strings')
        if not isinstance(method, dict):
            raise ValueError('Benchmark method entry must be an object')  # noqa: TRY004
        status = method.get('status')
        if type(status) is not str or not status:
            raise ValueError('Benchmark method status must be a non-empty string')
        if status != 'completed':
            reason = method.get('reason')
            if type(reason) is not str or not reason.strip():
                reason = f'Method is not completed (status={status}); evaluation was not computed'
            output_methods[name] = {'status': status, 'evaluation': None, 'reason': reason}
            continue
        ranking = method.get('ranking')
        if not isinstance(ranking, list) or not ranking:
            raise ValueError('Completed benchmark method needs non-empty ranking rows')
        ranked_ids = []
        for row in ranking:
            if not isinstance(row, dict) or 'paper_id' not in row:
                raise ValueError('Ranking rows must contain paper IDs')
            ranked_ids.append(row['paper_id'])
        output_methods[name] = {'status': status,
                                'evaluation': relevance_metrics(ranked_ids, labels, k=k)}

    labelled_count = sum(value is not None for value in labels.values())
    return {'pool_id': manifest['pool_id'], 'profile_version': profile_version,
            'labelled_count': labelled_count, 'candidate_count': len(labels),
            'status': 'labelled_pool_only' if labelled_count == len(labels) else 'awaiting_human_labels',
            'methods': output_methods, 'feedback_update_applied': False,
            'scientific_acceptance': False}
