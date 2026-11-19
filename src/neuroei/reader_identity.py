"""Pure identity and billing gate for reader results, shared by local and SSH execution backends."""
import re

_JOB_ID = re.compile(r'job-[a-f0-9]{32}')
_IDENTITY_KEYS = ('project_id', 'profile_version', 'paper_id', 'paper_version', 'code_revision')
_TERMINAL = ('completed', 'failed')
_JOB_KEYS = frozenset(('job_id', 'profile', 'paper', 'budget', 'code_revision'))


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError('Reader job ' + label + ' must be a non-empty string')


def _same(left, right):
    # Compare JSON types too: a bool must never satisfy an integer field, since True == 1.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _check_job(job, label):
    if not isinstance(job, dict):
        raise ValueError(label + ' must be an object')
    if set(job) != _JOB_KEYS:
        raise ValueError(label + ' must contain exactly job_id, profile, paper, budget and code_revision')
    if type(job['job_id']) is not str or not _JOB_ID.fullmatch(job['job_id']):
        raise ValueError(label + ' job_id must match job-[a-f0-9]{32}')
    budget = job['budget']
    if type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError(label + ' budget must be an integer between 1 and 6')
    if not isinstance(job['profile'], dict) or not isinstance(job['paper'], dict):
        raise ValueError(label + ' profile and paper must be objects')
    _text(job['profile'].get('project_id'), label + ' profile project_id')
    _text(job['profile'].get('version'), label + ' profile version')
    _text(job['paper'].get('paper_id'), label + ' paper_id')
    _text(job['paper'].get('version'), label + ' paper version')
    _text(job['code_revision'], label + ' code_revision')


def _identity(job):
    return {'project_id': job['profile']['project_id'], 'profile_version': job['profile']['version'],
            'paper_id': job['paper']['paper_id'], 'paper_version': job['paper']['version'],
            'code_revision': job['code_revision']}


def validate_reader_result(body: dict, saved_job: dict, expected_job: dict) -> dict:
    """Return a new, identity-checked result body; raise ValueError on any mismatch."""
    _check_job(expected_job, 'Expected reader job')
    _check_job(saved_job, 'Saved reader job')
    if not _same(saved_job, expected_job):
        raise ValueError('Saved reader job is not identical to the expected job')
    if not isinstance(body, dict):
        raise ValueError('Reader result must be an object')
    job_id = expected_job['job_id']
    if body.get('job_id') != job_id:
        raise ValueError('Reader result job_id differs from the submitted job')
    identity = _identity(expected_job)
    present = [key for key in _IDENTITY_KEYS if key in body]
    if not present:
        # Old terminal files predate the envelope; their exact saved input was verified above.
        result = {**body, **identity, 'result_schema': 'legacy-exact-input-validated'}
    elif len(present) == len(_IDENTITY_KEYS):
        result = dict(body)
    else:
        raise ValueError('Reader result has partial identity fields')
    actual = tuple(result.get(key) for key in ('job_id', *_IDENTITY_KEYS))
    if actual != (job_id, *(identity[key] for key in _IDENTITY_KEYS)):
        raise ValueError('Reader result identities differ from the submitted job')
    if result.get('status') not in _TERMINAL:
        raise ValueError('Reader result status must be completed or failed')
    count = result.get('llm_requests')
    if type(count) is not int or not 0 <= count <= expected_job['budget']:
        raise ValueError('Reader result request count must be a real integer within budget')
    if result['status'] == 'completed':
        analysis = result.get('analysis')
        if not isinstance(analysis, dict):
            raise ValueError('Completed reader result must contain an analysis object')
        if (analysis.get('run_id'), analysis.get('profile_version'), analysis.get('paper_id'),
                analysis.get('paper_version')) != (job_id, identity['profile_version'],
                                                   identity['paper_id'], identity['paper_version']):
            raise ValueError('Reader analysis identities differ from the job')
    return result
