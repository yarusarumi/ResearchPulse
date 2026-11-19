"""Model-free CPU ranking features on the trusted server; only compact JSON returns locally."""
import base64
import json
import os
import re
import shlex

import paramiko

from .paperflow_adapter import validate_mapping
from .remote_reader import ROOT

_TERMINAL = ('completed', 'failed')


def _same(left, right):
    # Compare JSON types too: a bool must never satisfy an integer field, since True == 1.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _job_payload(profile, papers, mapping, *, release, job_id):
    needed = {bid for f in profile.facts for bid in f.source_blocks}
    compact = profile.model_copy(update={'blocks': [b for b in profile.blocks if b.block_id in needed]})
    return {'job_id': job_id, 'code_revision': release, 'profile': compact.model_dump(mode='json'),
            'papers': [p.model_dump(mode='json') for p in papers], 'mapping': mapping.model_dump()}


def _validated_result(body, *, job_id, profile_version, paper_versions, mapping_version, release, paper_ids):
    if not isinstance(body, dict):
        raise ValueError('Ranking result must be an object')
    expected = {'job_id': job_id, 'profile_version': profile_version,
                'mapping_version': mapping_version, 'paper_versions': paper_versions}
    if 'code_revision' in body:
        actual = {key: body.get(key) for key in expected} | {'code_revision': body['code_revision']}
        if not _same(actual, expected | {'code_revision': release}):
            raise ValueError('Ranking result identities differ from the submitted job')
        result = dict(body)
    else:
        actual = {key: body.get(key) for key in expected}
        if not _same(actual, expected):
            raise ValueError('Ranking result has partial or mismatched identity fields')
        # Old terminal files predate code_revision; the saved input was verified exactly before this call.
        result = {**body, 'code_revision': release, 'result_schema': 'legacy-missing-code-revision'}
    if result.get('status') not in _TERMINAL:
        raise ValueError('Ranking result status must be completed or failed')
    if type(result.get('llm_requests')) is not int or result['llm_requests'] != 0:
        raise ValueError('Ranking result must report zero real LLM requests')
    if 'candidate_ids' in result:
        ids = result['candidate_ids']
        if not isinstance(ids, list) or any(type(pid) is not str for pid in ids):
            raise ValueError('Ranking result candidate_ids must be a list of strings')
        if len(ids) != len(set(ids)):
            raise ValueError('Ranking result contains duplicate candidates')
        if not set(ids) <= paper_ids:
            raise ValueError('Ranking result contains candidates outside the submitted papers')
    elif result['status'] == 'completed':
        raise ValueError('Completed ranking result must include candidate_ids')
    return result


def rank_remote(profile, papers, mapping, *, release, job_id, resume_only=False):
    mapping = validate_mapping(profile, mapping)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', release) or not re.fullmatch(r'rankjob-[a-f0-9]{32}', job_id):
        raise ValueError('Explicit safe ranking job/release required')
    if type(resume_only) is not bool:
        raise ValueError('Explicit ranking resume mode required')
    payload = _job_payload(profile, papers, mapping, release=release, job_id=job_id)
    paper_versions = [[p.paper_id, p.version] for p in papers]
    paper_ids = {p.paper_id for p in papers}
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    directory = ROOT + '/data/workflow-rank-jobs'
    job_path = directory + '/' + job_id + '.json'
    result_path = directory + '/' + job_id + '.result.json'
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    dispatched = False
    terminal_observed = False

    def saved_input(sftp):
        with sftp.open(job_path, 'rb') as stream:
            return json.loads(stream.read())

    def require_saved_input(sftp):
        if not _same(saved_input(sftp), payload):
            raise ValueError('Existing ranking job has different inputs')

    def result(sftp):
        with sftp.open(result_path, 'rb') as stream:
            raw = stream.read()
        # A terminal file is accepted only when its exact saved input still matches this request.
        try:
            saved = saved_input(sftp)
        except FileNotFoundError as exc:
            raise ValueError('Terminal ranking result has no saved input identity') from exc
        if not _same(saved, payload):
            raise ValueError('Terminal ranking job input differs from the submitted job')
        return _validated_result(json.loads(raw), job_id=job_id, profile_version=profile.version,
                                 paper_versions=paper_versions, mapping_version=mapping.mapping_version,
                                 release=release, paper_ids=paper_ids)

    try:
        ssh.connect('researchpulse-host.example.invalid', username='researchpulse', password=os.environ['NEUROEI_SSH_PASSWORD'],
                    timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
        with ssh.open_sftp() as sftp:
            if resume_only:
                try:
                    with sftp.open(job_path, 'rb'):
                        pass
                except FileNotFoundError:
                    return {'status': 'awaiting_result', 'job_id': job_id, 'llm_requests': 0,
                            'billing_unknown': False, 'live_verified': False,
                            'terminal_observed': False,
                            'error_type': 'MissingOriginalRankingInput',
                            'detail': 'Original input is missing; inspection required before any new dispatch'}
            try:
                return result(sftp)
            except FileNotFoundError:
                pass
            created = False
            try:
                require_saved_input(sftp)
            except FileNotFoundError:
                if resume_only:
                    return {'status': 'awaiting_result', 'job_id': job_id, 'llm_requests': 0,
                            'billing_unknown': False, 'live_verified': False,
                            'terminal_observed': False,
                            'error_type': 'MissingOriginalRankingInput',
                            'detail': 'Original input is missing; inspection required before any new dispatch'}
                try:
                    sftp.stat(directory)
                except FileNotFoundError:
                    sftp.mkdir(directory)
                try:
                    # Paramiko only adds WRITE for 'w', 'a' or '+'; plain 'x' is CREATE|EXCL and read-less,
                    # write-less. 'x+' is READ|WRITE|CREATE|EXCL, so it stays exclusive and actually writable.
                    with sftp.open(job_path, 'x+') as stream:
                        stream.write(data)
                    created = True
                except (OSError, paramiko.SFTPError):
                    # A concurrent creator may have won the exclusive create; never truncate its input.
                    pass
                sftp.chmod(job_path, 0o600)
                require_saved_input(sftp)
            if not created:
                try:
                    with sftp.open(directory+'/'+job_id+'.pid', 'rb') as stream:
                        pid = stream.read().decode('ascii').strip()
                except FileNotFoundError:
                    # An input file alone is not evidence of a live process.
                    return {'status': 'awaiting_result', 'job_id': job_id, 'llm_requests': 0,
                            'live_verified': False, 'detail': 'No terminal result/process handle could be verified'}
                if not pid.isdigit():
                    raise ValueError('Invalid ranking PID handle')
                command = None
                try:
                    with sftp.open('/proc/'+pid+'/cmdline', 'rb') as stream:
                        command = stream.read()
                except FileNotFoundError:
                    pass
                if command is None:
                    live = False
                else:
                    arguments = command.rstrip(b'\0').split(b'\0')
                    live = (any(arg.rsplit(b'/', 1)[-1] == b'workflow_ranker.py' for arg in arguments)
                            and any(a == b'--job' and b == job_path.encode('utf-8')
                                    for a, b in zip(arguments, arguments[1:], strict=False)))
                # The worker may have committed its result between the first read and the process check.
                try:
                    return result(sftp)
                except FileNotFoundError:
                    pass
                if command is None:
                    return {'status': 'failed', 'job_id': job_id, 'llm_requests': 0, 'live_verified': False,
                            'detail': 'Original ranking worker exited without a terminal result'}
                return {'status': 'awaiting_result' if live else 'failed', 'job_id': job_id,
                        'llm_requests': 0, 'live_verified': live, 'detail': 'Observed the original worker process'
                        if live else 'Original PID no longer identifies this job'}
        code = ROOT + '/releases/' + release
        script = ('set -euo pipefail\nexport LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES="" '
            'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
            'PYTHON_DOTENV_DISABLED=1 PAPERFLOW_WIKI_INGEST=false\n'
            'export NEUROEI_DATA_DIR=' + ROOT + '/data\nexport PYTHONPATH=' + shlex.quote(code+'/src') + '\n'
            'export PAPERFLOW_HOME=' + shlex.quote(directory+'/'+job_id+'-upstream-state') + '\n'
            'cd '+shlex.quote(code)+'\n'+ROOT+'/.venv/bin/python -u scripts/workflow_ranker.py --job '
            +shlex.quote(job_path)+'\n')
        encoded = base64.b64encode(script.encode('utf-8')).decode('ascii')
        stdin, stdout, stderr = ssh.exec_command('bash -s', timeout=600)
        dispatched = True
        stdin.write("printf %s '"+encoded+"' | base64 -d | bash\n")
        stdin.channel.shutdown_write()
        stdout.read()
        stderr.read()
        rc = stdout.channel.recv_exit_status()
        terminal_observed = True
        with ssh.open_sftp() as sftp:
            body = result(sftp)
        if rc and body['status'] == 'completed':
            raise ValueError('Ranking terminal status and process exit disagree')
        return body
    except Exception as exc:
        return {'status': 'awaiting_result' if dispatched and not terminal_observed else 'failed', 'job_id': job_id,
                'llm_requests': 0, 'error_type': type(exc).__name__}
    finally:
        ssh.close()
