"""Server-local CPU ranking: same job contract as the SSH ranking relay, without transport."""
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired

from .local_reader import _open_log, _write_input
from .paperflow_adapter import validate_mapping
from .remote_ranking import _job_payload, _same, _validated_result

ROOT = '/srv/researchpulse'
PLATFORM = sys.platform


def _linux_host():
    return PLATFORM.startswith('linux')


def _awaiting(job_id, error_type, detail, *, live=False, terminal_observed=False):
    return {'status': 'awaiting_result', 'job_id': job_id, 'llm_requests': 0,
            'billing_unknown': False, 'live_verified': live,
            'terminal_observed': terminal_observed, 'error_type': error_type, 'detail': detail}


def _failed(job_id, error_type, stage, detail, *, terminal_observed=False):
    return {'status': 'failed', 'job_id': job_id, 'llm_requests': 0,
            'billing_unknown': False, 'live_verified': False,
            'terminal_observed': terminal_observed, 'error_type': error_type,
            'stage': stage, 'detail': detail}


def _unavailable(job_id, error_type, stage, detail):
    return {'status': 'unavailable', 'job_id': job_id, 'llm_requests': 0,
            'billing_unknown': False, 'live_verified': False,
            'terminal_observed': False, 'error_type': error_type,
            'stage': stage, 'detail': detail}


def _read_terminal(job_path, result_path, payload, *, job_id, profile_version,
                   paper_versions, mapping_version, release, paper_ids):
    if not result_path.exists():
        return None
    # Read and validate the saved input before accepting anything from the terminal file.
    try:
        saved_raw = job_path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise ValueError('Terminal ranking result has no saved input identity') from exc
    try:
        saved = json.loads(saved_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError('Terminal ranking saved input is unreadable') from exc
    if not _same(saved, payload):
        raise ValueError('Terminal ranking job input differs from the submitted job')
    try:
        raw = result_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError('Terminal ranking result is unreadable') from exc
    return _validated_result(body, job_id=job_id, profile_version=profile_version,
                             paper_versions=paper_versions, mapping_version=mapping_version,
                             release=release, paper_ids=paper_ids)


def _proc_cmdline(process_id):
    return Path('/proc', str(process_id), 'cmdline').read_bytes()


def _worker_alive(process_id, job_path):
    try:
        command = _proc_cmdline(process_id)
    except (FileNotFoundError, ProcessLookupError):
        return False
    arguments = command.rstrip(b'\0').split(b'\0')
    target = str(job_path).encode('utf-8')
    return (any(argument.rsplit(b'/', 1)[-1] == b'workflow_ranker.py' for argument in arguments)
            and any(left == b'--job' and right == target
                    for left, right in itertools.pairwise(arguments)))


def rank_local(config, profile, papers, mapping, *, release, job_id,
               resume_only=False, observation_timeout=600):
    """Run or re-observe one bounded CPU ranking job on the Linux server itself."""
    if not isinstance(release, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', release):
        raise ValueError('Explicit deployed ranking release required')
    if not isinstance(job_id, str) or not re.fullmatch(r'rankjob-[a-f0-9]{32}', job_id):
        raise ValueError('Explicit safe ranking job required')
    if type(resume_only) is not bool:
        raise ValueError('Explicit ranking resume mode required')
    if (isinstance(observation_timeout, bool)
            or not isinstance(observation_timeout, (int, float))
            or not math.isfinite(observation_timeout)
            or not 0 <= observation_timeout <= 600):
        raise ValueError('Observation timeout must be a finite number of seconds within 0..600')
    mapping = validate_mapping(profile, mapping)
    payload = _job_payload(profile, papers, mapping, release=release, job_id=job_id)
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    paper_versions = [[p.paper_id, p.version] for p in papers]
    paper_ids = {p.paper_id for p in papers}
    if not _linux_host():
        return _unavailable(job_id, 'UnsupportedLocalRankingHost', 'host',
                            'The server-local backend requires the Linux server itself')
    try:
        data_dir = Path(config.data_dir).resolve()
    except (ValueError, OSError) as exc:
        return _unavailable(job_id, 'LocalRankingRootMismatch', 'host', str(exc))
    expected_dir = (Path(ROOT) / 'data').resolve()
    if data_dir != expected_dir:
        return _unavailable(job_id, 'LocalRankingRootMismatch', 'host',
                            'data_dir must be ' + str(expected_dir))
    directory = Path(ROOT) / 'data' / 'workflow-rank-jobs'
    job_path = directory / (job_id + '.json')
    result_path = directory / (job_id + '.result.json')
    if resume_only and not job_path.exists():
        return _awaiting(job_id, 'MissingOriginalRankingInput',
                         'Original ranking input is missing; inspection required '
                         'before any new dispatch')
    try:
        terminal = _read_terminal(
            job_path, result_path, payload, job_id=job_id, profile_version=profile.version,
            paper_versions=paper_versions, mapping_version=mapping.mapping_version,
            release=release, paper_ids=paper_ids)
    except (ValueError, OSError) as exc:
        return _failed(job_id, type(exc).__name__, 'local_terminal',
                       'Terminal ranking result rejected: ' + str(exc), terminal_observed=True)
    if terminal is not None:
        return terminal
    if not job_path.exists():
        if resume_only:
            return _awaiting(job_id, 'MissingOriginalRankingInput',
                             'Original ranking input is missing; inspection required '
                             'before any new dispatch')
        return _launch(
            job_id, payload, data, job_path, result_path,
            profile_version=profile.version, paper_versions=paper_versions,
            mapping_version=mapping.mapping_version, release=release, paper_ids=paper_ids,
            observation_timeout=observation_timeout)
    return _resume(job_path, result_path, payload, job_id=job_id,
                   profile_version=profile.version, paper_versions=paper_versions,
                   mapping_version=mapping.mapping_version, release=release,
                   paper_ids=paper_ids)


def _resume(job_path, result_path, payload, *, job_id, profile_version, paper_versions,
            mapping_version, release, paper_ids):
    try:
        raw = job_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return _awaiting(job_id, 'ObservationPending', 'Existing ranking input disappeared')
    except OSError as exc:
        return _awaiting(job_id, type(exc).__name__, 'Existing ranking input unreadable')
    try:
        saved = json.loads(raw)
    except (ValueError, TypeError):
        return _awaiting(job_id, 'InvalidRankingInput',
                         'Existing ranking input is not valid JSON')
    if not _same(saved, payload):
        return _awaiting(job_id, 'ExistingRankingInputDiffers',
                         'Existing ranking job has different inputs; input is never overwritten')
    try:
        terminal = _read_terminal(
            job_path, result_path, payload, job_id=job_id, profile_version=profile_version,
            paper_versions=paper_versions, mapping_version=mapping_version,
            release=release, paper_ids=paper_ids)
    except (ValueError, OSError) as exc:
        return _failed(job_id, type(exc).__name__, 'local_terminal',
                       'Terminal ranking result rejected: ' + str(exc), terminal_observed=True)
    if terminal is not None:
        return terminal
    pid_path = job_path.with_suffix('.pid')
    try:
        pid_text = pid_path.read_text(encoding='ascii').strip()
    except FileNotFoundError:
        # The worker may commit its result between the first read and this handle check.
        try:
            terminal = _read_terminal(
                job_path, result_path, payload, job_id=job_id, profile_version=profile_version,
                paper_versions=paper_versions, mapping_version=mapping_version,
                release=release, paper_ids=paper_ids)
        except (ValueError, OSError) as exc:
            return _failed(job_id, type(exc).__name__, 'local_terminal',
                           'Terminal ranking result rejected: ' + str(exc),
                           terminal_observed=True)
        if terminal is not None:
            return terminal
        return _awaiting(job_id, 'ObservationPending',
                         'Input exists; no terminal result/process handle verified')
    except (OSError, UnicodeError):
        return _awaiting(job_id, 'InvalidRankingPidHandle', 'Ranking PID handle unreadable')
    if not pid_text.isdigit():
        return _awaiting(job_id, 'InvalidRankingPidHandle',
                         'Ranking PID handle is not a process id')
    try:
        live = _worker_alive(int(pid_text), job_path)
    except OSError as exc:
        return _awaiting(job_id, type(exc).__name__, 'Ranking process is not observable')
    # The worker may commit its result between the first read and the process check.
    try:
        terminal = _read_terminal(
            job_path, result_path, payload, job_id=job_id, profile_version=profile_version,
            paper_versions=paper_versions, mapping_version=mapping_version,
            release=release, paper_ids=paper_ids)
    except (ValueError, OSError) as exc:
        return _failed(job_id, type(exc).__name__, 'local_terminal',
                       'Terminal ranking result rejected: ' + str(exc), terminal_observed=True)
    if terminal is not None:
        return terminal
    if live:
        return _awaiting(job_id, 'ObservationPending',
                         'The original ranking worker process is still running', live=True)
    return _failed(job_id, 'OriginalWorkerExitedWithoutResult', 'local_worker',
                   'Input exists; the process is absent or its PID was reused')


def _launch(job_id, payload, data, job_path, result_path, *, profile_version, paper_versions,
            mapping_version, release, paper_ids, observation_timeout):
    directory = job_path.parent
    code = Path(ROOT) / 'releases' / release
    python = Path(ROOT) / '.venv' / 'bin' / 'python'
    try:
        if not code.is_dir() or not (code / 'scripts' / 'workflow_ranker.py').is_file():
            raise ValueError('Deployed ranking release is incomplete; no job was created')
        if not python.is_file():
            raise ValueError('Deployed ranking interpreter is missing; no job was created')
    except (OSError, ValueError) as exc:
        return _failed(job_id, type(exc).__name__, 'local_job_validation', str(exc))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _failed(job_id, type(exc).__name__, 'local_job_validation',
                       'Job directory unavailable: ' + str(exc))
    stdout = stderr = None
    try:
        try:
            stdout = _open_log(directory / (job_id + '.stdout.log'))
        except OSError as exc:
            return _failed(job_id, type(exc).__name__, 'local_job_validation',
                           'stdout log unavailable: ' + str(exc))
        try:
            stderr = _open_log(directory / (job_id + '.stderr.log'))
        except OSError as exc:
            return _failed(job_id, type(exc).__name__, 'local_job_validation',
                           'stderr log unavailable: ' + str(exc))
        try:
            _write_input(job_path, data)
        except FileExistsError:
            # A concurrent creator may have won; never truncate or replace its input.
            return _resume(job_path, result_path, payload, job_id=job_id,
                           profile_version=profile_version, paper_versions=paper_versions,
                           mapping_version=mapping_version, release=release, paper_ids=paper_ids)
        except OSError as exc:
            return _failed(job_id, type(exc).__name__, 'local_job_validation',
                           'Ranking input write failed; never retried automatically: ' + str(exc))
        try:
            if job_path.read_bytes() != data:
                raise ValueError('Ranking job readback failed')
        except (OSError, ValueError) as exc:
            return _failed(job_id, type(exc).__name__, 'local_job_validation',
                           'Ranking input readback failed; never retried automatically: ' + str(exc))
        command = [str(python), '-B', '-u', 'scripts/workflow_ranker.py', '--job', str(job_path)]
        environment = dict(os.environ)
        environment.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                            'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1',
                            'TRANSFORMERS_OFFLINE': '1', 'PAPERFLOW_WIKI_INGEST': 'false',
                            'PYTHON_DOTENV_DISABLED': '1',
                            'PAPERFLOW_HOME': str(directory / (job_id + '-upstream-state')),
                            'NEUROEI_DATA_DIR': str(Path(ROOT) / 'data'),
                            'PYTHONPATH': str(code / 'src'),
                            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'})
        for secret in ('DEEPSEEK_API_KEY', 'FEISHU_APP_ID', 'FEISHU_APP_SECRET',
                       'NEUROEI_SSH_PASSWORD'):
            environment.pop(secret, None)
        try:
            process = Popen(command, cwd=str(code), env=environment, stdin=DEVNULL,
                            stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as exc:
            return _failed(job_id, type(exc).__name__, 'local_job_launch',
                           'Worker launch failed; the input is retained: ' + str(exc))
    finally:
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
    try:
        process.wait(timeout=observation_timeout)
    except TimeoutExpired:
        if process.poll() is None:
            return _awaiting(job_id, 'ObservationPending',
                             'Observation timeout; the original process was left running',
                             live=True)
    except Exception as exc:  # noqa: BLE001 - never lose a worker that may still be running
        return _awaiting(job_id, type(exc).__name__,
                         'Worker wait failed; the original process was left running')
    try:
        terminal = _read_terminal(
            job_path, result_path, payload, job_id=job_id, profile_version=profile_version,
            paper_versions=paper_versions, mapping_version=mapping_version,
            release=release, paper_ids=paper_ids)
    except (ValueError, OSError) as exc:
        return _failed(job_id, type(exc).__name__, 'local_worker',
                       'Terminal ranking result rejected: ' + str(exc), terminal_observed=True)
    if terminal is not None:
        return terminal
    return _failed(job_id, 'OriginalWorkerExitedWithoutResult', 'local_worker',
                   'Worker exited without a terminal result', terminal_observed=True)
