"""Server-local CPU reader: same job contract as the SSH relay, without transport."""
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired
from urllib.request import Request, urlopen

from .reader_identity import validate_reader_result

ROOT = '/srv/researchpulse'
PLATFORM = sys.platform
PDF_LIMIT = 70_000_000
_CHUNK = 256 * 1024


def _linux_host():
    return PLATFORM.startswith('linux')


def _awaiting(job_id, budget, error_type, detail, *, live=False, terminal_observed=False):
    return {'status': 'awaiting_result', 'job_id': job_id, 'llm_requests': budget,
            'billing_unknown': True, 'live_verified': live,
            'terminal_observed': terminal_observed, 'error_type': error_type, 'detail': detail}


def _failed(job_id, budget, error_type, stage, detail, *, billing_unknown, terminal_observed=False):
    return {'status': 'failed', 'job_id': job_id,
            'llm_requests': budget if billing_unknown else 0,
            'billing_unknown': billing_unknown, 'live_verified': False,
            'terminal_observed': terminal_observed, 'error_type': error_type,
            'stage': stage, 'detail': detail}


def _unavailable(job_id, error_type, stage, detail):
    return {'status': 'unavailable', 'job_id': job_id, 'llm_requests': 0,
            'billing_unknown': False, 'live_verified': False,
            'terminal_observed': False, 'error_type': error_type,
            'stage': stage, 'detail': detail}


def _read_terminal(job_path, result_path, job):
    try:
        body = json.loads(result_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    try:
        saved_job = json.loads(job_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError('Terminal reader result is missing its input identity') from exc
    return validate_reader_result(body, saved_job, job)


def _same_input(raw, job):
    data = json.dumps(job, ensure_ascii=False).encode('utf-8')
    if raw == data:
        return True
    try:
        return (json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
                == json.dumps(job, sort_keys=True, ensure_ascii=False))
    except (ValueError, TypeError):
        return False


def _proc_cmdline(process_id):
    return Path('/proc', str(process_id), 'cmdline').read_bytes()


def _worker_alive(process_id, job_path):
    try:
        command = _proc_cmdline(process_id)
    except (FileNotFoundError, ProcessLookupError):
        return False
    arguments = command.rstrip(b'\0').split(b'\0')
    target = str(job_path).encode('utf-8')
    return (any(argument.rsplit(b'/', 1)[-1] == b'workflow_reader.py' for argument in arguments)
            and any(left == b'--job' and right == target
                    for left, right in zip(arguments, arguments[1:], strict=False)))


def _ensure_pdf(pdf_path, pid, job_id):
    if pdf_path.exists():
        with pdf_path.open('rb') as stream:
            if stream.read(5) != b'%PDF-':
                raise ValueError('Cached PDF header invalid; an existing PDF is never overwritten')
        return
    part = pdf_path.with_name(pdf_path.name + '.' + job_id + '.part')
    if part.exists():
        raise ValueError('An existing partial PDF is retained and never overwritten')
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request('https://arxiv.org/pdf/' + pid, headers={'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=45) as response:
        expected = int(response.headers.get('content-length', '0'))
        count = 0
        with part.open('xb') as stream:
            while chunk := response.read(_CHUNK):
                if not count and b'%PDF-' not in chunk[:1024]:
                    raise ValueError('Official response is not PDF')
                count += len(chunk)
                if count > PDF_LIMIT:
                    raise ValueError('PDF size budget exceeded')
                stream.write(chunk)
    if not count or part.stat().st_size != count or (expected and count != expected):
        raise ValueError('Incomplete PDF retained as partial')
    if pdf_path.exists():
        raise ValueError('A PDF appeared during download; the partial file is retained')
    os.replace(part, pdf_path)


def _write_input(job_path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(job_path, flags, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
    os.chmod(job_path, 0o600)


def read_local(config, profile, paper, *, release, budget=4, job_id=None,
               resume_only=False, observation_timeout=600):
    """Run or re-observe one bounded CPU reader job on the Linux server itself."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', release):
        raise ValueError('Explicit deployed reader release required')
    if (not re.fullmatch(r'arxiv:\d{4}\.\d{4,5}', paper.paper_id)
            or not re.fullmatch(r'v\d+', paper.version)):
        return {'status': 'unavailable', 'error_type': 'UnsupportedOfficialPdfAcquisition',
                'llm_requests': 0}
    job_id = job_id or 'job-' + uuid.uuid4().hex
    if not re.fullmatch(r'job-[a-f0-9]{32}', job_id) or type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError('Explicit safe reader job and bounded budget required')
    if type(resume_only) is not bool:
        raise ValueError('Explicit reader resume mode required')
    if (isinstance(observation_timeout, bool)
            or not isinstance(observation_timeout, (int, float))
            or not math.isfinite(observation_timeout)
            or not 0 <= observation_timeout <= 600):
        raise ValueError('Observation timeout must be a finite number of seconds within 0..600')
    if not _linux_host():
        return _unavailable(job_id, 'UnsupportedLocalReaderHost', 'host',
                            'The server-local backend requires the Linux server itself')
    try:
        data_dir = config.data_dir
    except (ValueError, OSError) as exc:
        return _unavailable(job_id, 'LocalReaderRootMismatch', 'host', str(exc))
    expected_dir = (Path(ROOT) / 'data').resolve()
    if data_dir != expected_dir:
        return _unavailable(job_id, 'LocalReaderRootMismatch', 'host',
                            'data_dir must be ' + str(expected_dir))
    needed = {bid for fact in profile.facts for bid in fact.source_blocks}
    compact = profile.model_copy(
        update={'blocks': [block for block in profile.blocks if block.block_id in needed]})
    job = {'job_id': job_id, 'profile': compact.model_dump(mode='json'),
           'paper': paper.model_dump(mode='json'), 'budget': budget, 'code_revision': release}
    directory = Path(ROOT) / 'data' / 'workflow-jobs'
    job_path = directory / (job_id + '.json')
    result_path = directory / (job_id + '.result.json')
    pid = paper.paper_id.split(':')[1] + paper.version
    pdf_path = Path(ROOT) / 'data' / 'public-pdf' / (pid + '.pdf')
    try:
        terminal = _read_terminal(job_path, result_path, job)
    except (ValueError, OSError) as exc:
        return _awaiting(job_id, budget, type(exc).__name__,
                         'Terminal reader result rejected: ' + str(exc))
    if terminal is not None:
        return terminal
    if not job_path.exists():
        if resume_only:
            return _awaiting(job_id, budget, 'MissingOriginalReaderInput',
                             'Original input is missing; inspection required before any new dispatch')
        return _launch(job_id, budget, job, job_path, result_path, pdf_path, pid, release,
                       observation_timeout)
    return _resume(job_id, budget, job, job_path, result_path)


def _resume(job_id, budget, job, job_path, result_path):
    try:
        raw = job_path.read_bytes()
    except FileNotFoundError:
        return _awaiting(job_id, budget, 'ObservationPending', 'Existing reader input disappeared')
    except OSError as exc:
        return _awaiting(job_id, budget, type(exc).__name__, 'Existing reader input unreadable')
    if not _same_input(raw, job):
        return _awaiting(job_id, budget, 'ExistingReaderInputDiffers',
                         'Existing reader job has different inputs; input is never overwritten')
    try:
        terminal = _read_terminal(job_path, result_path, job)
    except (ValueError, OSError) as exc:
        return _awaiting(job_id, budget, type(exc).__name__,
                         'Terminal reader result rejected: ' + str(exc))
    if terminal is not None:
        return terminal
    pid_path = job_path.with_suffix('.pid')
    try:
        pid_text = pid_path.read_text(encoding='ascii').strip()
    except FileNotFoundError:
        return _awaiting(job_id, budget, 'ObservationPending',
                         'Input exists; no terminal result/process handle verified')
    except (OSError, UnicodeError):
        return _awaiting(job_id, budget, 'InvalidReaderPidHandle', 'Reader PID handle unreadable')
    if not pid_text.isdigit():
        return _awaiting(job_id, budget, 'InvalidReaderPidHandle',
                         'Reader PID handle is not a process id')
    try:
        live = _worker_alive(int(pid_text), job_path)
    except OSError as exc:
        return _awaiting(job_id, budget, type(exc).__name__, 'Reader process is not observable')
    # The worker may commit its result between the first read and the process check.
    try:
        terminal = _read_terminal(job_path, result_path, job)
    except (ValueError, OSError) as exc:
        return _awaiting(job_id, budget, type(exc).__name__,
                         'Terminal reader result rejected: ' + str(exc))
    if terminal is not None:
        return terminal
    if live:
        return _awaiting(job_id, budget, 'ObservationPending',
                         'The original worker process is still running', live=True)
    return _failed(job_id, budget, 'OriginalWorkerExitedWithoutResult', 'local_worker',
                   'Input exists; the process is absent or its PID was reused',
                   billing_unknown=True)


def _open_log(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(path, flags, 0o600)
    os.chmod(path, 0o600)
    return os.fdopen(descriptor, 'ab')


def _launch(job_id, budget, job, job_path, result_path, pdf_path, pid, release, observation_timeout):
    directory = job_path.parent
    code = Path(ROOT) / 'releases' / release
    python = Path(ROOT) / '.venv' / 'bin' / 'python'
    try:
        if not code.is_dir() or not (code / 'scripts' / 'workflow_reader.py').is_file():
            raise ValueError('Deployed reader release is incomplete; no job was created')
        if not python.is_file():
            raise ValueError('Deployed reader interpreter is missing; no job was created')
    except Exception as exc:
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation', str(exc),
                       billing_unknown=False)
    try:
        _ensure_pdf(pdf_path, pid, job_id)
    except Exception as exc:
        return _failed(job_id, budget, type(exc).__name__, 'official_pdf_acquisition', str(exc),
                       billing_unknown=False)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation',
                       'Job directory unavailable: ' + str(exc), billing_unknown=False)
    try:
        stdout = _open_log(directory / (job_id + '.stdout.log'))
    except OSError as exc:
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation',
                       'stdout log unavailable: ' + str(exc), billing_unknown=False)
    try:
        stderr = _open_log(directory / (job_id + '.stderr.log'))
    except OSError as exc:
        stdout.close()
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation',
                       'stderr log unavailable: ' + str(exc), billing_unknown=False)
    data = json.dumps(job, ensure_ascii=False).encode('utf-8')
    try:
        _write_input(job_path, data)
    except FileExistsError:
        stdout.close()
        stderr.close()
        return _resume(job_id, budget, job, job_path, result_path)
    except OSError as exc:
        stdout.close()
        stderr.close()
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation',
                       'Reader input write failed; never retried automatically: ' + str(exc),
                       billing_unknown=False)
    try:
        if job_path.read_bytes() != data:
            raise ValueError('Reader job readback failed')
    except (OSError, ValueError) as exc:
        stdout.close()
        stderr.close()
        return _failed(job_id, budget, type(exc).__name__, 'local_job_validation',
                       'Reader input readback failed; never retried automatically: ' + str(exc),
                       billing_unknown=False)
    command = [str(python), '-B', '-u', 'scripts/workflow_reader.py', '--job', str(job_path)]
    environment = dict(os.environ)
    environment.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                        'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
                        'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
                        'NEUROEI_DATA_DIR': str(Path(ROOT) / 'data'),
                        'PYTHONPATH': str(code / 'src')})
    try:
        process = Popen(command, cwd=str(code), env=environment, stdin=DEVNULL,
                        stdout=stdout, stderr=stderr, start_new_session=True)
    except OSError as exc:
        return _failed(job_id, budget, type(exc).__name__, 'local_job_launch',
                       'Worker launch failed; the input is retained: ' + str(exc),
                       billing_unknown=True)
    finally:
        stdout.close()
        stderr.close()
    try:
        process.wait(timeout=observation_timeout)
    except TimeoutExpired:
        if process.poll() is None:
            return _awaiting(job_id, budget, 'ObservationPending',
                             'Observation timeout; the original process was left running',
                             live=True)
    except Exception as exc:
        return _awaiting(job_id, budget, type(exc).__name__,
                         'Worker wait failed; the original process was left running')
    try:
        terminal = _read_terminal(job_path, result_path, job)
    except (ValueError, OSError) as exc:
        return _failed(job_id, budget, type(exc).__name__, 'local_worker',
                       'Terminal reader result rejected: ' + str(exc),
                       billing_unknown=True, terminal_observed=True)
    if terminal is not None:
        return terminal
    return _failed(job_id, budget, 'OriginalWorkerExitedWithoutResult', 'local_worker',
                   'Worker exited without a terminal result',
                   billing_unknown=True, terminal_observed=True)
