"""Strict SSH relay for official PDFs and a bounded PaperQA reader job."""
import base64
import json
import os
import re
import shlex
import uuid
from urllib.request import Request, urlopen

import paramiko

from .reader_identity import validate_reader_result

ROOT = '/srv/researchpulse'


def read_remote(config, profile, paper, *, release, budget=4, job_id=None, resume_only=False):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', release):
        raise ValueError('Explicit deployed reader release required')
    if not re.fullmatch(r'arxiv:\d{4}\.\d{4,5}', paper.paper_id) or not re.fullmatch(r'v\d+', paper.version):
        return {'status':'unavailable','error_type':'UnsupportedOfficialPdfAcquisition','llm_requests':0}
    job_id = job_id or 'job-'+uuid.uuid4().hex
    if not re.fullmatch(r'job-[a-f0-9]{32}', job_id) or type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError('Explicit safe reader job and bounded budget required')
    if type(resume_only) is not bool:
        raise ValueError('Explicit reader resume mode required')
    needed = {bid for f in profile.facts for bid in f.source_blocks}
    compact = profile.model_copy(update={'blocks':[b for b in profile.blocks if b.block_id in needed]})
    job = {'job_id':job_id,'profile':compact.model_dump(mode='json'),'paper':paper.model_dump(mode='json'),
           'budget':budget,'code_revision':release}
    data = json.dumps(job,ensure_ascii=False).encode('utf-8')
    directory = ROOT+'/data/workflow-jobs'
    job_path = directory+'/'+job_id+'.json'
    result_path = directory+'/'+job_id+'.result.json'
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    pid = paper.paper_id.split(':')[1]+paper.version
    pdf = ROOT+'/data/public-pdf/'+pid+'.pdf'
    # Until the server is observed, a connection error cannot prove an old job never ran.
    uncertain = True
    terminal_observed = False

    def result(sftp):
        with sftp.open(result_path, 'rb') as stream:
            body = json.loads(stream.read())
        try:
            with sftp.open(job_path, 'rb') as stream:
                saved_job = json.loads(stream.read())
        except FileNotFoundError as exc:
            raise ValueError('Terminal reader result is missing its input identity') from exc
        return validate_reader_result(body, saved_job, job)

    try:
        ssh.connect('researchpulse-host.example.invalid', username='researchpulse', password=os.environ['NEUROEI_SSH_PASSWORD'],
                    timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
        with ssh.open_sftp() as sftp:
            try:
                sftp.stat(directory)
            except FileNotFoundError:
                sftp.mkdir(directory)
            try:
                return result(sftp)
            except FileNotFoundError:
                pass
            try:
                with sftp.open(job_path, 'rb') as stream:
                    if stream.read() != data:
                        raise ValueError('Existing reader job has different inputs')
            except FileNotFoundError:
                if resume_only:
                    return {'status':'awaiting_result','job_id':job_id,'llm_requests':budget,
                            'billing_unknown':True,'live_verified':False,
                            'error_type':'MissingOriginalReaderInput',
                            'detail':'Original input is missing; inspection required before any new dispatch'}
                uncertain = False
            else:
                try:
                    with sftp.open(directory+'/'+job_id+'.pid', 'rb') as stream:
                        worker_pid = stream.read().decode('ascii').strip()
                except FileNotFoundError:
                    return {'status':'awaiting_result','job_id':job_id,'llm_requests':budget,
                            'billing_unknown':True,'live_verified':False,
                            'detail':'Input exists; no terminal result/process handle verified'}
                if not worker_pid.isdigit():
                    raise ValueError('Invalid reader PID handle')
                try:
                    with sftp.open('/proc/'+worker_pid+'/cmdline','rb') as stream:
                        command = stream.read()
                    arguments = command.rstrip(b'\0').split(b'\0')
                    live = (any(arg.rsplit(b'/',1)[-1] == b'workflow_reader.py' for arg in arguments)
                            and any(a == b'--job' and b == job_path.encode('utf-8')
                                    for a,b in zip(arguments,arguments[1:],strict=False)))
                except FileNotFoundError:
                    live = False
                # The worker may have committed its result between the first read and process check.
                try:
                    return result(sftp)
                except FileNotFoundError:
                    pass
                return {'status':'awaiting_result' if live else 'failed','job_id':job_id,
                        'llm_requests':budget,'billing_unknown':True,'live_verified':live,
                        'error_type':'ObservationPending' if live else 'OriginalWorkerExitedWithoutResult'}
            try:
                with sftp.open(pdf, 'rb') as stream:
                    if stream.read(5)!=b'%PDF-':
                        raise ValueError('Cached PDF header invalid')
            except FileNotFoundError:
                part = pdf+'.'+job_id+'.part'
                with urlopen(Request('https://arxiv.org/pdf/'+pid, headers={'Accept-Encoding':'identity'}), timeout=45) as response:
                    count = 0
                    expected = int(response.headers.get('content-length','0'))
                    with sftp.open(part, 'wb') as stream:
                        stream.set_pipelined(True)
                        while chunk := response.read(256*1024):
                            if not count and b'%PDF-' not in chunk[:1024]:
                                raise ValueError('Official response is not PDF')
                            count += len(chunk)
                            if count>70_000_000:
                                raise ValueError('PDF size budget exceeded')
                            stream.write(chunk)
                if not count or sftp.stat(part).st_size!=count or (expected and count!=expected):
                    raise ValueError('Incomplete PDF retained as partial')
                sftp.rename(part,pdf)
            # A saved input prevents automatic redispatch, even if the parent dies before exec.
            uncertain = True
            with sftp.open(job_path,'wb') as stream:
                stream.write(data)
            sftp.chmod(job_path,0o600)
            with sftp.open(job_path,'rb') as stream:
                if stream.read()!=data:
                    raise ValueError('Reader job readback failed')
        code = ROOT+'/releases/'+release
        script = ('set -euo pipefail\nexport LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES="" '
                  'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True\n'
                  'export NEUROEI_DATA_DIR='+ROOT+'/data\nexport PYTHONPATH='+shlex.quote(code+'/src')+'\n'
                  'cd '+shlex.quote(code)+'\nset -a\n. '+ROOT+'/runtime.env\nset +a\n'+ROOT+
                  '/.venv/bin/python -u scripts/workflow_reader.py --job '+shlex.quote(job_path)+'\n')
        encoded = base64.b64encode(script.encode()).decode()
        stdin, stdout, stderr = ssh.exec_command('bash -s',timeout=600)
        stdin.write("printf %s '"+encoded+"' | base64 -d | bash\n")
        stdin.channel.shutdown_write()
        stdout.read()
        stderr.read()
        stdout.channel.recv_exit_status()
        terminal_observed = True
        with ssh.open_sftp() as sftp:
            return result(sftp)
    except Exception as exc:
        return {'status':'awaiting_result' if uncertain and not terminal_observed else 'failed',
                'error_type':type(exc).__name__,'terminal_observed':terminal_observed,
                'job_id':job_id, 'llm_requests':budget if uncertain else 0,
                'billing_unknown':uncertain, 'stage':'remote_job' if uncertain else 'official_pdf_acquisition',
                'live_verified':False}
    finally:
        ssh.close()
