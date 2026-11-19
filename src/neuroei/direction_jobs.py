"""Start one bounded manual direction job using the existing credential loader; no services."""
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .direction_queries import SEARCH_POLICY_VERSION
from .direction_workflow import ZONE
from .directions import load_direction, optional, require_direction_project
from .execution import project_lock
from .models import now


def start_direction_job(store, user, direction_id):
    direction = load_direction(store, user, direction_id)
    require_direction_project(store, direction)
    if not direction['enabled']:
        raise ValueError('This direction is paused')
    did = direction['direction_id']
    day = datetime.now(ZONE).date().isoformat()
    with project_lock(store.root, 'direction-launch:' + did):
        existing = [r for r in store.list('direction_run') if (r['direction_id'], r['date'], r['direction_version'])
                    == (did, day, direction['version'])
                    and (r.get('search_policy_version') == SEARCH_POLICY_VERSION or r['status'] == 'running')]
        if existing:
            run = max(existing, key=lambda r: r['started_at'])
            return {'run_id': run['run_id'], 'status': run['status'], 'reused': True}
        key = did + ':' + day + ':' + direction['version'] + ':' + SEARCH_POLICY_VERSION
        prior = optional(store, 'direction_request', key)
        if prior:
            return {**prior, 'reused': True}
        root = Path(__file__).resolve().parents[2]
        shell = shutil.which('pwsh')
        credentials = root / 'scripts' / 'with_credentials.ps1'
        rid = 'directionrun-' + uuid4().hex
        command = ['-B', '-X', 'utf8', '-m', 'neuroei', 'direction', 'daily', '--user', direction['user_id'],
                   '--id', did, '--run-id', rid, '--read-limit', '1']
        if shell and credentials.is_file():
            argv = [shell, '-NoProfile', '-File', str(credentials), sys.executable, *command]
        else:
            # Direct dispatch uses the service environment; on the server DEEPSEEK_API_KEY comes
            # from the systemd EnvironmentFile, so no credential loader is required.
            argv = [sys.executable, *command]
        request = {'run_id': rid, 'status': 'starting', 'user_id': direction['user_id'], 'direction_id': did,
                   'direction_version': direction['version'], 'date': day, 'created_at': now(),
                   'search_policy_version': SEARCH_POLICY_VERSION,
                   'manual_request': True, 'max_llm_requests': 8, 'read_limit': 1}
        store.put('direction_request', key, request)
        folder = store.root / 'directions' / did
        folder.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'NEUROEI_DATA_DIR': str(store.root)}
        try:
            with (folder / (rid + '.log')).open('xb') as log:
                child = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, start_new_session=os.name != 'nt',
                    **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
            request.update(status='starting', pid=child.pid)
        except OSError as exc:
            request.update(status='failed', error_type=type(exc).__name__)
        store.put('direction_request', key, request)
        if 'pid' in request:
            def observe_exit():
                code = child.wait()
                observed = store.get('direction_request', key)
                run = optional(store, 'direction_run', rid)
                observed.update(process_exit=code, ended_at=now(),
                                status=run['status'] if run else 'failed')
                if not run:
                    observed['error_type'] = 'WorkerExitedBeforeRunRecord'
                store.put('direction_request', key, observed)

            threading.Thread(target=observe_exit, daemon=True).start()
        return {**request, 'reused': False}
