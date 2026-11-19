"""Linux-server controller entry: validate deployment settings, then render or run the worker.

The controller_id in the settings file is a deployment identifier only. It is not a
distributed lock and this module makes no cross-host coordination claim. Nothing is
created, downloaded, registered, activated or backgrounded here: execution remains a
single foreground child process whose exit code is propagated unchanged.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/srv/researchpulse')
SETTINGS = ROOT / 'control' / 'settings.json'
MODES = ('plan', 'server', 'register', 'once', 'serve')
PLATFORM = sys.platform
API_URL = 'http://127.0.0.1:4217/api'
DEFAULT_MODE = 'plan'

_SAFE = re.compile(r'[A-Za-z0-9_-]+')
_COMMIT = re.compile(r'[a-f0-9]{7,40}')
_REQUIRED = ('controller_id', 'project', 'source_commit',
             'controller_release', 'reader_release', 'read_limit')
_OPTIONAL = ('publication_mode', 'feishu_generated_upload_enabled')


def _token(value, limit=64):
    return isinstance(value, str) and 0 < len(value) <= limit and bool(_SAFE.fullmatch(value))


def validate_settings(raw):
    """Return the settings mapping when every field is safe and forced; ValueError otherwise."""
    if type(raw) is not dict:
        raise ValueError('Controller settings must be a JSON object')
    unknown = set(raw) - set(_REQUIRED) - set(_OPTIONAL)
    if unknown:
        raise ValueError('Controller settings contain unknown fields')
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise ValueError('Controller settings are incomplete')
    if raw['controller_id'] != 'neuroei-server':
        raise ValueError('controller_id must identify the neuroei-server deployment')
    if not _token(raw['project']):
        raise ValueError('project must be a safe non-empty token')
    commit = raw['source_commit']
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError('source_commit must be 7..40 lowercase hex characters')
    controller_release = raw['controller_release']
    if not _token(controller_release, 128):
        raise ValueError('controller_release must be a safe release token')
    if not controller_release.endswith(commit[:7]):
        raise ValueError('controller_release must end with the first 7 source_commit characters')
    if not _token(raw['reader_release'], 128):
        raise ValueError('reader_release must be a safe release token')
    limit = raw['read_limit']
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError('read_limit must be an integer within 1..5')
    if 'publication_mode' in raw and raw['publication_mode'] != 'draft':
        raise ValueError('publication_mode is fixed to draft')
    if ('feishu_generated_upload_enabled' in raw
            and raw['feishu_generated_upload_enabled'] is not False):
        raise ValueError('feishu_generated_upload_enabled is fixed to false')
    return dict(raw)


def load_settings(path=SETTINGS):
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('Controller settings file is missing or not JSON') from exc
    return validate_settings(raw)


def _release(settings):
    return ROOT / 'releases' / settings['controller_release']


def _orchestrator():
    return ROOT / '.orchestration-venv' / 'bin' / 'python'


def check_deployment(settings):
    """Fail closed before any child process when the immutable deployment is incomplete."""
    if not PLATFORM.startswith('linux'):
        raise RuntimeError('Controller execution is Linux-only')
    release = _release(settings)
    if not (release / 'config.toml').is_file():
        raise FileNotFoundError('Controller release is missing config.toml')
    if not (release / 'scripts' / 'prefect_worker.py').is_file():
        raise FileNotFoundError('Controller release is missing scripts/prefect_worker.py')
    if not _orchestrator().is_file():
        raise FileNotFoundError('Isolated orchestration interpreter is missing')
    reader = ROOT / 'releases' / settings['reader_release']
    if not reader.is_dir():
        raise FileNotFoundError('Requested reader release is missing')


def build_command(settings, mode, *, weekly=False, activate=False):
    """Render the worker argument list; no shell and no relative release path are used."""
    if mode not in MODES:
        raise ValueError('Unknown controller mode')
    if weekly and mode != 'once':
        raise ValueError('--weekly is only valid for once')
    if activate and mode != 'serve':
        raise ValueError('--activate is only valid for serve')
    config = _release(settings) / 'config.toml'
    command = [str(_orchestrator()), '-B', 'scripts/prefect_worker.py', mode,
               '--config', str(config),
               '--project', settings['project'],
               '--reader-release', settings['reader_release'],
               '--read-limit', str(settings['read_limit'])]
    if weekly:
        command.append('--weekly')
    if activate:
        command.append('--activate')
    return command


def build_env(settings):
    """Fixed CPU/offline/loopback environment; inherited secrets are never printed."""
    environment = dict(os.environ)
    environment.pop('PREFECT_API_KEY', None)
    environment.update({
        'NEUROEI_DATA_DIR': str(ROOT / 'data'),
        'NEUROEI_CODE_REVISION': settings['source_commit'],
        'NEUROEI_READER_BACKEND': 'server-local',
        'PYTHONPATH': str(_release(settings) / 'src'),
        'CUDA_VISIBLE_DEVICES': '',
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
        'PREFECT_API_URL': API_URL,
    })
    return environment


def render_plan(settings):
    """Safe, non-mutating plan: settings, one command and activate=false; never env values."""
    return {'mode': 'plan', 'active': False, 'activate': False,
            'controller_id': settings['controller_id'],
            'project': settings['project'],
            'source_commit': settings['source_commit'],
            'controller_release': settings['controller_release'],
            'reader_release': settings['reader_release'],
            'read_limit': settings['read_limit'],
            'cwd': str(_release(settings)),
            'command': build_command(settings, 'plan')}


def _run(command, settings):
    completed = subprocess.run(command, cwd=str(_release(settings)),
                               env=build_env(settings), check=False)
    return completed.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m neuroei.controller',
                                     description='Reviewable ResearchPulse server controller entry')
    parser.add_argument('mode', nargs='?', default=DEFAULT_MODE, choices=MODES)
    parser.add_argument('--settings', default=str(SETTINGS))
    parser.add_argument('--weekly', action='store_true')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        if args.activate and args.mode != 'serve':
            raise ValueError('--activate is only valid for serve')
        if args.weekly and args.mode != 'once':
            raise ValueError('--weekly is only valid for once')
        if args.mode == 'plan':
            print(json.dumps(render_plan(settings), ensure_ascii=False, indent=2))
            return 0
        check_deployment(settings)
        command = build_command(settings, args.mode,
                                weekly=args.weekly, activate=args.activate)
        return _run(command, settings)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print('controller: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
