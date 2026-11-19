"""Desktop-triggered entry point for the server-side daily workflow.

One foreground SSH dispatch to the fixed GPU server. The module never loads the
local Store/FlashClient, never reads project text or credentials apart from the
single password looked up in ``main`` for ``run``, and never starts a
background process of its own.
"""

import argparse
import base64
import json
import os
import re
import shlex
import sys
import time
from datetime import date

import paramiko

HOST = "researchpulse-host.example.invalid"
USER = "researchpulse"
ROOT = "/srv/researchpulse"
SETTINGS = ROOT + "/control/settings.json"
ORCHESTRATOR = ROOT + "/.orchestration-venv/bin/python"
STRATEGIES = ("baseline", "paperflow-experimental")
MODES = ("plan", "run")
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
MAPPING_ID = re.compile(r"[A-Za-z0-9:_-]+")
CONNECT_TIMEOUT = 15
OBSERVATION_TIMEOUT = 1800
CONNECTION_FAILED = 3
OBSERVATION_TIMEOUT_CODE = 4


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m neuroei.server_entry",
        description="Render or dispatch the fixed server daily workflow.",
    )
    parser.add_argument("mode", nargs="?", choices=MODES, default=None)
    parser.add_argument("--mode", dest="mode_option", choices=MODES, default=None)
    parser.add_argument("--project", default="bnn-arm")
    parser.add_argument("--date", default="")
    parser.add_argument("--read-limit", type=int, default=3)
    parser.add_argument("--ranking-strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--interest-mapping", default="")
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--new-version", action="store_true")
    return parser


def _iso_date(value):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be an ISO YYYY-MM-DD calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be an ISO YYYY-MM-DD calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must be an ISO YYYY-MM-DD calendar date")
    return value


def validate_payload(args):
    """Validate the local arguments and return the immutable dispatch payload."""
    project = args.project
    if not isinstance(project, str) or not SAFE_TOKEN.fullmatch(project):
        raise ValueError("project must be a safe token of letters, digits, '-' or '_'")
    report_date = args.date
    if report_date:
        if not isinstance(report_date, str):
            raise ValueError("date must be an ISO YYYY-MM-DD calendar date")
        report_date = _iso_date(report_date)
    read_limit = args.read_limit
    if type(read_limit) is not int or not 1 <= read_limit <= 5:
        raise ValueError("read-limit must be a real integer from 1 through 5")
    if args.ranking_strategy not in STRATEGIES:
        raise ValueError("ranking-strategy must be baseline or paperflow-experimental")
    mapping = args.interest_mapping
    if mapping and (not isinstance(mapping, str) or not MAPPING_ID.fullmatch(mapping)):
        raise ValueError("interest-mapping must contain only letters, digits, ':', '_' or '-'")
    if args.ranking_strategy == "paperflow-experimental":
        if not mapping:
            raise ValueError("paperflow-experimental requires an explicit interest-mapping")
        if args.weekly:
            raise ValueError("paperflow-experimental cannot be combined with weekly")
    plain_baseline = not mapping and not args.new_version and args.ranking_strategy == "baseline"
    task = "controlled-run" if plain_baseline else "workflow"
    return {
        "project": project,
        "date": report_date,
        "read_limit": read_limit,
        "ranking_strategy": args.ranking_strategy,
        "interest_mapping": mapping,
        "weekly": bool(args.weekly),
        "new_version": bool(args.new_version),
        "task": task,
    }


def render_plan(payload):
    """Safe, secret-free rendering used by ``mode=plan`` (no SSH, no env read)."""
    return {
        "mode": "plan",
        "host": HOST,
        "user": USER,
        "root": ROOT,
        "project": payload["project"],
        "date": payload["date"] or None,
        "read_limit": payload["read_limit"],
        "ranking_strategy": payload["ranking_strategy"],
        "interest_mapping": payload["interest_mapping"] or None,
        "weekly": payload["weekly"],
        "new_version": payload["new_version"],
        "task": payload["task"],
        "reader_release": "resolved on the server from control/settings.json",
        "dispatch": "none; plan does not open SSH or read an environment secret",
    }


def render_remote_program(payload):
    """Small self-contained program executed on the server through base64 stdin."""
    payload_literal = repr(json.dumps(payload, ensure_ascii=True))
    return f'''import json
import subprocess
import sys

ROOT = {ROOT!r}
payload = json.loads({payload_literal})

from neuroei.controller import build_env, check_deployment, load_settings

settings = load_settings(ROOT + '/control/settings.json')
if settings['project'] != payload['project']:
    sys.stderr.write(json.dumps({{'status': 'failed', 'error_type': 'ProjectMismatch',
                                  'detail': 'Requested project does not match the '
                                            'deployed settings'}}) + '\\n')
    raise SystemExit(2)
check_deployment(settings)
release = ROOT + '/releases/' + settings['controller_release']
config = release + '/config.toml'
python = ROOT + '/.orchestration-venv/bin/python'
if payload['task'] == 'controlled-run':
    argv = [python, '-B', '-m', 'neuroei', '--config', config, 'controlled-run',
            '--project', payload['project'],
            '--reader-release', settings['reader_release'],
            '--read-limit', str(payload['read_limit'])]
    if payload['date']:
        argv.extend(['--date', payload['date']])
    if payload['weekly']:
        argv.append('--weekly')
else:
    argv = [python, '-B', '-m', 'neuroei', '--config', config, 'workflow',
            '--project', payload['project'],
            '--reader-release', settings['reader_release'],
            '--read-limit', str(payload['read_limit']),
            '--ranking-strategy', payload['ranking_strategy']]
    if payload['date']:
        argv.extend(['--date', payload['date']])
    if payload['interest_mapping']:
        argv.extend(['--interest-mapping', payload['interest_mapping']])
    if payload['new_version']:
        argv.append('--new-version')
    if payload['weekly']:
        argv.append('--weekly')
completed = subprocess.run(argv, cwd=release, env=build_env(settings), check=False)
raise SystemExit(completed.returncode)
'''


def render_wrapper(program):
    """POSIX wrapper sourcing the server secrets, then executing the program bytes."""
    encoded = base64.b64encode(program.encode("utf-8")).decode("ascii")
    return (
        "set -euo pipefail\n"
        "umask 077\n"
        "cd " + shlex.quote(ROOT) + "\n"
        "set -a\n"
        ". " + shlex.quote(ROOT + "/runtime.env") + "\n"
        "set +a\n"
        "printf %s " + shlex.quote(encoded) + " | base64 -d | "
        + shlex.quote(ORCHESTRATOR) + " -B -\n"
    )


def _fail(error_type, detail):
    payload = {"status": "failed", "error_type": error_type, "detail": detail}
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.stderr.flush()


def _drain(channel, deadline):
    """Read at most one block per stream per pass under one global deadline."""
    out = bytearray()
    err = bytearray()
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("remote observation deadline exceeded")
        progressed = False
        if channel.recv_ready():
            chunk = channel.recv(65536)
            if chunk:
                out.extend(chunk)
                progressed = True
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(65536)
            if chunk:
                err.extend(chunk)
                progressed = True
        if (
            channel.exit_status_ready()
            and not channel.recv_ready()
            and not channel.recv_stderr_ready()
        ):
            break
        if not progressed:
            time.sleep(0.02)
    return bytes(out), bytes(err)


def dispatch(payload, password):
    """One foreground SSH call: no retry, no cancel, exit code passed through."""
    client = paramiko.SSHClient()
    try:
        try:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(
                HOST,
                username=USER,
                password=password,
                timeout=CONNECT_TIMEOUT,
                banner_timeout=CONNECT_TIMEOUT,
                auth_timeout=CONNECT_TIMEOUT,
                look_for_keys=False,
                allow_agent=False,
            )
        except (OSError, EOFError, paramiko.SSHException) as exc:
            _fail(
                "SSHConnectionFailed",
                "No task was dispatched; " + type(exc).__name__ + " is not retried",
            )
            return CONNECTION_FAILED
        wrapper = render_wrapper(render_remote_program(payload))
        deadline = time.monotonic() + OBSERVATION_TIMEOUT
        try:
            stdin, stdout, _ = client.exec_command(
                "bash -s", timeout=OBSERVATION_TIMEOUT
            )
            stdin.write(wrapper)
            stdin.channel.shutdown_write()
            out, err = _drain(stdout.channel, deadline)
            code = stdout.channel.recv_exit_status()
        except TimeoutError:
            _fail(
                "ObservationTimeout",
                "Remote observation timed out; the dispatched task is not cancelled or retried",
            )
            return OBSERVATION_TIMEOUT_CODE
        except (OSError, EOFError, paramiko.SSHException) as exc:
            _fail(
                "RemoteObservationFailed",
                "Remote observation failed; " + type(exc).__name__ + " is not retried",
            )
            return CONNECTION_FAILED
        sys.stdout.write(out.decode("utf-8", "replace"))
        sys.stderr.write(err.decode("utf-8", "replace"))
        sys.stdout.flush()
        sys.stderr.flush()
        return code
    finally:
        client.close()


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.mode_option:
        if args.mode and args.mode != args.mode_option:
            sys.stderr.write("server_entry: conflicting mode arguments\n")
            return 2
        args.mode = args.mode_option
    if not args.mode:
        args.mode = "run"
    try:
        payload = validate_payload(args)
    except ValueError as exc:
        sys.stderr.write("server_entry: " + str(exc) + "\n")
        return 2
    if args.mode == "plan":
        sys.stdout.write(json.dumps(render_plan(payload), ensure_ascii=False, indent=2) + "\n")
        return 0
    try:
        password = os.environ["NEUROEI_SSH_PASSWORD"]
    except KeyError:
        sys.stderr.write("server_entry: NEUROEI_SSH_PASSWORD is not set; nothing was dispatched\n")
        return 2
    return dispatch(payload, password)


if __name__ == "__main__":
    raise SystemExit(main())
