"""Launch a private Windows research workspace and its local worker."""
import argparse
import json
import os
import secrets
import site
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from zoneinfo import ZoneInfo

from cryptography.fernet import Fernet

from . import accounts, desktop_models
from .feishu_user_auth import DESKTOP_ORIGIN
from .portal import make_server
from .store import Store

PORT = 8768
HOST = f'localhost:{PORT}'
ORIGIN = DESKTOP_ORIGIN
LOCAL_ACCOUNT = 'local-researcher'
TOKEN_SECONDS = 120
BEIJING = ZoneInfo('Asia/Shanghai')
EMBEDDING_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'


def default_profile():
    return Path(sys.executable).resolve().parents[1] / 'data'


class Desktop:
    def __init__(self, profile):
        self.profile = profile.resolve()
        self.profile.mkdir(parents=True, exist_ok=True)
        self.store_path = self.profile / 'workspace'
        self.host = HOST
        self.origin = ORIGIN
        self._token_lock = threading.Lock()
        self.open_file = self.profile / 'browser-open.json'
        self.instance_file = self.profile / 'browser-instance'

    def initialize(self, store):
        if (owner := accounts.by_login(store, LOCAL_ACCOUNT)) is None:
            if accounts.accounts(store):
                raise ValueError('请选择全新的本机数据目录。')
            owner = accounts.create(store, LOCAL_ACCOUNT, '本机研究者', secrets.token_urlsafe(40), 'member')
        if not self.instance_file.exists():
            self.instance_file.write_text(secrets.token_urlsafe(32), encoding='ascii')
        return owner

    def instance(self):
        return self.instance_file.read_text(encoding='ascii').strip()

    def existing_instance(self):
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(self.origin + '/api/desktop/instance', timeout=2) as response:
                return secrets.compare_digest(json.load(response).get('instance', ''), self.instance())
        except (OSError, TypeError, ValueError, URLError):
            return False

    def open_browser(self):
        token = secrets.token_urlsafe(40)
        with tempfile.NamedTemporaryFile(mode='w', encoding='ascii', dir=self.profile, delete=False) as output:
            json.dump({'token': token, 'expires': time.time() + TOKEN_SECONDS}, output)
        Path(output.name).replace(self.open_file)
        webbrowser.open(self.origin + '/desktop/open#' + token)

    def redeem(self, token):
        if not isinstance(token, str) or not 40 <= len(token) <= 80:
            return False
        with self._token_lock:
            try:
                saved = json.loads(self.open_file.read_text(encoding='ascii'))
            except (FileNotFoundError, ValueError):
                return False
            if saved['expires'] < time.time() or not secrets.compare_digest(saved['token'], token):
                return False
            self.open_file.unlink()
            return True

    def prepare(self):
        desktop_models.prepare(self.profile)
        key_file = self.profile / 'feishu-key.dpapi'
        if not key_file.exists():
            key_file.write_text(desktop_models.protect_secret(Fernet.generate_key().decode('ascii')), encoding='ascii')
        for name in ('FEISHU_APP_ID', 'FEISHU_APP_SECRET', 'NEUROEI_COOKIE_SECURE', 'NEUROEI_ALLOWED_HOSTS',
                     'NEUROEI_CODEX_SCOPE_USERS', 'NEUROEI_CODEX_SCOPE_RUNNER'):
            os.environ.pop(name, None)
        os.environ.update(NEUROEI_DESKTOP='1', NEUROEI_BIND='127.0.0.1', NEUROEI_PUBLIC_ORIGIN=self.origin,
                          NEUROEI_FEISHU_TOKEN_KEY=desktop_models.unprotect_secret(key_file.read_text(encoding='ascii')),
                          NEUROEI_READING_FOLLOWUP='full', NEUROEI_DISCOVERY_FOLLOWUP='per_question',
                          NEUROEI_FEISHU_IMAGE_OCR='1', CUDA_VISIBLE_DEVICES='',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', LITELLM_LOCAL_MODEL_COST_MAP='True',
                          LITELLM_TELEMETRY='False')

    def model_settings(self):
        return desktop_models.load(self.profile)

    def model_configured(self):
        state = self.model_settings()
        return state['deepseek_configured'] if state['backend'] == 'deepseek' else state['codex_installed']

    def docling_paths(self):
        bundle = Path(sys.executable).resolve().parents[1]
        for root in (bundle, self.profile):
            packages = root / 'reader-packages'
            models = root / 'models'
            if ((packages / '.neuroei-ready').is_file() and (models / 'docling').is_dir()
                    and any((models / name).is_file() for name in ('.neuroei-docling-ready', '.neuroei-ready'))):
                return packages, models / 'docling'
        return None

    def reader_paths(self):
        models = self.profile / 'models'
        bge = models / 'bge-m3'
        if ((bge / 'config.json').is_file()
                and any((models / name).is_file() for name in ('.neuroei-bge-ready', '.neuroei-ready'))
                and (paths := self.docling_paths())):
            packages, docling = paths
            return packages, bge, docling
        return None

    def worker(self, shutdown):
        from .recommendation_worker import main as worker_main

        args = ['--store', str(self.store_path), '--max-understanding-calls', '100']
        if paths := self.reader_paths():
            packages, bge, docling = paths
            sys.path.insert(0, str(packages))
            site.addsitedir(str(packages))
            args += ['--model-path', str(bge), '--model-revision', EMBEDDING_REVISION, '--journal-sources',
                     '--artifacts-path', str(docling), '--fetch-fulltexts', '--open-access-fulltexts',
                     '--max-discovery-review-calls', '1', '--max-model-calls', '100',
                     '--wechat-sources', '--web-fallback', '--web-search-limit', '64', '--web-fetch-limit', '128']
        worker_main(args, shutdown=shutdown)

    def scheduler(self, shutdown):
        from .project_daily_schedule import run
        from .recommendation_jobs import worker_status

        store = Store(self.store_path)
        checked_day = None
        try:
            while not shutdown.is_set():
                timestamp = (datetime.now(BEIJING) - timedelta(hours=5)).replace(hour=5, minute=0, second=0, microsecond=0)
                worker = worker_status(store)
                if (timestamp.date() != checked_day and self.model_configured()
                        and worker['status'] == 'ready' and worker.get('recommendations_enabled', True)):
                    run(store, as_of=timestamp.isoformat(), email=None)
                    checked_day = timestamp.date()
                shutdown.wait(30)
        finally:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, help='Private local profile directory')
    parser.add_argument('--background', action='store_true', help='Start the local scheduler without opening a browser')
    args = parser.parse_args(argv)
    if sys.platform != 'win32':
        parser.error('The standalone package requires Windows')
    desktop = Desktop(args.profile or default_profile())
    if desktop.existing_instance():
        if not args.background:
            desktop.open_browser()
        return

    store = Store(desktop.store_path)
    shutdown = threading.Event()
    try:
        try:
            server = make_server(store, PORT, legacy_execution_enabled=False, desktop=desktop)
        except OSError:
            parser.error(f'本机端口 {PORT} 正在使用，请关闭占用程序后再启动。')
        with server:
            desktop.initialize(store)
            desktop.prepare()
            threading.Thread(target=desktop.worker, args=(shutdown,), daemon=True, name='research-worker').start()
            threading.Thread(target=desktop.scheduler, args=(shutdown,), daemon=True, name='daily-scheduler').start()
            print(f'ResearchPulse {desktop.origin} | {desktop.profile}', flush=True)
            if not args.background:
                desktop.open_browser()
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()
        store.close()


if __name__ == '__main__':
    main()
