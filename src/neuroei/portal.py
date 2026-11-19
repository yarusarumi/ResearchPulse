"""Loopback research review, user direction settings and bounded manual worker dispatch."""
import json
import os
import secrets
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

from pydantic import Field, TypeAdapter

from . import accounts as auth
from .config import StrictModel
from .directions import direction_state, load_direction
from .execution import WorkflowBusy
from .models import FeedbackEvent, ProjectFact, now
from .recommendation_pipeline import QueryPlanningError
from .scope_deletion import ScopeBusy, deleted_scopes, require_active, scope_deleted
from .store import Store


class ReviewFeedback(StrictModel):
    event_id: UUID
    workflow_id: str
    paper_id: str
    paper_version: str
    profile_version: str
    report_id: str
    user_id: str = Field(min_length=1, max_length=100)
    action: str
    reason: str = Field(default='', max_length=4000)


_FACTS = TypeAdapter(list[ProjectFact])

# Error contracts shared by the portal page. The page switches on `code`, so the
# wording can change without breaking it; the messages stay in one place.
STALE_TOKEN_ERROR = '页面已过期，请刷新后重试。'
HOST_REJECTED_ERROR = '请通过实验室服务器地址打开页面。'
ORIGIN_REJECTED_ERROR = '请求来源不受信任，请刷新页面。'
FORBIDDEN_ERROR = '这个账号没有访问权限。'
INVALID_USERNAME_ERROR = '用户名需为 2–32 位小写字母、数字或 . _ -，且以字母或数字开头。'
WEAK_PASSWORD_ERROR = '初始密码至少 10 位。'
WEAK_NEW_PASSWORD_ERROR = '新密码至少 10 位。'

_ADMIN_ACTIONS = {'/api/admin/account': 'account.create',
                  '/api/admin/password': 'account.password_reset',
                  '/api/admin/status': 'account.status',
                  '/api/admin/access': 'account.access',
                  '/api/admin/model-backend': 'model_backend.set'}
_SELF_PASSWORD_ACTION = 'account.password_change'


class PortalError(ValueError):
    """A rejected request whose page-facing code lets the UI react precisely."""

    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def audit_admin(store, action, actor, target, result='ok', detail=None, reason=None):
    """Record operation metadata only: never passwords, tokens, hashes or request bodies."""
    body = {'actor': actor, 'action': action, 'target': target, 'result': result}
    if detail is not None:
        body['detail'] = detail
    if reason is not None:
        body['reason'] = reason
    store.audit(f'admin:{uuid4().hex}', body)


def account_create_error(message):
    """Map accounts.create() rejections onto the page's structured error contract."""
    if message.startswith('用户名需为'):
        return INVALID_USERNAME_ERROR, 'invalid_username'
    if message.startswith('密码至少需要'):
        return WEAK_PASSWORD_ERROR, 'weak_password'
    if message.startswith('密码过长'):
        return '密码过长。', 'weak_password'
    if message.startswith('角色只能是'):
        return '角色只能是成员或管理员。', 'invalid_role'
    if message.startswith('这个用户名已经存在'):
        return '这个用户名已存在。', 'username_taken'
    if message.startswith(('这个显示名已经被占用', '这个显示名曾用于已删除的账号',
                           '这个显示名下有历史方向或来源')):
        return '这个显示名已存在。', 'display_name_taken'
    if message.startswith('显示名需为'):
        return '显示名需为 1–100 个字符。', 'invalid_display_name'
    return message, 'invalid_account'


def active_projects(store):
    """Active profile pointer plus its project/version, without parsing profile bodies."""
    projects, deleted = [], deleted_scopes(store)
    for pointer in store.select('active_profile', ('key',)):
        row = store.select('profile', ('project_id', 'version'), key=pointer['key'])
        if not row:
            raise ValueError(f"Missing profile: {pointer['key']}")
        if not scope_deleted(row[0], deleted):
            projects.append({**row[0], 'key': pointer['key']})
    return sorted(projects, key=lambda row: row['project_id'])


def profile_display(store, project, active_key, active_version, version=None):
    """The profile fields the portal shows, without validating its source blocks.

    Profiles are validated when written and need an approval event before they can
    become active, so the first-screen read only has to check the displayed facts
    (20 items) instead of re-validating all ~300 immutable source blocks on every
    request. Facts still go through ProjectFact validation.
    """
    active = version is None or version == active_version
    key = active_key if active else f'{project}:{version}'
    fields = ('project_id', 'facts', 'unresolved') if active else ('project_id', 'facts', 'unresolved', 'approved_by')
    row = store.select('profile', fields, key=key, decode=('facts', 'unresolved'))
    if not row:
        raise ValueError(f"Missing profile: {key}")
    body = row[0]
    if body.get('project_id') != project:
        raise ValueError('Profile project mismatch')
    approved_by = store.get('approval', active_key)['reviewer'] if active else body.get('approved_by')
    facts = _FACTS.validate_python(body.get('facts') or [])
    return {'version': active_version if active else version, 'active_version': active_version,
            'approved_by': approved_by, 'facts': [fact.model_dump() for fact in facts],
            'unresolved': list(body.get('unresolved') or [])}


def workflow_paper(store, workflow_id, paper_id, *, workflow=None, pool=None, papers=None):
    workflow = store.get('workflow', workflow_id) if workflow is None else workflow
    rec = next((r for r in workflow.get('recommendations', []) if r['paper_id'] == paper_id), None)
    if rec is None:
        raise ValueError('Paper is not a recommendation in this workflow')
    pool = store.get('pool', workflow['pool_id']) if pool is None else pool
    if pool['project_id'] != workflow['project_id']:
        raise ValueError('Workflow pool project mismatch')
    if papers is None:
        from .workflow import latest_papers
        papers = latest_papers(pool)
    paper = next((p for p in papers if p.paper_id == paper_id), None)
    if paper is None:
        raise ValueError('Missing recommended paper')
    reading = next((r for r in workflow['papers'] if r['paper_id'] == paper_id), {})
    if (not reading and workflow.get('current_paper') == paper_id and workflow.get('phase') == 'read'
            and workflow.get('status') in ('running', 'failed')):
        reading = {'paper_id': paper_id, 'version': paper.version,
                   'status': 'reading_pending' if workflow['status'] == 'running' else 'reading_interrupted'}
    if reading and reading['version'] != paper.version:
        raise ValueError('Reading and displayed paper versions differ')
    return workflow, paper, rec, reading


def paper_analysis(store, workflow_id, paper_id):
    workflow, paper, _, reading = workflow_paper(store, workflow_id, paper_id)
    if reading.get('status') != 'completed' or not reading.get('analysis_id'):
        return {'available': False, 'status': reading.get('status', 'not_requested')}
    analysis = store.get('analysis', reading['analysis_id'])
    if (analysis['paper_id'], analysis['paper_version'], analysis['profile_version']) != (
            paper.paper_id, paper.version, workflow['profile_version']):
        raise ValueError('Analysis belongs to another paper or profile version')
    # Reuse the same evidence/reference gates as the generated report.
    from .models import ProjectProfile
    from .reports import reading_report
    profile = ProjectProfile.model_validate(store.get('profile', f"{workflow['project_id']}:{workflow['profile_version']}"))
    reading_report(project=profile, report_date=workflow['date'], run_id=workflow_id,
                   entries=[{'paper': paper, 'analysis': analysis}])
    return {'available': True, 'analysis': analysis}


def save_feedback(store, payload, user=None, allowed=None):
    if user is not None:
        payload = {**payload, 'user_id': user}
    request = ReviewFeedback.model_validate(payload)
    workflow_body = store.get('workflow', request.workflow_id)
    if allowed is not None:
        project = workflow_body.get('project_id')
        if project not in allowed:
            raise PermissionError('这个账号没有这个项目的访问权限')
    workflow, paper, _, _ = workflow_paper(store, request.workflow_id, request.paper_id,
                                           workflow=workflow_body)
    report = store.get('report', request.report_id)
    if (paper.version, workflow['profile_version'], workflow.get('report_id')) != (
            request.paper_version, request.profile_version, request.report_id):
        raise ValueError('Feedback context changed; refresh the displayed workflow')
    if (report['project_id'], report['profile_version'], report.get('workflow_id')) != (
            workflow['project_id'], workflow['profile_version'], request.workflow_id):
        raise ValueError('Feedback report context mismatch')
    user = request.user_id.strip()
    if not user:
        raise ValueError('Enter your name before submitting feedback')
    if request.action in ('fact_correction', 'report_quality') and not request.reason.strip():
        raise ValueError('Corrections and report quality feedback need a reason')
    event_id = str(request.event_id)
    # Preserve the original occurrence time on HTTP retries; same ID/different intent is rejected.
    try:
        prior = store.get('feedback', event_id)
    except ValueError:
        prior = None
    event = FeedbackEvent(project_id=workflow['project_id'], user_id=user, paper_id=paper.paper_id,
        action=request.action, reason=request.reason.strip(), occurred_at=prior['occurred_at'] if prior else now(),
        paper_version=paper.version, profile_version=workflow['profile_version'],
        workflow_id=request.workflow_id, report_id=request.report_id)
    body = event.model_dump(mode='json')
    if prior and prior != body:
        raise ValueError('Feedback ID already records a different action')
    store.put('feedback', event_id, body, immutable=True)
    return {'feedback_id': event_id, 'saved': True, 'learning_enabled': False}


def portal_state(store, project=None, workflow_id=None, allowed=None, only_user=None):
    projects = active_projects(store)
    if allowed is not None:
        projects = [p for p in projects if p['project_id'] in allowed]
    choices = [{'id': p['project_id'], 'version': p['version']} for p in projects]
    if not choices:
        return {'projects': [], 'workflow': None, 'workflows': [], 'reports': [], 'cards': [],
                'sources': [], 'feedback': [], 'profile': None}
    project = project or choices[0]['id']
    if allowed is not None and project not in allowed:
        raise ValueError('No access to this project')
    current = next((p for p in projects if p['project_id'] == project), None)
    if current is None:
        raise ValueError(f"Missing active_profile: {project}")
    profile = profile_display(store, project, current['key'], current['version'])
    workflows = sorted([w for w in store.list('workflow') if w['project_id'] == project],
                       key=lambda w: w['started_at'], reverse=True)
    standard = [w for w in workflows if not w.get('experimental')]
    selected = next((w for w in workflows if w['run_id'] == workflow_id), None) if workflow_id else next(iter(standard or workflows), None)
    if workflow_id and selected is None:
        raise ValueError('Workflow does not belong to this project')
    if selected and selected['profile_version'] != profile['active_version']:
        profile = profile_display(store, project, current['key'], current['version'], selected['profile_version'])
    pointers = {p['report_id'] for p in store.select('report_latest', ('report_id',))}
    # Metadata projection: report bodies carry the markdown and are only read by /api/report.
    reports = [r for r in store.select('report', ('report_id', 'date', 'kind', 'profile_version', 'project_id'))
               if r['project_id'] == project and r['report_id'] in pointers]
    reports.sort(key=lambda r: (r['date'], r['kind'] or ''), reverse=True)
    inspection = None
    inspection_path = store.root / 'feishu-inputs' / 'inspection-latest.json'
    if inspection_path.exists():
        try:
            inspection = json.loads(inspection_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            inspection = {'at': None, 'results': [], 'status': 'inspection_file_unavailable'}
    cards, sources = [], []
    if selected and selected.get('pool_id'):
        pool = store.get('pool', selected['pool_id'])
        sources = [{k: s.get(k) for k in ('source', 'status', 'query', 'since', 'until', 'error', 'truncated')}
                   | {'count': len(s['papers'])} for s in pool['sources']]
        from .workflow import latest_papers
        papers = latest_papers(pool)
        for raw in selected.get('recommendations', []):
            _, paper, rec, reading = workflow_paper(store, selected['run_id'], raw['paper_id'],
                                                    workflow=selected, pool=pool, papers=papers)
            degraded = False
            if reading.get('status') == 'completed' and reading.get('analysis_id'):
                try:
                    degraded = bool(store.get('analysis', reading['analysis_id']).get('degraded_output'))
                except (OSError, ValueError):
                    degraded = False
            cards.append({'paper': paper.model_dump(), 'recommendation': rec, 'reading': reading,
                          'reading_status': reading.get('status', 'not_requested'),
                          'degraded': degraded})
    return {'projects': choices, 'project': project,
        'profile': profile,
        'workflows': [{k: w.get(k) for k in ('run_id', 'date', 'status', 'phase', 'profile_version', 'started_at', 'experimental')}
                      for w in workflows[:30]],
        'workflow': {k: v for k, v in selected.items() if k not in ('synthesis', 'recommendations')} if selected else None,
        'reports': [{k: r.get(k) for k in ('report_id', 'date', 'kind', 'profile_version')} for r in reports[:30]],
        'cards': cards, 'sources': sources, 'input_inspection': inspection,
        'feedback': [e for e in store.list('feedback')
                     if e['project_id'] == project
                     and (only_user is None or e.get('user_id') == only_user)],
        'policy': {'learning_enabled': False, 'feishu_generated_upload_enabled': False}}


def bind_address():
    return os.environ.get('NEUROEI_BIND', '127.0.0.1').strip() or '127.0.0.1'


def extra_hosts():
    raw = os.environ.get('NEUROEI_ALLOWED_HOSTS', '')
    return {item.strip().lower() for item in raw.split(',') if item.strip()}


CSRF_TOKEN_TTL_SECONDS = 7 * 24 * 3600
CSRF_TOKEN_HISTORY = 32
_CSRF_LOCK = threading.Lock()


def csrf_tokens(root, current):
    """Keep recent page tokens across restarts so open tabs keep working.

    The token is only a CSRF nonce: every request still needs a valid session
    cookie. Persisting a short history therefore does not grant access, but it
    stops a portal restart from turning every already-open tab into 403.
    """
    path = root / "csrf-tokens.json"
    now_ts = time.time()
    with _CSRF_LOCK:
        items = []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                known = item.get("token")
                at = item.get("at")
                if (isinstance(known, str) and isinstance(at, (int, float))
                        and now_ts - at <= CSRF_TOKEN_TTL_SECONDS):
                    items.append({"token": known, "at": at})
        if not any(secrets.compare_digest(item["token"], current) for item in items):
            items.append({"token": current, "at": now_ts})
        items = items[-CSRF_TOKEN_HISTORY:]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(items), encoding="utf-8")
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
        except OSError:
            pass
        return {item["token"] for item in items}


def make_server(store, port=8768, *, legacy_execution_enabled=True, desktop=None):
    import logging
    import smtplib

    from . import account_email

    logger = logging.getLogger(__name__)
    try:
        mailer = None if desktop else account_email.SMTPMailer.configured()
    except ValueError:
        logger.error('Account email settings are invalid')
        mailer = None
    account_paths = {'/account/register', '/account/forgot', '/account/reset', '/account/bind'}
    mail_paths = {f'/api/account/{action}/{step}': (action, step)
                  for action in ('register', 'reset', 'bind') for step in ('email', 'complete')}
    token = secrets.token_urlsafe(32)
    accepted_tokens = csrf_tokens(store.root, token)
    page = files('neuroei').joinpath('templates/portal.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
    page = page.replace('__LEGACY_EXECUTION__', json.dumps(legacy_execution_enabled))
    if not legacy_execution_enabled:
        page = page.replace('<body>', '<body><aside role="note" style="padding:12px 20px;background:#edf4f2">旧版报告生成已停用，历史资料仍可查看。<a href="/">返回文献研究工作台</a></aside>', 1)
    login_page = files('neuroei').joinpath('templates/login.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
    login_page = login_page.replace('__ACCOUNT_EMAIL_ENABLED__', json.dumps(mailer is not None))
    if desktop:
        login_page = files('neuroei').joinpath('templates/desktop_open.html').read_text(encoding='utf-8').replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        throttle = auth.LoginThrottle()
        bind_throttle = auth.LoginThrottle()

        def log_message(self, *_):
            pass

        def reply(self, status, body, content_type='application/json; charset=utf-8', download=False, cookies=()):
            encoded = (json.dumps(body, ensure_ascii=False) if content_type.startswith('application/json') else body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(encoded)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'nonce-"+token+"'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            for value in cookies:
                self.send_header('Set-Cookie', value)
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="neuroei-report.md"')
            self.end_headers()
            self.wfile.write(encoded)

        def redirect(self, location):
            self.send_response(303)
            self.send_header('Location', location)
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def host_allowed(self):
            host = (self.headers.get('Host') or '').strip().lower()
            if desktop:
                return host == desktop.host
            if host in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                return True
            return host in extra_hosts()

        def origin_allowed(self):
            origin = (self.headers.get('Origin') or '').strip().lower()
            if desktop:
                return origin == desktop.origin
            if origin in (f'http://127.0.0.1:{self.server.server_port}',
                          f'http://localhost:{self.server.server_port}'):
                return True
            return any(origin in (f'http://{host}', f'https://{host}') for host in extra_hosts())

        def session_token(self):
            jar = {}
            for part in (self.headers.get('Cookie') or '').split(';'):
                if '=' in part:
                    key, value = part.split('=', 1)
                    jar[key.strip()] = value.strip()
            return jar.get(auth.SESSION_COOKIE, '')

        def account(self):
            return auth.resolve(store, self.session_token())

        def account_payload(self, account):
            return {'user_id': account['user_id'], 'username': account['username'],
                    'display_name': account['display_name'], 'role': account['role']}

        def unauthorized(self):
            message = '请从启动器重新打开研究台。' if desktop else '登录已过期，请重新登录。'
            self.reply(401, {'error': message, 'login': True})

        def session_cookie(self, value, max_age):
            secure = '; Secure' if os.environ.get('NEUROEI_COOKIE_SECURE') == '1' else ''
            return f'{auth.SESSION_COOKIE}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}{secure}'

        def project_ids(self, account):
            deleted = deleted_scopes(store)
            ids = list(dict.fromkeys([p['project_id'] for p in active_projects(store)]
                                     + [p['project_id'] for p in store.list('research_project')
                                        if not scope_deleted(p, deleted)]))
            allowed = auth.projects_for(store, account)
            return ids if allowed is None else [i for i in ids if i in allowed]

        def require_project(self, account, project):
            require_active(store, project_id=project)
            allowed = auth.projects_for(store, account)
            if allowed is not None and project not in allowed:
                raise PermissionError('No access to this project')

        def reject_post(self, status, message, code=None):
            # Consume a bounded rejected body before closing the socket. On Windows an
            # unread request body can reset the connection before the client sees 403.
            previous_timeout = self.connection.gettimeout()
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if 0 < size <= 700_000:
                    self.connection.settimeout(2)
                    self.rfile.read(size)
            except (ValueError, OSError):
                pass
            finally:
                self.connection.settimeout(previous_timeout)
            body = {'error': message}
            if code is not None:
                body['code'] = code
            self.reply(status, body)

        def token_ok(self):
            supplied = self.headers.get('X-Review-Token') or ''
            return any(secrets.compare_digest(supplied, known) for known in accepted_tokens)

        def audit_action(self):
            return _ADMIN_ACTIONS.get(self.path, _SELF_PASSWORD_ACTION)

        def admin_target(self, body):
            """The acted-on username only; never echo a request body into the audit log."""
            if self.path == '/api/admin/model-backend':
                return 'global'
            if not isinstance(body, dict):
                return ''
            user_id = body.get('user_id')
            if isinstance(user_id, str) and user_id:
                target = auth.find(store, user_id)
                if target is not None:
                    return target['username']
            username = body.get('username')
            return username.strip()[:64] if isinstance(username, str) else ''

        def create_account(self, body):
            # Validate the raw username: uppercase must be rejected, not silently folded.
            username = body.get('username')
            login = username.strip() if isinstance(username, str) else ''
            if not isinstance(username, str) or username != login or not auth._USERNAME.match(login):
                raise PortalError(INVALID_USERNAME_ERROR, 'invalid_username')
            password = body.get('password')
            if not isinstance(password, str) or len(password) < auth._MIN_PASSWORD:
                raise PortalError(WEAK_PASSWORD_ERROR, 'weak_password')
            if len(password) > 200:
                raise PortalError('密码过长。', 'weak_password')
            role = body.get('role', 'member')
            if role not in ('admin', 'member'):
                raise PortalError('角色只能是成员或管理员。', 'invalid_role')
            display_name = body.get('display_name')
            name = display_name.strip() if isinstance(display_name, str) else ''
            if not 1 <= len(name) <= 100:
                raise PortalError('显示名需为 1–100 个字符。', 'invalid_display_name')
            try:
                return auth.create(store, login, name, password, role)
            except ValueError as exc:
                raise PortalError(*account_create_error(str(exc))) from exc

        def reset_password(self, target, body):
            try:
                auth.set_password(store, target['user_id'], body.get('password'))
            except ValueError as exc:
                message = str(exc)
                if message.startswith('密码至少需要'):
                    raise PortalError(WEAK_NEW_PASSWORD_ERROR, 'weak_password') from exc
                if message.startswith('密码过长'):
                    raise PortalError('密码过长。', 'weak_password') from exc
                if message.startswith('账号不存在'):
                    raise PortalError('账号不存在。', 'account_not_found') from exc
                raise PortalError(message, 'invalid_password') from exc

        def change_own_password(self, account, body):
            try:
                auth.change_password(store, account['user_id'], body.get('current_password'),
                                     body.get('new_password'))
            except ValueError as exc:
                message = str(exc)
                if message.startswith('当前密码不正确'):
                    raise PortalError('当前密码不正确。', 'wrong_password') from exc
                if message.startswith('密码至少需要'):
                    raise PortalError(WEAK_NEW_PASSWORD_ERROR, 'weak_password') from exc
                if message.startswith('密码过长'):
                    raise PortalError('密码过长。', 'weak_password') from exc
                if message.startswith('账号不存在'):
                    raise PortalError('账号不存在。', 'account_not_found') from exc
                raise PortalError(message, 'invalid_password') from exc
            audit_admin(store, _SELF_PASSWORD_ACTION, account['username'], account['username'])
            self.reply(200, {'changed': True, 'relogin': True}, cookies=[self.session_cookie('', 0)])

        def admin_action(self, account, body):
            action = self.audit_action()
            if account['role'] != 'admin':
                audit_admin(store, action, account['username'], self.admin_target(body),
                            'denied', reason='not_admin')
                raise PermissionError('Administrator account required')
            if self.path == '/api/admin/model-backend':
                from . import ai_backend
                from .llm import ModelUnavailable

                backend = body.get('backend')
                if set(body) != {'backend'} or backend not in ('sol-high', 'deepseek'):
                    raise PortalError('请选择 SOL 或 DeepSeek。', 'invalid_model_backend')
                try:
                    ai_backend.set_selected(backend)
                except (ModelUnavailable, OSError) as exc:
                    missing_key = getattr(exc, 'reason', None) == 'model_not_configured'
                    message = 'DeepSeek API 尚未配置。' if missing_key else '模型设置暂不可保存，请稍后重试。'
                    audit_admin(store, action, account['username'], 'global', 'error',
                                reason='model_not_configured' if missing_key else 'model_backend_unavailable')
                    self.reject_post(503, message, code='model_backend_unavailable')
                    return
                audit_admin(store, action, account['username'], 'global', detail={'backend': backend})
                self.reply(200, {'backend': backend})
                return
            if self.path == '/api/admin/account':
                created = self.create_account(body)
                audit_admin(store, action, account['username'], created['username'],
                            detail={'role': created['role']})
                self.reply(200, {'account': self.account_payload(created)})
                return
            target = auth.find(store, body.get('user_id'))
            if target is None:
                raise PortalError('账号不存在。', 'account_not_found')
            if self.path == '/api/admin/password':
                self.reset_password(target, body)
                audit_admin(store, action, account['username'], target['username'])
                self.reply(200, {'changed': True})
                return
            if self.path == '/api/admin/status':
                updated = auth.set_disabled(store, target['user_id'], body.get('disabled'))
                audit_admin(store, action, account['username'], updated['username'],
                            detail={'enabled': not updated['disabled']})
                self.reply(200, {'disabled': updated['disabled']})
                return
            project = str(body.get('project_id') or '')
            if project not in self.project_ids(account):
                audit_admin(store, action, account['username'], target['username'],
                            'denied', reason='project_not_found')
                self.reject_post(400, '项目不存在。', code='project_not_found')
                return
            granted = bool(body.get('granted'))
            if granted:
                auth.grant(store, project, target['user_id'])
            else:
                auth.revoke(store, project, target['user_id'])
            audit_admin(store, action, account['username'], target['username'],
                        detail={'project_id': project, 'granted': granted})
            self.reply(200, {'granted': granted})

        def do_GET(self):
            if not self.host_allowed():
                self.reply(403, {'error': 'Use this server address'})
                return
            parsed = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            account = self.account()
            try:
                if desktop and parsed.path == '/api/desktop/instance':
                    self.reply(200, {'instance': desktop.instance()})
                    return
                if desktop and parsed.path in {'/desktop/open', '/login'}:
                    if account and parsed.path == '/login':
                        self.redirect('/')
                    else:
                        self.reply(200, login_page, 'text/html; charset=utf-8')
                    return
                if desktop and parsed.path in account_paths:
                    self.reply(404, {'error': '本机版仅使用当前 Windows 用户的资料目录。'})
                    return
                if parsed.path == '/oauth/feishu/callback':
                    if account is None:
                        self.reply(401, '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>飞书授权</title>'
                                   '<p>登录状态已过期，请重新登录后再次发起飞书授权。</p><a href="/login">返回登录</a>',
                                   'text/html; charset=utf-8')
                        return
                    from .feishu_user_auth import FeishuOAuthError, complete

                    try:
                        complete(store, account, self.session_token(), query.get('state', ''), query.get('code', ''),
                                 error=query.get('error'))
                    except (FeishuOAuthError, PermissionError) as exc:
                        message = str(exc) if isinstance(exc, FeishuOAuthError) else '账号状态已变更，请重新登录后再试。'
                        logger.warning('Feishu OAuth callback failed: %s', message)
                        self.reply(400, '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>飞书授权</title>'
                                   f'<h1>飞书授权未完成</h1><p>{escape(message)}</p>'
                                   '<a href="/?feishu=incomplete#today">返回研究台</a>',
                                   'text/html; charset=utf-8')
                    else:
                        self.redirect('/?feishu=connected#today')
                    return
                if parsed.path == '/login':
                    self.reply(200, login_page, 'text/html; charset=utf-8') if not account else self.redirect('/')
                    return
                if parsed.path in account_paths:
                    if parsed.path == '/account/bind' and not account:
                        self.redirect('/login?next=/account/bind')
                    else:
                        self.reply(200, login_page, 'text/html; charset=utf-8')
                    return
                if parsed.path == '/':
                    if not account:
                        self.redirect('/login')
                    else:
                        from pathlib import Path
                        home = Path(__file__).with_name('templates').joinpath('recommendations.html').read_text(encoding='utf-8')
                        account_links = ''
                        if desktop:
                            account_links += '<a href="/desktop/settings">模型设置</a>'
                            home = home.replace('>退出登录</button>', '>锁定研究台</button>')
                            home = home.replace('的完整 HTTPS 回调地址', '的完整本机回调地址')
                        if mailer:
                            account_links += '<a href="/account/bind">绑定邮箱</a>'
                        if account['role'] == 'admin':
                            account_links += '<a href="/legacy#members">管理设置</a>'
                        if account_links:
                            home = home.replace('id="account"></span>', 'id="account"></span>' + account_links)
                        self.reply(200, home.replace('__TOKEN__', token), 'text/html; charset=utf-8')
                    return
                if account is None:
                    if desktop and parsed.path == '/desktop/settings':
                        self.redirect('/login')
                        return
                    self.unauthorized()
                    return
                user = account['display_name']
                if desktop and parsed.path == '/desktop/settings':
                    settings = files('neuroei').joinpath('templates/desktop_settings.html').read_text(encoding='utf-8')
                    self.reply(200, settings.replace('__TOKEN__', token), 'text/html; charset=utf-8')
                elif desktop and parsed.path == '/api/desktop/models':
                    self.reply(200, {**desktop.model_settings(), 'docling_installed': desktop.docling_paths() is not None,
                                     'reader_installed': desktop.reader_paths() is not None})
                elif desktop and parsed.path == '/api/desktop/codex/models':
                    from .desktop_models import codex_catalog

                    self.reply(200, codex_catalog(desktop.profile))
                elif parsed.path == '/legacy':
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif parsed.path == '/api/research-workspace':
                    from .research_workspace import workspace_state
                    self.reply(200, workspace_state(store, account))
                elif parsed.path == '/api/research-project':
                    from .research_workspace import editable_project, project_for
                    self.reply(200, editable_project(project_for(store, account, query.get('project_id', '')), account))
                elif parsed.path == '/api/delete-challenge-image':
                    from .scope_deletion import challenge_image

                    self.reply(200, challenge_image(store, account, self.session_token(), query.get('challenge_id')),
                               'image/svg+xml; charset=utf-8')
                elif parsed.path == '/api/research-feishu-folder':
                    from .feishu_folders import FolderUnavailable, browse
                    from .feishu_user_auth import FeishuAuthorizationRequired

                    try:
                        self.reply(200, browse(store, account, query.get('url', '')))
                    except FeishuAuthorizationRequired as exc:
                        self.reply(409, {'error': str(exc), 'code': 'feishu_authorization_required'})
                    except FolderUnavailable as exc:
                        self.reply(400, {'error': str(exc)})
                elif parsed.path == '/api/research-journal-preview':
                    from .feishu_folders import ManualSelectionError, manual_preview
                    from .research_workspace import project_for

                    project = project_for(store, account, query.get('project_id', ''))
                    if project.get('owner_id') != account['user_id']:
                        raise PermissionError('只有项目所有者可以选择日报段落。')
                    try:
                        self.reply(200, manual_preview(store, account, project.get('feishu'),
                                                       query.get('source_id', ''),
                                                       offset=query.get('offset', 0),
                                                       limit=query.get('limit', 200)))
                    except ManualSelectionError as exc:
                        self.reply(400, {'error': str(exc)})
                elif parsed.path == '/api/recommendation-reading':
                    from .recommendation_feedback import reading_state
                    self.reply(200, reading_state(store, account, auth.projects_for(store, account),
                                                  query.get('event', ''), query.get('paper', '')))
                elif parsed.path == '/api/recommendation-events':
                    from .recommendation_events import member_event, visible_events
                    events = visible_events(store, user_id=account['user_id'], projects=auth.projects_for(store, account))
                    self.reply(200, {'events': [member_event(event) for event in events],
                                     'modules_connected': True, 'live_triggers_enabled': False})
                elif parsed.path == '/api/recommendation-deliveries':
                    from .recommendation_handoff import initial_review_questions

                    allowed = auth.projects_for(store, account)
                    deleted = deleted_scopes(store)
                    rows = [row for row in store.list('recommendation_delivery')
                            if row['user_id'] == account['user_id'] and not scope_deleted(row, deleted)
                            and (allowed is None or row['project_id'] in allowed)]
                    rows = [{**row, 'suggested_question_ids': initial_review_questions(store, row)}
                            if row['status'] == 'awaiting_review' else row for row in rows]
                    self.reply(200, {'deliveries': rows})
                elif parsed.path == '/api/recommendation-jobs':
                    from .recommendation_jobs import (
                        jobs_with_progress,
                        member_jobs,
                        visible_jobs,
                        worker_status,
                    )
                    worker = worker_status(store)
                    if desktop:
                        worker = {**worker, 'desktop': True, 'model_configured': desktop.model_configured()}
                    jobs = jobs_with_progress(store, visible_jobs(store, account), worker_ready=worker['status'] == 'ready')
                    capabilities = ('status', 'paid_model_calls_enabled', 'fulltext_download_enabled', 'understanding_enabled')
                    if desktop:
                        capabilities += ('desktop', 'model_configured', 'recommendations_enabled')
                    self.reply(200, {'jobs': member_jobs(store, jobs, worker),
                                     'worker': {key: worker[key] for key in capabilities if key in worker}})
                elif parsed.path == '/api/source-capabilities':
                    from .recommendation_jobs import worker_status
                    from .wechat_accounts import WECHAT_ACCOUNTS
                    worker = worker_status(store)
                    automatic_pdf = worker['status'] == 'ready' and worker.get('fulltext_download_enabled', False)
                    official_enabled = worker['status'] == 'ready' and worker.get('official_sources_enabled', False)
                    fallback_enabled = worker['status'] == 'ready' and worker.get('web_fallback_enabled', False)
                    if desktop:
                        fallback_enabled = fallback_enabled and desktop.model_settings()['backend'] == 'deepseek' and desktop.model_configured()
                    journals_enabled = worker['status'] == 'ready' and worker.get('journal_sources_enabled', False)
                    open_pdf = automatic_pdf and worker.get('open_access_fulltexts_enabled', False)
                    self.reply(200, {'sources': [
                        {'name': 'arXiv', 'type': '预印本索引', 'discovery': True,
                         'abstract': '原始摘要', 'fulltext': ('已配置按论文版本获取官方PDF，实际结果见阅读记录'
                                                          if automatic_pdf else '提供官方候选链接，自动获取未启用')},
                        {'name': 'Crossref', 'type': '出版元数据', 'discovery': True,
                         'abstract': '可能缺失', 'fulltext': '仅落地页，不代表全文访问权限'},
                        {'name': 'PubMed Central 开放仓储', 'type': '全文补充', 'discovery': False,
                         'abstract': '按已发现论文的 DOI 核对身份',
                         'fulltext': '可尝试获取经核实的开放获取正式论文 PDF' if open_pdf else '自动读取尚未启用'},
                        {'name': 'PubMed 期刊摘要补查', 'type': '官网访问失败时补充', 'discovery': journals_enabled,
                         'abstract': '按已知 DOI 核对期刊、论文及电子发表日期后读取摘要',
                         'fulltext': '摘要入口，全文可用性以阅读记录为准'},
                        *({'name': name, 'type': '官方论文集', 'discovery': official_enabled,
                           'abstract': '有界检索指定年份／卷，读取详情页原始摘要',
                           'fulltext': '仅在官方文档身份核验后读取；实际取得范围见阅读记录'
                               if automatic_pdf and official_enabled else '提供官方候选链接，自动读取尚未启用'}
                          for name in ('RSS 机器人会议', 'PMLR／ICML', 'NeurIPS')),
                        *({'name': name, 'type': '期刊官方来源', 'discovery': journals_enabled,
                           'abstract': '读取可取得的论文原始摘要，缺失信息保留未知',
                           'fulltext': '核实官方文档身份后读取可取得的PDF；实际结果见阅读记录'
                               if automatic_pdf and journals_enabled else '自动读取尚未启用'}
                          for name in ('Nature', 'Science Robotics', 'Cell')),
                        {'name': '其他期刊订阅源' if journals_enabled else '期刊订阅源', 'type': '订阅源', 'discovery': False,
                         'abstract': '摘要与日期依据未充分核实，暂未启用', 'fulltext': '未获取'},
                        {'name': 'DeepSeek 备用检索', 'type': '主来源读取失败时补充', 'discovery': fallback_enabled,
                         'abstract': '从可取得的原始页面提取；缺失信息保留未知',
                         'fulltext': '提供原始页面线索；实际全文范围见论文阅读记录'},
                        {'name': '公众号文章导入', 'type': '用户授权资料', 'discovery': True,
                         'abstract': '仅解析明确论文标识，不把文章当论文摘要',
                         'fulltext': '手动导入可访问的文章正文'}], 'wechat_accounts': WECHAT_ACCOUNTS,
                        'wechat_direction_search_enabled': fallback_enabled and worker.get('wechat_sources_enabled', False)})
                elif parsed.path == '/api/me':
                    projects = self.project_ids(account)
                    current = {key for key,_ in store.items('research_project')}
                    removable = [p for p in projects if p not in current] if account['role'] == 'member' else []
                    from .daily_scope_backend import settings
                    self.reply(200, {'user': self.account_payload(account), 'projects': projects,
                                     'removable_legacy_projects': removable,
                                     'daily_scope_settings': None if desktop else settings(store, account['user_id'])})
                elif parsed.path == '/api/state':
                    self.reply(200, portal_state(store, query.get('project'), query.get('workflow'),
                                                  allowed=auth.projects_for(store, account), only_user=user))
                elif parsed.path == '/api/admin/accounts':
                    if account['role'] != 'admin':
                        raise PermissionError('Administrator account required')
                    rows = []
                    for item in auth.accounts(store):
                        rows.append({**self.account_payload(item), 'disabled': item['disabled'],
                                     'created_at': item['created_at'],
                                     'projects': sorted(r['project_id'] for r in auth.access_rows(store)
                                                        if r['user_id'] == item['user_id'])})
                    self.reply(200, {'accounts': rows, 'projects': self.project_ids(account),
                                     'current': account['user_id']})
                elif parsed.path == '/api/admin/model-backend':
                    if account['role'] != 'admin':
                        raise PermissionError('Administrator account required')
                    from . import ai_backend
                    from .llm import ModelUnavailable

                    try:
                        self.reply(200, {'backend': ai_backend.selected()})
                    except ModelUnavailable:
                        self.reply(503, {'error': '暂时无法读取模型设置。'})
                elif parsed.path == '/api/directions':
                    data = direction_state(store, user, auth.projects_for(store, account))
                    from .recommendation_pipeline import load_query_plan
                    for setting in data['directions']:
                        plan = load_query_plan(store, setting, account)
                        if plan:
                            setting['query_question_links'] = plan['query_question_links']
                            setting['query_plan_version'] = plan['version']
                    self.reply(200, data)
                elif parsed.path == '/api/direction-feedback':
                    from .direction_feedback import direction_feedback_state
                    self.reply(200, direction_feedback_state(store, user, query['direction'],
                        run_id=query.get('run'), report_id=query.get('report')))
                elif parsed.path == '/api/direction-weekly':
                    from .direction_weekly import weekly_freshness
                    report = store.get('direction_weekly', query['id'])
                    load_direction(store, user, report['direction_id'])
                    if report['user_id'] != user:
                        raise ValueError('Direction weekly owner mismatch')
                    if report.get('project_context'):
                        self.require_project(account, report['project_context']['project_id'])
                    self.reply(200, {**report, 'freshness': weekly_freshness(store, report)})
                elif parsed.path == '/api/media':
                    from .media import media_state
                    self.reply(200, media_state(store, user, query['direction']))
                elif parsed.path == '/api/media-item':
                    from .media import load_media
                    self.reply(200, load_media(store, user, query['direction'], query['id']))
                elif parsed.path == '/api/media-report':
                    from .media import media_snapshot, render_media_digest
                    snapshot = media_snapshot(store, user, query['direction'])
                    self.reply(200, {'content': render_media_digest(snapshot),
                                     'captured_at': snapshot['captured_at'], 'policy': snapshot['policy']})
                elif parsed.path == '/api/media-links':
                    from .media_links import analyze_media, render_media_links
                    analysis = analyze_media(store, user, query['direction'])
                    self.reply(200, {'content': render_media_links(analysis),
                                     'captured_at': analysis['captured_at'],
                                     'pool_id': analysis['pool_id'], 'counts': analysis['counts'],
                                     'policy': analysis['policy']})
                elif parsed.path == '/api/direction-report':
                    from .directions import optional
                    run = optional(store, 'direction_run', query['run'])
                    key = (run or {}).get('report_id') or query['run']
                    report = store.get('direction_report', key)
                    load_direction(store, user, report['direction_id'])
                    if report['user_id'] != user:
                        raise ValueError('Direction report owner mismatch')
                    if report.get('project_context'):
                        self.require_project(account, report['project_context']['project_id'])
                    self.reply(200, {**report, 'report_id': report.get('report_id') or key})
                elif parsed.path == '/api/analysis':
                    workflow = store.get('workflow', query['workflow'])
                    self.require_project(account, workflow['project_id'])
                    self.reply(200, paper_analysis(store, query['workflow'], query['paper']))
                elif parsed.path == '/api/project-source':
                    self.require_project(account, query['project'])
                    profile = store.get('profile', f"{query['project']}:{query['version']}")
                    block = next((b for b in profile['blocks'] if b['block_id'] == query['block']), None)
                    if block is None:
                        raise ValueError('Missing profile source block')
                    self.reply(200, block)
                elif parsed.path == '/api/report':
                    self.require_project(account, query['project'])
                    report = store.get('report', query['id'])
                    if report['project_id'] != query['project']:
                        raise ValueError('Report project mismatch')
                    if query.get('download') == '1':
                        self.reply(200, report['content'], 'text/markdown; charset=utf-8', download=True)
                    else:
                        self.reply(200, {k: report.get(k) for k in ('report_id', 'date', 'kind', 'content', 'profile_version')})
                else:
                    self.reply(404, {'error': 'Unknown review page'})
            except PermissionError:
                self.reply(403, {'error': '这个账号没有访问权限。'})
            except (ValueError, KeyError) as exc:
                if self.path.startswith('/api/research-journal-preview') and isinstance(exc, ValueError):
                    self.reply(400, {'error': str(exc)})
                else:
                    self.reply(400, {'error': '无法读取此版本，请刷新页面或检查运行记录。'})

        def do_POST(self):
            if not self.host_allowed():
                self.reject_post(403, HOST_REJECTED_ERROR, code='host_rejected')
                return
            if not self.origin_allowed():
                self.reject_post(403, ORIGIN_REJECTED_ERROR, code='origin_rejected')
                return
            if not desktop and self.path.startswith('/api/desktop/'):
                self.reject_post(404, 'Unknown review action')
                return
            if desktop and self.path == '/api/desktop/open':
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 256:
                        raise ValueError
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict) or set(body) != {'token'} or not desktop.redeem(body['token']):
                        raise ValueError
                except (OSError, ValueError):
                    self.reply(401, {'error': '本次打开链接已失效。请双击启动器重新打开研究台。'})
                    return
                owner = auth.by_login(store, 'local-researcher')
                session = auth.create_session(store, owner)
                self.reply(200, {'opened': True}, cookies=[self.session_cookie(session, auth.SESSION_SECONDS)])
                return
            if desktop and (self.path in mail_paths or self.path in {'/api/login', '/api/password'}):
                self.reject_post(404, '本机版使用启动器打开研究台。')
                return
            # Login has no session yet, so its token gate comes first. Every other
            # action resolves the session before the token, so an old page whose
            # token died with the server is told to refresh instead of being
            # mistaken for a cross-site request.
            if self.path == '/api/login' or self.path in mail_paths:
                if not self.token_ok():
                    self.reject_post(403, STALE_TOKEN_ERROR, code='stale_token')
                    return
                if self.path == '/api/login':
                    self.handle_login()
                else:
                    self.handle_account_email(*mail_paths[self.path])
                return
            account = self.account()
            if account is None:
                self.reject_post(401, '登录已过期，请重新登录。')
                return
            if not self.token_ok():
                self.reject_post(403, STALE_TOKEN_ERROR, code='stale_token')
                return
            if self.path not in ('/api/daily-scope-model', '/api/logout', '/api/password', '/api/feedback', '/api/direction', '/api/direction-suggestions',
                                 '/api/direction-run', '/api/direction-feedback', '/api/media',
                                 '/api/media-collect', '/api/media-collect-article', '/api/media-browser-import',
                                 '/api/recommendation-feedback', '/api/recommendation-job', '/api/recommendation-job-cancel', '/api/research-project', '/api/research-project-edit', '/api/research-journal',
                                 '/api/delete-challenge', '/api/delete-scope',
                                 '/api/research-project-feishu', '/api/research-project-automation', '/api/research-project-brief', '/api/feishu-oauth/start', '/api/feishu-oauth/disconnect',
                                 '/api/admin/account', '/api/admin/password', '/api/admin/status', '/api/admin/access',
                                 '/api/admin/model-backend', '/api/desktop/models', '/api/desktop/models/test',
                                 '/api/desktop/codex/login'):
                self.reject_post(404, 'Unknown review action')
                return
            if self.path == '/api/direction-run' and not legacy_execution_enabled:
                self.reject_post(409, '旧版报告生成已停用，请在关注方向发起推荐。', code='legacy_execution_disabled')
                return
            body = {}
            try:
                size = int(self.headers.get('Content-Length', '0'))
                limit = 700_000 if self.path in ('/api/media', '/api/media-browser-import', '/api/research-project',
                                                  '/api/research-project-edit', '/api/research-journal') else 20_000
                if self.path == '/api/research-project-brief':
                    limit = 80_000
                if size == 0:
                    body = {}
                elif not 0 < size <= limit:
                    raise ValueError('Request size')
                else:
                    body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise TypeError('Object request required')
                user = account['display_name']
                if desktop and self.path == '/api/desktop/codex/login':
                    from .desktop_models import start_codex_login

                    if body:
                        self.reply(400, {'error': '请在 Codex 官方页面完成登录。'})
                    elif any(row['status'] == 'processing' for row in store.select('recommendation_job', ('status',))):
                        self.reply(409, {'error': '研究任务正在运行，请在任务结束后登录或切换 Codex 账号。'})
                    else:
                        self.reply(200, start_codex_login(desktop.profile))
                elif desktop and self.path in {'/api/desktop/models', '/api/desktop/models/test'}:
                    from . import desktop_models

                    try:
                        if self.path.endswith('/test'):
                            if set(body) != {'backend'}:
                                raise ValueError('请选择要测试的模型。')
                            result = desktop_models.probe(desktop.profile, body['backend'])
                        else:
                            result = desktop_models.save(desktop.profile, body)
                    except ValueError as exc:
                        self.reply(400, {'error': str(exc)})
                        return
                    self.reply(200, result)
                elif self.path == '/api/logout':
                    auth.revoke_session(store, self.session_token())
                    self.reply(200, {'signed_out': True}, cookies=[self.session_cookie('', 0)])
                elif self.path == '/api/daily-scope-model':
                    from .daily_scope_backend import save_preference
                    self.reply(200, save_preference(store, account['user_id'], body))
                elif self.path == '/api/password':
                    self.change_own_password(account, body)
                elif self.path == '/api/feishu-oauth/start':
                    from .feishu_user_auth import begin

                    if body and (set(body) != {'app_id', 'app_secret'} or not all(isinstance(v, str) for v in body.values())):
                        raise ValueError('请同时填写飞书应用 App ID 和 App Secret。')
                    self.reply(200, {'authorize_url': begin(store, account, self.session_token(), **body)})
                elif self.path == '/api/feishu-oauth/disconnect':
                    from .feishu_folders import invalidate_previews
                    from .feishu_user_auth import disconnect

                    if body:
                        raise ValueError('断开飞书授权请求不接受额外参数。')
                    disconnect(store, account['user_id'])
                    invalidate_previews(account['user_id'])
                    self.reply(200, {'connected': False, 'app_configured': False})
                elif self.path.startswith('/api/admin/'):
                    self.admin_action(account, body)
                elif self.path == '/api/research-project':
                    from .research_workspace import create_project
                    self.reply(200, create_project(store, account, body))
                elif self.path == '/api/research-project-edit':
                    from .research_workspace import ProjectChanged, update_project
                    try:
                        self.reply(200, update_project(store, account, body))
                    except ProjectChanged as exc:
                        self.reply(409, {'error': str(exc)})
                elif self.path == '/api/research-project-brief':
                    from .research_workspace import ProjectChanged, update_project_brief
                    try:
                        self.reply(200, update_project_brief(store, account, body))
                    except ProjectChanged as exc:
                        self.reply(409, {'error': str(exc)})
                elif self.path == '/api/delete-challenge':
                    from .scope_deletion import create_challenge

                    self.reply(200, create_challenge(store, account, self.session_token(), **body))
                elif self.path == '/api/delete-scope':
                    from .scope_deletion import delete_scope

                    self.reply(200, delete_scope(store, account, self.session_token(), **body))
                elif self.path == '/api/research-project-feishu':
                    from .research_workspace import update_feishu_source
                    self.reply(200, update_feishu_source(store, account, body))
                elif self.path == '/api/research-project-automation':
                    from .research_workspace import update_project_automation
                    self.reply(200, update_project_automation(store, account, body))
                elif self.path == '/api/research-journal':
                    from .research_workspace import import_journal
                    self.reply(200, import_journal(store, account, body))
                elif self.path == '/api/recommendation-job-cancel':
                    from .recommendation_jobs import cancel
                    try:
                        job = cancel(store, account, body.get('job_id'))
                    except PermissionError:
                        self.reject_post(403, '没有找到可停止的任务。')
                    except ValueError:
                        self.reject_post(409, '这个任务已经结束，不需要停止。')
                    else:
                        self.reply(200, {'job_id': job['job_id'], 'status': job['status']})
                elif self.path == '/api/recommendation-job':
                    from .recommendation_jobs import enqueue, member_job, worker_status
                    if desktop and body.get('auto_email'):
                        self.reply(400, {'error': '本机版自动检测只在研究台保存结果。'})
                        return
                    if desktop and not desktop.model_configured():
                        self.reply(409, {'error': '请先在「模型设置」接通一个模型。'})
                        return
                    if (desktop and body.get('action') not in ('project_understanding', 'project_understanding_retry')
                            and not worker_status(store).get('recommendations_enabled')):
                        self.reply(409, {'error': '请先安装论文阅读组件，并重新启动研究台。'})
                        return
                    self.reply(200, member_job(enqueue(store, account, body)))
                elif self.path == '/api/media':
                    from .media import save_media
                    result = save_media(store, {**body, 'user_id': user})
                    self.reply(200, result)
                elif self.path == '/api/media-browser-import':
                    from .media import import_browser_article
                    result = import_browser_article(store, {**body, 'user_id': user})
                    self.reply(200, result)
                elif self.path == '/api/media-collect-article':
                    from .media import collect_media_article
                    result = collect_media_article(store, {**body, 'user_id': user})
                    self.reply(200, result)
                elif self.path == '/api/media-collect':
                    from .media import collect_media_channel
                    result = collect_media_channel(store, {**body, 'user_id': user})
                    self.reply(200, result)
                elif self.path == '/api/direction':
                    if body.get('project_id'):
                        self.require_project(account, body['project_id'])
                    from .recommendation_pipeline import save_direction_for_account
                    result = save_direction_for_account(store, account, body)
                    self.reply(200, result)
                elif self.path == '/api/direction-suggestions':
                    from .direction_suggestions import suggest_direction
                    from .llm import ModelFailure

                    try:
                        self.reply(200, suggest_direction(store, account, body))
                    except ModelFailure as exc:
                        if exc.reason == 'model_timeout':
                            self.reject_post(504, '生成方向建议超时，请稍后重试或手动填写。',
                                             code='direction_suggestions_timeout')
                        else:
                            self.reject_post(503, '暂时无法生成方向建议，可以重试或手动填写。',
                                             code='direction_suggestions_unavailable')
                elif self.path == '/api/recommendation-feedback':
                    from .recommendation_feedback import ReadingSnapshotChanged, save_reading_feedback
                    try:
                        self.reply(200, save_reading_feedback(store, account, auth.projects_for(store, account), body))
                    except ReadingSnapshotChanged as exc:
                        self.reject_post(409, str(exc), code='reading_snapshot_changed')
                elif self.path == '/api/direction-feedback':
                    from .direction_feedback import save_direction_feedback
                    result = save_direction_feedback(store, {**body, 'user_id': user})
                    self.reply(200, result)
                elif self.path == '/api/direction-run':
                    from .direction_jobs import start_direction_job
                    if set(body) != {'user_id', 'direction_id'}:
                        raise ValueError('Explicit direction and user required')
                    if not all(isinstance(value, str) for value in body.values()):
                        raise TypeError('Explicit direction and user strings required')
                    result = start_direction_job(store, user, body['direction_id'])
                    self.reply(200, result)
                else:
                    result = save_feedback(store, body, user=user, allowed=self.project_ids(account))
                    self.reply(200, result)
            except PortalError as exc:
                if self.path.startswith('/api/admin/'):
                    audit_admin(store, self.audit_action(), account['username'],
                                self.admin_target(body), 'error', reason=exc.code)
                elif self.path == '/api/password':
                    audit_admin(store, _SELF_PASSWORD_ACTION, account['username'],
                                account['username'], 'error', reason=exc.code)
                self.reject_post(400, exc.message, code=exc.code)
            except PermissionError:
                self.reject_post(403, FORBIDDEN_ERROR, code='forbidden')
            except (ValueError, KeyError, TypeError, WorkflowBusy) as exc:
                from .feishu_user_auth import FeishuAuthorizationRequired, FeishuOAuthError

                if isinstance(exc, FeishuOAuthError):
                    code = 'feishu_authorization_required' if isinstance(exc, FeishuAuthorizationRequired) else 'feishu_oauth_failed'
                    self.reject_post(400, str(exc), code=code)
                elif isinstance(exc, QueryPlanningError):
                    self.reject_post(400, str(exc), code='query_planning_required')
                elif isinstance(exc, ValueError) and self.path.startswith('/api/admin/'):
                    audit_admin(store, self.audit_action(), account['username'],
                                self.admin_target(body), 'error', reason='invalid_request')
                    self.reject_post(400, str(exc))
                elif self.path == '/api/password':
                    audit_admin(store, _SELF_PASSWORD_ACTION, account['username'],
                                account['username'], 'error', reason='invalid_request')
                    self.reject_post(400, '密码未更改，请核对当前密码和新密码。', code='invalid_password')
                elif self.path in ('/api/feedback', '/api/direction-feedback', '/api/recommendation-feedback'):
                    self.reject_post(400, '反馈未保存。请核对论文和报告版本；纠错需填写说明。')
                elif self.path == '/api/research-project-edit':
                    self.reject_post(400, '项目未保存，请核对项目名称和研究背景。')
                elif self.path == '/api/research-project-brief':
                    self.reject_post(400, '项目背景未保存，请核对编辑栏目及内容。每栏最多 500 字。')
                elif self.path in ('/api/research-project', '/api/research-journal', '/api/research-project-feishu', '/api/research-project-automation'):
                    from .feishu_folders import FolderUnavailable

                    message = ('飞书日报来源未保存，请核对项目、链接和所选日报。'
                               if self.path == '/api/research-project-feishu' else '资料未保存，请核对名称、正文和日报日期。')
                    self.reject_post(400, str(exc) if isinstance(exc, FolderUnavailable) else message)
                elif self.path == '/api/recommendation-job' and body.get('action') in ('project_understanding', 'project_understanding_retry'):
                    if body.get('analysis_mode') == 'manual' and isinstance(exc, ValueError):
                        self.reject_post(409 if '更新' in str(exc) else 400, str(exc))
                    else:
                        self.reject_post(400, '项目分析未提交，请刷新项目和服务状态后重试。')
                elif self.path == '/api/recommendation-job':
                    self.reject_post(400, '任务未提交，请刷新原推送状态后重试。已执行的恢复不会再次运行。')
                elif self.path == '/api/direction-suggestions':
                    if isinstance(exc, WorkflowBusy):
                        self.reject_post(409, '正在生成方向建议，请稍后再试。', code='direction_suggestions_busy')
                    else:
                        self.reject_post(400, '请填写 1–200 字的方向名称。', code='invalid_direction_title')
                elif self.path in ('/api/delete-challenge', '/api/delete-scope'):
                    message = str(exc) if isinstance(exc, ValueError) else '删除请求不完整，请刷新后重试。'
                    self.reject_post(409 if isinstance(exc, ScopeBusy) else 400, message)
                elif self.path.startswith('/api/media'):
                    self.reject_post(400, '来源操作未完成。请核对所属方向、平台链接、读取范围及带时区日期／字幕时间。')
                else:
                    self.reject_post(400, '方向操作未完成。请刷新设置和运行状态后重试；已暂停的方向不能生成进展。')

        def handle_account_email(self, action, step):
            account = self.account()
            if action == 'bind' and not account:
                self.reject_post(401, '请先登录，再绑定邮箱。')
                return
            if not mailer:
                self.reject_post(503, '邮箱服务尚未启用，请联系实验室管理员。')
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                size = 0
            if not 0 < size <= 4096:
                self.reject_post(400, '请填写完整的账号信息。')
                return
            try:
                body = json.loads(self.rfile.read(size))
                fields = {
                    ('register', 'email'): {'email'}, ('reset', 'email'): {'email'},
                    ('bind', 'email'): {'email', 'password'},
                    ('register', 'complete'): {'token', 'username', 'display_name', 'password'},
                    ('reset', 'complete'): {'token', 'password'}, ('bind', 'complete'): {'token'},
                }[action, step]
                if not isinstance(body, dict) or set(body) != fields or any(
                        not isinstance(v, str) or len(v) > 254 for v in body.values()):
                    raise ValueError('请填写完整的账号信息。')
                if step == 'email':
                    if action == 'bind' and not self.bind_throttle.allowed(account['username']):
                        self.reply(429, {'error': '尝试次数过多，请 15 分钟后再试。'})
                        return
                    try:
                        issued = account_email.issue(store, action, **body, account=account)
                    except account_email.WrongCurrentPassword:
                        self.bind_throttle.record(account['username'], False)
                        raise
                    except account_email.TooManyEmails:
                        if action == 'bind':
                            raise
                        issued = None
                    if action == 'bind':
                        self.bind_throttle.record(account['username'], True)
                    if issued:
                        def send():
                            try:
                                account_email.deliver(mailer, account_email.normalize_email(body['email']), action, *issued)
                            except (OSError, smtplib.SMTPException):
                                logger.warning('Account email delivery failed (%s)', action)

                        threading.Thread(target=send, daemon=True).start()
                    self.reply(202, {'message': account_email.EMAIL_RESPONSE})
                else:
                    account_email.complete(store, action, **body, account=account)
                    self.reply(200, {'done': True})
            except account_email.TooManyEmails as exc:
                self.reply(429, {'error': str(exc)})
            except (json.JSONDecodeError, UnicodeError):
                self.reply(400, {'error': '请检查输入后重试。'})
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})

        def handle_login(self):
            try:
                size = int(self.headers.get('Content-Length', '0') or 0)
            except ValueError:
                size = 0
            if not 0 < size <= 4000:
                self.reject_post(400, '登录信息不完整。')
                return
            try:
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise TypeError('Object request required')
            except (ValueError, TypeError):
                self.reject_post(400, '登录信息不完整。')
                return
            username = str(body.get('username', '')).strip()
            password = str(body.get('password', ''))
            if not self.throttle.allowed(username):
                self.reject_post(429, '尝试次数过多，请 15 分钟后再试，或联系管理员。')
                return
            try:
                account = auth.authenticate(store, username, password)
            except ValueError as exc:
                self.throttle.record(username, False)
                self.reject_post(401, str(exc))
                return
            self.throttle.record(username, True)
            session = auth.create_session(store, account)
            self.reply(200, {'user': self.account_payload(account)},
                       cookies=[self.session_cookie(session, auth.SESSION_SECONDS)])

    # One thread per connection: the page opens several API requests at once (state,
    # me, directions) and a single-threaded server serialised them behind each other.
    # Store keeps one SQLite connection per thread, so concurrent handlers stay safe.
    class PortalServer(ThreadingHTTPServer):
        daemon_threads = True

    return PortalServer(('127.0.0.1' if desktop else bind_address(), port), Handler)


def serve(config, port):
    if not 1 <= port <= 65535:
        raise ValueError('Port must be 1–65535')
    # The ranking layer imports paramiko/invoke and costs ~0.75 s on first use; pay that
    # while starting instead of during the first /api/state request.
    from . import workflow as _workflow  # noqa: F401
    store = Store(config.data_dir)
    if not auth.accounts(store):
        print(json.dumps({'accounts': 0,
                          'next': 'python -m neuroei user create --username admin --name 管理员 --role admin'},
                         ensure_ascii=False), flush=True)
    server = make_server(store, port)
    host = bind_address()
    shown = '127.0.0.1' if host in ('0.0.0.0', '') else host
    print(json.dumps({'review_url': f'http://{shown}:{server.server_port}/',
                      'bind': host,
                      'multi_user': True,
                      'model_calls': 0, 'manual_direction_worker_enabled': True}), flush=True)
    if host not in ('127.0.0.1', 'localhost', '::1'):
        print(json.dumps({'notice': '对外提供服务时建议放在 HTTPS 反向代理后，并设置 NEUROEI_COOKIE_SECURE=1；'
                                    '同时把访问域名写入 NEUROEI_ALLOWED_HOSTS。'}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
