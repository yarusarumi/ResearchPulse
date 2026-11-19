"""Local accounts, sessions and project membership for the shared lab portal.

The portal runs on a lab server, so identity must come from the server side:
passwords are stored as scrypt hashes, sessions are opaque random tokens whose
SHA-256 digests are persisted, and every request resolves the account from the
session cookie before touching research data. Client-supplied names are never
trusted for authorization.
"""
import hashlib
import hmac
import json
import re
import secrets
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta

from .models import now

SESSION_COOKIE = 'neuroei_session'
SESSION_SECONDS = 12 * 60 * 60
_SCRYPT = {'n': 2 ** 14, 'r': 8, 'p': 1, 'dklen': 32}
_USERNAME = re.compile(r'^[a-z0-9][a-z0-9._-]{1,31}$')
_MIN_PASSWORD = 10
_FAILED_LOGIN_LIMIT = 5


def _hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt), **_SCRYPT).hex()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt}${digest}"


def _matches(stored, password):
    try:
        scheme, n, r, p, salt, digest = stored.split('$')
        if scheme != 'scrypt':
            return False
        candidate = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt),
                                   n=int(n), r=int(r), p=int(p), dklen=len(digest) // 2).hex()
    except (AttributeError, ValueError):
        return False
    return hmac.compare_digest(candidate, digest)


def _require_password(password):
    if not isinstance(password, str) or len(password) < _MIN_PASSWORD:
        raise ValueError(f'密码至少需要 {_MIN_PASSWORD} 位')
    if len(password) > 200:
        raise ValueError('密码过长')


def _token_digest(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def accounts(store):
    return sorted(store.list('account'), key=lambda a: (a['role'] != 'admin', a['username']))


def find(store, user_id):
    # Accounts are stored under their user_id, so the primary-key read avoids
    # parsing every account on each authenticated request. The scan is kept as a
    # fallback for records written under a legacy key.
    try:
        return store.get('account', user_id)
    except ValueError:
        return next((a for a in store.list('account') if a['user_id'] == user_id), None)


def by_login(store, username):
    login = (username or '').strip().lower()
    for account in store.list('account'):
        if account['username'] == login:
            return account
    return None


def _name_key(name):
    return ' '.join((name or '').split()).casefold()


def by_display_name(store, name):
    target = (name or '').strip()
    return next((a for a in store.list('account') if a['display_name'] == target), None)


def _optional(store, kind, key):
    try:
        return store.get(kind, key)
    except ValueError:
        return None


def _legacy_owner(store, name):
    """Directions and media saved before accounts existed are keyed by display name."""
    for kind in ('direction_version', 'media_version'):
        for _, body in store.items(kind):
            if body.get('user_id') == name:
                return True
    return False


def create(store, username, display_name, password, role='member', adopt_legacy=False, *, db=None, email=None):
    login = (username or '').strip().lower()
    if not _USERNAME.match(login):
        raise ValueError('用户名需为 2–32 位小写字母、数字、点、下划线或连字符')
    name = (display_name or '').strip()
    if not 1 <= len(name) <= 100:
        raise ValueError('显示名需为 1–100 个字符')
    if role not in ('admin', 'member'):
        raise ValueError('角色只能是管理员或成员')
    _require_password(password)
    account = {'user_id': secrets.token_hex(16), 'username': login, 'display_name': name,
               'password_hash': _hash(password), 'role': role, 'disabled': False,
               'created_at': now()}
    if email:
        account['verified_email'] = email
    with (store.transaction() if db is None else nullcontext(db)) as connection:
        if db is None:
            connection.execute('BEGIN IMMEDIATE')
        if by_login(store, login):
            raise ValueError('这个用户名已经存在')
        if by_display_name(store, name):
            raise ValueError('这个显示名已经被占用')
        if _optional(store, 'reserved_name', _name_key(name)) is not None:
            raise ValueError('这个显示名曾用于已删除的账号，请换一个')
        if not adopt_legacy and _legacy_owner(store, name):
            raise ValueError('这个显示名下有历史方向或来源，请换一个，或让管理员先归档')
        connection.execute('INSERT INTO objects VALUES (?,?,?)',
                           ('account', account['user_id'], json.dumps(account)))
    return account


def set_password(store, user_id, password, *, current=None):
    _require_password(password)
    digest = _hash(password)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        account = find(store, user_id)
        if account is None:
            raise ValueError('账号不存在')
        if current is not None and not _matches(account['password_hash'], current):
            raise ValueError('当前密码不正确')
        replace_password(db, account, digest)


def replace_password(db, account, digest):
    updated = {**account, 'password_hash': digest}
    db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',
               ('account', account['user_id'], json.dumps(updated)))
    db.execute("DELETE FROM objects WHERE kind='session' AND json_extract(body,'$.user_id')=?",
               (account['user_id'],))


def change_password(store, user_id, current, password):
    set_password(store, user_id, password, current=current or '')


def delete(store, user_id):
    from .feishu_user_auth import disconnect

    disconnect(store, user_id)
    account = find(store, user_id)
    if account is not None:
        store.put('reserved_name', _name_key(account['display_name']),
                  {'display_name': account['display_name'], 'user_id': user_id, 'deleted_at': now()})
    for session_id, body in store.items('session'):
        if body.get('user_id') == user_id:
            store.delete('session', session_id)
    for key, body in store.items('project_access'):
        if body.get('user_id') == user_id:
            store.delete('project_access', key)
    return store.delete('account', user_id) > 0


def set_disabled(store, user_id, disabled):
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        account = find(store, user_id)
        if account is None:
            raise ValueError('账号不存在')
        account['disabled'] = bool(disabled)
        db.execute("UPDATE objects SET body=? WHERE kind='account' AND id=?", (json.dumps(account), user_id))
        if disabled:
            from .feishu_user_auth import revoke_account

            db.execute("DELETE FROM objects WHERE kind='session' AND json_extract(body,'$.user_id')=?", (user_id,))
            revoke_account(db, user_id)
    return account


def authenticate(store, username, password):
    account = by_login(store, username)
    if account is None:
        _matches(_hash('placeholder-password-for-timing'), password or '')
        raise ValueError('用户名或密码不正确')
    if not _matches(account['password_hash'], password or ''):
        raise ValueError('用户名或密码不正确')
    if account['disabled']:
        raise ValueError('账号已停用，请联系管理员')
    return account


def create_session(store, account):
    token = secrets.token_urlsafe(32)
    store.put('session', _token_digest(token), {
        'user_id': account['user_id'],
        'credential': _token_digest(account['password_hash']),
        'created_at': now(),
        'expires_at': (datetime.now(UTC) + timedelta(seconds=SESSION_SECONDS)).isoformat()})
    return token


def resolve(store, token):
    if not token or len(token) > 200:
        return None
    try:
        session = store.get('session', _token_digest(token))
    except ValueError:
        return None
    if session['expires_at'] <= now():
        revoke_session(store, token)
        return None
    account = find(store, session['user_id'])
    if account is None or account.get('disabled'):
        return None
    credential = _token_digest(account['password_hash'])
    if session.get('credential', credential) != credential:
        return None
    return account


def revoke_session(store, token):
    with store.transaction() as db:
        db.execute("DELETE FROM objects WHERE kind='session' AND id=?", (_token_digest(token),))


def revoke_all(store, user_id):
    digests = [key for key, body in store.items('session') if body['user_id'] == user_id]
    with store.transaction() as db:
        for digest in digests:
            db.execute("DELETE FROM objects WHERE kind='session' AND id=?", (digest,))


def grant(store, project_id, user_id):
    from .scope_deletion import require_active

    require_active(store, project_id=project_id)
    key = f'{project_id}:{user_id}'
    store.put('project_access', key, {'project_id': project_id, 'user_id': user_id, 'granted_at': now()})


def revoke(store, project_id, user_id):
    with store.transaction() as db:
        db.execute("DELETE FROM objects WHERE kind='project_access' AND id=?", (f'{project_id}:{user_id}',))


def projects_for(store, account):
    if account['role'] == 'admin':
        return None
    return {row['project_id'] for row in access_rows(store) if row['user_id'] == account['user_id']}


def access_rows(store):
    from .scope_deletion import deleted_scopes, scope_deleted

    deleted = deleted_scopes(store)
    return [row for row in store.list('project_access') if not scope_deleted(row, deleted)]


class LoginThrottle:
    """Bound failed attempts in a rolling window; a restart clears counters."""

    def __init__(self, window_seconds=15 * 60):
        self._window = window_seconds
        self._failures = {}

    def _recent(self, login, now):
        attempts = [stamp for stamp in self._failures.get(login, ()) if now - stamp < self._window]
        if attempts:
            self._failures[login] = attempts
        else:
            self._failures.pop(login, None)
        return attempts

    def allowed(self, username):
        login = (username or '').strip().lower()
        now = time.monotonic()
        if len(self._failures) > 1000:
            for key in [k for k in self._failures if not self._recent(k, now)]:
                self._failures.pop(key, None)
        return len(self._recent(login, now)) < _FAILED_LOGIN_LIMIT

    def record(self, username, ok):
        login = (username or '').strip().lower()
        if ok:
            self._failures.pop(login, None)
            return
        now = time.monotonic()
        self._failures.setdefault(login, []).append(now)
        self._failures[login] = self._recent(login, now)
