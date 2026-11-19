"""Per-user Feishu OAuth with encrypted, revocable offline credentials."""
import json
import logging
import os
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken

from . import accounts
from .models import now

SCOPES = ('offline_access', 'drive:drive:readonly', 'docx:document:readonly')
STATE_TTL_SECONDS = 600
AUTH_REQUIRED = '请先由当前账号本人完成飞书授权。'
DESKTOP_ORIGIN = 'http://localhost:8768'
_STATE = re.compile(r'[A-Za-z0-9_-]{43}')
_CODE = re.compile(r'[A-Za-z0-9_-]{1,256}')
_APP_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
FEISHU_TIMEOUT = httpx.Timeout(25, connect=10)


class FeishuOAuthError(ValueError):
    """A safe OAuth failure without provider response bodies or credentials."""


class FeishuAuthorizationRequired(FeishuOAuthError):
    """The current ResearchPulse account must authorize (or reauthorize) Feishu."""


class TokenRejected(FeishuOAuthError):
    """The provider rejected a refresh token; the local grant must be destroyed."""


def configured():
    return bool(os.environ.get('NEUROEI_PUBLIC_ORIGIN') and os.environ.get('NEUROEI_FEISHU_TOKEN_KEY'))


def _redirect_uri():
    if not configured():
        raise FeishuOAuthError('管理员尚未完整配置飞书用户授权。')
    origin = os.environ['NEUROEI_PUBLIC_ORIGIN']
    if os.environ.get('NEUROEI_DESKTOP') == '1':
        if origin != DESKTOP_ORIGIN:
            raise FeishuOAuthError('单机版授权回调必须使用 http://localhost:8768。')
        return origin + '/oauth/feishu/callback'
    origin = origin.rstrip('/')
    parsed = urlsplit(origin)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
            or parsed.path or parsed.query or parsed.fragment):
        raise FeishuOAuthError('NEUROEI_PUBLIC_ORIGIN 必须是受信任的 HTTPS 站点地址。')
    return origin + '/oauth/feishu/callback'


def _cipher():
    try:
        return Fernet(os.environ.get('NEUROEI_FEISHU_TOKEN_KEY', '').encode())
    except (TypeError, ValueError) as exc:
        raise FeishuOAuthError('NEUROEI_FEISHU_TOKEN_KEY 配置无效。') from exc


def _digest(value):
    return sha256(value.encode()).hexdigest()


def _encrypt(payload):
    return _cipher().encrypt(json.dumps(payload, separators=(',', ':')).encode()).decode()


def _app_input(app_id, app_secret):
    app_id = app_id.strip() if isinstance(app_id, str) else ''
    app_secret = app_secret.strip() if isinstance(app_secret, str) else ''
    if not _APP_ID.fullmatch(app_id) or not app_secret or len(app_secret) > 512:
        raise FeishuOAuthError('请填写有效的飞书应用 App ID 和 App Secret。')
    return {'client_id': app_id, 'client_secret': app_secret}


def _saved_app(store, user_id):
    try:
        record = store.get('feishu_user_app', user_id)
        payload = json.loads(_cipher().decrypt(record['ciphertext'].encode()))
        if payload['user_id'] == user_id:
            return payload['app']
    except (InvalidToken, KeyError, TypeError, ValueError):
        pass
    raise FeishuAuthorizationRequired('请填写当前账号的飞书应用 App ID 和 App Secret。')


def _legacy_app():
    app_id, secret = os.environ.get('FEISHU_APP_ID'), os.environ.get('FEISHU_APP_SECRET')
    if not app_id or not secret:
        raise FeishuAuthorizationRequired('请用自己的飞书应用重新授权。')
    return {'client_id': app_id, 'client_secret': secret}


def _decrypt(grant):
    try:
        payload = json.loads(_cipher().decrypt(grant['ciphertext'].encode()))
    except (InvalidToken, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FeishuAuthorizationRequired('飞书授权凭据不可用，请重新授权。') from exc
    if (payload.get('user_id'), payload.get('grant_id')) != (grant.get('user_id'), grant.get('grant_id')):
        raise FeishuAuthorizationRequired('飞书授权身份不匹配，请重新授权。')
    return payload


def connection_state(store, account):
    try:
        grant = store.get('feishu_user_grant', account['user_id'])
    except ValueError:
        grant = None
    try:
        _cipher()
        redirect_uri = _redirect_uri()
    except FeishuOAuthError:
        redirect_uri = None
    return {'configured': redirect_uri is not None, 'connected': grant is not None,
            'app_configured': bool(store.select('feishu_user_app', ('user_id',), key=account['user_id'])),
            'redirect_uri': redirect_uri,
            'display_name': grant.get('feishu_display_name') if grant else None,
            'connected_at': grant.get('connected_at') if grant else None}


def begin(store, account, session_token, *, app_id=None, app_secret=None):
    redirect_uri = _redirect_uri()
    if not session_token:
        raise PermissionError('登录状态已失效。')
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = __import__('base64').urlsafe_b64encode(
        sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    clock = datetime.now(UTC)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        app = (_saved_app(store, account['user_id']) if app_id is None and app_secret is None
               else _app_input(app_id, app_secret))
        record = {'user_id': account['user_id'], 'session_digest': _digest(session_token),
                  'redirect_uri': redirect_uri, 'code_verifier': _encrypt({
                      'user_id': account['user_id'], 'verifier': verifier, 'app': app}),
                  'created_at': clock.isoformat(),
                  'expires_at': (clock + timedelta(seconds=STATE_TTL_SECONDS)).isoformat()}
        db.execute("DELETE FROM objects WHERE kind='feishu_oauth_state' AND json_extract(body,'$.expires_at') <= ?",
                   (clock.isoformat(),))
        db.execute("DELETE FROM objects WHERE kind='feishu_oauth_state' AND json_extract(body,'$.user_id')=?",
                   (account['user_id'],))
        db.execute('INSERT INTO objects VALUES (?,?,?)',
                   ('feishu_oauth_state', _digest(state), json.dumps(record)))
    return 'https://accounts.feishu.cn/open-apis/authen/v1/authorize?' + urlencode({
        'response_type': 'code', 'client_id': app['client_id'],
        'redirect_uri': redirect_uri, 'scope': ' '.join(SCOPES),
        'state': state, 'code_challenge': challenge, 'code_challenge_method': 'S256',
        'prompt': 'consent'})


def _token_response(http, payload):
    refreshing = payload['grant_type'] == 'refresh_token'
    try:
        # Refresh holds the SQLite write lease while rotating the token.
        response = http.post('/authen/v2/oauth/token', json=payload,
                             timeout=10 if refreshing else FEISHU_TIMEOUT)
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise FeishuOAuthError('暂时无法连接飞书授权服务，请稍后重试。') from exc
    provider_code = body.get('code') if isinstance(body, dict) else None
    if response.status_code != 200 or provider_code != 0:
        if refreshing:
            raise TokenRejected('飞书授权已失效，请重新授权。')
        logging.getLogger(__name__).warning('Feishu token exchange rejected: HTTP %d, code=%s',
                                            response.status_code, provider_code if type(provider_code) is int else '-')
        raise FeishuOAuthError('飞书授权码交换失败，请重新发起授权。')
    return body


def _valid_tokens(body):
    access, refresh = body.get('access_token'), body.get('refresh_token')
    if (not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh
            or not isinstance(body.get('expires_in'), (int, float))
            or not isinstance(body.get('refresh_token_expires_in'), (int, float))
            or body['expires_in'] <= 0 or body['refresh_token_expires_in'] <= 0):
        raise FeishuOAuthError('飞书未返回完整的用户授权，请重新授权。')
    granted = set((body.get('scope') or '').split())
    if not set(SCOPES).issubset(granted):
        raise FeishuOAuthError('飞书授权范围不完整，请重新授权并确认只读权限。')
    return body


def _user_info(http, access_token):
    try:
        response = http.get('/authen/v1/user_info', headers={'Authorization': 'Bearer ' + access_token})
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise FeishuOAuthError('无法核对飞书用户身份，请重新授权。') from exc
    if response.status_code != 200 or body.get('code') != 0:
        raise FeishuOAuthError('无法核对飞书用户身份，请重新授权。')
    data = body.get('data') or {}
    if not isinstance(data.get('open_id'), str) or not data['open_id']:
        raise FeishuOAuthError('飞书用户身份信息不完整，请重新授权。')
    return data


def save_grant(store, account, token_data, user_info, *, app=None, state_key=None):
    """Atomically store a new grant generation; only ciphertext contains tokens."""
    token_data = _valid_tokens(token_data)
    user_id = account['user_id']
    grant_id = secrets.token_urlsafe(18)
    clock = time.time()
    payload = {'user_id': account['user_id'], 'grant_id': grant_id,
               'access_token': token_data['access_token'], 'refresh_token': token_data['refresh_token'],
               'access_expires_at': clock + token_data['expires_in'],
               'refresh_expires_at': clock + token_data['refresh_token_expires_in'],
               'scope': sorted(set(token_data['scope'].split()))}
    outer = {'user_id': account['user_id'], 'grant_id': grant_id,
             'feishu_open_id': user_info['open_id'],
             'feishu_display_name': str(user_info.get('name') or '')[:100],
             'connected_at': now(), 'updated_at': now(), 'ciphertext': _encrypt(payload)}
    legacy_scope = _digest(os.environ.get('FEISHU_APP_ID', ''))
    app_scope = _digest(app['client_id']) if app else legacy_scope
    if app:
        outer['feishu_app_scope'] = app_scope
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        current = accounts.find(store, user_id)
        if current is None or current['disabled']:
            raise PermissionError('账号已停用或不存在。')
        if state_key and not db.execute("SELECT 1 FROM objects WHERE kind='feishu_oauth_state' AND id=?",
                                        (state_key,)).fetchone():
            raise FeishuOAuthError('本次授权已取消或已被新请求替换，请重新发起授权。')
        other = db.execute("SELECT id FROM objects WHERE kind='feishu_user_grant' "
                           "AND json_extract(body,'$.feishu_open_id')=? AND id<>? "
                           "AND coalesce(json_extract(body,'$.feishu_app_scope'),?)=?",
                           (user_info['open_id'], user_id, legacy_scope, app_scope)).fetchone()
        if other:
            raise FeishuOAuthError('这个飞书账号已连接其他 ResearchPulse 账号。请使用原账号，或先在原账号解除飞书授权，再连接当前账号。')
        db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',
                   ('feishu_user_grant', user_id, json.dumps(outer)))
        if app:
            record = {'user_id': user_id, 'ciphertext': _encrypt({'user_id': user_id, 'app': app})}
            db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',
                       ('feishu_user_app', user_id, json.dumps(record)))
    return grant_id


def complete(store, account, session_token, state, code, *, error=None, transport=None):
    if not _STATE.fullmatch(state or '') or (not error and not _CODE.fullmatch(code or '')):
        raise FeishuOAuthError('飞书授权回调无效，请重新发起授权。')
    key = _digest(state)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT body FROM objects WHERE kind='feishu_oauth_state' AND id=?", (key,)).fetchone()
        record = json.loads(row[0]) if row else {}
        if 'code_verifier' not in record:
            raise FeishuOAuthError('授权链接已使用或不存在，请重新发起授权。')
        encrypted = record.pop('code_verifier')
        # Consume the verifier, keeping the row as a marker that can be cancelled
        # while the token exchange is in flight.
        db.execute("UPDATE objects SET body=? WHERE kind='feishu_oauth_state' AND id=?",
                   (json.dumps(record), key))
    try:
        if error:
            raise FeishuOAuthError('飞书授权未完成，请返回研究台重新操作。')
        return _exchange(store, account, session_token, record, encrypted, key, code, transport)
    finally:
        store.delete('feishu_oauth_state', key)


def _exchange(store, account, session_token, record, encrypted, state_key, code, transport):
    if (record.get('user_id') != account['user_id']
            or record.get('session_digest') != _digest(session_token)
            or record.get('expires_at', '') <= datetime.now(UTC).isoformat()):
        raise FeishuOAuthError('授权链接已过期或不属于当前登录账号，请重新发起授权。')
    try:
        verifier_record = json.loads(_cipher().decrypt(encrypted.encode()))
    except (InvalidToken, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FeishuOAuthError('授权状态凭据不可用，请重新发起授权。') from exc
    if verifier_record.get('user_id') != account['user_id'] or not isinstance(
            verifier_record.get('verifier'), str):
        raise FeishuOAuthError('授权身份不匹配，请重新发起授权。')
    verifier = verifier_record['verifier']
    app = verifier_record.get('app')
    if not app:
        raise FeishuOAuthError('请填写自己的飞书应用凭据并重新发起授权。')
    redirect_uri = record['redirect_uri']
    if redirect_uri != _redirect_uri():
        raise FeishuOAuthError('授权回调配置已改变，请重新发起授权。')
    with httpx.Client(base_url='https://open.feishu.cn/open-apis', timeout=FEISHU_TIMEOUT,
                      follow_redirects=False, transport=transport) as http:
        tokens = _token_response(http, {'grant_type': 'authorization_code',
            **app, 'code': code, 'redirect_uri': redirect_uri, 'code_verifier': verifier})
        tokens = _valid_tokens(tokens)
        identity = _user_info(http, tokens['access_token'])
    grant_id = save_grant(store, account, tokens, identity, app=app, state_key=state_key)
    return {'connected': True, 'grant_id': grant_id}


def _active_grant(store, account, expected_grant_id=None):
    current = accounts.find(store, account['user_id'])
    if current is None or current['disabled']:
        raise PermissionError('账号已停用或不存在。')
    try:
        grant = store.get('feishu_user_grant', account['user_id'])
    except ValueError as exc:
        raise FeishuAuthorizationRequired(AUTH_REQUIRED) from exc
    if grant.get('user_id') != account['user_id'] or (
            expected_grant_id is not None and grant.get('grant_id') != expected_grant_id):
        raise FeishuAuthorizationRequired('飞书授权已变更，请重新发起项目分析。')
    return grant


def current_grant_id(store, account):
    return _active_grant(store, account)['grant_id']


def authorization(store, account, *, expected_grant_id=None, http=None, transport=None):
    """Return this user's token, refreshing atomically across portal and worker processes."""
    grant = _active_grant(store, account, expected_grant_id)
    payload = _decrypt(grant)
    if payload['access_expires_at'] > time.time() + 60:
        return {'Authorization': 'Bearer ' + payload['access_token']}
    own = http is None
    client = httpx.Client(base_url='https://open.feishu.cn/open-apis', timeout=FEISHU_TIMEOUT,
                          follow_redirects=False, transport=transport) if own else http
    try:
        # SQLite's write lease serializes rotating refresh tokens across processes.
        rejected = None
        with store.transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT body FROM objects WHERE kind='feishu_user_grant' AND id=?",
                             (account['user_id'],)).fetchone()
            if not row:
                raise FeishuAuthorizationRequired(AUTH_REQUIRED)
            latest = json.loads(row[0])
            current = accounts.find(store, account['user_id'])
            if current is None or current['disabled']:
                raise PermissionError('账号已停用或不存在。')
            if expected_grant_id is not None and latest.get('grant_id') != expected_grant_id:
                raise FeishuAuthorizationRequired('飞书授权已变更，请重新发起项目分析。')
            payload = _decrypt(latest)
            if payload['access_expires_at'] > time.time() + 60:
                return {'Authorization': 'Bearer ' + payload['access_token']}
            try:
                if payload['refresh_expires_at'] <= time.time():
                    raise TokenRejected('飞书授权已过期，请重新授权。')
                app = (_saved_app(store, account['user_id']) if latest.get('feishu_app_scope')
                       else _legacy_app())
                refreshed = _token_response(client, {'grant_type': 'refresh_token', **app,
                    'refresh_token': payload['refresh_token']})
            except TokenRejected as exc:
                db.execute("DELETE FROM objects WHERE kind='feishu_user_grant' AND id=?", (account['user_id'],))
                rejected = exc
            else:
                refreshed = _valid_tokens(refreshed)
                updated = {'user_id': account['user_id'], 'grant_id': latest['grant_id'],
                           'access_token': refreshed['access_token'], 'refresh_token': refreshed['refresh_token'],
                           'access_expires_at': time.time() + refreshed['expires_in'],
                           'refresh_expires_at': time.time() + refreshed['refresh_token_expires_in'],
                           'scope': sorted(set(refreshed['scope'].split()))}
                latest.update(updated_at=now(), ciphertext=_encrypt(updated))
                db.execute("UPDATE objects SET body=? WHERE kind='feishu_user_grant' AND id=?",
                           (json.dumps(latest), account['user_id']))
                return {'Authorization': 'Bearer ' + updated['access_token']}
        # Commit credential destruction before converting the provider error.
        if rejected:
            raise rejected
    except TokenRejected as exc:
        raise FeishuAuthorizationRequired('飞书授权已失效，请重新授权。') from exc
    finally:
        if own:
            client.close()


def disconnect(store, user_id):
    with store.transaction() as db:
        revoke_account(db, user_id)


def invalidate_grant(store, user_id, grant_id):
    with store.transaction() as db:
        db.execute("DELETE FROM objects WHERE kind='feishu_user_grant' AND id=? "
                   "AND json_extract(body,'$.grant_id')=?", (user_id, grant_id))


def revoke_account(db, user_id):
    """Destroy credentials and pending OAuth, including obsolete managed-folder grants."""
    for kind in ('feishu_user_grant', 'feishu_user_app', 'feishu_oauth_state', 'feishu_folder_access'):
        if kind == 'feishu_oauth_state':
            db.execute("DELETE FROM objects WHERE kind=? AND json_extract(body,'$.user_id')=?", (kind, user_id))
        else:
            db.execute('DELETE FROM objects WHERE kind=? AND id=?', (kind, user_id))
