import base64
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet

from neuroei import accounts, feishu_user_auth
from neuroei.feishu_user_auth import (
    FeishuAuthorizationRequired,
    FeishuOAuthError,
    authorization,
    begin,
    complete,
    connection_state,
    disconnect,
    save_grant,
)
from neuroei.portal import make_server
from neuroei.store import Store

SCOPE = 'offline_access drive:drive:readonly docx:document:readonly'
PASSWORD = 'review-password-1'
APP_INPUT = {'app_id': 'cli_alice', 'app_secret': 'alice-app-secret'}
PERSONAL_APP = {'client_id': APP_INPUT['app_id'], 'client_secret': APP_INPUT['app_secret']}


@pytest.fixture
def oauth_env(monkeypatch):
    monkeypatch.delenv('NEUROEI_DESKTOP', raising=False)
    monkeypatch.setenv('FEISHU_APP_ID', 'offline-app-id')
    monkeypatch.setenv('FEISHU_APP_SECRET', 'offline-app-secret')
    monkeypatch.setenv('NEUROEI_PUBLIC_ORIGIN', 'https://research.example.test')
    monkeypatch.setenv('NEUROEI_FEISHU_TOKEN_KEY', Fernet.generate_key().decode())


def tokens(prefix='user', *, expires=7200):
    return {'access_token': prefix+'-access-secret', 'refresh_token': prefix+'-refresh-secret',
            'expires_in': expires, 'refresh_token_expires_in': 604800, 'scope': SCOPE}


def identity(login='alice'):
    return {'open_id': 'open-'+login, 'name': login.title()}


def member(store, login='alice'):
    return accounts.create(store, login, login.title(), PASSWORD, 'member')


@pytest.mark.parametrize('desktop', [False, True])
def test_oauth_uses_pkce_and_stores_only_encrypted_user_tokens(oauth_env, monkeypatch, tmp_path, desktop):
    origin = 'https://research.example.test'
    if desktop:
        origin = feishu_user_auth.DESKTOP_ORIGIN
        monkeypatch.setenv('NEUROEI_DESKTOP', '1')
        monkeypatch.setenv('NEUROEI_PUBLIC_ORIGIN', origin)
    store = Store(tmp_path)
    account = member(store)
    with pytest.raises(FeishuAuthorizationRequired):
        begin(store, account, 'session-one')
    url = begin(store, account, 'session-one', **APP_INPUT)
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert (parts.scheme, parts.hostname, parts.path) == (
        'https', 'accounts.feishu.cn', '/open-apis/authen/v1/authorize')
    assert query['redirect_uri'] == [origin + '/oauth/feishu/callback']
    assert query['client_id'] == [APP_INPUT['app_id']]
    assert query['scope'] == [SCOPE]
    assert query['code_challenge_method'] == ['S256'] and query['prompt'] == ['consent']
    assert APP_INPUT['app_secret'] not in url
    state = query['state'][0]
    state_body = store.get('feishu_oauth_state', sha256(state.encode()).hexdigest())
    assert state_body['user_id'] == account['user_id']
    assert APP_INPUT['app_secret'] not in json.dumps(state_body)
    assert not connection_state(store, account)['app_configured']
    calls = []

    def respond(request):
        path = request.url.path.removeprefix('/open-apis')
        if path == '/authen/v2/oauth/token':
            payload = json.loads(request.content)
            calls.append(payload)
            assert payload['grant_type'] == 'authorization_code'
            assert payload['redirect_uri'] == query['redirect_uri'][0]
            assert payload['client_id'] == PERSONAL_APP['client_id']
            assert payload['client_secret'] == PERSONAL_APP['client_secret']
            verifier = payload['code_verifier']
            challenge = base64.urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b'=').decode()
            assert challenge == query['code_challenge'][0]
            return httpx.Response(200, json={'code': 0, **tokens()})
        assert path == '/authen/v1/user_info'
        assert request.headers['Authorization'] == 'Bearer user-access-secret'
        return httpx.Response(200, json={'code': 0, 'data': identity()})

    result = complete(store, account, 'session-one', state, 'valid_code', transport=httpx.MockTransport(respond))
    assert result['connected'] and result['grant_id']
    connection = connection_state(store, account)
    assert connection['connected'] and connection['app_configured']
    assert connection['redirect_uri'] == query['redirect_uri'][0]
    assert APP_INPUT['app_secret'] not in json.dumps(connection)
    assert authorization(store, account) == {'Authorization': 'Bearer user-access-secret'}
    raw = store.path.read_bytes()
    assert b'user-access-secret' not in raw and b'user-refresh-secret' not in raw
    assert APP_INPUT['app_secret'].encode() not in raw and APP_INPUT['app_id'].encode() not in raw
    assert b'valid_code' not in raw and calls[0]['code_verifier'].encode() not in raw
    with pytest.raises(FeishuOAuthError):
        complete(store, account, 'session-one', state, 'valid_code', transport=httpx.MockTransport(respond))
    assert len(calls) == 1
    renewed = parse_qs(urlsplit(begin(store, account, 'session-one')).query)
    assert renewed['client_id'] == query['client_id']
    with pytest.raises(FeishuAuthorizationRequired):
        begin(store, member(store, 'bob'), 'session-two')


@pytest.mark.parametrize('origin', [
    'http://127.0.0.1:8768',
    'http://localhost:8769',
    'http://localhost:8768/',
    'http://localhost.example.test:8768',
    'http://localhost:8768@remote.example.test',
    'https://remote.example.test',
])
def test_desktop_oauth_requires_exact_local_origin(oauth_env, monkeypatch, tmp_path, origin):
    monkeypatch.setenv('NEUROEI_DESKTOP', '1')
    monkeypatch.setenv('NEUROEI_PUBLIC_ORIGIN', origin)
    with closing(Store(tmp_path)) as store:
        account = member(store)
        assert not connection_state(store, account)['configured']
        with pytest.raises(FeishuOAuthError, match='localhost:8768'):
            begin(store, account, 'session', **APP_INPUT)


def test_server_oauth_still_rejects_http_loopback(oauth_env, monkeypatch, tmp_path):
    monkeypatch.setenv('NEUROEI_PUBLIC_ORIGIN', feishu_user_auth.DESKTOP_ORIGIN)
    with closing(Store(tmp_path)) as store:
        account = member(store)
        assert not connection_state(store, account)['configured']
        with pytest.raises(FeishuOAuthError, match='HTTPS'):
            begin(store, account, 'session', **APP_INPUT)


def test_state_is_bound_to_user_session_and_expiry(oauth_env, tmp_path):
    store = Store(tmp_path)
    alice, bob = member(store), member(store, 'bob')
    first = begin(store, alice, 'session-a', **APP_INPUT)
    second = begin(store, alice, 'session-a', **APP_INPUT)
    first_state = parse_qs(urlsplit(first).query)['state'][0]
    second_state = parse_qs(urlsplit(second).query)['state'][0]
    with pytest.raises(FeishuOAuthError):
        complete(store, alice, 'session-a', first_state, 'code')
    with pytest.raises(FeishuOAuthError):
        complete(store, bob, 'session-b', second_state, 'code')
    with pytest.raises(FeishuOAuthError):
        complete(store, alice, 'wrong-session', second_state, 'code')
    third = begin(store, alice, 'session-a', **APP_INPUT)
    third_state = parse_qs(urlsplit(third).query)['state'][0]
    key = sha256(third_state.encode()).hexdigest()
    record = store.get('feishu_oauth_state', key)
    store.put('feishu_oauth_state', key, {**record, 'expires_at': '2000-01-01T00:00:00+00:00'})
    with pytest.raises(FeishuOAuthError):
        complete(store, alice, 'session-a', third_state, 'code')
    assert not store.list('feishu_oauth_state')


def test_new_authorization_cancels_an_inflight_exchange(oauth_env, tmp_path):
    store = Store(tmp_path)
    account = member(store)
    original = save_grant(store, account, tokens('old'), identity(), app=PERSONAL_APP)
    candidate = {'app_id': 'cli_candidate', 'app_secret': 'candidate-secret'}
    url = begin(store, account, 'session', **candidate)
    state = parse_qs(urlsplit(url).query)['state'][0]

    def provider(request):
        if request.url.path.endswith('/oauth/token'):
            begin(store, account, 'session')
            return httpx.Response(200, json={'code': 0, **tokens('candidate')})
        return httpx.Response(200, json={'code': 0, 'data': identity()})

    with pytest.raises(FeishuOAuthError, match='新请求替换'):
        complete(store, account, 'session', state, 'code', transport=httpx.MockTransport(provider))
    assert store.get('feishu_user_grant', account['user_id'])['grant_id'] == original
    assert len(store.list('feishu_oauth_state')) == 1


@pytest.mark.parametrize('app', [None, PERSONAL_APP])
def test_refresh_is_rotated_once_across_process_stores(oauth_env, monkeypatch, tmp_path, app):
    monkeypatch.setattr(feishu_user_auth.time, 'time', lambda: 1_000_000)
    store = Store(tmp_path)
    account = member(store)
    grant_id = save_grant(store, account, tokens(expires=1), identity(), app=app)
    if app:
        monkeypatch.delenv('FEISHU_APP_ID')
        monkeypatch.delenv('FEISHU_APP_SECRET')
    calls = []

    def respond(request):
        assert request.url.path == '/open-apis/authen/v2/oauth/token'
        payload = json.loads(request.content)
        calls.append(payload)
        expected = app or {'client_id': 'offline-app-id', 'client_secret': 'offline-app-secret'}
        assert payload == {'grant_type': 'refresh_token', **expected, 'refresh_token': 'user-refresh-secret'}
        return httpx.Response(200, json={'code': 0, **tokens('fresh')})

    transport = httpx.MockTransport(respond)
    stores = [Store(tmp_path), Store(tmp_path)]
    barrier = threading.Barrier(2)

    def read(current):
        barrier.wait(timeout=3)
        return authorization(current, account, expected_grant_id=grant_id, transport=transport)

    with ThreadPoolExecutor(max_workers=2) as pool:
        headers = list(pool.map(read, stores))
    assert headers == [{'Authorization': 'Bearer fresh-access-secret'}] * 2
    assert len(calls) == 1
    assert store.get('feishu_user_grant', account['user_id'])['grant_id'] == grant_id
    raw = store.path.read_bytes()
    assert b'fresh-access-secret' not in raw and b'fresh-refresh-secret' not in raw


def test_rejected_refresh_destroys_grant_without_tenant_fallback(oauth_env, monkeypatch, tmp_path):
    monkeypatch.setattr(feishu_user_auth.time, 'time', lambda: 1_000_000)
    store = Store(tmp_path)
    account = member(store)
    save_grant(store, account, tokens(expires=1), identity(), app=PERSONAL_APP)
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(401, json={'code': 20005, 'msg': 'invalid user token'})

    with pytest.raises(FeishuAuthorizationRequired):
        authorization(store, account, transport=httpx.MockTransport(respond))
    assert paths == ['/open-apis/authen/v2/oauth/token']
    assert not store.list('feishu_user_grant')
    assert connection_state(store, account)['app_configured']
    with pytest.raises(FeishuAuthorizationRequired):
        authorization(store, account, transport=httpx.MockTransport(respond))
    assert len(paths) == 1


def test_disconnect_disable_and_delete_destroy_authorization(oauth_env, tmp_path):
    store = Store(tmp_path)
    account = member(store)
    save_grant(store, account, tokens(), identity(), app=PERSONAL_APP)
    begin(store, account, 'session-a')
    disconnect(store, account['user_id'])
    assert not store.list('feishu_user_grant') and not store.list('feishu_oauth_state')
    assert not store.list('feishu_user_app')
    save_grant(store, account, tokens(), identity(), app=PERSONAL_APP)
    begin(store, account, 'session-a')
    accounts.set_disabled(store, account['user_id'], True)
    assert not store.list('feishu_user_grant') and not store.list('feishu_oauth_state')
    assert not store.list('feishu_user_app')
    accounts.set_disabled(store, account['user_id'], False)
    with pytest.raises(FeishuAuthorizationRequired):
        authorization(store, account)
    save_grant(store, account, tokens(), identity(), app=PERSONAL_APP)
    assert accounts.delete(store, account['user_id'])
    assert not store.list('feishu_user_grant')
    assert not store.list('feishu_user_app')


def test_same_feishu_identity_cannot_connect_two_neuroei_accounts(oauth_env, tmp_path):
    store = Store(tmp_path)
    alice, bob = member(store), member(store, 'bob')
    save_grant(store, alice, tokens('alice'), identity('shared'))
    with pytest.raises(FeishuOAuthError):
        save_grant(store, bob, tokens('bob'), identity('shared'))
    assert connection_state(store, bob)['connected'] is False
    save_grant(store, alice, tokens('alice'), identity('shared'), app=PERSONAL_APP)
    with pytest.raises(FeishuOAuthError, match='已连接其他'):
        save_grant(store, bob, tokens('bob'), identity('shared'), app=PERSONAL_APP)
    save_grant(store, bob, tokens('bob'), identity('shared'), app={**PERSONAL_APP, 'client_id': 'cli_bob'})
    assert connection_state(store, bob)['connected'] is True


def test_portal_starts_oauth_and_callback_cleans_browser_url(oauth_env, monkeypatch, tmp_path, caplog):
    store = Store(tmp_path)
    account = member(store)
    server = make_server(store, 0, legacy_execution_enabled=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    seen = []

    def completed(store, current, session, state, code, *, error=None):
        if code == 'duplicate_identity':
            raise FeishuOAuthError('这个飞书账号已连接其他 ResearchPulse 账号。')
        seen.append((current['user_id'], session, state, code, error))
        return {'connected': True}

    monkeypatch.setattr(feishu_user_auth, 'complete', completed)
    try:
        with httpx.Client(base_url=base, trust_env=False) as client:
            login = client.get('/login')
            assert login.status_code == 200
            token = re.search(r"const token='([^']+)'", login.text)[1]
            headers = {'Origin': base, 'X-Review-Token': token}
            assert client.post('/api/login', headers=headers, json={'username': 'alice', 'password': PASSWORD}).status_code == 200
            missing = client.post('/api/feishu-oauth/start', json={}, headers=headers)
            assert missing.status_code == 400 and 'authorize_url' not in missing.json()
            partial = client.post('/api/feishu-oauth/start', json={'app_secret': APP_INPUT['app_secret']}, headers=headers)
            assert partial.status_code == 400 and APP_INPUT['app_secret'] not in partial.text
            started = client.post('/api/feishu-oauth/start', json=APP_INPUT, headers=headers)
            assert started.status_code == 200
            url = started.json()['authorize_url']
            assert APP_INPUT['app_secret'] not in url and 'client_id=cli_alice' in url
            state = parse_qs(urlsplit(url).query)['state'][0]
            callback = '/oauth/feishu/callback?' + urlencode({
                'state': state, 'code': 'valid_code'})
            with httpx.Client(base_url=base, trust_env=False) as anonymous:
                assert anonymous.get(callback, follow_redirects=False).status_code == 401
            result = client.get(callback, headers=headers, follow_redirects=False)
            assert result.status_code == 303
            assert result.headers['location'] == '/?feishu=connected#today'
            assert seen == [(account['user_id'], client.cookies.get('neuroei_session'), state, 'valid_code', None)]
            assert 'valid_code' not in result.headers['location'] and state not in result.headers['location']
            rejected = client.get(callback.replace('valid_code', 'duplicate_identity'), headers=headers)
            assert rejected.status_code == 400
            assert '这个飞书账号已连接其他 ResearchPulse 账号。' in rejected.text
            for sensitive in (state, 'duplicate_identity', client.cookies.get('neuroei_session')):
                assert sensitive not in rejected.text and sensitive not in caplog.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
