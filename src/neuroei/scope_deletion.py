"""Hide deleted project/direction scopes and retain their immutable records."""
import json
import secrets
import time
from hashlib import sha256

from .models import now


class ScopeBusy(ValueError):
    """A queued or running job still owns a scope."""


def scope_keys(row):
    return [kind+':'+row[field] for kind, field in (('project', 'project_id'), ('direction', 'direction_id'))
            if row.get(field)]


def deleted_scopes(store):
    return {key for key, _ in store.items('deleted_research_scope')}


def scope_deleted(row, deleted):
    return any(key in deleted for key in scope_keys(row))


def require_active(store, *, project_id=None, direction_id=None, db=None):
    row = {'project_id': project_id, 'direction_id': direction_id}
    if db is not None:
        deleted = any(db.execute("SELECT 1 FROM objects WHERE kind='deleted_research_scope' AND id=?", (key,)).fetchone()
                      for key in scope_keys(row))
    else:
        deleted = scope_deleted(row, deleted_scopes(store))
    if deleted:
        raise PermissionError('项目或方向已删除。')


def _target(store, account, kind, target_id, *, db=None):
    if kind not in ('project', 'direction', 'legacy_project') or not isinstance(target_id, str) or not 1 <= len(target_id) <= 100:
        raise ValueError('请选择要删除的项目或方向。')
    def read(namespace, key):
        if db is None:
            return store.get(namespace, key)
        row = db.execute('SELECT body FROM objects WHERE kind=? AND id=?', (namespace, key)).fetchone()
        if row is None:
            raise ValueError('目标不存在。')
        return json.loads(row[0])

    try:
        if kind == 'project':
            target = read('research_project', target_id)
            if target['owner_id'] != account['user_id']:
                raise PermissionError('仅项目所有者可删除。')
            require_active(store, project_id=target_id, db=db)
        elif kind == 'legacy_project':
            if account['role'] != 'member':
                raise PermissionError('只可移除本人获得授权的旧项目。')
            read('project_access', target_id+':'+account['user_id'])
            target = read('active_profile', target_id)
            require_active(store, project_id=target_id, db=db)
            try:
                read('research_project', target_id)
            except ValueError:
                pass
            else:
                raise PermissionError('请选择项目工作台中的删除操作。')
        else:
            pointer = read('direction_current', target_id)
            if pointer['user_id'] != account['display_name']:
                raise PermissionError('仅方向所有者可删除。')
            target = read('direction_version', pointer['key'])
            require_active(store, project_id=target.get('project_id'), direction_id=target_id, db=db)
    except ValueError as exc:
        raise PermissionError('目标不可用。') from exc
    return target


_SEGMENTS = ('M3 2h14','M19 4v12','M19 20v12','M3 34h14','M1 20v12','M1 4v12','M3 18h14')
_DIGITS = ('012345','12','01643','01623','5612','05623','056234','012','0123456','015623')


def _captcha_svg(answer):
    digits = ''.join(
        f'<path transform="translate({15 + i*32} 11) rotate({secrets.randbelow(19)-9} 10 18)" '
        f'd="{" ".join(_SEGMENTS[int(s)] for s in _DIGITS[int(digit)])}"/>'
        for i, digit in enumerate(answer))
    noise = ' '.join(f'M{secrets.randbelow(168)} {secrets.randbelow(55)}h{2+secrets.randbelow(5)}' for _ in range(20))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="178" height="58" viewBox="0 0 178 58">'
            '<rect width="178" height="58" rx="6" fill="#edf6f3"/>'
            f'<path d="{noise}" stroke="#a5c5bf" stroke-width="1.5"/>'
            f'<g fill="none" stroke="#24524d" stroke-width="4" stroke-linecap="round">{digits}</g></svg>')


def _answer():
    return ''.join(secrets.choice('0123456789') for _ in range(5))


def _digest(challenge_id, answer):
    return sha256((challenge_id+':'+answer).encode()).hexdigest()


def create_challenge(store, account, session, kind, target_id):
    _target(store, account, kind, target_id)
    challenge_id, answer = secrets.token_urlsafe(24), _answer()
    store.put('scope_delete_challenge', account['user_id'], {
        'id': challenge_id, 'session': sha256(session.encode()).hexdigest(),
        'kind': kind, 'target_id': target_id, 'expires': time.time()+300,
        'answer': _digest(challenge_id, answer), 'svg': _captcha_svg(answer),
    })
    return {'challenge_id': challenge_id}


def _matches(challenge, session, challenge_id):
    return (challenge and challenge['id'] == challenge_id and challenge['expires'] >= time.time()
            and secrets.compare_digest(challenge['session'], sha256(session.encode()).hexdigest()))


def challenge_image(store, account, session, challenge_id):
    try:
        challenge = store.get('scope_delete_challenge', account['user_id'])
    except ValueError:
        challenge = None
    if not _matches(challenge, session, challenge_id):
        raise ValueError('验证码已失效，请刷新。')
    return challenge['svg']


def delete_scope(store, account, session, *, kind, target_id, challenge_id, answer):
    if not isinstance(answer, str) or len(answer.strip()) != 5 or not answer.strip().isascii() or not answer.strip().isdigit():
        raise ValueError('请输入图中的 5 位数字。')
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        _target(store, account, kind, target_id, db=db)
        row = db.execute("SELECT body FROM objects WHERE kind='scope_delete_challenge' AND id=?",
                         (account['user_id'],)).fetchone()
        challenge = json.loads(row[0]) if row else None
        valid = (_matches(challenge, session, challenge_id)
                 and challenge['kind'] == kind and challenge['target_id'] == target_id)
        if not valid:
            raise ValueError('验证码已失效，请刷新。')
        db.execute("DELETE FROM objects WHERE kind='scope_delete_challenge' AND id=?", (account['user_id'],))
        correct = secrets.compare_digest(challenge['answer'], _digest(challenge_id, answer.strip()))
        if correct:
            directions = [target_id] if kind == 'direction' else [row[0] for row in db.execute(
                "SELECT p.id FROM objects p JOIN objects v ON v.kind='direction_version' "
                "AND v.id=json_extract(p.body,'$.key') WHERE p.kind='direction_current' "
                "AND json_extract(v.body,'$.project_id')=?", (target_id,))]
            project = target_id if kind != 'direction' else None
            personal = account['user_id'] if kind == 'legacy_project' else None
            active = db.execute(
                "SELECT 1 FROM objects WHERE kind='recommendation_job' "
                "AND json_extract(body,'$.status') IN ('queued','processing') "
                "AND (json_extract(body,'$.snapshot.project_id')=? OR "
                "json_extract(body,'$.snapshot.direction_id') IN (SELECT value FROM json_each(?))) "
                "AND (? IS NULL OR json_extract(body,'$.user_id')=?) LIMIT 1",
                (project, json.dumps(directions), personal, personal)).fetchone()
            if active:
                raise ScopeBusy('该项目或方向还有排队或运行中的任务，请等任务结束后再删除。')
            if personal:
                db.execute("DELETE FROM objects WHERE kind='project_access' AND id=?", (target_id+':'+personal,))
                db.execute('INSERT INTO audit VALUES (?,?,?)',(now(),target_id,
                           json.dumps({'event':'legacy_project_removed_from_account','user_id':personal})))
            else:
                keys = ([f'project:{target_id}'] if project else []) + [f'direction:{item}' for item in directions]
                data = json.dumps({'deleted_at': now(), 'deleted_by': account['user_id']})
                db.executemany('INSERT OR IGNORE INTO objects VALUES (?,?,?)',
                               [('deleted_research_scope', key, data) for key in keys])
    if not correct:
        raise ValueError('验证码不正确，请刷新后重试。')
    return {'deleted': True, 'kind': kind, 'target_id': target_id}
