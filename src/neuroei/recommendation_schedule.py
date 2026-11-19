"""One-shot direction scheduling. Preview by default; no models, timers or services."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import accounts
from .directions import load_direction
from .models import now
from .recommendation_jobs import enqueue, visible_jobs
from .recommendation_pipeline import saved_direction_configuration
from .scope_deletion import deleted_scopes
from .store import Store

TZ = ZoneInfo('Asia/Shanghai')


def schedule(store, account, *, as_of, submit=False):
    if account['disabled']:
        raise PermissionError('Account disabled')
    clock = datetime.fromisoformat(as_of)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError('Timezone-aware scheduling time required')
    day = clock.astimezone(TZ).date()
    allowed = accounts.projects_for(store, account)
    jobs = visible_jobs(store, account)
    rows = []
    deleted = deleted_scopes(store)
    for pointer in store.list('direction_current'):
        if pointer['user_id'] != account['display_name']:
            continue
        did = pointer['key'].split(':')[0]
        if 'direction:'+did in deleted:
            continue
        try:
            setting = load_direction(store, account['display_name'], did)
        except PermissionError:
            rows.append({'direction_id':did,'date':day.isoformat(),'action':None,'reason':'project_access_required'})
            continue
        row = {'direction_id': did, 'date': day.isoformat(), 'action': None, 'reason': None}
        rows.append(row)
        if not setting['enabled']:
            row['reason'] = 'paused'
            continue
        if setting['project_id'] and allowed is not None and setting['project_id'] not in allowed:
            row['reason'] = 'project_access_required'
            continue
        try:
            config = saved_direction_configuration(store, account, did)
        except ValueError:
            row['reason'] = 'settings_incomplete'
            continue
        matching = [j for j in jobs if j['action'].startswith('direction_')
                    and j['snapshot']['direction_id'] == did and j['snapshot']['version'] == config['version']]
        first = next((j for j in matching if j['action'] == 'direction_bootstrap'), None)
        if first is None:
            row['action'] = 'direction_bootstrap'
        elif first['status'] in ('completed', 'partial'):
            daily = next((j for j in matching if j['action'] == 'direction_daily'
                          and datetime.fromisoformat(j['created_at']).astimezone(TZ).date() == day), None)
            if daily:
                row.update(reason='already_recorded', job_id=daily['job_id'], status=daily['status'])
                continue
            row['action'] = 'direction_daily'
        else:
            row.update(reason='initial_failed' if first['status'] == 'failed' else 'initial_pending',
                       job_id=first['job_id'], status=first['status'])
            continue
        if submit:
            job = enqueue(store, account, {'action': row['action'], 'direction_id': did}, as_of=as_of)
            row.update(job_id=job['job_id'], status=job['status'])
    return {'as_of': as_of, 'timezone': 'Asia/Shanghai', 'submitted': submit, 'directions': rows,
            'model_calls': 0, 'services_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--as-of', default=now())
    parser.add_argument('--enqueue', action='store_true', help='Persist due jobs; never execute models')
    args = parser.parse_args()
    if not (args.store/'neuroei.sqlite3').is_file():
        parser.error('Existing isolated account workspace required')
    store = Store(args.store)
    try:
        account = accounts.find(store, args.user_id)
        if not account:
            parser.error('Account not found')
        print(json.dumps(schedule(store, account, as_of=args.as_of, submit=args.enqueue), ensure_ascii=False))
    finally:
        store.close()


if __name__ == '__main__':
    main()
