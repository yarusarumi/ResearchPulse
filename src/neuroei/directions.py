"""User-declared research directions, separate from laboratory facts and project activation."""
import hashlib
import json
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import Field, StringConstraints, field_validator

from .config import StrictModel
from .execution import project_lock
from .models import ProjectFact, ProjectProfile, SourceBlock, now
from .run_diagnostics import describe_run
from .scope_deletion import deleted_scopes, require_active, scope_deleted

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class DirectionSettings(StrictModel):
    direction_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default='', max_length=4000)
    questions: list[ShortText] = Field(default_factory=list, max_length=10)
    public_queries: list[ShortText] = Field(default_factory=list, max_length=3)
    exclusions: list[ShortText] = Field(default_factory=list, max_length=10)
    priority: int = Field(default=2, ge=1, le=3)
    enabled: bool = True
    project_id: str | None = Field(default=None, max_length=100)

    @field_validator('user_id', 'title')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Nonblank user and direction title required')
        return value.strip()


class DirectionSave(DirectionSettings):
    expected_version: str | None = None


def optional(store, kind, key):
    try:
        return store.get(kind, key)
    except ValueError as exc:
        if not str(exc).startswith(f'Missing {kind}:'):
            raise
        return None


def load_direction(store, user, direction_id, *, check_project_access=True):
    if not isinstance(user, str) or not isinstance(direction_id, (str, UUID)):
        raise TypeError('Explicit direction ID and user name required')
    did = str(UUID(str(direction_id)))
    pointer = store.get('direction_current', did)
    if pointer['user_id'] != user.strip():
        raise ValueError('Direction belongs to another user')
    setting = store.get('direction_version', pointer['key'])
    require_active(store, project_id=setting.get('project_id'), direction_id=did)
    if check_project_access and setting.get('project_id'):
        from . import accounts

        account = accounts.by_display_name(store, user.strip())
        if account:
            allowed = accounts.projects_for(store, account)
            if allowed is not None and setting['project_id'] not in allowed:
                raise PermissionError('No access to linked project')
    return {**setting, 'activated_at': pointer['activated_at']}


def save_direction(store, payload, *, allow_research_project=False):
    request = DirectionSave.model_validate(payload)
    settings = request.model_dump(mode='json', exclude={'expected_version'})
    did = settings['direction_id']
    version = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    with project_lock(store.root, 'direction-save:' + did):
        pointer = optional(store, 'direction_current', did)
        current = load_direction(store, settings['user_id'], did, check_project_access=False) if pointer else None
        if current and current['version'] == version:
            return {**current, 'reused': True}
        if (current and request.expected_version != current['version']) or (not current and request.expected_version):
            raise ValueError('Direction settings changed; refresh before saving')
        if settings['project_id']:
            require_direction_project(store, settings)
            if not (allow_research_project and optional(store, 'research_project', settings['project_id'])):
                store.get('active_profile', settings['project_id'])
        key = did + ':' + version
        body = optional(store, 'direction_version', key)
        if body is None:
            body = {**settings, 'version': version, 'saved_at': now(),
                    'provenance': 'explicit_user_direction_not_scientific_observation'}
            store.put('direction_version', key, body, immutable=True)
        pointer = {'key': key, 'user_id': settings['user_id'], 'activated_at': now()}
        with store.transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            require_active(store, project_id=settings['project_id'], direction_id=did, db=db)
            db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)', ('direction_current', did, json.dumps(pointer)))
        return {**load_direction(store, settings['user_id'], did), 'reused': False}


def direction_state(store, user, projects=None):
    user = user.strip()
    if not user or len(user) > 100:
        raise ValueError('User name required')
    deleted = deleted_scopes(store)
    directions = []
    for pointer in store.list('direction_current'):
        if pointer['user_id'] == user:
            setting = store.get('direction_version', pointer['key'])
            if (not scope_deleted(setting, deleted)
                    and (not setting.get('project_id') or projects is None or setting['project_id'] in projects)):
                directions.append({**setting, 'activated_at': pointer['activated_at']})
    directions.sort(key=lambda item: (item['priority'], item['title']))
    active_ids = {setting['direction_id'] for setting in directions}
    runs = sorted([run for run in store.list('direction_run')
                   if run['user_id'] == user and run['direction_id'] in active_ids],
                  key=lambda item: item['started_at'], reverse=True)
    known_runs = {run['run_id'] for run in runs}
    runs += [{**request, 'phase': 'starting', 'started_at': request['created_at']}
             for request in store.list('direction_request')
             if request['user_id'] == user and request['run_id'] not in known_runs
             and request['direction_id'] in active_ids]
    runs.sort(key=lambda item: item['started_at'], reverse=True)
    fields = ('run_id', 'direction_id', 'direction_version', 'date', 'status', 'phase', 'started_at',
              'ended_at', 'error_type', 'llm_requests', 'new_papers', 'new_fulltext_analyses', 'report_id',
              'summary', 'result_kind', 'effective_queries', 'query_plan_origin', 'source_summary', 'search_policy_version',
              'history_policy_version', 'candidate_history_complete', 'first_seen_recommendations',
              'sources_completed', 'sources_planned')
    rendered = []
    for run in runs[:50]:
        rendered.append({**{k: run.get(k) for k in fields},
            'recommended_count': len(run.get('recommendations', [])),
            **describe_run(run)})
    weeklies = sorted([r for r in store.list('direction_weekly')
                       if r['user_id'] == user and r['direction_id'] in active_ids],
                      key=lambda r: (r['date'], r.get('created_at', ''), r['report_id']), reverse=True)
    # Keep the current compilation per evidence context/window; retain every saved version in storage.
    latest_weeklies = {}
    for report in weeklies:
        period = report.get('period') or {}
        identity = (report['direction_id'], report.get('context_version') or report['direction_version'],
                    period.get('since'), period.get('until'))
        latest_weeklies.setdefault(identity, report)
    return {'directions': directions, 'runs': rendered,
            'weekly_reports': [{k: r.get(k) for k in ('report_id', 'direction_id', 'direction_version',
                'date', 'status', 'period', 'covered_dates', 'missing_dates')}
                for r in list(latest_weeklies.values())[:50]],
            'policy': {'automatic_schedule_enabled': False, 'generated_feishu_upload_enabled': False,
                       'identity_scope': 'local_user_label_not_authenticated_multiuser_service'}}


def direction_paper_history(store, runs, user, direction_id, until):
    """Candidate discovery and recommendation are different events, scoped to this direction."""
    seen, recommended, missing = {}, {}, set()
    for run in sorted(runs, key=lambda r: (r.get('date', ''), r.get('started_at', ''))):
        if (run.get('user_id') != user or run.get('direction_id') != direction_id
                or not run.get('date') or run['date'] > until):
            continue
        day = run['date']
        for paper in run.get('papers', []):
            key = (paper['paper_id'], paper['version'])
            seen.setdefault(key, day)
            if run.get('status') in ('completed', 'partial'):
                recommended.setdefault(key, day)
        if run.get('pool_id'):
            pool = optional(store, 'pool', run['pool_id'])
            if pool is None or pool.get('until', day) > day:
                missing.add(run['pool_id'])
            else:
                for paper in pool.get('papers', []):
                    seen.setdefault((paper['paper_id'], paper['version']), day)
        elif run.get('candidate_count') or run.get('recommendations'):
            missing.add('unrecorded-pool:' + run['run_id'])
    return {'first_seen': seen, 'first_recommended': recommended,
            'candidate_history_complete': not missing, 'missing_pool_ids': sorted(missing)}


def require_direction_project(store, direction):
    """Apply account project grants to background jobs as well as HTTP requests."""
    project = direction.get('project_id')
    if not project:
        return
    require_active(store, project_id=project)
    from . import accounts
    owners = store.list('account')
    if not owners:  # Existing local-label preview stores have no account system.
        return
    owner = next((a for a in owners if a['display_name'] == direction['user_id']), None)
    if not owner or owner.get('disabled'):
        raise PermissionError('Direction owner has no active account')
    allowed = accounts.projects_for(store, owner)
    if allowed is not None and project not in allowed:
        raise PermissionError('No access to linked project')


def resolve_direction_context(store, direction):
    project = direction.get('project_id')
    if not project:
        return direction_context(direction), None
    require_direction_project(store, direction)
    pointer = store.get('active_profile', project)
    profile = ProjectProfile.model_validate(store.get('profile', pointer['key']))
    if profile.project_id != project or pointer['key'] != project + ':' + profile.version:
        raise ValueError('Linked project profile identity mismatch')
    approval = optional(store, 'approval', pointer['key'])
    if not profile.approved_at and not (approval and approval.get('at') and approval.get('reviewer')):
        raise ValueError('Linked project requires an approved active profile')
    context = direction_context(direction, project_profile=profile)
    dates = [f.source_date for f in profile.facts if f.source_date]
    return context, {'project_id': project, 'profile_key': pointer['key'], 'profile_version': profile.version,
                     'fact_count': len(profile.facts), 'source_date': max(dates) if dates else None,
                     'policy': 'approved_profile_with_historical_source_dates_not_current_experiment_state'}


def direction_context(direction, *, project_profile=None):
    """In-memory adapter for existing evidence tools. Never persists a lab ProjectProfile."""
    basis = {key: direction[key] for key in ('direction_id', 'title', 'description', 'questions', 'exclusions')}
    if direction.get('project_id'):
        if project_profile is None:
            raise ValueError('Resolve the linked project profile before building direction context')
        basis['linked_project'] = {'project_id': project_profile.project_id, 'version': project_profile.version}
    revision = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    block_id = 'direction-' + direction['direction_id']
    block = SourceBlock(document_id=block_id, block_id=block_id, revision=revision,
        text='使用者声明的关注方向与问题，不包含科研观察或实验结论。\n' + json.dumps(basis, ensure_ascii=False),
        section='使用者关注设置')
    questions = direction['questions'] or ['这个方向近期有哪些有原始证据支持的新进展、方法与局限？']
    facts = [ProjectFact(id='direction-goal', text=direction['title'] + '\n' + direction['description'],
                        kind='goal', source_blocks=[block_id])]
    facts += [ProjectFact(id=f'direction-q{i}', text=question, kind='question', source_blocks=[block_id],
                         provenance='direct' if direction['questions'] else 'derived_question')
              for i, question in enumerate(questions)]
    facts += [ProjectFact(id=f'direction-exclusion{i}', text=text, kind='exclusion', source_blocks=[block_id])
              for i, text in enumerate(direction['exclusions'])]
    blocks = [block]
    if project_profile is not None:
        prefix = 'project:' + project_profile.project_id + ':'
        referenced = {key for fact in project_profile.facts for key in fact.source_blocks}
        blocks += [b.model_copy(update={'block_id': prefix + b.block_id})
                   for b in project_profile.blocks if b.block_id in referenced]
        facts += [f.model_copy(update={'id': prefix + f.id,
                                      'source_blocks': [prefix + key for key in f.source_blocks]})
                  for f in project_profile.facts]
    return ProjectProfile(project_id=block_id, version=revision, blocks=blocks, facts=facts)
