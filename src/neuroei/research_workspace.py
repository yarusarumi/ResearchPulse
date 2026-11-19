"""Account-scoped manual research inputs and snapshots for the existing worker."""
import json
from datetime import date, datetime
from hashlib import sha256
from typing import Annotated
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import Field, StringConstraints, field_validator

from . import accounts
from .config import StrictModel
from .models import now
from .project_understanding import MAX_PROJECT_CONTEXT_CHARS, JournalBatch, current_items, identity
from .scope_deletion import deleted_scopes, require_active, scope_deleted

TZ = ZoneInfo('Asia/Shanghai')
BACKGROUND_LABELS = {'goal': '研究目标', 'question': '研究问题', 'method': '研究方法', 'data': '数据',
                     'setting': '实验设置', 'difficulty': '困难', 'plan': '计划', 'result': '已记录结果'}


class FeishuSourceInput(StrictModel):
    feishu_folder: str | None = Field(default=None, max_length=1000)
    feishu_selection: list[str] = Field(default_factory=list, max_length=100)


class ProjectDetails(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    background: str = Field(max_length=100000)

    @field_validator('title')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('请输入项目名称。')
        return value

    @field_validator('background')
    @classmethod
    def empty_background(cls, value):
        return value if value.strip() else ''


class ProjectInput(ProjectDetails, FeishuSourceInput):
    background: str = Field(default='', max_length=100000)


class ProjectUpdate(ProjectDetails):
    project_id: str
    version: str


class ProjectChanged(ValueError):
    pass


class ProjectBriefUpdate(StrictModel):
    project_id: str
    version: str
    edits: dict[str, Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(max_length=11)


class ProjectAutomationUpdate(StrictModel):
    project_id: str
    enabled: bool


class ProjectFeishuInput(FeishuSourceInput):
    project_id: str
    feishu_folder: str = Field(min_length=1, max_length=1000)


class JournalInput(StrictModel):
    project_id: str
    title: str = Field(min_length=1, max_length=200)
    record_date: date
    text: str = Field(min_length=1, max_length=100000)

    @field_validator('title', 'text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('请输入日报标题和正文。')
        return value


def project_for(store, account, project_id):
    require_active(store, project_id=project_id)
    project = store.get('research_project', project_id)
    allowed = accounts.projects_for(store, account)
    if allowed is not None and project_id not in allowed:
        raise PermissionError('无权访问这个项目。')
    return project


def record(user_id, project_id, source_id, text, record_date, stamp):
    version = sha256(text.encode()).hexdigest()[:24]
    return {'user_id': user_id, 'project_id': project_id, 'source_id': source_id,
            'record_date': record_date, 'available_at': stamp, 'source_modified_at': None,
            'synced_at': stamp, 'blocks': [{'document_id': source_id, 'block_id': 'text',
                'revision': version, 'section': '用户导入原文', 'text': text}]}


def feishu_binding(store, account, data):
    from .feishu_folders import browse, selected_rows

    catalog = browse(store, account, data.feishu_folder)
    selected = sorted(set(data.feishu_selection))
    selected_rows({row['token']: row for row in catalog['entries']}, selected)
    return {'url': catalog['url'], 'selection': selected}


def create_project(store, account, payload):
    data = ProjectInput.model_validate(payload)
    project_id, stamp = 'project-'+uuid4().hex[:16], now()
    background = (record(account['user_id'], project_id, project_id+':background', data.background, None, stamp)
                  if data.background else None)
    project = {'project_id': project_id, 'title': data.title, 'owner_id': account['user_id'],
               'created_at': stamp, 'background': background, 'automation': {'daily_enabled': False, 'daily_hour': 5, 'timezone': 'Asia/Shanghai'}}
    if data.feishu_folder:
        project['feishu'] = feishu_binding(store, account, data)
    elif data.feishu_selection:
        raise ValueError('请先选择飞书文件夹。')
    # The project and creator's membership become visible together.
    with store.transaction() as db:
        db.execute('INSERT INTO objects VALUES (?,?,?)', ('research_project', project_id, json.dumps(project)))
        membership = {'project_id': project_id, 'user_id': account['user_id'], 'granted_at': stamp}
        db.execute('INSERT INTO objects VALUES (?,?,?)',
                   ('project_access', project_id+':'+account['user_id'], json.dumps(membership)))
    return {'project_id': project_id, 'title': data.title}


def editable_project(project, account):
    if project['owner_id'] != account['user_id']:
        raise PermissionError('请使用项目创建账号编辑项目。')
    background = project['background']
    return {'project_id': project['project_id'], 'title': project['title'],
            'background': background['blocks'][0]['text'] if background else '',
            'version': identity([project['title'], project['background']])}


def update_project(store, account, payload):
    data = ProjectUpdate.model_validate(payload)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        require_active(store, project_id=data.project_id, db=db)
        allowed = accounts.projects_for(store, account)
        if allowed is not None and data.project_id not in allowed:
            raise PermissionError('无权访问这个项目。')
        row = db.execute("SELECT body FROM objects WHERE kind='research_project' AND id=?",
                         (data.project_id,)).fetchone()
        if not row:
            raise ValueError('项目不存在。')
        project = json.loads(row[0])
        current = editable_project(project, account)
        if (data.title, data.background) == (current['title'], current['background']):
            return current
        if data.version != current['version']:
            raise ProjectChanged('项目已在其他页面修改。请复制需要保留的内容，关闭后重新打开编辑。')
        project['title'] = data.title
        if data.background != current['background']:
            project['background'] = (record(account['user_id'], data.project_id, data.project_id+':background',
                                            data.background, None, now()) if data.background else None)
            project['background_edited'] = True
        db.execute("UPDATE objects SET body=? WHERE kind='research_project' AND id=?",
                   (json.dumps(project), data.project_id))
    return editable_project(project, account)


def update_feishu_source(store, account, payload):
    data = ProjectFeishuInput.model_validate(payload)
    project = project_for(store, account, data.project_id)
    if project['owner_id'] != account['user_id']:
        raise PermissionError('请使用项目创建账号编辑飞书日报来源。')
    binding = feishu_binding(store, account, data)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        require_active(store, project_id=data.project_id, db=db)
        db.execute("UPDATE objects SET body=json_set(body,'$.feishu',json(?)) WHERE kind='research_project' AND id=?",
                   (json.dumps(binding), data.project_id))
    return {'project_id': data.project_id, 'feishu': binding}


def update_project_automation(store, account, payload):
    data = ProjectAutomationUpdate.model_validate(payload)
    project = project_for(store, account, data.project_id)
    if project['owner_id'] != account['user_id']:
        raise PermissionError('请使用项目创建账号修改自动检测设置。')
    automation = {'daily_enabled': data.enabled, 'daily_hour': 5, 'timezone': 'Asia/Shanghai',
                  'updated_at': now()}
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        require_active(store, project_id=data.project_id, db=db)
        db.execute("UPDATE objects SET body=json_set(body,'$.automation',json(?)) WHERE kind='research_project' AND id=?",
                   (json.dumps(automation), data.project_id))
    return {'project_id': data.project_id, 'automation': automation}


def import_journal(store, account, payload):
    data = JournalInput.model_validate(payload)
    project_for(store, account, data.project_id)
    if data.record_date > datetime.now(TZ).date():
        raise ValueError('日报日期不能晚于今天。')
    source_id = 'journal:'+identity([data.project_id, account['user_id'], str(data.record_date), data.title])
    body = record(account['user_id'], data.project_id, source_id, data.text, data.record_date.isoformat(), now())
    checked = JournalBatch.model_validate({'records': [body], 'source_runs': [{'source_id': source_id, 'status': 'read'}]})
    version = checked.records[0].content_version
    key = source_id+':'+version
    saved = {'journal_id': key, 'title': data.title, 'user_id': account['user_id'],
             'project_id': data.project_id, 'record': body, 'content_version': version}
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        require_active(store, project_id=data.project_id, db=db)
        db.execute('INSERT OR IGNORE INTO objects VALUES (?,?,?)', ('research_journal', key, json.dumps(saved)))
        db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?)',
                   ('research_journal_current', source_id, json.dumps({'journal_id': key})))
    return {'journal_id': key, 'content_version': version, 'project_id': data.project_id}


def current_journals(store, account, project_id):
    project_for(store, account, project_id)
    rows = []
    for pointer in store.list('research_journal_current'):
        journal = store.get('research_journal', pointer['journal_id'])
        if (journal['user_id'], journal['project_id']) == (account['user_id'], project_id):
            rows.append(journal)
    return sorted(rows, key=lambda row: (row['record']['record_date'], row['journal_id']))


def generated_background_sections(project_id, summary):
    """Select the latest sourced entry for each background section."""
    latest = {}
    for item in reversed(summary):
        kind = item['kind']
        if (kind in BACKGROUND_LABELS and kind not in latest and item['state_status'] == 'recorded_candidate'
                and item['epistemic'] == 'recorded'
                and any(ref['source_id'] != project_id+':background' for ref in item['citations'])):
            latest[kind] = {'kind': kind, 'title': BACKGROUND_LABELS[kind], 'text': item['text'],
                            'conditions': item['conditions']}
    return [latest[kind] for kind in BACKGROUND_LABELS if kind in latest]


def project_brief_sections(project, summary, brief):
    from .project_brief import TITLES

    originals = (brief['sections'] if brief else
                 [] if project['background'] else generated_background_sections(project['project_id'], summary))
    edits = project.get('brief_edits', {})
    sections = {section['kind']: {**section, 'ai_text': section['text'], 'manually_edited': section['kind'] in edits,
                                  'text': edits.get(section['kind'], section['text'])}
                for section in originals}
    # Older briefs use additional categories. Keep edits if a later synthesis has no matching category.
    for kind, text in edits.items():
        if kind not in sections:
            sections[kind] = {'kind': kind, 'title': TITLES.get(kind) or BACKGROUND_LABELS[kind],
                              'ai_text': '', 'text': text, 'manually_edited': True,
                              'conditions': [], 'sources': []}
    version = identity([brief['state_version'] if brief else None, originals, edits]) if sections else None
    return list(sections.values()), version


def update_project_brief(store, account, payload):
    from .project_brief import saved_brief

    data = ProjectBriefUpdate.model_validate(payload)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        project = project_for(store, account, data.project_id)
        if project['owner_id'] != account['user_id']:
            raise PermissionError('请使用项目创建账号编辑项目背景。')
        heads = store.select('understanding_head', ('key',), key=identity([account['user_id'], data.project_id]))
        summary = current_items(store.get('understanding_state', heads[0]['key'])) if heads else []
        brief = saved_brief(store, account['user_id'], data.project_id) if heads else None
        sections, version = project_brief_sections(project, summary, brief)
        if version is None:
            raise ProjectChanged('请先生成项目背景，再编辑各节内容。')
        if version != data.version:
            raise ProjectChanged('项目背景已更新。请保留你的修改并重新打开编辑。')
        by_kind = {section['kind']: section for section in sections}
        if not data.edits.keys() <= by_kind.keys():
            raise ValueError('Invalid project brief section')
        project['brief_edits'] = {kind: text for kind, text in data.edits.items()
                                  if text != by_kind[kind]['ai_text'].strip()}
        db.execute("UPDATE objects SET body=? WHERE kind='research_project' AND id=?",
                   (json.dumps(project, ensure_ascii=False), data.project_id))
        sections, version = project_brief_sections(project, summary, brief)
    return {'project_id': data.project_id, 'brief_edits_version': version,
            'generated_background_sections': sections, 'generated_background': generated_background(sections)}


def generated_background(sections):
    """Keep model context compact while the workspace can display each complete section."""
    lines = []
    for section in sorted(sections, key=lambda section: not section.get('manually_edited', False)):
        text = section['text']
        if section['conditions'] and not section.get('manually_edited'):
            text += '（条件：' + '；'.join(section['conditions']) + '）'
        if len(text) > 260:
            text = text[:259] + '…'
        lines.append(f"{section['title']}：{text}")
    return '；'.join(lines)[:MAX_PROJECT_CONTEXT_CHARS]


def analysis_snapshot(store, account, project_id):
    project = project_for(store, account, project_id)
    if project['owner_id'] != account['user_id']:
        raise PermissionError('请使用项目创建账号分析其导入资料。')
    journals = current_journals(store, account, project_id)
    snapshot = {'project_id': project_id, 'project_title': project['title'],
            'background': project['background'], 'journal_versions': [j['journal_id'] for j in journals],
            'batch': {'records': [j['record'] for j in journals],
                      'source_runs': [{'source_id': project_id+':manual-imports', 'status': 'read'}]}}
    if not snapshot['background']:
        heads = store.select('understanding_head', ('key',), key=identity([account['user_id'], project_id]))
        if heads:
            from .project_brief import saved_brief

            state = store.get('understanding_state', heads[0]['key'])
            brief = saved_brief(store, account['user_id'], project_id)
            sections, _ = project_brief_sections(project, current_items(state), brief)
            snapshot['project_context'] = generated_background(sections)
    if project.get('feishu'):
        from .feishu_user_auth import current_grant_id

        snapshot['feishu'] = project['feishu']
        snapshot['feishu_grant_id'] = current_grant_id(store, account)
    if project.get('background_edited'):
        snapshot['background_edited'] = True
    return snapshot


def workspace_state(store, account):
    from .feishu_user_auth import configured, connection_state
    from .project_brief import saved_brief

    allowed = accounts.projects_for(store, account)
    deleted = deleted_scopes(store)
    projects = []
    heads = {key: value for key, value in store.items('understanding_head')}
    for project in store.list('research_project'):
        if (project['owner_id'] != account['user_id'] or scope_deleted(project, deleted)
                or (allowed is not None and project['project_id'] not in allowed)):
            continue
        project_id = project['project_id']
        pointer = heads.get(identity([account['user_id'], project_id]))
        state = store.get('understanding_state', pointer['key']) if pointer else None
        background = project['background']
        summary = current_items(state) if state else []
        brief = saved_brief(store, account['user_id'], project_id) if state else None
        sections, brief_edits_version = project_brief_sections(project, summary, brief)
        projects.append({'project_id': project_id, 'title': project['title'], 'initialized': state is not None,
                         'background': background['blocks'][0]['text'] if background else '',
                         'generated_background': generated_background(sections), 'generated_background_sections': sections,
                         'generated_background_policy': brief['policy'] if brief else None,
                         'brief_edits_version': brief_edits_version,
                         'state_version': state['version'] if state else None,
                         'summary': summary,
                         'automation': project.get('automation', {'daily_enabled': False, 'daily_hour': 5, 'timezone': 'Asia/Shanghai'}),
                         'journals': [{'journal_id': j['journal_id'], 'title': j['title'],
                                       'record_date': j['record']['record_date'], 'content_version': j['content_version']}
                                      for j in current_journals(store, account, project_id)]})
        if project.get('feishu'):
            projects[-1]['feishu'] = project['feishu']
    result = {'projects': projects}
    if configured():
        result['feishu_folders_enabled'] = True
        result['feishu_connection'] = connection_state(store, account)
    return result
