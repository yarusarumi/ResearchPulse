import argparse
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load_config
from .feedback import feedback_channels
from .feishu import Feishu
from .llm import FlashClient
from .models import EvidenceRecord, FeedbackEvent, PaperRecord, now
from .projects import approve, import_project, load_profile
from .readers import compatibility_status
from .reports import daily_report
from .research import analyze, recommend
from .sources import Sources, deduplicate
from .store import Store


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def code_version():
    # Caller must have passed the user's local health check before running Git.
    # Runtime never starts Git scans. Deployment injects the committed revision.
    return os.environ.get("NEUROEI_CODE_REVISION", "unrecorded-working-tree")


def doctor(config):
    readers = compatibility_status(config.data_dir)
    versions = {}
    for package in ("neuroei-literature-agent", "pydantic", "httpx", "jinja2", "paperflow", "paper-qa", "docling"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "data_dir": str(config.data_dir),
            "credentials_present": {k: bool(os.environ.get(k)) for k in
                ("DEEPSEEK_API_KEY", "FEISHU_APP_ID", "FEISHU_APP_SECRET")},
            "versions": versions, "model": config.llm.model, "endpoint": config.llm.base_url,
            "publication_mode": config.runtime.publication_mode,
            "feishu_allowlist": config.feishu.allowed_document_ids,
            "feishu_generated_upload_enabled": config.feishu.generated_upload_enabled,
            "reader_compatibility": readers, "bge_m3": readers["bge_m3"],
            "code_revision": code_version()}


def collect(config, store, project, query, since, until):
    pool_id = uuid.uuid4().hex
    adapter = Sources(config.sources.max_results)
    adapter.coordination_dir = store.root
    try:
        # Each source is independent; retain failure and zero-result states separately.
        results = []
        with ThreadPoolExecutor(max_workers=3) as workers:
            tasks = [workers.submit(adapter.collect, s, query, since, until) for s in config.sources.enabled]
            for future in as_completed(tasks):
                result = future.result()
                results.append(result)
                emit({"source_completed": result.source, "status": result.status,
                      "count": len(result.papers), "error": result.error, "truncated": result.truncated})
                sys.stdout.flush()
                store.put("source_result", f"{pool_id}:{result.source}",
                          {"pool_id": pool_id, "project_id": project, **result.model_dump()}, immutable=True)
        results.sort(key=lambda r: r.source)
    finally:
        adapter.close()
    papers = deduplicate([p for result in results for p in result.papers])
    store.put("pool", pool_id, {"pool_id": pool_id, "project_id": project,
        "created_at": now(), "query": query, "since": since, "until": until,
        "sources": [r.model_dump() for r in results], "papers": [p.model_dump() for p in papers]}, immutable=True)
    return pool_id, results, papers


def run(args, config):
    if args.command == 'media':
        from .media import collect_media_channel, media_snapshot, media_state, render_media_digest, save_media
        store = Store(config.data_dir)
        if args.action == 'import':
            emit(save_media(store, json.loads(Path(args.file).read_text(encoding='utf-8-sig'))))
        elif args.action == 'collect-youtube':
            emit(collect_media_channel(store, {'user_id': args.user, 'direction_id': args.direction,
                                              'channel_id': args.channel}))
        elif args.action == 'collect-wechat':
            from .media import collect_media_article
            emit(collect_media_article(store, {'user_id': args.user, 'direction_id': args.direction,
                                              'url': args.url}))
        elif args.action == 'report':
            print(render_media_digest(media_snapshot(store, args.user, args.direction)))
        elif args.action == 'links':
            from .media_links import analyze_media, render_media_links
            print(render_media_links(analyze_media(store, args.user, args.direction)))
        else:
            emit(media_state(store, args.user, args.direction))
        return
    if args.command == 'direction':
        from .directions import direction_state, save_direction
        store = Store(config.data_dir)
        if args.action == 'save':
            emit(save_direction(store, json.loads(Path(args.file).read_text(encoding='utf-8-sig'))))
        elif args.action == 'list':
            emit(direction_state(store, args.user))
        elif args.action == 'weekly':
            from .direction_weekly import run_direction_weekly
            result = run_direction_weekly(config, args.user, args.id, args.until, since=args.since,
                                           allow_model=not args.reuse_only)
            emit({k: result.get(k) for k in ('report_id', 'path', 'period', 'covered_dates',
                                             'missing_dates', 'llm_requests', 'reused', 'scientific_acceptance')})
            if result.get('status') in ('failed', 'cancelled'):
                raise ValueError('Direction weekly synthesis stopped; inspect its saved attempt')
        else:
            from .direction_workflow import run_direction_daily
            result = run_direction_daily(config, args.user, args.id, report_date=args.date,
                pool_id=args.pool, read_limit=args.read_limit, run_id=args.run_id,
                progress=lambda message: print(json.dumps(message, ensure_ascii=False), flush=True))
            emit({k: v for k, v in result.items() if k != 'synthesis'})
            if result['status'] in ('failed', 'cancelled'):
                raise ValueError('Direction run stopped; inspect its saved state')
        return
    if args.command == 'user':
        from . import accounts
        store = Store(config.data_dir)
        if args.action == 'create':
            password = args.password or prompt_password()
            account = accounts.create(store, args.username, args.name, password, args.role,
                                      adopt_legacy=args.adopt_legacy)
            emit({k: account[k] for k in ('user_id', 'username', 'display_name', 'role')})
        elif args.action == 'list':
            emit([{k: a[k] for k in ('user_id', 'username', 'display_name', 'role', 'disabled')}
                  for a in accounts.accounts(store)])
        elif args.action == 'password':
            account = accounts.by_login(store, args.username)
            if account is None:
                raise ValueError('账号不存在')
            accounts.set_password(store, account['user_id'], args.password or prompt_password())
            emit({'changed': True, 'username': account['username']})
        elif args.action in ('disable', 'enable'):
            account = accounts.by_login(store, args.username)
            if account is None:
                raise ValueError('账号不存在')
            accounts.set_disabled(store, account['user_id'], args.action == 'disable')
            emit({'username': account['username'], 'disabled': args.action == 'disable'})
        elif args.action == 'delete':
            account = accounts.by_login(store, args.username)
            if account is None:
                raise ValueError('账号不存在')
            admins = [a for a in accounts.accounts(store) if a['role'] == 'admin' and not a['disabled']]
            if account['role'] == 'admin' and len(admins) <= 1:
                raise ValueError('不能删除唯一的管理员账号')
            accounts.delete(store, account['user_id'])
            emit({'deleted': True, 'username': account['username']})
        else:
            account = accounts.by_login(store, args.username)
            if account is None:
                raise ValueError('账号不存在')
            if args.revoke:
                accounts.revoke(store, args.project, account['user_id'])
            else:
                accounts.grant(store, args.project, account['user_id'])
            emit({'username': account['username'], 'project': args.project, 'granted': not args.revoke})
        return
    if args.command == 'serve':
        from .portal import serve
        serve(config, args.port)
        return
    if args.command == "doctor":
        emit(doctor(config))
        return
    if args.command == "project" and args.action == "changes":
        from .profile_changes import compare_profiles
        store = Store(config.data_dir, read_only=True)
        emit(compare_profiles(load_profile(store, args.project, approved=True),
                              load_profile(store, args.project, approved=False)))
        return
    if args.command == "evaluate-benchmark":
        from .evaluation import evaluate_benchmark
        payloads = [json.loads(Path(path).read_text(encoding="utf-8-sig"))
                    for path in (args.benchmark, args.review, args.manifest, args.mapping)]
        emit(evaluate_benchmark(*payloads, args.profile_version, k=args.k))
        return
    if args.command == "audit-analysis":
        from .evidence_audit import audit_analysis
        from .models import ProjectProfile
        store = Store(config.data_dir, read_only=True)
        analysis = store.get("analysis", args.analysis)
        version = analysis.get("profile_version")
        profile = ProjectProfile.model_validate(store.get("profile", f"{args.project}:{version}"))
        if profile.project_id != args.project:
            raise ValueError("Analysis profile belongs to another project")
        candidates = [paper for pool in store.list("pool") if pool.get("project_id") == args.project
                      for paper in pool.get("papers", [])
                      if (paper.get("paper_id"), paper.get("version")) ==
                         (analysis.get("paper_id"), analysis.get("paper_version"))]
        if not candidates:
            raise ValueError("Analysis paper/version is absent from this project's stored pools")
        emit({"analysis_id": args.analysis, "project_id": args.project,
              **audit_analysis(analysis, profile, PaperRecord.model_validate(candidates[0]))})
        return
    if args.command == "fetch-evidence":
        store = Store(config.data_dir, read_only=True)
        pool = store.get("pool", args.pool)
        if pool.get("project_id") != args.project:
            raise ValueError("Candidate pool belongs to another project")
        candidates = [p for p in pool.get("papers", [])
                      if (p.get("paper_id"), p.get("version")) == (args.paper, args.version)]
        if len(candidates) != 1:
            raise ValueError("Exactly one stored paper/version is required")
        target = Path(args.output).expanduser().resolve()
        if target.exists():
            raise ValueError("Evidence output already exists; choose a new artifact path")
        paper = PaperRecord.model_validate(candidates[0])
        if paper.paper_id.startswith("arxiv:"):
            from .html_reader import fetch_arxiv_html
            payload = fetch_arxiv_html(paper)
        else:
            from .pmc_reader import fetch_pmc_xml
            payload = fetch_pmc_xml(paper)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        emit({"output": str(target), "paper_id": paper.paper_id, "paper_version": paper.version,
              "evidence_blocks": len(payload["evidence"]), "url": payload["url"],
              "provenance": payload.get("provenance"), "model_calls": 0,
              "supplementary_materials_fetched": False})
        return
    if args.command == "fetch-supplements":
        from .supplement_materials import acquire_supplement_text
        payload = json.loads(Path(args.evidence).read_text(encoding="utf-8-sig"))
        result = acquire_supplement_text(payload, Path(args.output_dir), max_files=args.max_files,
                                          max_pages=args.max_pages)
        emit({k: v for k, v in result.items() if k not in ("evidence", "section_context")}
             | {"evidence_blocks": len(result["evidence"]), "model_calls": 0})
        return
    if args.command == 'trial-status':
        from .operations import trial_status
        emit(trial_status(Store(config.data_dir),args.project,since=args.since,until=args.until))
        return
    if args.command == 'controlled-run':
        from .operations import controlled_run
        result = controlled_run(config,args.project,report_date=args.date,reader_release=args.reader_release,
                                read_limit=args.read_limit,weekly=args.weekly,
                                progress=lambda item:print(json.dumps(item,ensure_ascii=False),flush=True))
        emit(result)
        if result['status'] not in ('completed','partial','busy','awaiting_remote_read','awaiting_remote_rank'):
            raise ValueError('Controlled run needs inspection; no automatic new version was started')
        return
    if args.command == "feedback-view":
        store = Store(config.data_dir)
        emit(feedback_channels([FeedbackEvent.model_validate(e) for e in store.list("feedback")],
             project_id=args.project, user_id=args.user, cutoff=datetime.fromisoformat(args.before)))
        return
    if args.command == "publish":
        raise ValueError("User policy prohibits uploading Agent-generated artifacts to Feishu; use local reports")
    if args.command == "workflow":
        from .workflow import run_daily
        if args.ranking_strategy != 'baseline' and args.weekly:
            raise ValueError('Experimental daily reports are excluded from regular weekly reports; run weekly separately')
        result=run_daily(config,args.project,args.date,pool_id=args.pool,reader_release=args.reader_release,
                         read_limit=args.read_limit,new_version=args.new_version,resume=args.resume,
                         ranking_strategy=args.ranking_strategy,interest_mapping=args.interest_mapping,
                         progress=lambda item: print(json.dumps(item,ensure_ascii=False),flush=True))
        emit({k:v for k,v in result.items() if k not in ('synthesis','recommendations')})
        if result['status']=='failed':
            raise ValueError('Workflow failed; inspect its sanitized checkpoint and resume that run')
        if args.weekly and result['status'] in ('completed','partial'):
            from .workflow import run_weekly
            report=run_weekly(config,args.project,args.date)
            emit({'weekly_report_id':report['report_id'],'path':report['path'],'llm_requests':report['llm_requests']})
        return
    if args.command == "weekly":
        from .workflow import run_weekly
        result=run_weekly(config,args.project,args.until,since=args.since,new_version=args.new_version)
        emit({k:v for k,v in result.items() if k not in ('content','synthesis')})
        return
    store = Store(config.data_dir)
    run_id = uuid.uuid4().hex
    record = {"run_id": run_id, "command": args.command, "started_at": now(),
              "code_revision": code_version(), "status": "running", "environment": doctor(config)}
    client = FlashClient(config.llm, store, run_id)
    store.put("run", run_id, record)
    try:
        if args.command == "model-smoke":
            from pydantic import BaseModel

            class Smoke(BaseModel):
                ok: bool
                evidence_id: str
            result = client.structured('Return {"ok":true,"evidence_id":"public-smoke-001"}.', Smoke, "兼容性样例")
            if not result.ok or result.evidence_id != "public-smoke-001":
                raise ValueError("Model smoke returned incorrect values")
            emit({"run_id": run_id, "structured_response": result.model_dump()})
        elif args.command == "project":
            if args.action == "update-inputs":
                from .project_updates import update_inputs
                emit(update_inputs(store, client, args.project, config.feishu, args.snapshot_key))
            elif args.action == "sync":
                adapter = Feishu(config.feishu.allowed_document_ids,allowed_wiki_tokens=config.feishu.allowed_wiki_tokens)
                try:
                    snapshot = adapter.read_wiki(args.document) if '/wiki/' in args.document or args.document in config.feishu.allowed_wiki_tokens else adapter.read(args.document)
                finally:
                    adapter.close()
                profile = import_project(store, client, args.project, snapshot)
                record["profile_version"] = profile.version
                emit({"run_id": run_id, "project": profile.project_id, "version": profile.version,
                      "facts": len(profile.facts), "unresolved": profile.unresolved,
                      "unread_items": snapshot["unread_items"], "approval": "pending"})
            elif args.action == "show":
                profile = load_profile(store, args.project, approved=not args.preview)
                emit(profile.model_dump())
            elif args.action == "approve":
                approve(store, args.project, args.version, args.reviewer)
                emit({"approved": args.version, "reviewer": args.reviewer})
            elif args.action == "import":
                # Explicit local export import for a document already authorized by the user.
                snapshot = json.loads(Path(args.file).read_text(encoding="utf-8"))
                if any(b["document_id"] not in config.feishu.allowed_document_ids for b in snapshot["blocks"]):
                    raise ValueError("Export contains documents outside allowlist")
                emit(import_project(store, client, args.project, snapshot).model_dump())
            elif args.action == "import-md":
                from .markdown_import import snapshot_markdown
                snapshot = snapshot_markdown([Path(path) for path in args.file])
                profile = import_project(store, client, args.project, snapshot)
                record["profile_version"] = profile.version
                out = config.data_dir / "profiles"
                out.mkdir(exist_ok=True)
                target = out / f"{args.project}-{profile.version}.json"
                target.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
                emit({"run_id": run_id, "project": args.project, "version": profile.version,
                      "path": str(target), "facts": len(profile.facts),
                      "questions": sum(f.kind == "question" for f in profile.facts),
                      "unread_images": len(snapshot["unread_items"]), "approval": "pending"})
        elif args.command == "collect":
            pool_id, results, papers = collect(config, store, args.project, args.query, args.since, args.until)
            record["pool_id"] = pool_id
            emit({"run_id": run_id, "pool_id": pool_id, "candidate_count": len(papers),
                  "sources": [{"source": r.source, "status": r.status, "count": len(r.papers),
                               "error": r.error, "truncated": r.truncated} for r in results]})
        elif args.command in ("recommend", "daily"):
            profile = load_profile(store, args.project, approved=not args.preview)
            record["profile_version"] = profile.version
            pool = store.get("pool", args.pool)
            if pool["project_id"] != args.project:
                raise ValueError("Candidate pool belongs to another project")
            # Latest version per ID is preferred for recommendation, old versions remain in the pool.
            papers = [PaperRecord.model_validate(p) for p in pool["papers"]]
            latest = {}
            for paper in papers:
                score = int(paper.version[1:]) if paper.version.startswith("v") and paper.version[1:].isdigit() else 0
                if paper.paper_id not in latest or score > latest[paper.paper_id][0]:
                    latest[paper.paper_id] = (score, paper)
            papers = [item[1] for item in latest.values()]
            ranking = recommend(client, profile, papers, config.runtime.daily_deep_read_limit)
            result = {"run_id": run_id, "project_id": args.project, "profile_version": profile.version,
                      "pool_id": args.pool, "strategy": "exploratory-bm25-flash", "preview": args.preview,
                      "recommendations": [r.model_dump() for r in ranking]}
            store.put("ranking", run_id, result, immutable=True)
            record["pool_id"] = args.pool
            if args.command == "daily":
                from .models import SourceResult
                content = daily_report(project=profile, report_date=args.date, run_id=run_id,
                    sources=[SourceResult.model_validate(r) for r in pool["sources"]], papers=papers,
                    recommendations=ranking, preview=args.preview)
                # Each draft is versioned; retries never overwrite a reviewed artifact.
                report_id = f"{args.project}:{args.date}:daily:{run_id}"
                report_dir = config.data_dir / "reports"
                report_dir.mkdir(exist_ok=True)
                target = report_dir / f"daily-{args.date}-{run_id}.md"
                target.write_text(content, encoding="utf-8")
                store.put("report", report_id, {**result, "report_id": report_id, "date": args.date,
                    "path": str(target), "status": "draft", "fulltext_verified": False,
                    "content": content}, immutable=True)
                emit({"report_id": report_id, "path": str(target), "cards": len(ranking), "preview": args.preview})
            else:
                emit(result)
        elif args.command == "analyze":
            profile = load_profile(store, args.project, approved=not args.preview)
            payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
            paper = PaperRecord.model_validate(payload["paper"])
            evidence = [EvidenceRecord.model_validate(e) for e in payload["evidence"]]
            result = analyze(client, profile, paper, evidence)
            store.put("analysis", run_id, result, immutable=True)
            emit({"run_id": run_id, **result})
        elif args.command == "feedback":
            event = FeedbackEvent(project_id=args.project, user_id=args.user, paper_id=args.paper,
                                  action=args.action, reason=args.reason, occurred_at=datetime.fromisoformat(args.at))
            event_id = uuid.uuid4().hex
            store.put("feedback", event_id, event.model_dump(mode="json"), immutable=True)
            emit({"feedback_id": event_id, "learning_enabled": False})
        elif args.command == "runs":
            emit(sorted(store.list("run"), key=lambda record: record["started_at"])[-args.limit:])
        record["status"] = "completed"
    except Exception as exc:
        record["status"], record["error_type"] = "failed", type(exc).__name__
        raise
    finally:
        record["ended_at"], record["llm_requests"] = now(), client.calls
        store.put("run", run_id, record)
        client.close()


def prompt_password():
    import getpass
    first = getpass.getpass('Password: ')
    if first != getpass.getpass('Repeat: '):
        raise ValueError('两次输入的密码不一致')
    return first


def parser():
    p = argparse.ArgumentParser(description="ResearchPulse: draft-only project-grounded literature research")
    p.add_argument("--config", default="config.toml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    a = sub.add_parser('serve', help='Review stored local reports and submit explicit feedback; no model calls')
    a.add_argument('--port', type=int, default=8768)
    users = sub.add_parser('user', help='Local lab accounts, sessions and project membership')
    user_actions = users.add_subparsers(dest='action', required=True)
    a = user_actions.add_parser('create', help='Create the first administrator or a member account')
    a.add_argument('--username', required=True)
    a.add_argument('--name', required=True)
    a.add_argument('--role', choices=('admin', 'member'), default='member')
    a.add_argument('--password', help='Omit to be prompted without echoing')
    a.add_argument('--adopt-legacy', action='store_true',
                   help='Allow taking over directions saved before accounts existed')
    user_actions.add_parser('list')
    a = user_actions.add_parser('password')
    a.add_argument('--username', required=True)
    a.add_argument('--password', help='Omit to be prompted without echoing')
    for action in ('disable', 'enable'):
        a = user_actions.add_parser(action)
        a.add_argument('--username', required=True)
    a = user_actions.add_parser('delete', help='Delete an account together with its sessions and project access')
    a.add_argument('--username', required=True)
    a = user_actions.add_parser('access', help='Grant or revoke a member project visibility')
    a.add_argument('--username', required=True)
    a.add_argument('--project', required=True)
    a.add_argument('--revoke', action='store_true')
    sub.add_parser("model-smoke")
    media = sub.add_parser('media', help='Import source text/transcripts and manually collect public video metadata')
    media_actions = media.add_subparsers(dest='action', required=True)
    a = media_actions.add_parser('import')
    a.add_argument('--file', required=True)
    for action in ('list', 'report', 'links', 'collect-youtube', 'collect-wechat'):
        a = media_actions.add_parser(action)
        a.add_argument('--user', required=True)
        a.add_argument('--direction', required=True)
        if action == 'collect-youtube':
            a.add_argument('--channel', required=True)
        if action == 'collect-wechat':
            a.add_argument('--url', required=True)
    directions = sub.add_parser('direction', help='Independent user directions and bounded manual daily progress')
    direction_actions = directions.add_subparsers(dest='action', required=True)
    a = direction_actions.add_parser('save')
    a.add_argument('--file', required=True)
    a = direction_actions.add_parser('list')
    a.add_argument('--user', required=True)
    a = direction_actions.add_parser('daily')
    a.add_argument('--user', required=True)
    a.add_argument('--id', required=True)
    a.add_argument('--date')
    a.add_argument('--pool')
    a.add_argument('--read-limit', type=int, default=1)
    a.add_argument('--run-id')
    a = direction_actions.add_parser('weekly', help='Synthesize existing direction evidence; at most two model calls')
    a.add_argument('--user', required=True)
    a.add_argument('--id', required=True)
    a.add_argument('--until', required=True)
    a.add_argument('--since')
    a.add_argument('--reuse-only', action='store_true', help='Use matching saved judgments; forbid new model calls')
    project = sub.add_parser("project").add_subparsers(dest="action", required=True)
    for action in ("sync", "show", "approve", "import", "import-md", "changes", "update-inputs"):
        a = project.add_parser(action)
        a.add_argument("--project", required=True)
        if action == "sync":
            a.add_argument("--document", required=True)
        if action == "update-inputs":
            a.add_argument("--snapshot-key", action="append", required=True,
                           help="Exact stored snapshots from explicitly authorized Feishu nodes")
        if action == "show":
            a.add_argument("--preview", action="store_true")
        if action == "approve":
            a.add_argument("--version", required=True)
            a.add_argument("--reviewer", required=True)
        if action == "import":
            a.add_argument("--file", required=True)
        if action == "import-md":
            a.add_argument("--file", nargs="+", required=True, help="Explicit authorized Markdown source paths")
    a = sub.add_parser("collect")
    a.add_argument("--project", required=True)
    a.add_argument("--query", required=True, help="Reviewed public search terms only, not internal project text")
    a.add_argument("--since", required=True)
    a.add_argument("--until", default=date.today().isoformat())
    for action in ("recommend", "daily"):
        a = sub.add_parser(action)
        a.add_argument("--project", required=True)
        a.add_argument("--pool", required=True)
        a.add_argument("--preview", action="store_true")
        if action == "daily":
            a.add_argument("--draft", action="store_true", required=True)
            a.add_argument("--date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    a = sub.add_parser("analyze")
    a.add_argument("--project", required=True)
    a.add_argument("--evidence", required=True, help="JSON: paper + versioned located evidence, no abstract substitution")
    a.add_argument("--preview", action="store_true")
    a = sub.add_parser("audit-analysis", help="Recheck stored evidence without model calls or database writes")
    a.add_argument("--project", required=True)
    a.add_argument("--analysis", required=True)
    a = sub.add_parser("fetch-evidence", help="Acquire located public text for a stored paper; no model calls")
    for name in ("project", "pool", "paper", "version", "output"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("fetch-supplements", help="Acquire and parse bounded OA PDF supplements; no model calls")
    a.add_argument("--evidence", required=True, help="Saved Europe PMC evidence payload with original identities")
    a.add_argument("--output-dir", required=True)
    a.add_argument("--max-files", type=int, default=3)
    a.add_argument("--max-pages", type=int, default=40)
    a = sub.add_parser("evaluate-benchmark", help="Evaluate frozen rankings against explicit human labels")
    for name in ("benchmark", "review", "manifest", "mapping", "profile-version"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--k", type=int, default=5)
    a = sub.add_parser("feedback")
    for name in ("project", "user", "paper", "action"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--reason", default="")
    a.add_argument("--at", default=now())
    a = sub.add_parser("feedback-view", help="Read explicit feedback channels without updating a profile")
    for name in ("project", "user", "before"):
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("runs")
    a.add_argument("--limit", type=int, default=10)
    sub.add_parser("publish")
    a=sub.add_parser("workflow",help="Acquire, recommend, deep-read and create a local research draft")
    a.add_argument('--project',required=True)
    a.add_argument('--date',default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    a.add_argument('--pool',help="Replay an explicit stored pool instead of collecting current sources")
    a.add_argument('--reader-release',help="Explicit immutable server release containing the generic PaperQA reader")
    a.add_argument('--ranking-strategy',choices=('baseline','paperflow-experimental'),default='baseline')
    a.add_argument('--interest-mapping',help='Explicit stored source-linked mapping ID for experimental PaperFlow')
    a.add_argument('--read-limit',type=int,default=3)
    a.add_argument('--new-version',action='store_true')
    a.add_argument('--resume')
    a.add_argument('--weekly',action='store_true',help='Also synthesize the available daily evidence in the seven-day window')
    a=sub.add_parser('weekly',help='Synthesize existing evidence-bearing daily drafts')
    a.add_argument('--project',required=True)
    a.add_argument('--since')
    a.add_argument('--until',default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    a.add_argument('--new-version',action='store_true')
    a=sub.add_parser('trial-status',help='Observed execution days, separate from scientific acceptance')
    a.add_argument('--project',required=True)
    a.add_argument('--since')
    a.add_argument('--until')
    a=sub.add_parser('controlled-run',help='Bounded baseline drafts; no blind checkpoint restart')
    a.add_argument('--project',required=True)
    a.add_argument('--date')
    a.add_argument('--reader-release',required=True)
    a.add_argument('--read-limit',type=int,default=2)
    a.add_argument('--weekly',action='store_true')
    return p


def main():
    args = parser().parse_args()
    try:
        run(args, load_config(args.config))
    except Exception as exc:  # noqa: BLE001 - CLI boundary sanitizes upstream secrets
        # Upstream exception bodies may contain secrets: give sanitized diagnostics only.
        emit({"status": "failed", "error_type": type(exc).__name__,
              "message": str(exc) if type(exc) is ValueError
              else "Operation failed; inspect sanitized run/audit metadata"})
        sys.exit(1)


if __name__ == "__main__":
    main()
