"""Queue enabled project daily checks at the configured Asia/Shanghai morning time."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import accounts
from .models import now
from .recommendation_jobs import enqueue
from .store import Store

TZ = ZoneInfo("Asia/Shanghai")


def run(store, *, submit=True, as_of=None, project_id=None, email=True):
    stamp = as_of or now()
    rows = []
    from .project_daily_email import retry_pending

    email_retries = retry_pending(store) if email and submit and project_id is None else []
    for project in store.list("research_project"):
        automation = project.get("automation") or {}
        if project_id is not None and project["project_id"] != project_id:
            continue
        if project_id is None and not automation.get("daily_enabled"):
            continue
        owner = accounts.find(store, project.get("owner_id"))
        row = {"project_id": project.get("project_id"), "title": project.get("title", "")}
        if not owner or owner.get("disabled"):
            row["reason"] = "account_unavailable"
            rows.append(row)
            continue
        if not project.get("feishu"):
            row["reason"] = "feishu_not_configured"
            rows.append(row)
            continue
        target_day = datetime.fromisoformat(stamp).astimezone(TZ).date().isoformat()
        prior = next((job for job in store.list("recommendation_job")
                      if job.get("user_id") == owner["user_id"]
                      and job.get("action") == "project_understanding"
                      and job.get("snapshot", {}).get("project_id") == project["project_id"]
                      and job.get("snapshot", {}).get("analysis_mode") == "daily"
                      and job["snapshot"].get("auto_dispatch")
                      and datetime.fromisoformat(job.get("created_at", stamp)).astimezone(TZ).date().isoformat() == target_day
                      and job.get("status") in ("queued", "processing", "completed", "partial", "skipped")), None)
        if prior:
            row.update(reason="already_recorded", job_id=prior["job_id"], status=prior["status"])
            rows.append(row)
            continue
        if not submit:
            row["action"] = "project_understanding"
            rows.append(row)
            continue
        try:
            job = enqueue(store, owner, {
                "action": "project_understanding",
                "project_id": project["project_id"],
                "analysis_mode": "daily",
                "auto_dispatch": True,
                "auto_email": email,
            }, as_of=stamp, scheduled_project_daily=project_id is None)
            row.update(action="project_understanding", job_id=job["job_id"], status=job["status"])
        except (ValueError, PermissionError) as exc:
            row.update(reason=type(exc).__name__, detail=str(exc))
        rows.append(row)
    return {"as_of": stamp, "timezone": "Asia/Shanghai", "submitted": submit,
            "projects": rows, "email_retries": email_retries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path,
                        default=Path(os.environ.get("NEUROEI_DATA_DIR", "data")))
    parser.add_argument("--as-of", default=now())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-id", help="Run one project immediately without changing its recurring schedule")
    args = parser.parse_args()
    store = Store(args.store, read_only=args.dry_run)
    try:
        print(json.dumps(run(store, submit=not args.dry_run, as_of=args.as_of,
                             project_id=args.project_id), ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()

