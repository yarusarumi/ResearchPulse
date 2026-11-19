"""Render and send the saved daily recommendation and reading report."""
from datetime import datetime
from urllib.parse import urlencode, urlsplit
from zoneinfo import ZoneInfo

from . import account_email, accounts
from .execution import WorkflowBusy, project_lock
from .models import now
from .reading_evidence_gaps import evidence_gaps, reading_unanswered_notes

TZ = ZoneInfo("Asia/Shanghai")
STATUS = {"completed": "已完成", "partial": "部分完成", "skipped": "已跳过",
          "failed": "失败", "unavailable": "全文不可用", "canceled": "已停止"}
REASONS = {
    "no_update": "昨天没有新增或更新的日报，已跳过检索和深读。",
    "no_results": "已分析日报，本次检索没有找到符合条件的论文。",
    "discovery_incomplete": "部分检索来源或筛选证据未通过核验。以下是已保留的结果。",
    "analysis_incomplete": "部分推荐论文的全文或深读证据不足。以下是已保留的结果。",
    "source_failed": "本次未能读取日报，请在研究台检查飞书授权和文件夹权限。",
    "upstream_failed": "日报分析未完成，详情见研究台。",
    "context_too_large": "项目资料超出本次处理范围，未能完成日报分析。",
    "budget_exhausted": "本次处理预算已用完，已保存的内容见下方或研究台。",
    "paper_version_unresolved": "尚未核实全文版本，暂无法深读。",
    "canceled_by_user": "本次任务已停止。",
    "insufficient_cited_papers": "可引用的论文少于两篇，本次未进行跨论文综合。",
    "no_question_with_multiple_cited_papers": "没有同一问题得到多篇论文的引用支持，本次未进行跨论文综合。",
    "synthesis_failed": "跨论文综合未完成，单篇深读结果已保留。",
}
CLAIM_KINDS = {"paper_conclusion": "论文结论", "author_interpretation": "作者解释",
               "inference": "基于证据的推断"}
CONDITIONS = {"applicability": "适用范围", "comparator": "对照", "dataset": "数据集",
              "evaluation": "评估", "metric": "指标", "statistical_unit": "统计单位",
              "task": "任务", "training": "训练设置"}
SYNTHESIS_KINDS = {"transferable_method": "可借用的方法", "comparison": "论文之间的差异",
                   "open_question": "仍待核实", "experiment_proposal": "建议补测"}
VISUAL_KINDS = {"figure": "图", "table": "表", "formula": "公式", "scan_page": "扫描页"}


def _public_url(links):
    for link in links:
        if not link or any(ch.isspace() for ch in link):
            continue
        try:
            url = urlsplit(link)
        except ValueError:
            continue
        if url.scheme in ("https", "http") and url.hostname and not url.username and not url.password:
            return link
    return ""


def _questions(ids, numbers):
    return "、".join(f"问题 {numbers[qid]}" for qid in ids if qid in numbers)


def _reading_lines(analysis, paper, numbers, paper_number, question_ids=()):
    claims = {row["claim_id"]: row for row in analysis.get("claims", [])}
    evidence = {row["evidence_id"]: row for row in analysis.get("evidence", [])}
    labels = {cid: f"{paper_number}.{i}" for i, cid in enumerate(claims, 1)}
    fulltext = _public_url(paper.get("fulltext_links") or [])
    lines = [f"深读状态：{STATUS.get(analysis.get('status'), '状态未知')}"]
    if not claims:
        lines.append("本次尚无通过证据校验的深读结论。")
    shown = set()
    sections = [("论文研究的问题", analysis.get("problem", [])),
                ("研究方法", analysis.get("methods", [])),
                ("主要发现", analysis.get("findings", [])),
                ("局限与适用条件", analysis.get("limitations", [])),
                ("其他已保存结论", list(claims))]
    for heading, ids in sections:
        ids = list(dict.fromkeys(cid for cid in ids if cid in claims and cid not in shown))
        if not ids:
            continue
        lines.extend(["", heading])
        for cid in ids:
            shown.add(cid)
            claim = claims[cid]
            qids = [qid for qid, relevant in (analysis.get("relevance_to_questions") or {}).items() if cid in relevant]
            tag = " · ".join(filter(None, (_questions(qids, numbers), CLAIM_KINDS.get(claim.get("kind")))))
            lines.append(f"[结论 {labels[cid]}]{f'（{tag}）' if tag else ''} {claim['text']}")
            conditions = [f"{CONDITIONS.get(key, key)}：{value}"
                          for key, value in (claim.get("conditions") or {}).items() if value]
            if conditions:
                lines.append("  条件：" + "；".join(conditions))
            for support in claim.get("supports", []):
                source = evidence.get(support.get("evidence_id")) or {}
                pages = sorted({row["page_no"] for row in source.get("provenance", [])
                                if isinstance(row.get("page_no"), int) and row["page_no"] > 0})
                location = " · ".join(filter(None, ("第 " + "、".join(map(str, pages)) + " 页" if pages else "",
                                                     source.get("section"))))
                lines.append(f"  原文依据{f'（{location}）' if location else ''}：{support['quote']}")
                source_url = _public_url([source.get("file")]) or fulltext
                path = urlsplit(source_url).path.lower()
                if pages and (path.endswith(".pdf") or path.startswith("/pdf/")):
                    lines.append(f"  原文定位：{source_url.split('#', 1)[0]}#page={pages[0]}")
    if suggestions := analysis.get("research_suggestions"):
        lines.extend(["", "对项目的研究建议"])
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"{i}. {suggestion['text']}")
            if why := suggestion.get("why"):
                lines.append(f"   理由：{why}")
            refs = "、".join(labels[cid] for cid in suggestion.get("claim_ids", []) if cid in labels)
            if refs:
                lines.append(f"   参考结论：{refs}")
            if parameters := suggestion.get("proposed_parameters"):
                lines.append("   建议参数：" + "；".join(f"{key}：{value}" for key, value in parameters.items()))
    scope = analysis.get("reading_scope") or {}
    if visuals := scope.get("visual_review"):
        lines.extend(["", "论文原图辅助解读（待核对）"])
        for visual in visuals:
            kind = VISUAL_KINDS.get(visual.get("label"), "原图")
            page = visual.get("page_no")
            heading = kind + (f" · 第 {page} 页" if isinstance(page, int) and page > 0 else "")
            lines.append(f"- {heading}")
            if visual.get("status") == "unavailable":
                lines.append("  本次未取得辅助识别结果。")
                continue
            details = visual.get("details") or {}
            if caption := visual.get("caption"):
                lines.append(f"  {caption}")
            for cell in details.get("cells", []):
                if label := " · ".join(filter(None, (cell.get("row"), cell.get("column"), cell.get("value")))):
                    lines.append(f"  {label}")
            for item in details.get("observations", []):
                if observation := item.get("text"):
                    question = _questions([item.get("question_id")], numbers)
                    lines.append(f"  {question} · {observation}" if question else f"  {observation}")
            lines.extend(f"  {note}" for note in [*details.get("legend", []), *details.get("gaps", [])])
    if classified := evidence_gaps(analysis, question_ids):
        lines.extend(["", "证据缺口"])
        for gap in classified:
            if gap["kind"] == "fulltext_unavailable":
                lines.append("- 全文暂不可用，当前无法核对方法和实验。")
            elif gap["kind"] == "author_explicit":
                source = evidence[gap["evidence_id"]]
                location = source.get("section") or source.get("block_id") or "原文"
                lines.append(f"- 原文明示的未报告或未评估范围（{location}）：{gap['quote']}")
            else:
                items = []
                if questions := _questions(gap["question_ids"], numbers):
                    items.append(questions)
                items.extend(f"结论 {labels[row['claim_id']]}的" + "、".join(CONDITIONS[field] for field in row["fields"])
                             for row in gap["conditions"])
                items.extend(gap["notes"])
                if items:
                    lines.append("- 本次未定位到可核验依据：" + "；".join(items))
    warnings = analysis.get("warnings", [])
    unclassified_notes = set(reading_unanswered_notes(scope))
    # The evidence-gap section carries the reading notes that are scoped to this pass.
    gaps = list(dict.fromkeys(note for note in warnings
                              if note not in unclassified_notes and not note.startswith("首次阅读记录")
                              and any("\u4e00" <= char <= "\u9fff" for char in note)))
    review = (scope.get("paperqa_checkpoint") or {}).get("semantic_review_status")
    if review == "model_checked_not_human_verified":
        gaps.append("已完成模型语义复核，引用前请核对原文。")
    elif any(note.split(";", 1)[0] in ("semantic_review_pending", "semantic_review_unavailable") for note in warnings):
        gaps.append("语义核验仍有缺口，引用前请核对原文。")
    if any(note.startswith("Figures/formulas not semantically verified") for note in warnings):
        gaps.append("图表和公式尚未经过语义核验。")
    if gaps:
        lines.extend(["", "未解决的问题与阅读缺口", *(f"- {note}" for note in gaps)])
    return lines, labels


def _body(project, event, job, origin):
    date = datetime.fromisoformat(job.get("created_at", now())).astimezone(TZ).date().isoformat()
    trigger = (event.get("basis") or {}).get("trigger_details") or {}
    window = trigger.get("window") or {}
    journal_day = window.get("since")
    questions = (event.get("input") or {}).get("research_questions") or []
    numbers = {question["id"]: i for i, question in enumerate(questions, 1)}
    discovery = event.get("discovery") or {}
    papers = {paper["paper_id"]: paper for paper in discovery.get("candidates", [])}
    recommendations = discovery.get("recommendations") or []
    analyses = {(analysis["paper_id"], analysis.get("paper_version")): analysis
                for analysis in event.get("analyses") or []}
    errors = {error["paper_id"]: error.get("reason") for error in event.get("errors", [])}
    lines = [
        "ResearchPulse 自动日报 · 研究问题与论文深读",
        "",
        f"项目：{project.get('title', project.get('project_id', ''))}",
        f"运行日期：{date}（北京时间）",
    ]
    if job.get('model_fallback') == 'sol_quota_deepseek':
        from .ai_backend import FALLBACK_NOTICE

        lines.append(FALLBACK_NOTICE)
    if journal_day:
        lines.append(f"日报范围：{datetime.fromisoformat(journal_day).astimezone(TZ).date().isoformat()}（北京时间）")
    lines.append(f"本次状态：{STATUS.get(event.get('status'), '状态未知')}")
    if reason := REASONS.get(event.get("reason")):
        lines.append(reason)
    if questions:
        lines.extend(["", "一、研究问题"])
        for i, question in enumerate(questions, 1):
            description = question["description"].replace("；所需方法：", "\n   所需方法：").replace(
                "；下一步决策（建议）：", "\n   下一步决策（建议）：")
            lines.append(f"问题 {i}：{description.removeprefix('研究问题：')}")
    claim_labels = {}
    if recommendations:
        lines.extend(["", f"二、推荐与深读（{len(recommendations)} 篇）"])
        for i, item in enumerate(recommendations, 1):
            paper_id = item["paper_id"]
            paper = papers.get(paper_id) or {}
            version = paper.get("paper_version")
            lines.extend(["", f"{i}. {paper.get('title') or paper_id}"])
            if author := "、".join(paper.get("authors") or []):
                lines.append(f"作者：{author}")
            if related := _questions(item.get("question_ids", []), numbers):
                lines.append(f"对应研究问题：{related}")
            lines.append(f"推荐理由：{item['reason']}")
            if original := _public_url(paper.get("original_links") or paper.get("fulltext_links") or []):
                lines.append(f"论文原文：{original}")
            for proof in item.get("matched_evidence", []):
                page = proof.get("page_no")
                if proof.get("scope") == "fulltext" and isinstance(page, int) and page > 0:
                    lines.append(f"推荐依据（正文第 {page} 页）：{proof['text']}")
                    url = proof.get("url")
                    if (url in (paper.get("fulltext_links") or []) and proof.get("paper_version") == version
                            and (link := _public_url([url]))):
                        lines.append(f"正文定位：{link.split('#', 1)[0]}#page={page}")
            if analysis := analyses.get((paper_id, version)):
                section, labels = _reading_lines(analysis, paper, numbers, i, item.get('question_ids', ()))
                lines.extend(section)
                claim_labels[paper_id] = labels
            else:
                lines.append(REASONS.get(errors.get(paper_id), "本次没有可用的深读结果。"))
            if event.get("event_id"):
                query = urlencode({"event": event["event_id"], "paper": paper_id})
                lines.append(f"在研究台查看本篇：{origin}/#reading?{query}")
    synthesis = event.get("synthesis") or {}
    if insights := synthesis.get("insights"):
        lines.extend(["", "三、跨论文综合"])
        for insight in insights:
            label = SYNTHESIS_KINDS.get(insight.get("category"), "综合结论")
            question = _questions([insight.get("question_id")], numbers)
            lines.append(f"- {label}{f'（{question}）' if question else ''}：{insight['text']}")
            lines.append(f"  适用条件与限制：{insight['conditions_and_limits']}")
            refs = [claim_labels.get(ref["paper_id"], {}).get(ref["claim_id"]) for ref in insight["references"]]
            if any(refs):
                lines.append("  参考结论：" + "、".join(filter(None, refs)))
    elif recommendations and synthesis.get("status") == "failed":
        lines.extend(["", "三、跨论文综合", REASONS["synthesis_failed"]])
    elif recommendations and (reason := REASONS.get(synthesis.get("reason"))):
        lines.extend(["", "三、跨论文综合", reason])
    if questions and not recommendations and event.get("reason") != "no_results":
        lines.extend(["", "本次没有可发送的论文推荐或深读结果。"])
    lines.extend(["", f"打开研究台：{origin}/#today"])
    return "\n".join(lines)


def send_daily_report(store, account, job):
    if not job.get("snapshot", {}).get("auto_email"):
        return {"status": "not_requested"}
    key = job["job_id"]
    try:
        with project_lock(store.root, "project-daily-email:" + key):
            return _deliver(store, account, job)
    except WorkflowBusy:
        return {"job_id": key, "status": "sending"}


def _deliver(store, account, job):
    from .recommendation_jobs import automatic_project_enabled

    key = job["job_id"]
    prior = store.select("auto_email_delivery", ("status",), key=key)
    if prior and prior[0]["status"] == "sent":
        return store.get("auto_email_delivery", key)
    current = accounts.find(store, account["user_id"])
    reason = ("account_unavailable" if not current or current["disabled"] else
              "automation_disabled" if not automatic_project_enabled(store, job) else
              "verified_email_missing" if not current.get("verified_email") else None)
    if reason:
        result = {"job_id": key, "status": "skipped", "reason": reason, "updated_at": now()}
        store.put("auto_email_delivery", key, result)
        return result
    try:
        from .research_workspace import project_for

        event = store.get("recommendation_event", job["event_id"])
        project = project_for(store, current, job["snapshot"]["project_id"])
        if (job["user_id"] != current["user_id"] or event["user_id"] != current["user_id"]
                or event["project_id"] != project["project_id"] or project["owner_id"] != current["user_id"]):
            raise PermissionError("Daily email scope mismatch")
        mailer = account_email.SMTPMailer.configured()
        if mailer is None:
            raise RuntimeError("smtp_unconfigured")
        date = datetime.fromisoformat(job.get("created_at", now())).astimezone(TZ).date().isoformat()
        title = " ".join(project["title"].split())
        subject = f"ResearchPulse 自动日报 · {title} · {date}"
        body = _body(project, event, job, mailer.origin)
        store.put("auto_email_delivery", key, {"job_id": key, "status": "sending", "updated_at": now()})
        mailer.send_daily_report(current["verified_email"], subject, body)
        result = {"job_id": key, "status": "sent", "email": current["verified_email"], "sent_at": now(),
                  "event_id": job["event_id"]}
    except Exception as exc:
        result = {"job_id": key, "status": "failed", "reason": type(exc).__name__, "updated_at": now(),
                  "event_id": job.get("event_id")}
    store.put("auto_email_delivery", key, result)
    return result


def retry_pending(store):
    results = []
    for row in store.list("auto_email_delivery"):
        if row.get("status") not in ("failed", "sending"):
            continue
        try:
            job = store.get("recommendation_job", row["job_id"])
            account = accounts.find(store, job["user_id"])
            if account and job.get("event_id"):
                results.append(send_daily_report(store, account, job))
        except Exception as exc:
            results.append({"job_id": row.get("job_id"), "status": "failed", "reason": type(exc).__name__})
    return results

