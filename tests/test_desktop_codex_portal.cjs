const assert = require('node:assert/strict');
const {chromium} = require('playwright');

async function checkWorkerStartup(page, url) {
    let ready = false;
    let dailyFailed = false;
    let workspaceReads = 0;
    let submitted = 0;
    const workspacePath = '**/api/research-workspace';
    const jobsPath = '**/api/recommendation-jobs';
    const deliveryPath = '**/api/recommendation-deliveries';
    const stamp = '2026-11-20T01:00:00+00:00';
    const project = {project_id:'startup-project', title:'本机启动验收', initialized:false, summary:[],
        journals:[{journal_id:'local-journal', title:'测试日报', record_date:'2026-11-20'}]};
    const canceled = {job_id:'stopped-before-restart', action:'project_understanding',
        snapshot:{project_id:project.project_id, analysis_mode:'imported'}, status:'canceled',
        reason:'canceled_by_user', created_at:stamp, started_at:stamp, ended_at:stamp};
    const watchSubmissions = request => {
        if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/recommendation-job') submitted++;
    };
    page.on('request', watchSubmissions);
    await page.route(workspacePath, route => {
        workspaceReads++;
        return route.fulfill({json:{projects:[project], feishu_folders_enabled:false}});
    });
    const failureStamp = new Date().toISOString();
    const failed = {...canceled, job_id:'daily-failed', snapshot:{...canceled.snapshot, analysis_mode:'daily'},
        status:'failed', reason:'worker_failed', created_at:failureStamp, started_at:failureStamp, ended_at:failureStamp};
    await page.route(jobsPath, route => route.fulfill({json:{jobs:dailyFailed?[failed,canceled]:[canceled], worker:{
        desktop:true, model_configured:true, status:ready?'ready':'starting', understanding_enabled:ready,
    }}}));
    await page.route(deliveryPath, route => route.fulfill({json:{deliveries:dailyFailed?[{
        project_id:project.project_id, status:'awaiting_review', request_id:'imported-delivery',
        received_at:stamp, trigger_details:{selection:'explicit_import'},
    }]:[]}}));
    try {
        await page.goto(url);
        const generate = page.getByRole('button', {name:'生成项目背景', exact:true});
        await generate.waitFor();
        assert.equal(await generate.isDisabled(), true);
        const preparing = page.getByText('正在准备分析服务，就绪后可开始分析。', {exact:true});
        await preparing.waitFor();
        assert.equal(await page.getByText('项目分析暂不可用，请稍后刷新。', {exact:true}).count(), 0);
        await page.waitForResponse(response => new URL(response.url()).pathname === '/api/recommendation-jobs');
        assert.equal(workspaceReads, 1);
        ready = true;
        await page.waitForFunction(() => {
            const button = document.querySelector('button[data-action="project_understanding"][data-analysis-mode="imported"]');
            return button && !button.disabled;
        }, null, {timeout:12000});
        await preparing.waitFor({state:'detached'});
        assert.equal(submitted, 0);
        dailyFailed = true;
        project.initialized = true;
        project.feishu = {url:'https://example.feishu.cn/drive/folder/journals'};
        await page.reload();
        await page.locator('#project-work').getByText(/运行失败 · 本机后台遇到异常/).waitFor();
        await page.getByText(/首次分析已纳入项目背景的日报时，请用「手动选择段落」/).waitFor();
        assert.equal(await page.getByText(/下方待确认的问题来自/).count(), 0);
        assert.equal(await page.getByRole('button', {name:/研究问题确认/}).count(), 0);
    } finally {
        page.off('request', watchSubmissions);
        await page.unroute(workspacePath);
        await page.unroute(jobsPath);
        await page.unroute(deliveryPath);
    }
}

async function run() {
    const browser = await chromium.launch({channel:process.env.NEUROEI_BROWSER_CHANNEL || 'msedge'});
    try {
        const page = await browser.newPage();
        await page.route('**/api/desktop/codex/models', route => route.fulfill({
            json: {installed:true, signed_in:false, models:[]}
        }), {times:1});
        await checkWorkerStartup(page, process.argv[2]);
        await page.getByRole('link', {name:'模型设置', exact:true}).click();
        await page.getByText(/研究台后台仍在使用旧版/).waitFor();
        await page.getByRole('button', {name:'刷新模型列表', exact:true}).click();
        await page.getByText('请在 Codex 打开的官方浏览器页面登录。', {exact:true}).waitFor();
        await page.getByRole('button', {name:'登录 Codex', exact:true}).click();
        await page.getByText('请在 OpenAI 页面完成登录。', {exact:true}).waitFor();
        await page.getByRole('button', {name:'刷新模型列表', exact:true}).click();
        await page.getByText('已登录 Codex。以下是当前账号可选的模型。', {exact:true}).waitFor();
        await page.getByText('account-1@example.test', {exact:true}).waitFor();
        await page.getByText(/5 小时额度剩余 80%.*重置/).waitFor();
        await page.getByText('周额度剩余 62.5%', {exact:true}).waitFor();
        const model = page.getByLabel('Codex 模型', {exact:true});
        const effort = page.getByLabel('推理强度', {exact:true});
        const speed = page.getByLabel('响应速度', {exact:true});
        const imageWorkers = page.getByLabel('Codex 图片识别并发');
        assert.equal(await imageWorkers.inputValue(), '4');
        assert.deepEqual(await imageWorkers.locator('option').evaluateAll(nodes => nodes.map(option => option.value)), ['1','2','3','4']);
        await model.selectOption('example-strong');
        assert.deepEqual(await effort.locator('option').evaluateAll(nodes => nodes.map(option => option.value)), ['', 'medium', 'high']);
        assert.deepEqual(await effort.locator('option').allTextContents(), ['跟随模型默认值','中（medium）','高（high）']);
        assert.equal(await speed.isDisabled(), true);
        await effort.selectOption('high');
        const save = page.getByRole('button', {name:'保存设置', exact:true});
        const saved = response => response.request().method() === 'POST'
            && new URL(response.url()).pathname === '/api/desktop/models' && response.ok();
        await Promise.all([page.waitForResponse(saved), save.click()]);
        await page.getByText('已保存，对下次启动的任务生效。', {exact:true}).waitFor();
        await model.selectOption('');
        await effort.selectOption('medium');
        await speed.selectOption('fast');
        await imageWorkers.selectOption('3');
        await Promise.all([page.waitForResponse(saved), save.click()]);
        await page.getByText('已保存，对下次启动的任务生效。', {exact:true}).waitFor();
        await page.getByRole('button', {name:'重新登录或切换账号', exact:true}).click();
        await page.getByRole('button', {name:'刷新模型列表', exact:true}).click();
        await page.getByText('account-2@example.test', {exact:true}).waitFor();
        await page.getByRole('link', {name:'返回研究台'}).click();
        await page.getByRole('link', {name:'邮件设置', exact:true}).click();
        await page.getByLabel('邮箱地址').fill('owner@example.invalid');
        await page.getByLabel('SMTP 服务器').fill('smtp.example.invalid');
        const code = page.getByLabel('SMTP 授权码');
        await code.fill('mail-ui-fixture-code');
        await page.getByRole('button', {name:'保存设置', exact:true}).click();
        await page.getByText('已保存。请发送一封测试邮件验证设置。', {exact:true}).waitFor();
        assert.equal(await code.inputValue(), '');
        await page.route('**/api/desktop/mail/test', route => route.fulfill({json:{sent:true}}), {times:1});
        await page.getByRole('button', {name:'发送测试邮件', exact:true}).click();
        await page.getByText(/测试邮件已发送到已保存的邮箱/).waitFor();
        await page.reload();
        await page.getByText('已保存邮件设置。', {exact:true}).waitFor();
        assert.equal(await page.getByLabel('邮箱地址').inputValue(), 'owner@example.invalid');
        assert.equal(await code.inputValue(), '');
    } finally {
        await browser.close();
    }
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
