const assert = require('node:assert/strict');
const {chromium} = require('playwright');

async function run() {
    const browser = await chromium.launch({channel:process.env.NEUROEI_BROWSER_CHANNEL || 'msedge'});
    try {
        const page = await browser.newPage();
        await page.route('**/api/desktop/codex/models', route => route.fulfill({
            json: {installed:true, signed_in:false, models:[]}
        }), {times:1});
        await page.goto(process.argv[2]);
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
        await Promise.all([page.waitForResponse(saved), save.click()]);
        await page.getByText('已保存，对下次启动的任务生效。', {exact:true}).waitFor();
        await page.getByRole('button', {name:'重新登录或切换账号', exact:true}).click();
        await page.getByRole('button', {name:'刷新模型列表', exact:true}).click();
        await page.getByText('account-2@example.test', {exact:true}).waitFor();
    } finally {
        await browser.close();
    }
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
