const assert = require('node:assert/strict');
const {chromium} = require('playwright');

async function run() {
    const browser = await chromium.launch({channel:process.env.NEUROEI_BROWSER_CHANNEL || 'msedge'});
    try {
        const page = await browser.newPage();
        await page.goto(process.argv[2]);
        await page.getByRole('link', {name:'模型设置', exact:true}).click();
        await page.getByRole('button', {name:'登录 Codex', exact:true}).click();
        await page.getByText('请在 OpenAI 页面完成登录。', {exact:true}).waitFor();
        await page.getByRole('button', {name:'刷新模型列表', exact:true}).click();
        await page.getByText('已登录 Codex。以下是当前账号可选的模型。', {exact:true}).waitFor();
        const model = page.getByLabel('Codex 模型', {exact:true});
        const effort = page.getByLabel('推理强度', {exact:true});
        await model.selectOption('example-strong');
        assert.deepEqual(await effort.locator('option').evaluateAll(nodes => nodes.map(option => option.value)), ['medium', 'high']);
        await effort.selectOption('high');
        const save = page.getByRole('button', {name:'保存设置', exact:true});
        const saved = response => response.request().method() === 'POST'
            && new URL(response.url()).pathname === '/api/desktop/models' && response.ok();
        await Promise.all([page.waitForResponse(saved), save.click()]);
        await page.getByText('已保存，对下次启动的任务生效。', {exact:true}).waitFor();
        await model.selectOption('');
        assert.equal(await effort.isDisabled(), true);
        await Promise.all([page.waitForResponse(saved), save.click()]);
        await page.getByText('已保存，对下次启动的任务生效。', {exact:true}).waitFor();
    } finally {
        await browser.close();
    }
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
