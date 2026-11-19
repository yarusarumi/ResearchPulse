const assert = require('node:assert/strict');
const {waitForPageState} = require('./browser_wait.cjs');
const path = require('node:path');
const {chromium} = require('playwright');
const root = process.argv[2];
const loopback = process.argv[3] === 'loopback';

async function run() {
    const browser = await chromium.launch({
        channel: process.env.NEUROEI_BROWSER_CHANNEL,
        args: ['--disable-gpu'],
    });
    try {
        for (const [width, username] of [[375,'mobile-onboarding'],[1440,'desktop-onboarding']]) {
            const page = await browser.newPage({viewport:{width,height:800}});
            page.setDefaultTimeout(10000);
            const errors = [];
            page.on('pageerror',error=>errors.push(error.message));
            let authorize = 0;
            await page.route('https://accounts.feishu.cn/**',route=>{
                const query = new URL(route.request().url()).searchParams;
                const state = query.get('state');
                assert.ok(state);
                assert.equal(query.get('client_id'), 'cli_fixture_'+width);
                assert.ok(!route.request().url().includes('fixture-secret'));
                const callback = root+'/oauth/feishu/callback';
                if (loopback) assert.equal(query.get('redirect_uri'), callback);
                authorize++;
                const outcome = authorize === 1 ? 'error=access_denied' : 'code=fixture-code';
                return route.fulfill({status:302,headers:{
                    location:callback+'?state='+encodeURIComponent(state)+'&'+outcome,
                },body:''});
            });
            await page.goto(root+'/login');
            await page.locator('#loginForm #username').fill(username);
            await page.locator('#loginForm #password').fill('review-password-1');
            await page.getByRole('button',{name:'登录',exact:true}).click();
            const card = page.locator('#onboarding-card');
            const highlight = page.locator('#onboarding-highlight');
            const expectStep = async (text, target) => {
                await card.getByRole('heading',{name:text,exact:true}).waitFor();
                await highlight.waitFor();
                await waitForPageState(page,selector=>{
                    const target=document.querySelector(selector).getBoundingClientRect();
                    const highlight=document.querySelector('#onboarding-highlight').getBoundingClientRect();
                    const card=document.querySelector('#onboarding-card').getBoundingClientRect();
                    const intersects=(a,b)=>Math.min(a.right,b.right)>Math.max(a.left,b.left)
                        && Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top);
                    return intersects(target,highlight) && !intersects(target,card)
                        && card.left>=0 && card.right<=innerWidth && card.bottom<=innerHeight;
                },target);
            };
            const create = '[data-research-action="create"]';
            await expectStep('新建研究项目',create);
            if (loopback) {
                const session = (await page.context().cookies(root)).find(cookie => cookie.name === 'neuroei_session');
                assert.ok(session);
                assert.equal(session.domain, 'localhost');
                assert.equal(session.sameSite, 'Lax');
                assert.equal(session.httpOnly, true);
            }
            await page.keyboard.press('Escape');
            await card.waitFor({state:'hidden'});
            await page.reload();
            await page.locator(create).waitFor();
            assert.ok(await card.isHidden());
            await page.getByRole('button',{name:'使用指南',exact:true}).click();
            await page.getByRole('button',{name:'开始飞书接入引导',exact:true}).click();
            await expectStep('新建研究项目',create);
            await page.locator(create).click();
            await expectStep('给项目起个名字','#research-title');
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            assert.equal(await page.evaluate(()=>document.activeElement.id),'research-title');
            await page.locator('#research-title').fill('日报项目 '+width);
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            await expectStep('粘贴飞书日报文件夹链接','#research-feishu-url');
            const link = page.locator('#research-feishu-url');
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            await expectStep('粘贴飞书日报文件夹链接','#research-feishu-url');
            assert.equal(await page.evaluate(()=>document.activeElement.id),'research-feishu-url');
            await link.fill('https://example.feishu.cn/drive/folder/missing');
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            await expectStep('用你自己的飞书账号授权','#research-feishu-connect');
            const authorizeOwnApp = async () => {
                await page.locator('#research-feishu-connect').click();
                const dialog = page.locator('#feishu-app-dialog');
                await dialog.waitFor();
                const secret = dialog.locator('#feishu-app-secret');
                assert.equal(await secret.inputValue(), '');
                assert.equal(await secret.getAttribute('type'), 'password');
                await dialog.locator('#feishu-app-id').fill('cli_fixture_'+width);
                await secret.fill('fixture-secret');
                await dialog.locator('#feishu-app-submit').click();
            };
            await authorizeOwnApp();
            await page.getByRole('link',{name:'返回研究台',exact:true}).click();
            await expectStep('用你自己的飞书账号授权','#research-feishu-connect');
            await page.locator('#research-error').getByText('飞书授权未完成。',{exact:false}).waitFor();
            assert.equal(await page.locator('#research-title').inputValue(),'日报项目 '+width);
            await authorizeOwnApp();
            await expectStep('预览可读取的日报','#research-feishu-browse');
            assert.ok(await page.evaluate(()=>![sessionStorage,localStorage].some(
                storage=>Object.values(storage).some(value=>value.includes('fixture-secret')))));
            assert.ok((await link.inputValue()).endsWith('/missing'));
            await page.locator('#research-feishu-browse').click();
            await page.getByText('当前飞书账号无权读取此目录。',{exact:true}).waitFor();
            assert.ok(await page.locator('#research-feishu-choices').isHidden());
            await link.fill('https://example.feishu.cn/drive/folder/journal');
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            await expectStep('预览可读取的日报','#research-feishu-browse');
            await page.locator('#research-feishu-browse').click();
            await expectStep('确认要分析的日报范围','#research-feishu-choices');
            if (process.env.NEUROEI_ONBOARDING_PREVIEW) {
                await page.screenshot({path:path.join(process.env.NEUROEI_ONBOARDING_PREVIEW,'feishu-onboarding-'+width+'.png')});
            }
            await page.getByRole('checkbox',{name:'近期日报'}).check();
            await page.getByRole('checkbox',{name:'历史日报'}).check();
            assert.ok(await page.getByRole('checkbox',{name:/旧快捷方式/}).isDisabled());
            await card.getByRole('button',{name:'确认范围',exact:true}).click();
            await expectStep('补充研究背景','#research-body');
            assert.equal(await page.locator('#research-body').getAttribute('required'), null);
            assert.equal(await page.locator('#research-body').inputValue(), '');
            await card.getByRole('button',{name:'下一步',exact:true}).click();
            await expectStep('保存项目和日报范围','#research-save');
            await page.locator('#research-save').click();
            await expectStep('飞书日报已接入','.project-flow [data-action="project_understanding"][data-analysis-mode="imported"]');
            await card.getByRole('button',{name:'完成',exact:true}).click();
            await card.waitFor({state:'hidden'});
            assert.deepEqual(errors,[]);
            console.log(width+'px: exit, relaunch, consent retry, denied folder, two-file preview and saving passed');
            await page.close();
        }
    } finally {await browser.close();}
}

run().catch(error=>{console.error(error);process.exitCode=1;});
