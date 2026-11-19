const {setTimeout: delay} = require('node:timers/promises');

// Keep the page's real CSP active. Playwright's waitForFunction polls via in-page eval.
async function waitForPageState(page, predicate, arg, timeout = 6000) {
    const deadline = Date.now() + timeout;
    do {
        if (await page.evaluate(predicate, arg)) return;
        await delay(50);
    } while (Date.now() < deadline);
    throw new Error(`Page state did not match within ${timeout} ms: ${predicate}`);
}

module.exports = {waitForPageState};
