async (page) => {
  const report = {checks:{}, errors:[], remoteRequests:[],
    transport:'localhost HTML; original response reloaded from memory with network offline'};
  const check = (name, value) => {
    report.checks[name] = value;
    if (!value) throw new Error('Go14 report check failed: ' + name);
  };
  page.on('pageerror', error => report.errors.push(String(error)));
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !request.url().startsWith('http://127.0.0.1:8784/')) report.remoteRequests.push(request.url());
  });
  const frozen = await (await page.request.get(page.url())).text();
  await page.route('**/go14_report.html', route => route.fulfill({status:200,contentType:'text/html; charset=utf-8',body:frozen}));
  await page.context().setOffline(true);
  await page.reload();
  await page.setViewportSize({width:1440, height:1000});
  await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
  report.dom = await page.evaluate(() => {
    const ids = [...document.querySelectorAll('[id]')].map(el => el.id);
    const data = JSON.parse(document.getElementById('evidence').textContent);
    return {
      math:document.querySelectorAll('math').length,
      displayMath:document.querySelectorAll('math[display="block"]').length,
      duplicateIds:ids.filter((id, i) => ids.indexOf(id) !== i),
      missingTargets:[...document.querySelectorAll('a[href^="#"]')].filter(el =>
        !document.getElementById(decodeURIComponent(el.hash.slice(1)))).map(el => el.hash),
      loadedImages:[...document.images].every(el => el.complete && el.naturalWidth > 0),
      externalAssets:document.querySelectorAll('script[src],link[rel="stylesheet"]').length,
      overflow:document.documentElement.scrollWidth > innerWidth,
      unresolvedMath:document.querySelector('main').innerText.includes('$$'),
      completeGames:data.complete_games,
      layouts:Object.keys(data.layouts).length,
      methods:Object.keys(data.table).length
    };
  });
  check('native_mathml',report.dom.math >= 100 && report.dom.displayMath >= 25);
  check('unique_ids_and_navigation',report.dom.duplicateIds.length === 0 && report.dom.missingTargets.length === 0);
  check('embedded_assets',report.dom.loadedImages && report.dom.externalAssets === 0);
  check('desktop_no_page_overflow',!report.dom.overflow);
  check('no_unconverted_tex',!report.dom.unresolvedMath);
  check('complete_evidence_matrix',report.dom.completeGames === 860 && report.dom.layouts === 9 && report.dom.methods === 22);
  const methods = await page.locator('#method option').evaluateAll(options => options.map(o => o.value));
  for (const method of methods) {
    await page.locator('#method').selectOption(method);
    const valid = await page.evaluate(name => {
      const row = JSON.parse(document.getElementById('evidence').textContent).table[name];
      const text = document.getElementById('joint-data').innerText;
      const pct = x => (100*x).toFixed(2) + '%';
      return text.includes(pct(row.a_after)) && text.includes(pct(row.b_after))
        && text.includes(row.accepted + ' / ' + row.attempts)
        && text.includes(row.joint_gate ? '通过' : '未通过');
    }, method);
    check('selector_' + method, valid);
  }
  await page.locator('#method').selectOption('spatial_adaptive');
  await page.locator('#method').focus();
  await page.keyboard.press('ArrowDown');
  check('keyboard_selector',await page.locator('#method').inputValue() !== 'spatial_adaptive');
  await page.locator('#method').selectOption('spatial_adaptive');
  await page.evaluate(() => window.scrollTo(0,0));
  await page.screenshot({path:'/tmp/go14-html-desktop.png'});
  await page.locator('math[display="block"]').nth(11).scrollIntoViewIfNeeded();
  await page.screenshot({path:'/tmp/go14-html-math.png'});
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(() => window.scrollTo(0,0));
  await page.screenshot({path:'/tmp/go14-html-mobile.png'});
  report.mobile = await page.evaluate(() => ({page:document.documentElement.scrollWidth, viewport:innerWidth}));
  check('mobile_no_page_overflow',report.mobile.page === report.mobile.viewport);
  await page.emulateMedia({media:'print'});
  check('print_hides_navigation',await page.evaluate(() => getComputedStyle(document.querySelector('aside')).display === 'none'));
  await page.emulateMedia({media:'screen'});
  check('no_script_errors_or_remote_requests',report.errors.length === 0 && report.remoteRequests.length === 0);
  await page.evaluate(value => {window.go14BrowserQA = value;}, report);
}
