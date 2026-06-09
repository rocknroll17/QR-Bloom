import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { extname, join } from 'path';
import puppeteer from 'puppeteer';

const ROOT = new URL('../docs/', import.meta.url).pathname;
const MIME = {'.html':'text/html','.js':'text/javascript','.json':'application/json',
  '.bin':'application/octet-stream','.wasm':'application/wasm','.png':'image/png'};
const server = createServer((req,res)=>{
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p==='/') p='/live.html';
  const f = join(ROOT, p);
  if (!existsSync(f)) { res.writeHead(404); return res.end('nf'); }
  res.writeHead(200, {'content-type': MIME[extname(f)]||'application/octet-stream',
    'cross-origin-opener-policy':'same-origin','cross-origin-embedder-policy':'require-corp'});
  res.end(readFileSync(f));
});
await new Promise(r=>server.listen(0,r));
const port = server.address().port;

const browser = await puppeteer.launch({ headless:'new',
  args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan'] });
const page = await browser.newPage();
page.on('console', m => { const t=m.text(); if(!/oneAPI|cpu_feature/.test(t)) console.log('  [page]', t); });
page.on('pageerror', e => console.log('  [pageerr]', e.message));

const backend = process.argv[2] || 'wasm';
const steps = process.argv[3] || '8';
await page.goto(`http://localhost:${port}/live.html?backend=${backend}&steps=${steps}`, {waitUntil:'load', timeout:60000});

// wait until status reports done/error/ready-then-done
try {
  await page.waitForFunction(() => {
    const s = document.getElementById('status')?.textContent || '';
    return s.startsWith('done') || s.startsWith('error') || s.startsWith('init failed');
  }, { timeout: 120000, polling: 500 });
} catch(e) { console.log('  TIMEOUT waiting for result'); }

const status = await page.$eval('#status', el=>el.textContent);
console.log('FINAL STATUS:', status);
await page.screenshot({ path: new URL('../docs/_live_test.png', import.meta.url).pathname });
console.log('screenshot -> docs/_live_test.png');
await browser.close(); server.close();
