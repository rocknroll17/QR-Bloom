import puppeteer from 'puppeteer';
const b=await puppeteer.launch({headless:'new',args:['--no-sandbox']});
const pg=await b.newPage();
await pg.goto('https://example.com',{waitUntil:'domcontentloaded',timeout:30000}); // some https origin
const url='https://huggingface.co/rocknroll17/QR-Bloom/resolve/main/tfjs/weights_v3.bin';
for (const opt of [{cache:'default'},{cache:'force-cache'}]) {
  const r = await pg.evaluate(async (u,o)=>{
    try { const res = await fetch(u, o); return {ok:res.ok, status:res.status, type:res.type, len:res.headers.get('content-length')}; }
    catch(e){ return {error:e.message}; }
  }, url, opt);
  console.log(`fetch(${JSON.stringify(opt)}) ->`, JSON.stringify(r));
}
await b.close();
