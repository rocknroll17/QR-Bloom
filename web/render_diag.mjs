import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { extname, join } from 'path';
import puppeteer from 'puppeteer';
const ROOT=new URL('../docs/',import.meta.url).pathname;
const MIME={'.html':'text/html','.js':'text/javascript','.json':'application/json'};
const srv=createServer((q,s)=>{let p=q.url.split('?')[0];if(p==='/')p='/_shot.html';const f=join(ROOT,p);
 if(!existsSync(f)){s.writeHead(404);return s.end('404 '+p)};s.writeHead(200,{'content-type':MIME[extname(f)]||'text/plain'});s.end(readFileSync(f));});
await new Promise(r=>srv.listen(0,r)); const port=srv.address().port;
const b=await puppeteer.launch({headless:'new',args:['--no-sandbox','--use-gl=swiftshader']});
const pg=await b.newPage(); await pg.setViewport({width:700,height:600});
pg.on('requestfailed',r=>console.log('[REQFAIL]',r.url(),r.failure()?.errorText));
pg.on('response',r=>{if(r.status()>=400)console.log('[HTTP',r.status()+']',r.url());});
pg.on('console',m=>console.log('[pg]',m.text())); pg.on('pageerror',e=>console.log('[ERR]',e.message));
await pg.goto(`http://localhost:${port}/_shot.html`,{waitUntil:'networkidle0',timeout:30000});
await new Promise(r=>setTimeout(r,2000));
const st=await pg.evaluate(()=>{
  const v=window.__v;
  const cv=document.querySelector('canvas');
  return {ready:window.__ready, hasV:!!v,
    meshes: v? v._meshInstances?.length : -1,
    rootKids: v? v._root?.children?.length : -1,
    canvas: cv? [cv.width,cv.height,cv.clientWidth,cv.clientHeight] : null,
    cam: v? v.camera.position.toArray().map(n=>+n.toFixed(1)) : null,
    iw: innerWidth, ih: innerHeight };
});
console.log('STATE', JSON.stringify(st));
await b.close(); srv.close();
