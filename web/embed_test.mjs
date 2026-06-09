import { createServer } from 'http';
import { readFileSync, existsSync, writeFileSync } from 'fs';
import { extname, join } from 'path';
import puppeteer from 'puppeteer';
const DOCS = new URL('../docs/', import.meta.url).pathname;
// build the snippet exactly like copy-embed does, wrap in a host page
const tmpl = readFileSync(join(DOCS,'embed-template.html'),'utf8');
const { cells } = JSON.parse(readFileSync(join(DOCS,'_cells.json'),'utf8'));
const snippet = tmpl.replace('__CELLS__', JSON.stringify(cells));
const host = `<!DOCTYPE html><html><head><meta charset=utf8><title>host</title></head>
<body style="font-family:sans-serif;padding:20px">
<h1>My website</h1><p>Some content above the embed.</p>
${snippet}
<p>Content below.</p></body></html>`;
writeFileSync(join(DOCS,'_embed_test.html'), host);

const MIME={'.html':'text/html','.js':'text/javascript','.json':'application/json'};
const srv=createServer((q,s)=>{let p=q.url.split('?')[0];const f=join(DOCS,p);
 if(!existsSync(f)){s.writeHead(404);return s.end('404')};s.writeHead(200,{'content-type':MIME[extname(f)]||'text/plain'});s.end(readFileSync(f));});
await new Promise(r=>srv.listen(0,r)); const port=srv.address().port;
const b=await puppeteer.launch({headless:'new',args:['--no-sandbox','--use-gl=swiftshader','--enable-unsafe-swiftshader']});
const pg=await b.newPage(); await pg.setViewport({width:640,height:640});
const errs=[];
pg.on('pageerror',e=>errs.push('PAGEERROR: '+e.message));
pg.on('console',m=>{if(m.type()==='error')errs.push('CONSOLE.ERR: '+m.text());});
pg.on('requestfailed',r=>errs.push('REQFAIL: '+r.url()+' '+(r.failure()?.errorText||'')));
await pg.goto(`http://localhost:${port}/_embed_test.html`,{waitUntil:'networkidle0',timeout:40000});
await new Promise(r=>setTimeout(r,3000));
const info=await pg.evaluate(()=>{
  const c=document.querySelector('#qrbloom-embed canvas');
  if(!c) return {canvas:false};
  // sample pixels: count non-background colors
  const tmp=document.createElement('canvas'); tmp.width=c.width; tmp.height=c.height;
  const ctx=tmp.getContext('2d'); ctx.drawImage(c,0,0);
  const d=ctx.getImageData(0,0,c.width,c.height).data; const set=new Set(); let nonbg=0;
  for(let i=0;i<d.length;i+=4*97){const k=(d[i]>>4)+','+(d[i+1]>>4)+','+(d[i+2]>>4); set.add(k);
    if(!(d[i]>240&&d[i+1]>244&&d[i+2]>250))nonbg++;}
  return {canvas:true, w:c.width,h:c.height, distinctColors:set.size, nonBgSamples:nonbg};
});
console.log('errors:', errs.length? errs.slice(0,8): 'NONE ✅');
console.log('render:', JSON.stringify(info));
await pg.screenshot({path:join(DOCS,'_embed_test.png')});
console.log('screenshot -> docs/_embed_test.png');
await b.close(); srv.close();
