import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, writeFileSync } from 'fs';
import { makeQRBloom } from './qrbloom.js';
const manifest=JSON.parse(readFileSync('./assets/manifest_v2.json'));
const wbuf=readFileSync('./assets/weights_v2.bin');
const cj=JSON.parse(readFileSync('./assets/cond_v2.json'));
const pal=JSON.parse(readFileSync('./assets/palette.json'));
const [gxy,_,gz]=cj.dims; const qe=25, off=(gxy-qe)>>1, ctr=qe>>1;
const net=makeQRBloom(tf,manifest,wbuf.buffer.slice(wbuf.byteOffset,wbuf.byteOffset+wbuf.byteLength));
const cond=tf.tensor(cj.cond,[1,gxy,gxy,gz,1]);
// rebuild core from cond (dark where any height set)
const cd=cond.dataSync(); const core=[];
for(let i=0;i<qe;i++){core.push([]);for(let j=0;j<qe;j++){const gi=off+i,gj=off+j;core[i].push(cd[((gi*gxy+gj)*gz)+0]>0.5?1:0);}}
const out=await net.sample(cond,0,[0.5,0.5,0.5],8,1); const data=out.dataSync();
const P=pal['cherryblossom']; const idx=(d,h,w,c)=>((d*gxy+h)*gz+w)*4+c;
const hx=x=>Math.max(0,Math.min(255,Math.round((x+1)*127.5)));
const cells=[];
for(let i=0;i<qe;i++)for(let j=0;j<qe;j++)cells.push([j-ctr,0,i-ctr,1.0,core[i][j]?P.qr_dark:P.qr_light]);
for(let d=0;d<gxy;d++)for(let h=0;h<gxy;h++)for(let w=1;w<gz;w++){if(data[idx(d,h,w,3)]<=0)continue;
  const c='#'+[hx(data[idx(d,h,w,0)]),hx(data[idx(d,h,w,1)]),hx(data[idx(d,h,w,2)])].map(v=>v.toString(16).padStart(2,'0')).join('');
  cells.push([h-(off+ctr),w,d-(off+ctr),1.0,c]);}
writeFileSync('../docs/_cells.json',JSON.stringify({cells}));
console.log('wrote',cells.length,'cells');
