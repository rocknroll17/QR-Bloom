import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';
import { makeQRBloom } from './qrbloom.js';
const manifest = JSON.parse(readFileSync('./assets/manifest_v2.json'));
const wbuf = readFileSync('./assets/weights_v2.bin');
const cj = JSON.parse(readFileSync('./assets/cond_v2.json'));
const pal = JSON.parse(readFileSync('./assets/palette.json'));
const [gxy,_,gz] = cj.dims;
const net = makeQRBloom(tf, manifest, wbuf.buffer.slice(wbuf.byteOffset, wbuf.byteOffset+wbuf.byteLength));
const cond = tf.tensor(cj.cond, [1,gxy,gxy,gz,1]);
const qe = 4*2+17, off=(gxy-qe)>>1, ctr=qe>>1;

const out = await net.sample(cond, 0, [0.5,0.5,0.5], 8, 1);
const data = out.dataSync();
const idx=(d,h,w,c)=>((d*gxy+h)*gz+w)*4+c;
const hx=(x)=>Math.max(0,Math.min(255,Math.round((x+1)*127.5)));
// base
let base=0, tree=0; const xs=[],ys=[],zs=[];
for(let i=0;i<qe;i++)for(let j=0;j<qe;j++){base++;}
for(let d=0;d<gxy;d++)for(let h=0;h<gxy;h++)for(let w=1;w<gz;w++){
  if(data[idx(d,h,w,3)]<=0)continue; tree++;
  xs.push(h-(off+ctr)); ys.push(w); zs.push(d-(off+ctr));
}
const rng=a=>a.length?`[${Math.min(...a)}..${Math.max(...a)}]`:'[]';
console.log(`base=${base} tree=${tree} total=${base+tree}`);
console.log(`x${rng(xs)} y${rng(ys)} z${rng(zs)}`);
// coarse top-down occupancy (max over height) on qe-ish grid
const topo=Array.from({length:gxy},()=>Array(gxy).fill(' '));
for(let d=0;d<gxy;d++)for(let h=0;h<gxy;h++){let hit=false;for(let w=1;w<gz;w++)if(data[idx(d,h,w,3)]>0){hit=true;break;}topo[d][h]=hit?'#':'.';}
console.log('top-down (D rows x H cols):');
console.log(topo.map(r=>r.join('')).join('\n'));
console.log('DONE');
