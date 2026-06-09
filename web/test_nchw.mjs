import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';
import { makeQRBloom } from '../docs/qrbloom-model.js';
const params = JSON.parse(readFileSync('../docs/assets/manifest_v2.json'));
const wbuf = readFileSync('../docs/assets/weights_v2.bin');
const cj = JSON.parse(readFileSync('./assets/cond_v2.json'));   // NDHWC cond flat from earlier
const [gx,_,gz]=cj.dims, D=gx,H=gx,W=gz, X0=4;
const net = makeQRBloom(tf, params, wbuf.buffer.slice(wbuf.byteOffset, wbuf.byteOffset+wbuf.byteLength));
// cond NDHWC tensor [1,D,H,W,1] from cj.cond (already NDHWC order d,h,w)
const condND = tf.tensor(cj.cond, [1,D,H,W,1]);
// forwardNCHW: x NCHW Float32 -> vPred NCHW Float32
async function fwd(xNCHW,t){
  const vT=tf.tidy(()=>tf.transpose(net.forward(tf.transpose(tf.tensor(xNCHW,[1,X0,D,H,W]),[0,2,3,4,1]),condND,t,0,[.5,.5,.5],1.0),[0,4,1,2,3]));
  const d=await vT.data(); vT.dispose(); return d;
}
// schedule
const T=params.schedule.T, acp=params.schedule.acp;
const sq=acp.map(Math.sqrt), sq1=acp.map(a=>Math.sqrt(1-a));
const voxN=X0*D*H*W; let x=new Float32Array(voxN); for(let i=0;i<voxN;i++)x[i]=(Math.random()*2-1);
const steps=8, seq=[]; for(let i=0;i<steps;i++)seq.push(Math.floor((T-1)-i*(T-1)/(steps-1)+1e-9));
const t0=Date.now();
for(let idx=0;idx<steps;idx++){const t=seq[idx];const v=await fwd(x,t);const a=sq[t],b=sq1[t];
  const x0=new Float32Array(voxN),eps=new Float32Array(voxN);
  for(let i=0;i<voxN;i++){const xi=x[i],vi=v[i];let xv=a*xi-b*vi;xv=xv>1?1:xv<-1?-1:xv;x0[i]=xv;eps[i]=b*xi+a*vi;}
  if(idx<steps-1){const at=acp[t],an=acp[seq[idx+1]];let s=Math.sqrt((1-an)/(1-at))*Math.sqrt(1-at/an);if(!isFinite(s))s=0;const c=Math.sqrt(Math.max(0,1-an-s*s));const sa=Math.sqrt(an);
    for(let i=0;i<voxN;i++)x[i]=sa*x0[i]+c*eps[i]+s*(Math.random()*2-1);}else x=x0;}
// occupancy from NCHW channel 3
const cs=D*H*W, occBase=3*cs; let occ=0,nan=false;
for(let i=0;i<cs;i++){const v=x[occBase+i]; if(Number.isNaN(v))nan=true; if(v>0)occ++;}
console.log(`NCHW loop 8 steps: ${(Date.now()-t0)/1000}s occ=${occ} hasNaN=${nan}`);
