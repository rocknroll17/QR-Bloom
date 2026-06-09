import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';
import { makeQRBloom } from './qrbloom.js';

const manifest = JSON.parse(readFileSync('./assets/manifest_v2.json'));
const wbuf = readFileSync('./assets/weights_v2.bin');
const ref = JSON.parse(readFileSync('./assets/ref_v2.json'));
const [D,H,W] = ref.dims;

const net = makeQRBloom(tf, manifest, wbuf.buffer.slice(wbuf.byteOffset, wbuf.byteOffset+wbuf.byteLength));

// ref tensors are NCDHW (torch). Convert x,cond -> NDHWC.
const toNDHWC = (flat, C) => tf.tidy(()=> tf.transpose(tf.tensor(flat, [1,C,D,H,W]), [0,2,3,4,1]));
const x = toNDHWC(ref.x, 4);
const cond = toNDHWC(ref.cond, 1);

const out = net.forward(x, cond, ref.t, ref.theme, ref.attr, ref.attr_mask); // [1,D,H,W,4]
// back to NCDHW for comparison
const outNCDHW = tf.tidy(()=> tf.transpose(out, [0,4,1,2,3]));
const got = outNCDHW.dataSync();
const want = ref.v;

let maxAbs=0, sumSq=0;
for (let i=0;i<want.length;i++){ const d=Math.abs(got[i]-want[i]); if(d>maxAbs)maxAbs=d; sumSq+=d*d; }
const rmse = Math.sqrt(sumSq/want.length);
const meanGot = got.reduce((a,b)=>a+b,0)/got.length;
console.log(`elements=${want.length}`);
console.log(`ref   mean=${(want.reduce((a,b)=>a+b,0)/want.length).toFixed(4)}`);
console.log(`tfjs  mean=${meanGot.toFixed(4)}`);
console.log(`max|diff|=${maxAbs.toExponential(3)}  RMSE=${rmse.toExponential(3)}`);
console.log(maxAbs < 2e-3 ? 'PASS ✅ forward matches PyTorch' : 'FAIL ❌ mismatch');
