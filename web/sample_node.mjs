import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';
import { makeQRBloom } from './qrbloom.js';

const manifest = JSON.parse(readFileSync('./assets/manifest_v2.json'));
const wbuf = readFileSync('./assets/weights_v2.bin');
const cj = JSON.parse(readFileSync('./assets/cond_v2.json'));
const [D,H,W] = cj.dims;
const net = makeQRBloom(tf, manifest, wbuf.buffer.slice(wbuf.byteOffset, wbuf.byteOffset+wbuf.byteLength));
const cond = tf.tensor(cj.cond, [1,D,H,W,1]);

for (const steps of [8, 20]) {
  const t0 = Date.now();
  const out = await net.sample(cond, /*theme*/3, [0.5,0.5,0.5], steps, /*seed*/1);
  const data = out.dataSync();   // NDHWC [1,D,H,W,4]
  const dt = (Date.now()-t0)/1000;
  // occupancy = channel 3 > 0
  let occ=0; for (let i=3;i<data.length;i+=4) if (data[i] > 0) occ++;
  console.log(`steps=${steps}  ${dt.toFixed(2)}s (tfjs-node CPU)  occ_voxels=${occ}  hasNaN=${data.some(Number.isNaN)}`);
  out.dispose();
}
console.log('DONE');
