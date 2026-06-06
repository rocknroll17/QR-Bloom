// UNet3D + v-prediction diffusion, ported to TensorFlow.js (NDHWC layout).
// Numerically matches the PyTorch reference (verified in verify_node.mjs).
// Pass the tf module in (so the same code runs under tfjs-node and the browser
// webgpu/wasm backends).

export function makeQRBloom(tf, manifest, weightsBuf) {
  const P = {};                      // name -> tf.tensor
  const f32 = new Float32Array(weightsBuf);
  for (const r of manifest.params) {
    const slice = f32.slice(r.offset, r.offset + r.count); // typed copy, no Array.from
    P[r.name] = tf.tensor(slice, r.shape, 'float32');
  }
  const CH = manifest.ch, TDIM = manifest.tdim, EPS = 1e-5;

  // --- helpers ----------------------------------------------------------
  const silu = (x) => tf.mul(x, tf.sigmoid(x));

  // tf.js WebGPU's conv3d/conv3dTranspose kernels are ~8x slower than conv2d
  // (measured). So every 3D conv is run as a stack of 2D convs over depth
  // slices (depth as batch) + a depth assembly. Numerically identical to the
  // 3D conv (verified vs PyTorch), but uses the fast conv2d kernels.

  // pad axes H,W (4D NHWC) by p via concat (4D pad -> vec4, WGSL-safe).
  function padHW(x, p) {
    if (p <= 0) return x;
    for (const ax of [1, 2]) {
      const zs = x.shape.slice(); zs[ax] = p;
      x = tf.concat([tf.zeros(zs), x, tf.zeros(zs)], ax);
    }
    return x;
  }

  // out[od] = sum_kd planes[kd][ idxOf(od,kd) ]  (out-of-range -> 0)
  function assembleDepth(planes, Dout, idxOf) {
    const Din = planes[0].shape[0];
    let out = null;
    for (let kd = 0; kd < planes.length; kd++) {
      const idx = new Array(Dout); const mask = new Float32Array(Dout);
      for (let od = 0; od < Dout; od++) {
        const i = idxOf(od, kd); const ok = i >= 0 && i < Din;
        idx[od] = ok ? i : 0; mask[od] = ok ? 1 : 0;
      }
      let g = tf.gather(planes[kd], idx, 0);              // [Dout,Ho,Wo,Cout]
      g = tf.mul(g, tf.tensor(mask, [Dout, 1, 1, 1]));
      out = out ? tf.add(out, g) : g;
    }
    return out;
  }

  // Conv3d via conv2d depth-unroll. W layout [kD,kH,kW,Cin,Cout].
  function conv3d(x, wname, bname, stride = 1, pad = 1) {
    return tf.tidy(() => conv3dImpl(x, wname, bname, stride, pad));
  }
  function conv3dImpl(x, wname, bname, stride, pad) {
    const W = P[wname];
    const [kD, kH, kW2, Cin, Cout] = W.shape;
    const [, D, H, Wd] = x.shape;
    const xb = tf.reshape(x, [D, H, Wd, Cin]);            // depth -> batch
    const planes = [];
    for (let kd = 0; kd < kD; kd++) {
      const W2 = tf.reshape(tf.slice(W, [kd,0,0,0,0], [1,kH,kW2,Cin,Cout]), [kH,kW2,Cin,Cout]);
      let a;
      if (stride === 1 && pad === ((kH - 1) >> 1)) a = tf.conv2d(xb, W2, 1, 'same');
      else if (pad === 0)                          a = tf.conv2d(xb, W2, stride, 'valid');
      else                                         a = tf.conv2d(padHW(xb, pad), W2, stride, 'valid');
      planes.push(a);
    }
    const Dout = Math.floor((D + 2*pad - kD) / stride) + 1;
    let out = assembleDepth(planes, Dout, (od, kd) => stride*od + kd - pad);
    out = tf.reshape(out, [1, Dout, out.shape[1], out.shape[2], Cout]);
    if (bname) out = tf.add(out, tf.reshape(P[bname], [1,1,1,1,-1]));
    return out;
  }

  // ConvTranspose3d (k4,s2,p1) via conv2dTranspose depth-unroll.
  // W layout [kD,kH,kW,Cout,Cin]. outShape [1,Dout,Ho,Wo,Cout].
  function convT3d(x, wname, bname, outShape) {
    return tf.tidy(() => convT3dImpl(x, wname, bname, outShape));
  }
  function convT3dImpl(x, wname, bname, outShape) {
    const W = P[wname];
    const [kD, kH, kW2, Cout, Cin] = W.shape;
    const [, D, H, Wd] = x.shape;
    const Dout = outShape[1], Ho = outShape[2], Wo = outShape[3];
    const xb = tf.reshape(x, [D, H, Wd, Cin]);
    const planes = [];
    for (let kd = 0; kd < kD; kd++) {
      const W2 = tf.reshape(tf.slice(W, [kd,0,0,0,0], [1,kH,kW2,Cout,Cin]), [kH,kW2,Cout,Cin]);
      planes.push(tf.conv2dTranspose(xb, W2, [D, Ho, Wo, Cout], 2, 'same')); // HW x2
    }
    // depth transpose-conv k4 s2 p1: od = 2*i + kd - 1  ->  i = (od-kd+1)/2
    let out = assembleDepth(planes, Dout, (od, kd) => {
      const n = od - kd + 1; return (n >= 0 && n % 2 === 0) ? n/2 : -1;
    });
    out = tf.reshape(out, [1, Dout, Ho, Wo, Cout]);
    if (bname) out = tf.add(out, tf.reshape(P[bname], [1,1,1,1,-1]));
    return out;
  }

  function groupNorm(x, wname, bname, groups = 8) {
    const [b, d, hh, ww, c] = x.shape;
    const g = groups, cg = c / g;
    // rank-4 reshape [b, spatial, g, cg] — avoids rank-6 (which the WebGPU
    // backend mishandles like the 5D Pad). Channels split as group*cg+within,
    // matching torch GroupNorm.
    let r = tf.reshape(x, [b, d*hh*ww, g, cg]);
    const mom = tf.moments(r, [1,3], true);        // per (b,group)
    r = tf.div(tf.sub(r, mom.mean), tf.sqrt(tf.add(mom.variance, EPS)));
    r = tf.reshape(r, [b, d, hh, ww, c]);
    const w = tf.reshape(P[wname], [1,1,1,1,c]);
    const bi = tf.reshape(P[bname], [1,1,1,1,c]);
    return tf.add(tf.mul(r, w), bi);
  }

  function linear(x, wname, bname) {            // weight pre-transposed to [in,out]
    let h = tf.matMul(x, P[wname]);
    if (bname) h = tf.add(h, P[bname]);
    return h;
  }

  function timestepEmbedding(tVal, dim) {
    const half = dim / 2;
    const freqs = [];
    for (let i = 0; i < half; i++) freqs.push(Math.exp(-Math.log(10000) * i / half));
    const cos = freqs.map(fr => Math.cos(tVal * fr));
    const sin = freqs.map(fr => Math.sin(tVal * fr));
    return tf.tensor(cos.concat(sin), [1, dim]);  // [1, dim]
  }

  function resblock(x, temb, prefix, cin, cout) {
    return tf.tidy(() => {
      let h = conv3d(silu(groupNorm(x, `${prefix}.norm1.weight`, `${prefix}.norm1.bias`)),
                     `${prefix}.conv1.weight`, `${prefix}.conv1.bias`);
      const tp = linear(temb, `${prefix}.temb.weight`, `${prefix}.temb.bias`); // [1,cout]
      h = tf.add(h, tf.reshape(tp, [1,1,1,1,cout]));
      h = conv3d(silu(groupNorm(h, `${prefix}.norm2.weight`, `${prefix}.norm2.bias`)),
                 `${prefix}.conv2.weight`, `${prefix}.conv2.bias`);
      let sk = x;
      if (cin !== cout) sk = conv3d(x, `${prefix}.skip.weight`, `${prefix}.skip.bias`, 1, 0);
      return tf.add(h, sk);
    });
  }

  // --- forward: x,cond NDHWC [1,D,H,W,C] ; t scalar; theme int; attr [3] ----
  function forward(x, cond, tVal, themeIdx, attrArr, attrMask) {
    return tf.tidy(() => {
      // conditioning embedding
      let temb = tf.add(linear(timestepEmbedding(tVal, CH), 'tmlp.0.weight','tmlp.0.bias'), 0);
      temb = linear(silu(temb), 'tmlp.2.weight','tmlp.2.bias');
      const themeRow = tf.reshape(tf.gather(P['theme_emb.weight'], [themeIdx]), [1, TDIM]);
      temb = tf.add(temb, themeRow);
      let attrEmb = linear(tf.tensor(attrArr, [1,3]), 'attr_mlp.0.weight','attr_mlp.0.bias');
      attrEmb = linear(silu(attrEmb), 'attr_mlp.2.weight','attr_mlp.2.bias');
      const m = attrMask;
      const attrNull = tf.reshape(P['attr_null'], [1, TDIM]);
      attrEmb = tf.add(tf.mul(attrEmb, m), tf.mul(attrNull, 1 - m));
      temb = tf.add(temb, attrEmb);

      const ch = CH;
      const inp = tf.concat([x, cond], 4);                       // [1,D,H,W,5]
      const h0 = conv3d(inp, 'in_conv.weight', 'in_conv.bias');  // ch
      const h1 = resblock(h0, temb, 'd1', ch, ch);
      const d1 = conv3d(h1, 'down1.weight', 'down1.bias', 2, 1); // ch, /2
      const h2 = resblock(d1, temb, 'd2', ch, ch*2);
      const d2 = conv3d(h2, 'down2.weight', 'down2.bias', 2, 1); // 2ch, /4
      const h3 = resblock(d2, temb, 'd3', ch*2, ch*4);
      let h = resblock(h3, temb, 'm1', ch*4, ch*4);
      h = resblock(h, temb, 'm2', ch*4, ch*4);
      const [B,D4,H4,W4] = [h.shape[0], h.shape[1], h.shape[2], h.shape[3]];
      h = convT3d(h, 'up2.weight','up2.bias', [B, D4*2, H4*2, W4*2, ch*4]); // 4ch, /2
      h = resblock(tf.concat([h, h2], 4), temb, 'u2', ch*4+ch*2, ch*2);
      const [Bb,D2,H2,W2] = [h.shape[0], h.shape[1], h.shape[2], h.shape[3]];
      h = convT3d(h, 'up1.weight','up1.bias', [Bb, D2*2, H2*2, W2*2, ch*2]); // 2ch, full
      h = resblock(tf.concat([h, h1], 4), temb, 'head_u1', ch*2+ch, ch);
      const out = conv3d(silu(groupNorm(h, 'head_norm.weight','head_norm.bias')),
                         'head_conv.weight','head_conv.bias');
      return out;                                                // [1,D,H,W,4]
    });
  }

  // --- DDIM/ancestral sample (mirrors Diffusion.sample, eta=1, cfg=1) -------
  async function sample(condNDHWC, themeIdx, attrArr, steps, seed = 1, onStep = null) {
    const acp = manifest.schedule.acp;            // length T+1? -> T values
    const T = manifest.schedule.T;
    const [_, D, H, W, __] = condNDHWC.shape;
    // seeded gaussian
    let rngState = seed >>> 0;
    const rand = () => { rngState = (rngState*1664525 + 1013904223) >>> 0; return rngState/4294967296; };
    const randn = (n) => { const a=new Float32Array(n);
      for(let i=0;i<n;i+=2){const u=Math.max(rand(),1e-9),v=rand();
        const r=Math.sqrt(-2*Math.log(u)); a[i]=r*Math.cos(2*Math.PI*v);
        if(i+1<n)a[i+1]=r*Math.sin(2*Math.PI*v);} return a; };
    const N = D*H*W*4;
    let x = tf.tensor(randn(N), [1,D,H,W,4]);
    const seq = [];
    for (let i=0;i<steps;i++) seq.push(Math.round((T-1) - (T-1)*i/(steps-1)));

    for (let idx=0; idx<steps; idx++) {
      await tf.nextFrame();             // yield to UI so the page stays responsive
      if (onStep) onStep(idx, steps);
      const t = seq[idx];
      const a = Math.sqrt(acp[t]), b = Math.sqrt(1-acp[t]);
      const v = forward(x, condNDHWC, t, themeIdx, attrArr, 1.0);
      const x0 = tf.tidy(()=> tf.clipByValue(tf.sub(tf.mul(x,a), tf.mul(v,b)), -1, 1));
      const eps = tf.tidy(()=> tf.add(tf.mul(x,b), tf.mul(v,a)));
      v.dispose();
      if (idx < steps-1) {
        const tn = seq[idx+1];
        const acpT = acp[t], acpN = acp[tn];
        let sigma = 1.0 * Math.sqrt((1-acpN)/(1-acpT)) * Math.sqrt(1 - acpT/acpN);
        if (!isFinite(sigma)) sigma = 0;
        const c = Math.sqrt(Math.max(0, 1 - acpN - sigma*sigma));
        const noise = tf.tensor(randn(N), [1,D,H,W,4]);
        const xn = tf.tidy(()=> tf.add(tf.add(tf.mul(x0, Math.sqrt(acpN)), tf.mul(eps, c)),
                                       tf.mul(noise, sigma)));
        x.dispose(); x0.dispose(); eps.dispose(); noise.dispose();
        x = xn;
      } else {
        x.dispose(); eps.dispose(); x = x0;
      }
    }
    // apply footprint mask: outside cond -> -1
    const out = tf.tidy(()=>{
      const mask = tf.greater(condNDHWC, 0.5).toFloat();   // [1,D,H,W,1]
      return tf.add(tf.mul(x, mask), tf.sub(mask, 1));
    });
    x.dispose();
    return out;   // [1,D,H,W,4]
  }

  return { forward, sample, P };
}
