// DiT3D (qrbloom/dit.py) ported to TensorFlow.js.
//
// One net serves every trained QR version: the grid-dependent pieces
// (patchify/unpatchify index maps, positional embeddings) are built per
// version on first use and cached, and the version itself enters the
// conditioning vector as Fourier features of the QR module count.
//
// The backbone is matmul/attention-only, so every tensor on the GPU is rank
// ≤ 4 — well inside tf.js WebGPU's comfort zone (its conv3d and rank-5+
// kernels are slow or broken). The two layout-heavy steps, patchify and
// unpatchify, run in plain JS on the CPU: for a single sample they are just
// ~300k float copies per forward, far below measurement noise.
//
// Pass the tf module in, so the same code runs under tfjs-node (verification)
// and the browser webgpu/wasm backends.

export function makeQRBloom(tf, manifest, weightsBuf) {
  const P = {};                      // name -> tf.tensor
  const f32 = new Float32Array(weightsBuf);
  for (const r of manifest.params) {
    const slice = f32.slice(r.offset, r.offset + r.count);
    P[r.name] = tf.tensor(slice, r.shape, 'float32');
  }
  const DIM = manifest.dim, DEPTH = manifest.depth, HEADS = manifest.heads;
  const PS = manifest.patch, HD = DIM / HEADS;
  const CIN = 5, C_OUT = 4;
  const HAS_VER = 'ver_mlp.0.weight' in P;

  // --- 3D sin-cos positional embedding (matches qrbloom.dit.posemb_3d) ----
  function posemb1d(dim, n) {
    const half = dim / 2, out = new Float32Array(n * dim);
    for (let i = 0; i < half; i++) {
      const fr = Math.exp(-Math.log(10000.0) * i / half);
      for (let p = 0; p < n; p++) {
        out[p * dim + i] = Math.sin(p * fr);
        out[p * dim + half + i] = Math.cos(p * fr);
      }
    }
    return out;
  }

  // --- per-version grid structures, built on first use --------------------
  // Token order is D-major (di, hi, wi), matching x.flatten(2) in PyTorch.
  // Token vector layout: channel-major (c, a, b, cc) — matches the exported
  // patch_embed matrix. Output vector layout: (a, b, cc, C) — matches the
  // reshape in DiT3D's unpatchify.
  const _grids = new Map();          // version -> structures
  function gridFor(version) {
    let g = _grids.get(version);
    if (g) return g;
    const dims = manifest.grids[String(version)];
    if (!dims) throw new Error(`version ${version} not in manifest grids`);
    const [GX, , GZ] = dims;
    const dp = GX / PS, hp = GX / PS, wp = GZ / PS;
    const L = dp * hp * wp;
    const PV = CIN * PS * PS * PS, TV = PS * PS * PS * C_OUT;

    const gatherIdx = new Int32Array(L * PV);
    {
      let o = 0;
      for (let di = 0; di < dp; di++) for (let hi = 0; hi < hp; hi++) for (let wi = 0; wi < wp; wi++)
        for (let c = 0; c < CIN; c++)
          for (let a = 0; a < PS; a++) for (let b = 0; b < PS; b++) for (let cc = 0; cc < PS; cc++)
            gatherIdx[o++] = ((c * GX + di * PS + a) * GX + hi * PS + b) * GZ + wi * PS + cc;
    }
    const scatterIdx = new Int32Array(L * TV);
    {
      let o = 0;
      for (let di = 0; di < dp; di++) for (let hi = 0; hi < hp; hi++) for (let wi = 0; wi < wp; wi++)
        for (let a = 0; a < PS; a++) for (let b = 0; b < PS; b++) for (let cc = 0; cc < PS; cc++)
          for (let c = 0; c < C_OUT; c++)
            scatterIdx[o++] = ((c * GX + di * PS + a) * GX + hi * PS + b) * GZ + wi * PS + cc;
    }
    let ad = Math.floor(DIM / 3); ad -= ad % 2;
    const aw = ad, ah = DIM - ad - aw;
    const pd = posemb1d(ad, dp), ph = posemb1d(ah, hp), pw = posemb1d(aw, wp);
    const pos = new Float32Array(L * DIM);
    let o = 0;
    for (let di = 0; di < dp; di++) for (let hi = 0; hi < hp; hi++) for (let wi = 0; wi < wp; wi++) {
      pos.set(pd.subarray(di * ad, (di + 1) * ad), o); o += ad;
      pos.set(ph.subarray(hi * ah, (hi + 1) * ah), o); o += ah;
      pos.set(pw.subarray(wi * aw, (wi + 1) * aw), o); o += aw;
    }
    g = { GX, GZ, L, PV, gatherIdx, scatterIdx, posT: tf.tensor(pos, [1, L, DIM]) };
    _grids.set(version, g);
    return g;
  }

  // --- tf helpers ----------------------------------------------------------
  const silu = (x) => tf.mul(x, tf.sigmoid(x));
  // GELU with tanh approximation — exact match for nn.GELU(approximate="tanh").
  const gelu = (x) => tf.tidy(() => {
    const inner = tf.mul(0.7978845608028654, tf.add(x, tf.mul(0.044715, tf.mul(x, tf.mul(x, x)))));
    return tf.mul(tf.mul(x, 0.5), tf.add(1, tf.tanh(inner)));
  });
  const linear = (x, w, b) => {
    let h = tf.matMul(x, P[w]);               // weights pre-transposed [in,out]
    if (b) h = tf.add(h, P[b]);
    return h;
  };
  // LayerNorm over the last axis, no affine (eps matches PyTorch's 1e-6).
  const layerNorm = (x) => tf.tidy(() => {
    const m = tf.mean(x, -1, true);
    const d = tf.sub(x, m);
    const v = tf.mean(tf.mul(d, d), -1, true);
    return tf.div(d, tf.sqrt(tf.add(v, 1e-6)));
  });

  function fourier(tVal, dim) {
    const half = dim / 2, arr = new Float32Array(dim);
    for (let i = 0; i < half; i++) {
      const fr = Math.exp(-Math.log(10000) * i / half);
      arr[i] = Math.cos(tVal * fr);
      arr[half + i] = Math.sin(tVal * fr);
    }
    return tf.tensor(arr, [1, dim]);
  }

  function attention(x, i, L) {                // x [1,L,DIM]
    return tf.tidy(() => {
      const qkv = linear(x, `blocks.${i}.qkv.weight`, `blocks.${i}.qkv.bias`); // [1,L,3·DIM]
      const r = tf.transpose(tf.reshape(qkv, [L, 3, HEADS, HD]), [1, 2, 0, 3]); // [3,H,L,HD]
      const q = tf.reshape(tf.slice(r, [0, 0, 0, 0], [1, HEADS, L, HD]), [HEADS, L, HD]);
      const k = tf.reshape(tf.slice(r, [1, 0, 0, 0], [1, HEADS, L, HD]), [HEADS, L, HD]);
      const v = tf.reshape(tf.slice(r, [2, 0, 0, 0], [1, HEADS, L, HD]), [HEADS, L, HD]);
      const scores = tf.softmax(tf.mul(tf.matMul(q, k, false, true), 1 / Math.sqrt(HD)), -1);
      const o = tf.reshape(tf.transpose(tf.matMul(scores, v), [1, 0, 2]), [1, L, DIM]);
      return linear(o, `blocks.${i}.proj.weight`, `blocks.${i}.proj.bias`);
    });
  }

  function block(x, c, i, L) {                 // x [1,L,DIM], c [1,DIM]
    return tf.tidy(() => {
      const ada = linear(silu(c), `blocks.${i}.ada.1.weight`, `blocks.${i}.ada.1.bias`); // [1,6·DIM]
      const m = tf.reshape(ada, [1, 1, 6 * DIM]);
      const [sh1, sc1, g1, sh2, sc2, g2] =
        [0, 1, 2, 3, 4, 5].map(j => tf.slice(m, [0, 0, j * DIM], [1, 1, DIM]));
      let h = tf.add(x, tf.mul(g1, attention(tf.add(tf.mul(layerNorm(x), tf.add(1, sc1)), sh1), i, L)));
      const mlpIn = tf.add(tf.mul(layerNorm(h), tf.add(1, sc2)), sh2);
      let f = gelu(linear(mlpIn, `blocks.${i}.mlp.0.weight`, `blocks.${i}.mlp.0.bias`));
      f = linear(f, `blocks.${i}.mlp.2.weight`, `blocks.${i}.mlp.2.bias`);
      return tf.add(h, tf.mul(g2, f));
    });
  }

  // --- forward: NCHW Float32Arrays in and out ------------------------------
  // x: (4, D, H, W) noised voxels; cond: (D, H, W) QR footprint;
  // version selects the grid and feeds the micro-conditioning.
  // Returns the v prediction (4, D, H, W).
  async function forward(xNCHW, condArr, tVal, themeIdx, attrArr, attrMask, version) {
    const g = gridFor(version);
    const vox = g.GX * g.GX * g.GZ;
    const xc = new Float32Array(CIN * vox);
    xc.set(xNCHW, 0);
    xc.set(condArr, 4 * vox);
    const tokens = new Float32Array(g.L * g.PV);
    for (let i = 0; i < tokens.length; i++) tokens[i] = xc[g.gatherIdx[i]];

    const out = tf.tidy(() => {
      // conditioning vector
      let temb = linear(fourier(tVal, 256), 'tmlp.0.weight', 'tmlp.0.bias');
      temb = linear(silu(temb), 'tmlp.2.weight', 'tmlp.2.bias');
      const themeRow = tf.reshape(tf.gather(P['theme_emb.weight'], [themeIdx]), [1, DIM]);
      let attrEmb = linear(tf.tensor(attrArr, [1, 3]), 'attr_mlp.0.weight', 'attr_mlp.0.bias');
      attrEmb = linear(silu(attrEmb), 'attr_mlp.2.weight', 'attr_mlp.2.bias');
      const attrNull = tf.reshape(P['attr_null'], [1, DIM]);
      attrEmb = tf.add(tf.mul(attrEmb, attrMask), tf.mul(attrNull, 1 - attrMask));
      let c = tf.add(tf.add(temb, themeRow), attrEmb);
      if (HAS_VER) {
        let verEmb = linear(fourier(4 * version + 17, 256), 'ver_mlp.0.weight', 'ver_mlp.0.bias');
        verEmb = linear(silu(verEmb), 'ver_mlp.2.weight', 'ver_mlp.2.bias');
        c = tf.add(c, verEmb);
      }

      let h = linear(tf.tensor(tokens, [1, g.L, g.PV]), 'patch_embed.weight', 'patch_embed.bias');
      h = tf.add(h, g.posT);
      for (let i = 0; i < DEPTH; i++) h = block(h, c, i, g.L);

      const adaF = tf.reshape(
        linear(silu(c), 'ada_f.1.weight', 'ada_f.1.bias'), [1, 1, 2 * DIM]);
      const sh = tf.slice(adaF, [0, 0, 0], [1, 1, DIM]);
      const sc = tf.slice(adaF, [0, 0, DIM], [1, 1, DIM]);
      const hf = tf.add(tf.mul(layerNorm(h), tf.add(1, sc)), sh);
      return linear(hf, 'head.weight', 'head.bias');       // [1, L, p³·4]
    });
    const tok = await out.data();
    out.dispose();

    const v = new Float32Array(C_OUT * vox);
    for (let i = 0; i < tok.length; i++) v[g.scatterIdx[i]] = tok[i];
    return v;
  }

  return { forward, P };
}
