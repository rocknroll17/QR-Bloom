// In-browser QR-Bloom inference.
//
// ModelClient owns the full pipeline behind the demo page: weight-host
// resolution, download + Cache API persistence, tf.js backend setup, the
// v-prediction diffusion sampling loop, and post-processing into the
// `[x, y, z, scale, color]` cell list the viewer expects.
//
// ONE model serves every trained QR version — the version is passed to the
// net at runtime (micro-conditioning), so a single weights download covers
// the whole trained range. Weights live on Hugging Face; the manifest at the
// same path supplies grid sizes, attribute stats, and theme metadata.
//
// ENGINE: tf.js on the WebGPU backend. The DiT backbone is matmul/attention
// only (qrbloom-model.js) — no 3-D convs, so every GPU tensor stays rank ≤ 4
// and runs on tf.js's fastest kernels. WebGPU needs no COOP/COEP, so plain
// static hosting works.

import qrcode from 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/+esm';
import { makeQRBloom } from './qrbloom-model.js?v=2';

const HF_HOST = 'https://huggingface.co/rocknroll17/QR-Bloom/resolve/main/tfjs';

const T_TOTAL  = 500;        // Diffusion.T
const STEPS    = 16;         // diffusion steps — tuned for browser (quality holds well below the gallery's 100)
const X0_CH    = 4;
const COSINE_S = 0.008;

// --- tf.js loader (runtime <script> injection — keeps index.html untouched) -

let tf = null;
function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.crossOrigin = 'anonymous';
    s.onload = () => res(); s.onerror = () => rej(new Error('failed to load ' + src));
    document.head.appendChild(s);
  });
}
let _backendPromise = null;
function ensureBackend() {
  if (_backendPromise) return _backendPromise;
  _backendPromise = (async () => {
    if (!self.tf) {
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js');
    }
    tf = self.tf;
    if (tf.wasm && tf.wasm.setWasmPaths)
      tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
    if (navigator.gpu) {
      try { await tf.setBackend('webgpu'); await tf.ready();
            if (tf.getBackend() !== 'webgpu') throw new Error('inactive'); }
      catch (e) { console.warn('[qrbloom] WebGPU unavailable, WASM fallback:', e.message);
                  await tf.setBackend('wasm'); await tf.ready(); }
    } else { await tf.setBackend('wasm'); await tf.ready(); }
    return tf.getBackend();
  })();
  return _backendPromise;
}

// --- fetch helpers -----------------------------------------------------------

// HF's Xet CDN intermittently returns a transient 403 on browser fetches, so
// every download is wrapped in exponential-backoff retries. (403 is normally
// not retryable, but HF Xet's is a known transient signed-URL / outage quirk;
// genuinely permanent statuses like 404 are marked fatal and not retried.)
const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const _RETRYABLE = new Set([403, 408, 429, 500, 502, 503, 504]);

async function _withRetry(fn, label, tries = 4) {
  for (let i = 0; ; i++) {
    try { return await fn(i); }
    catch (e) {
      if (e.fatal || i >= tries - 1) throw e;
      console.warn(`[qrbloom] ${label} retry ${i + 1}/${tries - 1}: ${e.message}`);
      await _sleep(Math.min(8000, 500 * 2 ** i) + Math.random() * 300); // backoff + jitter
    }
  }
}
async function _fetchOk(url, opts) {
  const r = await fetch(url, opts);            // network errors throw -> retried
  if (!r.ok) { const e = new Error(`${r.status}`); e.fatal = !_RETRYABLE.has(r.status); throw e; }
  return r;
}

// --- QR encoding -------------------------------------------------------------

// qrcode-generator throws when text doesn't fit; try increasing type numbers
// until it does, then clamp into the trained range from the manifest.
function encodeQR(text, trainedVersions) {
  const min = Math.min(...trainedVersions);
  const max = Math.max(...trainedVersions);
  let lastErr = null;
  for (let v = 1; v <= 40; v++) {
    try {
      const qr = qrcode(v, 'M');
      qr.addData(text);
      qr.make();
      const chosen = Math.max(min, v);
      if (chosen > max) throw new Error(`text too long for trained range (v${min}..v${max}); needs v${v}`);
      // Re-encode at the chosen version if we bumped up.
      const final = qrcode(chosen, 'M');
      final.addData(text);
      final.make();
      const n = final.getModuleCount();
      const matrix = new Uint8Array(n * n);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          matrix[r * n + c] = final.isDark(r, c) ? 1 : 0;
        }
      }
      return { version: chosen, modules: n, matrix };
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('QR encode failed');
}

// --- cosine β schedule + gaussians ------------------------------------------

function buildSchedule() {
  // Matches qrbloom/model.py:cosine_beta_schedule (T=500, s=0.008).
  const T = T_TOTAL;
  const acp = new Float32Array(T);
  const sqrt_acp = new Float32Array(T);
  const sqrt_one_minus_acp = new Float32Array(T);
  const f0 = Math.cos((COSINE_S / (1 + COSINE_S)) * Math.PI / 2) ** 2;
  for (let i = 0; i < T; i++) {
    const f = Math.cos(((i + 1) / T + COSINE_S) / (1 + COSINE_S) * Math.PI / 2) ** 2;
    acp[i] = Math.min(0.999, Math.max(1e-8, f / f0));
    sqrt_acp[i] = Math.sqrt(acp[i]);
    sqrt_one_minus_acp[i] = Math.sqrt(1 - acp[i]);
  }
  return { acp, sqrt_acp, sqrt_one_minus_acp };
}
const SCHED = buildSchedule();

function linspaceSteps(steps) {
  // torch.linspace(T-1, 0, steps).long()
  const out = new Int32Array(steps);
  for (let i = 0; i < steps; i++) {
    const f = (T_TOTAL - 1) - i * (T_TOTAL - 1) / (steps - 1);
    out[i] = Math.floor(f + 1e-9);  // .long() truncates toward zero for positives
  }
  return out;
}

function fillNormal(arr) {
  for (let i = 0; i < arr.length; i += 2) {
    const u1 = Math.max(1e-12, Math.random());
    const u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    arr[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < arr.length) arr[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
}

function randn() {
  const u = Math.max(1e-12, Math.random());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// --- post-processing: (occ, rgb) → viewer cell list --------------------------

function rgbToHex(r, g, b) {
  const hex = n => n.toString(16).padStart(2, '0');
  return '#' + hex(r) + hex(g) + hex(b);
}

function clampU8(v) {
  v = Math.round(v);
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

function buildCells(x0, D, H, W, qr, themeMeta) {
  const cx = D >> 1;
  const m = qr.modules;
  const off = Math.floor((D - m) / 2);
  const cells = [];
  // Base plane (y=0): per-QR-module color.
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const dark = qr.matrix[i * m + j] === 1;
      const color = dark ? themeMeta.qr_dark : themeMeta.qr_light;
      cells.push([(off + j) - cx, 0, (off + i) - cx, 1.0, color]);
    }
  }
  // Tree voxels at y>=1 where the occupancy channel > 0.
  const channelSize = D * H * W;
  const occBase = 3 * channelSize;
  const gBase = channelSize;
  const bBase = 2 * channelSize;
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < H; j++) {
      const baseDH = (i * H + j) * W;
      for (let k = 1; k < W; k++) {
        if (x0[occBase + baseDH + k] <= 0) continue;
        const r = clampU8((x0[baseDH + k] + 1) * 127.5);
        const g = clampU8((x0[gBase + baseDH + k] + 1) * 127.5);
        const b = clampU8((x0[bBase + baseDH + k] + 1) * 127.5);
        cells.push([j - cx, k, i - cx, 1.0, rgbToHex(r, g, b)]);
      }
    }
  }
  return cells;
}

// --- the client --------------------------------------------------------------

export class ModelClient {
  #manifest = null;
  #base = null;
  #phase = 'pending';
  #bytes = 0;
  #total = 0;
  #buf = null;
  #params = null;
  #net = null;
  #fetchPromise = null;
  #netPromise = null;

  // Weights are persisted in the Cache API keyed by their content hash
  // (sha256 from the manifest), so they download exactly once and survive
  // reloads — unlike the HTTP cache, which can't hold them (HF mints a new
  // signed URL per request, so the URL cache key never matches). A model
  // update changes the hash, so the key changes and the new weights are
  // fetched; stale entries from old revisions are pruned on load.
  static CACHE_NAME = 'qrbloom-weights-v1';

  async #openCache() {
    try { return await caches.open(ModelClient.CACHE_NAME); } catch { return null; }
  }
  #sha() { return this.#manifest.sha256 || String(this.#manifest.bytes); }
  #wKey(sha) { return `qrbloom-cache/weights-all-${sha}`; }
  #pKey(sha) { return `qrbloom-cache/params-all-${sha}`; }

  // Weight host resolution, in priority order:
  //   1. ?weights=<same-origin path> — explicit override. Relative paths
  //      only: accepting a full URL would let a crafted link make this page
  //      fetch and render arbitrary third-party weights.
  //   2. ./assets/ served next to the page — the local dev loop
  //      (scripts/export_for_tfjs.py writes there; the folder is gitignored,
  //      so the deployed Pages site never has it and falls through to HF).
  //   3. The Hugging Face weight host — production.
  async #resolveBase() {
    if (this.#base) return this.#base;
    const p = new URLSearchParams(location.search).get('weights');
    if (p && /^\.?\/(?!\/)/.test(p)) return (this.#base = p);
    try {
      const r = await fetch('./assets/manifest.json', { method: 'HEAD', cache: 'no-cache' });
      if (r.ok) return (this.#base = './assets');
    } catch { /* no local assets — fall through to HF */ }
    return (this.#base = HF_HOST);
  }

  /** 'local' when serving weights from this origin, 'hf' for production. */
  source() {
    return this.#base && this.#base !== HF_HOST ? 'local' : 'hf';
  }

  /** Download phase and byte progress, for the page's status bar. */
  state() {
    return { phase: this.#phase, bytes: this.#bytes, total: this.#total };
  }

  /** Theme list for the UI, sourced from the manifest. */
  async themes() {
    return (await this.manifest()).themes;
  }

  /** Kick off the model download (alias with the ApiClient interface). */
  async prepare(onProgress) {
    return this.download(onProgress);
  }

  /** Fetch (once) and return the manifest; prunes stale cached weights. */
  async manifest() {
    if (this.#manifest) return this.#manifest;
    const base = await this.#resolveBase();
    // Fetch fresh (revalidate) so a re-uploaded model is picked up immediately.
    const r = await fetch(`${base}/manifest.json`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
    this.#manifest = await r.json();
    this.#total = this.#manifest.bytes;
    const cache = await this.#openCache();
    if (cache) {
      const valid = new Set([this.#wKey(this.#sha()), this.#pKey(this.#sha())]);
      try {
        for (const req of await cache.keys()) {
          const tail = new URL(req.url).pathname.replace(/^\//, '');
          if (!valid.has(tail)) await cache.delete(req);
        }
      } catch { /* cache eviction races are non-fatal */ }
    }
    return this.#manifest;
  }

  /** Kick off (or await) the model download; safe to call repeatedly. */
  async download(onProgress) {
    await this.manifest();
    try { await this.#fetchBytes(onProgress); } catch { /* surfaced via phase */ }
  }

  // Phase 1: download bytes only (network — no GPU). Building the tf.js net
  // and the WebGPU warmup are deferred to first use so the page-load
  // download doesn't stutter the UI with GPU work.
  #fetchBytes(onProgress) {
    if (this.#buf || this.#net) return Promise.resolve();
    if (this.#fetchPromise) return this.#fetchPromise;
    this.#fetchPromise = (async () => {
      this.#phase = 'downloading';
      const sha = this.#sha();
      const cache = await this.#openCache();
      // Network downloads use cache:'no-store' (never touch the HTTP cache,
      // which would otherwise pin a transient 403); retries re-fetch fresh.
      // Persistence is handled by the Cache API, keyed by content hash.
      const params = await (cache && cache.match(this.#pKey(sha)).catch(() => null));
      if (params) this.#params = await params.json();
      else {
        this.#params = await _withRetry(
          () => _fetchOk(`${this.#base}/${this.#manifest.params}`, { cache: 'no-store' }).then((r) => r.json()),
          'params');
        if (cache) cache.put(this.#pKey(sha),
          new Response(JSON.stringify(this.#params), { headers: { 'content-type': 'application/json' } })).catch(() => {});
      }

      const hit = await (cache && cache.match(this.#wKey(sha)).catch(() => null));
      if (hit) {                                   // cache hit — no network
        this.#bytes = this.#manifest.bytes;
        if (onProgress) onProgress({ downloadedBytes: this.#bytes, totalBytes: this.#total });
        this.#buf = new Uint8Array(await hit.arrayBuffer());
      } else {
        // the whole stream lives inside the retry, so a mid-download failure restarts it
        this.#buf = await _withRetry(async () => {
          const resp = await _fetchOk(`${this.#base}/${this.#manifest.weights}`, { cache: 'no-store' });
          const reader = resp.body.getReader();
          const total = this.#manifest.bytes;
          const buf = new Uint8Array(total);
          let offset = 0; this.#bytes = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf.set(value, offset);
            offset += value.byteLength;
            this.#bytes = offset;
            if (onProgress) onProgress({ downloadedBytes: offset, totalBytes: total });
          }
          return buf;
        }, 'weights');
        if (cache) cache.put(this.#wKey(sha),
          new Response(this.#buf, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
      }
      this.#phase = 'downloaded';
    })().catch((e) => {
      this.#phase = 'error';
      this.#fetchPromise = null;   // clear so a later call can retry from scratch
      throw e;
    });
    return this.#fetchPromise;
  }

  // Phase 2: build the tf.js net + warmup (GPU). Lazy — first generation
  // only. Warmup runs on the smallest trained grid; larger grids compile
  // their pipelines on first use.
  async #buildNet() {
    if (this.#net) return this.#net;
    await this.#fetchBytes();
    this.#netPromise = this.#netPromise || (async () => {
      await ensureBackend();
      this.#phase = 'creating-session';
      const net = makeQRBloom(tf, this.#params, this.#buf.buffer);
      const vMin = Math.min(...this.#manifest.trained_versions);
      const g = this.#manifest.versions[String(vMin)];
      await net.forward(new Float32Array(4 * g.grid_xy * g.grid_xy * g.grid_z),
                        new Float32Array(g.grid_xy * g.grid_xy * g.grid_z),
                        0, 0, [0.5, 0.5, 0.5], 1.0, vMin);
      this.#net = net;
      this.#buf = null;   // free the raw bytes once on the GPU
      this.#phase = 'ready';
      return net;
    })().catch(e => { this.#phase = 'error'; throw e; });
    return this.#netPromise;
  }

  async #waitForNet(onWait) {
    if (this.#net) return this.#net;
    this.#fetchBytes().catch(() => {});
    while (!this.#buf && !this.#net) {
      if (this.#phase === 'error') throw new Error('model download failed');
      if (onWait) onWait({ downloadedBytes: this.#bytes, totalBytes: this.#total });
      await _sleep(250);
    }
    return this.#buildNet();
  }

  // Attribute conditioning: sample from the species' training distribution
  // (manifest mean ± std, clamped to ±2σ and [0,1]), so repeated generations
  // vary in proportions while staying in-distribution — values outside the
  // species' range produce floating-voxel artifacts.
  #sampleAttr(themeName, version) {
    const stats = this.#manifest.attrs
      && this.#manifest.attrs[themeName]
      && this.#manifest.attrs[themeName][String(version)];
    if (Array.isArray(stats)) return stats;        // mean-only manifest
    if (!stats) return [0.5, 0.5, 0.5];
    return stats.mean.map((mu, i) => {
      const z = Math.max(-2, Math.min(2, randn()));
      return Math.min(1, Math.max(0, mu + z * stats.std[i]));
    });
  }

  async #sampleVoxels(net, version, themeIdx, qr, attr, onPhase) {
    const meta = this.#manifest.versions[String(version)];
    const D = meta.grid_xy;
    const H = meta.grid_xy;
    const W = meta.grid_z;
    const voxN = X0_CH * D * H * W;

    // cond: 1 inside the QR footprint columns, 0 elsewhere.
    const condArr = new Float32Array(D * H * W);
    const m = qr.modules;
    const off = Math.floor((D - m) / 2);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        if (qr.matrix[i * m + j]) {
          const baseDH = (off + i) * H + (off + j);
          for (let k = 0; k < W; k++) condArr[baseDH * W + k] = 1.0;
        }
      }
    }

    let x = new Float32Array(voxN);
    fillNormal(x);
    const seq = linspaceSteps(STEPS);
    const noiseBuf = new Float32Array(voxN);

    for (let idx = 0; idx < STEPS; idx++) {
      const t = seq[idx];
      const vPred = await net.forward(x, condArr, t, themeIdx, attr, 1.0, version);
      const a = SCHED.sqrt_acp[t];
      const b = SCHED.sqrt_one_minus_acp[t];
      // x0_pred = a*x - b*v, clamp(-1,1); eps_pred = b*x + a*v
      const x0 = new Float32Array(voxN);
      const eps = new Float32Array(voxN);
      for (let i = 0; i < voxN; i++) {
        const xi = x[i], vi = vPred[i];
        let xv = a * xi - b * vi;
        if (xv > 1) xv = 1; else if (xv < -1) xv = -1;
        x0[i] = xv;
        eps[i] = b * xi + a * vi;
      }
      if (idx < STEPS - 1) {
        const acp_t = SCHED.acp[t];
        const acp_n = SCHED.acp[seq[idx + 1]];
        let sigma = Math.sqrt((1 - acp_n) / (1 - acp_t)) * Math.sqrt(1 - acp_t / acp_n);
        if (!isFinite(sigma)) sigma = 0;
        const c = Math.sqrt(Math.max(0, 1 - acp_n - sigma * sigma));
        const sqrt_acp_n = Math.sqrt(acp_n);
        fillNormal(noiseBuf);
        for (let i = 0; i < voxN; i++) {
          x[i] = sqrt_acp_n * x0[i] + c * eps[i] + sigma * noiseBuf[i];
        }
      } else {
        x = x0;
      }
      if (onPhase && (idx % 2 === 0 || idx === STEPS - 1)) {
        onPhase('sampling', { version, step: idx + 1, total: STEPS });
      }
    }

    // Enforce footprint: outside the QR mask, set to -1 (rejected by occ > 0).
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < H; j++) {
        const baseDH = i * H + j;
        if (condArr[baseDH * W] >= 0.5) continue;  // any k works — column mask
        for (let c = 0; c < X0_CH; c++) {
          const cBase = c * (D * H * W);
          for (let k = 0; k < W; k++) x[cBase + baseDH * W + k] = -1.0;
        }
      }
    }
    return { x0: x, D, H, W };
  }

  /** Generate a tree for `url` with `theme`; returns viewer cells. */
  async generate({ url, theme, onPhase }) {
    const m = await this.manifest();
    if (!url || typeof url !== 'string') throw new Error('url required');
    const themeMeta = m.themes.find(t => t.name === theme) || m.themes[0];
    const themeIdx = m.themes.findIndex(t => t.name === themeMeta.name);
    if (themeIdx < 0) throw new Error(`unknown theme ${theme}`);
    const qr = encodeQR(url, m.trained_versions);
    const net = await this.#waitForNet(info => {
      if (onPhase) onPhase('waiting-download', info);
    });
    const attr = this.#sampleAttr(themeMeta.name, qr.version);
    const { x0, D, H, W } = await this.#sampleVoxels(net, qr.version, themeIdx, qr, attr, onPhase);
    const cells = buildCells(x0, D, H, W, qr, themeMeta);
    return { cells, version: qr.version, theme: themeMeta.name, url };
  }
}

export const qrbloom = new ModelClient();
