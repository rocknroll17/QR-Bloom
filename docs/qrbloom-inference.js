// In-browser QR-Bloom inference.
//
// Mirrors the home-hosted gallery.py pipeline:
//   pick_version_for_text(text) → segno-compatible smallest standard QR clamped
//   into the trained-version range from manifest.json. Encode the QR matrix,
//   pad it to the grid, run v-prediction diffusion sampling against the
//   per-version UNet, and post-process the (occ, rgb) output into the same
//   `[x, y, z, scale, color]` cell list the viewer expects.
//
// Weights live on Hugging Face. Manifest at the same path supplies grid sizes
// and theme metadata. First page load kicks off background downloads of every
// trained version; per-version inference waits only for the one it needs.
//
// ENGINE: tf.js on the WebGPU backend. The DiT backbone is matmul/attention
// only (qrbloom-model.js) — no 3-D convs, so every GPU tensor stays rank ≤ 4
// and runs on tf.js's fastest kernels. WebGPU needs no COOP/COEP, so plain
// static hosting works.

import qrcode from 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/+esm';
import { makeQRBloom } from './qrbloom-model.js';

// ?weights=<base> overrides the weight host — e.g. ?weights=./assets to test
// a local export served next to the page, before it is uploaded to HF.
const HF_BASE = new URLSearchParams(location.search).get('weights')
  || 'https://huggingface.co/rocknroll17/QR-Bloom/resolve/main/tfjs';
const MANIFEST_URL = `${HF_BASE}/manifest.json`;

const T_TOTAL    = 500;        // Diffusion.T
const STEPS      = 16;         // diffusion steps — tuned for browser (quality holds well below the gallery's 100)
const X0_CH      = 4;
const N_THEMES   = 10;
const COSINE_S   = 0.008;

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

// --- manifest + downloads --------------------------------------------------

let _manifest = null;
const _versionState = new Map();  // version → { phase, bytes, total, fetchPromise, buf, params, net }

// Weights are persisted in the Cache API keyed by their content hash (sha256
// from the manifest), so they download exactly once and survive reloads —
// unlike the HTTP cache, which can't hold them (HF mints a new signed URL per
// request, so the URL cache key never matches). A model update changes the
// hash, so the key changes and the new weights are fetched; stale entries from
// old revisions are pruned below.
const _CACHE_NAME = 'qrbloom-weights-v1';
async function _openCache() { try { return await caches.open(_CACHE_NAME); } catch { return null; } }
const _verSha = (meta) => meta.sha256 || String(meta.bytes); // fallback if manifest lacks a hash
const _wKey = (v, sha) => `qrbloom-cache/weights-v${v}-${sha}`;
const _pKey = (v, sha) => `qrbloom-cache/params-v${v}-${sha}`;

async function _pruneCache() {
  const cache = await _openCache();
  if (!cache) return;
  const valid = new Set();
  for (const v of _manifest.trained_versions) {
    const sha = _verSha(_manifest.versions[String(v)]);
    valid.add(_wKey(v, sha)); valid.add(_pKey(v, sha));
  }
  try {
    for (const req of await cache.keys()) {
      const tail = new URL(req.url).pathname.replace(/^\//, '');
      if (!valid.has(tail)) await cache.delete(req);   // drop entries from old model revisions
    }
  } catch { /* cache eviction races are non-fatal */ }
}

export async function ensureManifest() {
  if (_manifest) return _manifest;
  // Fetch fresh (revalidate) so a re-uploaded model is picked up immediately.
  const r = await fetch(MANIFEST_URL, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
  _manifest = await r.json();
  for (const v of _manifest.trained_versions) {
    _versionState.set(v, {
      phase: 'pending', bytes: 0, total: _manifest.versions[String(v)].bytes,
      fetchPromise: null, buf: null, params: null, net: null,
    });
  }
  await _pruneCache();
  return _manifest;
}

export function getDownloadState() {
  const out = {};
  for (const [v, s] of _versionState) {
    out[v] = { phase: s.phase, bytes: s.bytes, total: s.total };
  }
  return out;
}

// HF's Xet CDN intermittently returns a transient 403 on browser fetches, so
// every download is wrapped in exponential-backoff retries. (403 is normally
// not retryable, but HF Xet's is a known transient signed-URL / outage quirk;
// genuinely permanent statuses like 404 are marked fatal and not retried.)
const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const _RETRYABLE = new Set([403, 408, 429, 500, 502, 503, 504]);

async function _withRetry(fn, label, tries = 4) {
  for (let i = 0; ; i++) {
    try { return await fn(i); }   // pass the attempt index so retries can bypass the cache
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

// Phase 1: download bytes only (network — no GPU). Building the tf.js net and
// the WebGPU warmup are deferred to first use (buildVersionNet) so streaming
// every version on page load doesn't stutter the UI with GPU work.
function fetchVersionBytes(version, onProgress) {
  const st = _versionState.get(version);
  if (!st) throw new Error(`unknown version ${version}`);
  if (st.buf || st.net) return Promise.resolve();
  if (st.fetchPromise) return st.fetchPromise;
  st.fetchPromise = (async () => {
    st.phase = 'downloading';
    const meta = _manifest.versions[String(version)];
    const sha = _verSha(meta);
    const cache = await _openCache();
    // Network downloads use cache:'no-store' (never touch the HTTP cache, which
    // would otherwise pin a transient 403); retries re-fetch fresh. Persistence
    // is handled by the Cache API below, keyed by content hash.
    const params = await (cache && cache.match(_pKey(version, sha)).catch(() => null));
    if (params) st.params = await params.json();
    else {
      st.params = await _withRetry(
        () => _fetchOk(`${HF_BASE}/${meta.params}`, { cache: 'no-store' }).then((r) => r.json()),
        `v${version} params`);
      if (cache) cache.put(_pKey(version, sha),
        new Response(JSON.stringify(st.params), { headers: { 'content-type': 'application/json' } })).catch(() => {});
    }

    const hit = await (cache && cache.match(_wKey(version, sha)).catch(() => null));
    if (hit) {                                   // cache hit — no network, no re-download
      st.bytes = meta.bytes;
      if (onProgress) onProgress({ version, downloadedBytes: meta.bytes, totalBytes: meta.bytes });
      st.buf = new Uint8Array(await hit.arrayBuffer());
    } else {
      // the whole stream lives inside the retry, so a mid-download failure restarts it
      st.buf = await _withRetry(async () => {
        const resp = await _fetchOk(`${HF_BASE}/${meta.weights}`, { cache: 'no-store' });
        const reader = resp.body.getReader();
        const total = meta.bytes;
        const buf = new Uint8Array(total);
        let offset = 0; st.bytes = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf.set(value, offset);
          offset += value.byteLength;
          st.bytes = offset;
          if (onProgress) onProgress({ version, downloadedBytes: offset, totalBytes: total });
        }
        return buf;
      }, `v${version} weights`);
      if (cache) cache.put(_wKey(version, sha),
        new Response(st.buf, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
    }
    st.phase = 'downloaded';
  })().catch((e) => {
    st.phase = 'error';
    st.fetchPromise = null;   // clear so a later call (e.g. Generate) can retry from scratch
    throw e;
  });
  return st.fetchPromise;
}

// Phase 2: build the tf.js net + warmup (GPU). Lazy — only when a version is
// actually used for generation.
let _buildChain = Promise.resolve();   // serialize builds so warmups don't pile on the GPU at once
async function buildVersionNet(version) {
  const st = _versionState.get(version);
  if (st.net) return st.net;
  await fetchVersionBytes(version);
  st.netPromise = st.netPromise || (async () => {
    await ensureBackend();
    // serialize against any other in-flight build
    const run = _buildChain.then(async () => {
      st.phase = 'creating-session';
      const net = makeQRBloom(tf, st.params, st.buf.buffer);
      const [gx, , gz] = st.params.grid;
      await net.forward(new Float32Array(4 * gx * gx * gz),
                        new Float32Array(gx * gx * gz), 0, 0, [0.5, 0.5, 0.5], 1.0);
      st.net = net; st.buf = null;   // free the raw bytes once on the GPU
      st.phase = 'ready';
      return net;
    });
    _buildChain = run.catch(() => {});
    return run;
  })().catch(e => { st.phase = 'error'; throw e; });
  return st.netPromise;
}

export async function downloadAllVersions(onProgress) {
  const m = await ensureManifest();
  const versions = m.trained_versions.slice();
  let ready = 0;
  const total = versions.length;
  const downloads = versions.map(v => fetchVersionBytes(v, p => {
    if (onProgress) onProgress({
      ready, total, current: v,
      downloadedBytes: p.downloadedBytes, totalBytes: p.totalBytes,
    });
  }).then(() => {
    ready += 1;
    if (onProgress) onProgress({
      ready, total, current: null, downloadedBytes: 0, totalBytes: 0,
    });
  }));
  await Promise.allSettled(downloads);
}

async function waitForVersion(version, onWait) {
  const st = _versionState.get(version);
  if (!st) throw new Error(`version ${version} not in manifest`);
  if (st.net) return st.net;
  // make sure bytes are downloading, surface progress, then build the net
  fetchVersionBytes(version).catch(() => {});
  while (!st.buf && !st.net) {
    if (st.phase === 'error') throw new Error(`v${version} download failed`);
    if (onWait) onWait({ version, downloadedBytes: st.bytes, totalBytes: st.total });
    await new Promise(r => setTimeout(r, 250));
  }
  return buildVersionNet(version);
}

// --- QR encoding -----------------------------------------------------------

// qrcode-generator returns a QR object once createImgTag/getModuleCount works.
// `make()` is the entry point; we try increasing type numbers until the text
// fits, then clamp to the trained range from the manifest.
function encodeQR(text, trainedVersions) {
  const min = Math.min(...trainedVersions);
  const max = Math.max(...trainedVersions);
  let lastErr = null;
  // Natural smallest fit.
  for (let v = 1; v <= 40; v++) {
    try {
      const qr = qrcode(v, 'M');
      qr.addData(text);
      qr.make();
      const naturalV = v;
      const chosen = Math.max(min, naturalV);
      if (chosen > max) throw new Error(`text too long for trained range (v${min}..v${max}); needs v${naturalV}`);
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
      // qrcode-generator throws when text doesn't fit; try larger version.
    }
  }
  throw lastErr || new Error('QR encode failed');
}

// --- cosine β schedule + per-step constants --------------------------------

function buildSchedule() {
  // Matches qrbloom/diffusion.py:cosine_beta_schedule (T=500, s=0.008).
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

// --- Box-Muller standard normal --------------------------------------------

function fillNormal(arr) {
  for (let i = 0; i < arr.length; i += 2) {
    const u1 = Math.max(1e-12, Math.random());
    const u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    arr[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < arr.length) arr[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
}

// --- model_generate-equivalent sampling loop -------------------------------

async function sampleVoxels(net, version, themeIdx, qr, onPhase) {
  const meta = _manifest.versions[String(version)];
  const D = meta.grid_xy;
  const H = meta.grid_xy;
  const W = meta.grid_z;
  const channels = X0_CH;
  const voxN = channels * D * H * W;

  // Build cond: 1 inside the QR footprint columns, 0 elsewhere.
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

  const steps = STEPS;
  const seq = linspaceSteps(steps);
  const noiseBuf = new Float32Array(voxN);

  for (let idx = 0; idx < steps; idx++) {
    const t = seq[idx];
    const vPred = await net.forward(x, condArr, t, themeIdx, [0.5, 0.5, 0.5], 1.0);  // Float32Array NCHW
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
    if (idx < steps - 1) {
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
    if (onPhase && (idx % 2 === 0 || idx === steps - 1)) {
      onPhase('sampling', { version, step: idx + 1, total: steps });
    }
  }

  // Enforce footprint: outside QR mask, set to -1 (will be rejected by occ > 0).
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < H; j++) {
      const baseDH = i * H + j;
      const masked = condArr[baseDH * W] < 0.5;  // any k works — column-wide mask
      if (!masked) continue;
      for (let c = 0; c < channels; c++) {
        const cBase = c * (D * H * W);
        for (let k = 0; k < W; k++) x[cBase + baseDH * W + k] = -1.0;
      }
    }
  }
  return { x0: x, D, H, W };
}

// --- post-processing: (occ, rgb) → viewer cell list ------------------------

function rgbToHex(r, g, b) {
  const hex = n => n.toString(16).padStart(2, '0');
  return '#' + hex(r) + hex(g) + hex(b);
}

function buildCells(x0, D, H, W, qr, themeMeta) {
  const cx = D >> 1;
  const m = qr.modules;
  const off = Math.floor((D - m) / 2);
  const cells = [];
  // Base plane (y=0) — same as gallery.py model_generate: per-QR-module color.
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const dark = qr.matrix[i * m + j] === 1;
      const color = dark ? themeMeta.qr_dark : themeMeta.qr_light;
      const gx = (off + j) - cx;
      const gz = (off + i) - cx;
      cells.push([gx, 0, gz, 1.0, color]);
    }
  }
  // Tree voxels at y>=1 where occ channel > 0.
  const channelSize = D * H * W;
  const occBase = 3 * channelSize;
  const rBase = 0;
  const gBase = channelSize;
  const bBase = 2 * channelSize;
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < H; j++) {
      const baseDH = (i * H + j) * W;
      for (let k = 1; k < W; k++) {
        if (x0[occBase + baseDH + k] <= 0) continue;
        const r = clampU8((x0[rBase + baseDH + k] + 1) * 127.5);
        const g = clampU8((x0[gBase + baseDH + k] + 1) * 127.5);
        const b = clampU8((x0[bBase + baseDH + k] + 1) * 127.5);
        const x = j - cx;
        const z = i - cx;
        cells.push([x, k, z, 1.0, rgbToHex(r, g, b)]);
      }
    }
  }
  return cells;
}

function clampU8(v) {
  v = Math.round(v);
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

// --- public generate -------------------------------------------------------

export async function generate({ url, theme, onPhase }) {
  const m = await ensureManifest();
  if (!url || typeof url !== 'string') throw new Error('url required');
  const themeMeta = m.themes.find(t => t.name === theme) || m.themes[0];
  const themeIdx = m.themes.findIndex(t => t.name === themeMeta.name);
  if (themeIdx < 0) throw new Error(`unknown theme ${theme}`);
  const qr = encodeQR(url, m.trained_versions);
  const net = await waitForVersion(qr.version, info => {
    if (onPhase) onPhase('waiting-download', info);
  });
  const { x0, D, H, W } = await sampleVoxels(net, qr.version, themeIdx, qr, onPhase);
  const cells = buildCells(x0, D, H, W, qr, themeMeta);
  return { cells, version: qr.version, theme: themeMeta.name, url };
}
