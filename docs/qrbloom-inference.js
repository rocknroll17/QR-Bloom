// In-browser QR-Bloom inference.
//
// Mirrors the home-hosted gallery.py pipeline:
//   pick_version_for_text(text) → segno-compatible smallest standard QR clamped
//   into the trained-version range from manifest.json. Encode the QR matrix,
//   pad it to the grid, run 100-step v-prediction diffusion sampling against
//   the per-version ONNX UNet (qrbloom_v{N}.fp32.onnx), and post-process the
//   (occ, rgb) output into the same `[x, y, z, scale, color]` cell list the
//   viewer expects.
//
// ONNX weights live on Hugging Face. Manifest at the same path supplies
// sha256, grid sizes, and theme metadata. First page load kicks off
// background downloads of every trained version; per-version inference waits
// only for the version it actually needs.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.webgpu.min.mjs';
import qrcode from 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/+esm';

const HF_BASE = 'https://huggingface.co/rocknroll17/QR-Bloom/resolve/main/onnx';
const MANIFEST_URL = `${HF_BASE}/manifest.json`;

// Allow ORT to fetch its wasm artifacts from the matching CDN bundle.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';

const T_TOTAL    = 500;        // Diffusion.T
const STEPS      = 100;        // Diffusion.sample default
const X0_CH      = 4;
const N_THEMES   = 10;
const COSINE_S   = 0.008;

// --- manifest + downloads --------------------------------------------------

let _manifest = null;
const _versionState = new Map();  // version → { phase, bytes, total, sessionPromise, bytesNumber }

export async function ensureManifest() {
  if (_manifest) return _manifest;
  const r = await fetch(MANIFEST_URL, { cache: 'force-cache' });
  if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
  _manifest = await r.json();
  for (const v of _manifest.trained_versions) {
    _versionState.set(v, {
      phase: 'pending', bytes: 0, total: _manifest.versions[String(v)].bytes,
      sessionPromise: null, session: null,
    });
  }
  return _manifest;
}

export function getDownloadState() {
  const out = {};
  for (const [v, s] of _versionState) {
    out[v] = { phase: s.phase, bytes: s.bytes, total: s.total };
  }
  return out;
}

async function downloadVersion(version, onProgress) {
  const st = _versionState.get(version);
  if (!st) throw new Error(`unknown version ${version}`);
  if (st.session) return st.session;
  if (st.sessionPromise) return st.sessionPromise;

  st.sessionPromise = (async () => {
    st.phase = 'downloading';
    const meta = _manifest.versions[String(version)];
    const url = `${HF_BASE}/${meta.file}`;
    const resp = await fetch(url, { cache: 'force-cache' });
    if (!resp.ok) throw new Error(`v${version} fetch ${resp.status}`);
    const reader = resp.body.getReader();
    const total = meta.bytes;
    const buf = new Uint8Array(total);
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf.set(value, offset);
      offset += value.byteLength;
      st.bytes = offset;
      if (onProgress) onProgress({ version, downloadedBytes: offset, totalBytes: total });
    }
    st.phase = 'creating-session';
    const session = await ort.InferenceSession.create(buf, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    });
    st.phase = 'ready';
    st.session = session;
    return session;
  })();
  return st.sessionPromise;
}

export async function downloadAllVersions(onProgress) {
  const m = await ensureManifest();
  const versions = m.trained_versions.slice();
  let ready = 0;
  const total = versions.length;
  const downloads = versions.map(v => downloadVersion(v, p => {
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
  if (st.session) return st.session;
  if (!st.sessionPromise) {
    // Page may not have kicked off background downloads yet — start now.
    downloadVersion(version);
  }
  // Poll progress for the status line until the session resolves.
  while (!st.session) {
    if (onWait) {
      onWait({ version, downloadedBytes: st.bytes, totalBytes: st.total });
    }
    await new Promise(r => setTimeout(r, 250));
    if (st.phase === 'error') throw new Error(`v${version} download failed`);
  }
  return st.session;
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

async function sampleVoxels(session, version, themeIdx, qr, onPhase) {
  const meta = _manifest.versions[String(version)];
  const D = meta.grid_xy;
  const H = meta.grid_xy;
  const W = meta.grid_z;
  const channels = X0_CH;
  const voxN = channels * D * H * W;

  // Build cond tensor: 1 inside the QR footprint columns, 0 elsewhere.
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
  const condTensor = new ort.Tensor('float32', condArr, [1, 1, D, H, W]);
  const themeTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(themeIdx)]), [1]);
  const attrTensor  = new ort.Tensor('float32', Float32Array.from([0.5, 0.5, 0.5]), [1, 3]);
  const m1Tensor    = new ort.Tensor('int64', BigInt64Array.from([1n]), [1]);

  let x = new Float32Array(voxN);
  fillNormal(x);

  const steps = STEPS;
  const seq = linspaceSteps(steps);
  const noiseBuf = new Float32Array(voxN);

  for (let idx = 0; idx < steps; idx++) {
    const t = seq[idx];
    const tTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(t)]), [1]);
    const xTensor = new ort.Tensor('float32', x, [1, channels, D, H, W]);
    const out = await session.run({
      x: xTensor, t: tTensor, cond: condTensor,
      theme: themeTensor, attr: attrTensor, attr_mask: m1Tensor,
    });
    const vPred = out.v_pred.data;   // Float32Array
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
    if (onPhase && (idx % 5 === 0 || idx === steps - 1)) {
      onPhase('sampling', { version, step: idx + 1, total: steps });
    }
    // Yield to the browser so the UI stays alive.
    if (idx % 4 === 0) await new Promise(r => setTimeout(r, 0));
  }

  // Enforce footprint: outside QR mask, set to -1 (will be rejected by occ > 0).
  // Mask is 1 inside columns where condArr > 0.5, else 0.
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
  const sliceDH = H * W;
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
  const session = await waitForVersion(qr.version, info => {
    if (onPhase) onPhase('waiting-download', info);
  });
  const { x0, D, H, W } = await sampleVoxels(session, qr.version, themeIdx, qr, onPhase);
  const cells = buildCells(x0, D, H, W, qr, themeMeta);
  return { cells, version: qr.version, theme: themeMeta.name, url };
}

// --- progress observers ----------------------------------------------------

const _listeners = new Set();
export function onDownloadProgress(fn) { _listeners.add(fn); return () => _listeners.delete(fn); }
