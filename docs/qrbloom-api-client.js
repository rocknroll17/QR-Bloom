// Server-side generation client for self-hosted deployments.
//
// The gallery injects `window.QRBLOOM_API` when it serves the page, and the
// page then loads this module instead of the in-browser diffusion stack —
// no tf.js, no weights download. It implements the same interface as
// ModelClient (qrbloom-inference.js): themes / prepare / state / source /
// generate.

export class ApiClient {
  constructor(base) {
    this.base = base;
  }

  async themes() {
    const r = await fetch(`${this.base}/themes`);
    if (!r.ok) throw new Error(`themes fetch ${r.status}`);
    return r.json();
  }

  /** Nothing to download — the model lives on the server. */
  async prepare() {}

  state() {
    return { phase: 'ready', bytes: 0, total: 0 };
  }

  source() {
    return 'server';
  }

  async generate({ url, theme, onPhase }) {
    if (onPhase) onPhase('sampling', { step: 0, total: 0 });   // indeterminate
    const r = await fetch(`${this.base}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, theme }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    return d;
  }
}
