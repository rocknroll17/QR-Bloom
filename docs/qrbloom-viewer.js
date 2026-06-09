// Shared 3D voxel viewer for QR-Bloom.
//
// One implementation, used by both:
//   - templates/index.html  (full gallery, served by gallery.py)
//   - docs/index.html       (static GitHub Pages demo)
//
// The gallery exposes this module at `/qrbloom-viewer.js` (route added in
// gallery.py); the Pages site fetches the same file from `./qrbloom-viewer.js`.
// Both pages thus share rendering, camera presets, color blending, and the
// tree<->QR transition behavior — no more drift between renderers.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// Canonical camera presets. Anything that builds a viewer of voxel trees on
// top of a QR plane should pick from these — no inventing new angles.
export const CAMERAS = {
  tree: { pos: [45, 20, 45], tgt: [0, 14, 0], up: [0, 1, 0], halfSize: 24 },
  qr:   { pos: [0, 100, 0],  tgt: [0, 0, 0],  up: [0, 0, -1], halfSize: 16.5 },
};

const DIR_BASE_INTENSITY = 1.2;

export class Viewer {
  /**
   * Build a viewer inside `container` (defaults to document.body).
   * The renderer's canvas is appended to that container.
   */
  constructor(container = document.body) {
    this.container = container;
    this._setupRenderer();
    this._setupScene();
    this._setupCamera();
    this._setupLights();
    this._setupControls();

    // Animation / transition state.
    this.mode = 'tree';
    this._lerpTo = CAMERAS.tree;
    this._isTransitioning = false;
    this._transitionFrames = 0;
    // blendFactor: 0 = tree colors, 1 = QR-base colors under each column.
    this._blendFactor = 0;
    this._targetBlend = 0;

    this._posT = new THREE.Vector3();
    this._tgtT = new THREE.Vector3();
    this._upT = new THREE.Vector3();
    this._tmpColor = new THREE.Color();

    this._meshInstances = [];   // [{mesh, items: [{treeCol, qrCol}]}]
    this._root = new THREE.Group();
    this._scene.add(this._root);
    this._boxGeo = new THREE.BoxGeometry(1, 1, 1);

    // Growth-reveal animation: new trees pop in voxel-by-voxel from the ground
    // up (see setCells(..., animate=true) and _updateGrowth).
    this._growthItems = [];
    this._growing = false;
    this._growthStart = 0;
    this._gM = new THREE.Matrix4();
    this._gP = new THREE.Vector3();
    this._gQ = new THREE.Quaternion();
    this._gS = new THREE.Vector3();

    this._onResize = this._onResize.bind(this);
    addEventListener('resize', this._onResize);

    this._running = true;
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  // --- internal setup ------------------------------------------------------

  _setupRenderer() {
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(innerWidth, innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(renderer.domElement);
    this.renderer = renderer;
  }

  _setupScene() {
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0xf8fafc);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    this._scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  }

  _setupCamera() {
    this._orthoHalf = CAMERAS.tree.halfSize;
    const aspect = innerWidth / innerHeight;
    this.camera = new THREE.OrthographicCamera(
      -this._orthoHalf * aspect, this._orthoHalf * aspect,
      this._orthoHalf, -this._orthoHalf, 0.1, 500
    );
    this.camera.position.fromArray(CAMERAS.tree.pos);
    this.camera.up.fromArray(CAMERAS.tree.up);
  }

  _setupLights() {
    this._amb = new THREE.AmbientLight(0xffffff, 0.6);
    this._scene.add(this._amb);
    this._dir = new THREE.DirectionalLight(0xffffff, DIR_BASE_INTENSITY);
    this._dir.position.set(20, 30, 20);
    this._dir.castShadow = true;
    this._dir.shadow.mapSize.set(1024, 1024);
    this._dir.shadow.camera.left = -30; this._dir.shadow.camera.right = 30;
    this._dir.shadow.camera.top = 30; this._dir.shadow.camera.bottom = -30;
    this._dir.shadow.camera.near = 1; this._dir.shadow.camera.far = 100;
    this._dir.shadow.bias = -0.0002; this._dir.shadow.normalBias = 0.02;
    this._scene.add(this._dir);
  }

  _setupControls() {
    const controls = new OrbitControls(this.camera, this.renderer.domElement);
    controls.target.fromArray(CAMERAS.tree.tgt);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    // Pan stays on in tree mode (right-click drag); switchMode disables it
    // while in QR mode so the top-down view stays centered.
    controls.enablePan = true;
    controls.minDistance = 15;
    controls.maxDistance = 260;
    controls.maxPolarAngle = Math.PI / 2;
    this.controls = controls;
  }

  // --- public API ----------------------------------------------------------

  /**
   * Replace the rendered voxels with a new cell list.
   * Each cell is `[x, y, z, scale, color]` where y == 0 marks QR base voxels.
   */
  setCells(cells, animate = false) {
    this._clearRoot();
    this._growthItems = [];
    this._growing = false;
    if (!cells || !cells.length) return;

    // Map (x, z) -> QR module color from the base layer (y == 0). Tree voxels
    // lerp to this color in QR mode so the top-down silhouette reads cleanly.
    const qrUnder = new Map();
    for (const c of cells) {
      if (c[1] === 0) qrUnder.set(c[0] + ',' + c[2], c[4]);
    }

    // Group by (scale, isBase) so base voxels can skip casting shadows while
    // tree voxels still cast onto them. All groups share a white material and
    // per-instance color (lerped between treeCol and qrCol).
    const groups = new Map();
    const treeSet = new Set();          // "x,y,z" of tree (non-base) voxels
    let maxY = 0, minTreeY = Infinity;
    for (const c of cells) {
      const [x, y, z, scale, color] = c;
      if (y > maxY) maxY = y;
      const isBase = (y === 0);
      if (!isBase) { treeSet.add(x + ',' + y + ',' + z); if (y < minTreeY) minTreeY = y; }
      const key = scale + '|' + (isBase ? 'b' : 't');
      if (!groups.has(key)) groups.set(key, { scale, isBase, items: [] });
      groups.get(key).items.push({
        pos: [x, y, z],
        treeCol: new THREE.Color(color),
        qrCol: new THREE.Color(isBase ? color : (qrUnder.get(x + ',' + z) || color)),
      });
    }
    const span = Math.max(1, maxY);
    // Deterministic hash for a little per-voxel reveal jitter.
    const hash = (x, y, z) => {
      const s = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
      return s - Math.floor(s);
    };

    // Reveal order: BFS over the *tree* voxels (26-connectivity) seeded from the
    // trunk base, so growth climbs the trunk and spreads out to the branches and
    // canopy like a real tree — not a flat layer-by-layer stack. dist = graph
    // steps from the base; disconnected clusters fall back to their height.
    const dist = new Map();
    let maxD = 1;
    if (animate && treeSet.size) {
      const q = [];
      for (const k of treeSet) {
        if (Number(k.split(',')[1]) <= minTreeY) { dist.set(k, 0); q.push(k); }
      }
      const NB = [];
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++)
        if (dx || dy || dz) NB.push([dx, dy, dz]);
      for (let head = 0; head < q.length; head++) {
        const k = q[head], d = dist.get(k);
        const [x, y, z] = k.split(',').map(Number);
        for (const [dx, dy, dz] of NB) {
          const nk = (x + dx) + ',' + (y + dy) + ',' + (z + dz);
          if (treeSet.has(nk) && !dist.has(nk)) {
            dist.set(nk, d + 1); if (d + 1 > maxD) maxD = d + 1; q.push(nk);
          }
        }
      }
    }

    const M = new THREE.Matrix4();
    const Q = new THREE.Quaternion();
    const P = new THREE.Vector3();
    const SV = new THREE.Vector3();
    for (const { scale, isBase, items } of groups.values()) {
      const mat = new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.9, metalness: 0.0,
      });
      const mesh = new THREE.InstancedMesh(this._boxGeo, mat, items.length);
      mesh.castShadow = !isBase;
      mesh.receiveShadow = true;
      for (let i = 0; i < items.length; i++) {
        const p = items[i].pos;
        P.set(p[0], p[1], p[2]);
        SV.setScalar(animate ? 0.0001 : scale);   // start hidden when animating
        M.compose(P, Q, SV);
        mesh.setMatrixAt(i, M);
        this._tmpColor.copy(items[i].treeCol).lerp(items[i].qrCol, this._blendFactor);
        mesh.setColorAt(i, this._tmpColor);
        if (animate) {
          let revealAt;
          if (p[1] === 0) {
            revealAt = 0.02;                                   // QR ground forms first
          } else {
            const k = p[0] + ',' + p[1] + ',' + p[2];
            const d = dist.has(k) ? dist.get(k) / maxD         // BFS growth distance
                                  : (p[1] - minTreeY) / span;  // disconnected -> height
            revealAt = 0.08 + 0.88 * d;
          }
          revealAt += (hash(p[0], p[1], p[2]) - 0.5) * 0.05;
          revealAt = Math.max(0, Math.min(0.95, revealAt));
          this._growthItems.push({ mesh, i, px: p[0], py: p[1], pz: p[2], scale, revealAt });
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      this._root.add(mesh);
      this._meshInstances.push({ mesh, items });
    }

    if (animate && this._growthItems.length) {
      this._growing = true;
      this._growthStart = performance.now();
    }
  }

  /** Switch between 'tree' (iso) and 'qr' (top-down) views. */
  switchMode(mode) {
    if (this.mode === mode) return;
    if (!CAMERAS[mode]) return;
    this.mode = mode;
    this._lerpTo = CAMERAS[mode];
    this._targetBlend = (mode === 'qr') ? 1.0 : 0.0;
    this._isTransitioning = true;
    this._transitionFrames = 0;
    this.controls.enabled = false;
  }

  /** Return the last-rendered scene as a PNG data URL (used by snapshot UIs). */
  snapshot() {
    this.renderer.render(this._scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  destroy() {
    this._running = false;
    removeEventListener('resize', this._onResize);
    this._clearRoot();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // --- internals -----------------------------------------------------------

  _clearRoot() {
    while (this._root.children.length) {
      const c = this._root.children[0]; this._root.remove(c);
      c.traverse(o => {
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material])
          .forEach(m => m.dispose());
      });
    }
    this._meshInstances = [];
  }

  _updateInstanceColors() {
    for (const { mesh, items } of this._meshInstances) {
      for (let i = 0; i < items.length; i++) {
        this._tmpColor.copy(items[i].treeCol).lerp(items[i].qrCol, this._blendFactor);
        mesh.setColorAt(i, this._tmpColor);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  // Advance the bottom-up grow-in. Each voxel scales 0 -> full (with a small
  // easeOutBack overshoot for a "pop") once the rising growth front passes its
  // height. Runs only for ~DURATION ms after a new animated tree, then stops.
  _updateGrowth() {
    const DURATION = 1400;   // ms for the whole tree to finish growing
    const WINDOW = 0.28;     // how long one voxel takes to pop (reveal units)
    const p = Math.min(1, (performance.now() - this._growthStart) / DURATION);
    // The reveal front sweeps 0 -> (1 + WINDOW), so even the last voxel
    // (revealAt up to ~1) completes its full window — its `local` reaches 1 and
    // it settles at exactly scale 1. (Capping the front at 1 left late canopy
    // voxels permanently undersized.) The easeOutBack still overshoots mid-pop
    // for the bounce, but lands cleanly on 1.
    const front = p * (1 + WINDOW);
    const M = this._gM, P = this._gP, Q = this._gQ, S = this._gS;
    const c1 = 1.70158, c3 = c1 + 1;
    const touched = new Set();
    for (const it of this._growthItems) {
      let local = (front - it.revealAt) / WINDOW;
      local = local <= 0 ? 0 : (local >= 1 ? 1 : local);
      let s;
      if (local <= 0) s = 0.0001;
      else if (local >= 1) s = 1;                                  // settle exactly at 1
      else { const x = local - 1; s = 1 + c3 * x * x * x + c1 * x * x; }  // easeOutBack bounce
      S.setScalar(it.scale * s);
      P.set(it.px, it.py, it.pz);
      M.compose(P, Q, S);
      it.mesh.setMatrixAt(it.i, M);
      touched.add(it.mesh);
    }
    for (const mesh of touched) mesh.instanceMatrix.needsUpdate = true;
    if (p >= 1) { this._growing = false; this._growthItems = []; }
  }

  _onResize() {
    this.renderer.setSize(innerWidth, innerHeight);
    const aspect = innerWidth / innerHeight;
    this.camera.left = -this._orthoHalf * aspect;
    this.camera.right = this._orthoHalf * aspect;
    this.camera.top = this._orthoHalf;
    this.camera.bottom = -this._orthoHalf;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    if (!this._running) return;
    requestAnimationFrame(this._animate);

    // QR mode keeps the camera locked at the top-down preset; tree mode lerps
    // only during a transition and otherwise allows free orbit.
    const shouldLerp = (this.mode === 'qr') || this._isTransitioning;
    const K = 0.032;

    // Color blend always lerps, independent of camera movement.
    const newBlend = this._blendFactor + (this._targetBlend - this._blendFactor) * K;
    if (Math.abs(newBlend - this._blendFactor) > 0.001) {
      this._blendFactor = newBlend;
      this._updateInstanceColors();
      // Fade directional, boost ambient as we approach QR mode for flat shading.
      this._dir.intensity = (1 - this._blendFactor) * DIR_BASE_INTENSITY;
      this._amb.intensity = 0.6 + this._blendFactor * 1.0;
    }

    if (shouldLerp) {
      this._posT.fromArray(this._lerpTo.pos);
      this._tgtT.fromArray(this._lerpTo.tgt);
      this._upT.fromArray(this._lerpTo.up);
      this.camera.position.lerp(this._posT, K);
      this.camera.up.lerp(this._upT, K).normalize();
      this.controls.target.lerp(this._tgtT, K);
      this._orthoHalf += (this._lerpTo.halfSize - this._orthoHalf) * K;
      this._onResize();

      if (this._isTransitioning) {
        this._transitionFrames++;
        const dist = this.camera.position.distanceTo(this._posT);
        if (dist < 0.3 || this._transitionFrames > 300) {
          // Snap to the exact target so the resting state is deterministic.
          this._isTransitioning = false;
          this.controls.enabled = true;
          this.controls.enableRotate = (this.mode === 'tree');
          this.controls.enablePan = (this.mode === 'tree');
          this.camera.position.fromArray(this._lerpTo.pos);
          this.camera.up.fromArray(this._lerpTo.up).normalize();
          this.controls.target.fromArray(this._lerpTo.tgt);
          this._orthoHalf = this._lerpTo.halfSize;
          this._onResize();
        }
      }
    }

    // While free-orbiting in tree mode: reduce rotateSpeed near the top pole
    // (prevents azimuth twisting), and start blending toward QR colors as the
    // camera tilts toward top-down so the transition feels continuous.
    if (!this._isTransitioning && this.mode === 'tree') {
      const v = new THREE.Vector3()
        .subVectors(this.camera.position, this.controls.target)
        .normalize();
      const polar = Math.acos(Math.max(-1, Math.min(1, v.y)));   // 0 = top
      this.controls.rotateSpeed = polar < 0.35 ? (0.15 + (polar / 0.35) * 0.85) : 1.0;
      // Smoothstep blend over the last ~10deg approaching top-down.
      let bt = (0.175 - polar) / 0.175;
      bt = Math.max(0, Math.min(1, bt));
      this._targetBlend = bt * bt * (3 - 2 * bt);
    }

    if (this._growing) this._updateGrowth();

    this.controls.update();
    this.renderer.render(this._scene, this.camera);
  }
}
