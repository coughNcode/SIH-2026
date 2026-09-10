/**
 * DepthWizard — app.js v2
 * - Zero img src errors: all images loaded via createImageBitmap/canvas
 * - IM2ELEVATION green hillshade dual panel
 * - Three.js 3D terrain from raw float heights (no 16-bit PNG trick)
 * - Click-to-inspect real meter values
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── API base — same origin (FastAPI serves static + API)
const API = window.location.origin;  // http://localhost:8000

// ── DOM refs ────────────────────────────────────────────────────────────────
const dropzone      = document.getElementById("dropzone");
const dzInner       = document.getElementById("dzInner");
const fileInput     = document.getElementById("fileInput");
const previewCanvas = document.getElementById("previewCanvas");
const runBtn        = document.getElementById("runBtn");
const statusWrap    = document.getElementById("statusWrap");
const statusMsg     = document.getElementById("statusMsg");
const progFill      = document.getElementById("progFill");
const metricsGrid   = document.getElementById("metricsGrid");
const resultCard    = document.getElementById("resultCard");
const resultStats   = document.getElementById("resultStats");
const inspectCard   = document.getElementById("inspectCard");
const inspectContent= document.getElementById("inspectContent");
const ctrlCard      = document.getElementById("ctrlCard");
const im2elevPanels = document.getElementById("im2elevPanels");
const canvasHolder  = document.getElementById("canvasHolder");
const emptyState    = document.getElementById("emptyState");
const viewerHud     = document.getElementById("viewerHud");
const metaHint      = document.getElementById("metaHint");
const rangeHint     = document.getElementById("rangeHint");
const hsLegend      = document.getElementById("hsLegend");
const hsMax         = document.getElementById("hsMax");
const plMax         = document.getElementById("plMax");
const loadingOverlay= document.getElementById("loadingOverlay");
const loaderMsg     = document.getElementById("loaderMsg");
const deviceChip    = document.getElementById("deviceChip");
const exagSlider    = document.getElementById("exagSlider");
const exagVal       = document.getElementById("exagVal");
const wireToggle    = document.getElementById("wireframeToggle");
const vbtn3d        = document.getElementById("vbtn3d");
const vbtnhs        = document.getElementById("vbtnhs");
const vbtnrgb       = document.getElementById("vbtnrgb");
const cRgb          = document.getElementById("c_rgb");
const cHs           = document.getElementById("c_hs");

// ── State ────────────────────────────────────────────────────────────────────
let selectedFile   = null;
let heightsArr     = null;   // Float32Array, meters, 224×224
let gridSize       = 224;
let predMax        = 30;
let currentMode    = "3d";
let heightScale    = 4.0;

// ── Upload UX ────────────────────────────────────────────────────────────────
dropzone.addEventListener("click",  () => fileInput.click());
dropzone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
dropzone.addEventListener("dragover",  e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", ()  => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

async function handleFile(file) {
  if (!file.type.startsWith("image/") && !file.name.match(/\.(tif|tiff)$/i)) {
    setStatus("Please choose an image file.", "err"); return;
  }
  selectedFile = file;
  runBtn.disabled = false;

  // Preview via canvas (avoids any img src MIME issue)
  try {
    const bmp = await createImageBitmap(file);
    previewCanvas.width  = bmp.width;
    previewCanvas.height = bmp.height;
    previewCanvas.getContext("2d").drawImage(bmp, 0, 0);
    previewCanvas.style.display = "block";
    dzInner.style.display = "none";
    bmp.close();
  } catch {
    dzInner.style.display = "block";
  }
  setStatus("");
}

// ── Inference ─────────────────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  showLoader("Reading image…");
  setStatus("Sending to HeightNet GPU backend…");

  try {
    // Use multipart form upload — no base64 MIME guessing
    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name);

    showLoader("Running HeightNet inference…");
    const res = await fetch(`${API}/api/infer-form`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(60000),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Inference failed (${res.status}): ${errText}`);
    }

    const data = await res.json();

    // Store heights
    heightsArr = new Float32Array(data.heights);
    gridSize   = data.width;
    predMax    = data.pred_max_m;

    // ── Render IM2ELEVATION panels ──
    showLoader("Rendering visualization…");
    await renderIM2ElevPanels(data);

    // ── Build 3D terrain ──
    showLoader("Building 3D terrain…");
    await buildTerrain(data);

    // ── Update UI ──
    hideLoader();
    setStatus("Done ✓ — orbit to explore, click terrain to inspect", "ok");

    resultCard.style.display  = "block";
    inspectCard.style.display = "block";
    ctrlCard.style.display    = "block";
    viewerHud.style.display   = "flex";
    hsLegend.style.display    = "flex";
    metaHint.textContent      = `${gridSize}×${gridSize} grid`;
    rangeHint.textContent     = `0 – ${predMax.toFixed(1)} m`;
    hsMax.textContent         = `${predMax.toFixed(1)} m`;
    plMax.textContent         = `${predMax.toFixed(1)} m`;

    resultStats.innerHTML = `
      <div class="rs-row"><span class="rs-key">Height range</span><span class="rs-val">${data.pred_min_m.toFixed(2)} – ${predMax.toFixed(2)} m</span></div>
      <div class="rs-row"><span class="rs-key">Mean height</span><span class="rs-val">${data.pred_mean_m.toFixed(2)} m</span></div>
      <div class="rs-row"><span class="rs-key">Grid size</span><span class="rs-val">${gridSize} × ${gridSize}</span></div>
      <div class="rs-row"><span class="rs-key">Source</span><span class="rs-val">HeightNet (GPU)</span></div>
    `;

  } catch (err) {
    console.error(err);
    hideLoader();
    setStatus(`Error: ${err.message}`, "err");
  } finally {
    runBtn.disabled = false;
  }
});

// ── IM2ELEVATION panels ─────────────────────────────────────────────────────
async function renderIM2ElevPanels(data) {
  im2elevPanels.style.display = "flex";
  canvasHolder.style.display  = "none";
  emptyState.style.display    = "none";

  // RGB panel
  await drawBase64ToCanvas(cRgb, data.rgb_b64);

  // Hillshade panel
  await drawBase64ToCanvas(cHs, data.hillshade_b64);
}

function drawBase64ToCanvas(canvas, b64) {
  return new Promise((resolve, reject) => {
    const imgEl = new Image();
    imgEl.onload = () => {
      canvas.width  = imgEl.naturalWidth;
      canvas.height = imgEl.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imgEl, 0, 0);
      resolve();
    };
    imgEl.onerror = reject;
    imgEl.src = `data:image/png;base64,${b64}`;
  });
}

// ── Three.js terrain ────────────────────────────────────────────────────────
let renderer, scene, camera, controls, terrainMesh, raycaster, rgbTexture;

function initThree() {
  if (renderer) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111a22);
  scene.fog = new THREE.FogExp2(0x111a22, 0.006);

  const w = canvasHolder.clientWidth;
  const h = canvasHolder.clientHeight;
  camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 2000);
  camera.position.set(0, 100, 160);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.shadowMap.enabled = true;
  canvasHolder.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.06;
  controls.minDistance    = 10;
  controls.maxDistance    = 500;
  controls.maxPolarAngle  = Math.PI * 0.84;
  controls.target.set(0, 0, 0);

  // Lighting — dramatic side light to emphasize terrain
  const hemi = new THREE.HemisphereLight(0x223344, 0x112233, 0.6);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffeedd, 1.8);
  sun.position.set(80, 150, 60);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  const rim = new THREE.DirectionalLight(0x4488bb, 0.4);
  rim.position.set(-100, 50, -80);
  scene.add(rim);

  raycaster = new THREE.Raycaster();
  renderer.domElement.addEventListener("click", onTerrainClick);
  window.addEventListener("resize", onResize);
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function onResize() {
  if (!renderer) return;
  const w = canvasHolder.clientWidth;
  const h = canvasHolder.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

async function buildTerrain(data) {
  // Show 3D canvas (behind panels initially, switch to it on 3D button click)
  initThree();
  canvasHolder.style.display = "block";
  canvasHolder.style.position = "absolute";
  canvasHolder.style.inset = "0";
  canvasHolder.style.zIndex = "-1";  // behind panels by default

  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
  }

  // Load hillshade as texture for 3D
  const tex = await loadTextureFromB64(data.hillshade_b64);

  const meshSize = 200;
  const geo = new THREE.PlaneGeometry(meshSize, meshSize, gridSize - 1, gridSize - 1);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes.position;
  const vs  = (meshSize / Math.max(predMax, 1)) * heightScale;

  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heightsArr[i] * vs);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    map:       tex,
    roughness: 0.88,
    metalness: 0.05,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.userData = { meshSize, vs };
  scene.add(terrainMesh);

  camera.position.set(0, 100, 160);
  controls.target.set(0, 20, 0);
  controls.update();
}

function loadTextureFromB64(b64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      resolve(tex);
    };
    img.onerror = reject;
    img.src = `data:image/png;base64,${b64}`;
  });
}

// ── Click-to-inspect ─────────────────────────────────────────────────────────
function onTerrainClick(e) {
  if (!terrainMesh || !heightsArr || currentMode !== "3d") return;
  const rect  = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width)  * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(terrainMesh);
  if (!hits.length) return;

  const p   = hits[0].point;
  const { meshSize } = terrainMesh.userData;
  const cx  = ((p.x / meshSize) + 0.5) * (gridSize - 1);
  const cy  = ((-p.z / meshSize) + 0.5) * (gridSize - 1);
  const ix  = Math.round(Math.max(0, Math.min(gridSize - 1, cx)));
  const iy  = Math.round(Math.max(0, Math.min(gridSize - 1, cy)));
  const hM  = heightsArr[iy * gridSize + ix] ?? 0;

  inspectCard.style.display = "block";
  inspectContent.innerHTML = `
    <div class="inspect-row"><span class="ik">Predicted height</span><span class="iv">${hM.toFixed(2)} m</span></div>
    <div class="inspect-row"><span class="ik">Grid position</span><span class="iv">(${ix}, ${iy})</span></div>
    <div class="inspect-row"><span class="ik">% of max</span><span class="iv">${(hM / predMax * 100).toFixed(1)}%</span></div>
    <div class="inspect-row"><span class="ik">World XZ</span><span class="iv">${p.x.toFixed(1)}, ${p.z.toFixed(1)}</span></div>
  `;
}

// ── View mode buttons ─────────────────────────────────────────────────────────
function setViewMode(mode) {
  currentMode = mode;
  [vbtn3d, vbtnhs, vbtnrgb].forEach(b => b.classList.remove("active"));

  if (mode === "3d") {
    vbtn3d.classList.add("active");
    im2elevPanels.style.display = "none";
    canvasHolder.style.display  = "block";
    canvasHolder.style.zIndex   = "1";
    canvasHolder.style.position = "relative";
    emptyState.style.display    = "none";
  } else if (mode === "hs") {
    vbtnhs.classList.add("active");
    // Full-screen hillshade panel
    im2elevPanels.style.display = "flex";
    canvasHolder.style.zIndex   = "-1";
    // Make hillshade fill the entire right area
    const row = document.querySelector(".panel-row");
    if (row) {
      document.getElementById("p_rgb").style.display = "none";
      document.getElementById("p_hs").style.flex = "1";
    }
  } else {
    vbtnrgb.classList.add("active");
    im2elevPanels.style.display = "flex";
    canvasHolder.style.zIndex   = "-1";
    const row = document.querySelector(".panel-row");
    if (row) {
      document.getElementById("p_rgb").style.display = "flex";
      document.getElementById("p_rgb").style.flex = "1";
      document.getElementById("p_hs").style.display = "none";
    }
  }
}

vbtn3d.addEventListener("click",  () => {
  // Reset panels
  ["p_rgb", "p_hs"].forEach(id => {
    const el = document.getElementById(id);
    el.style.display = "flex"; el.style.flex = "1";
  });
  setViewMode("3d");
});
vbtnhs.addEventListener("click",  () => setViewMode("hs"));
vbtnrgb.addEventListener("click", () => {
  ["p_rgb", "p_hs"].forEach(id => {
    const el = document.getElementById(id);
    el.style.display = "flex"; el.style.flex = "1";
  });
  setViewMode("rgb");
});

// ── Exaggeration ──────────────────────────────────────────────────────────────
exagSlider.addEventListener("input", () => {
  heightScale = parseFloat(exagSlider.value);
  exagVal.textContent = `${heightScale.toFixed(1)}×`;
  if (!terrainMesh || !heightsArr) return;
  const pos = terrainMesh.geometry.attributes.position;
  const { meshSize } = terrainMesh.userData;
  const vs = (meshSize / Math.max(predMax, 1)) * heightScale;
  for (let i = 0; i < pos.count; i++) pos.setY(i, heightsArr[i] * vs);
  pos.needsUpdate = true;
  terrainMesh.geometry.computeVertexNormals();
  terrainMesh.userData.vs = vs;
});

// ── Wireframe ──────────────────────────────────────────────────────────────────
wireToggle.addEventListener("change", () => {
  if (terrainMesh) terrainMesh.material.wireframe = wireToggle.checked;
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showLoader(msg) {
  loaderMsg.textContent = msg;
  loadingOverlay.style.display = "flex";
  statusWrap.style.display = "block";
  statusMsg.textContent = msg;
  statusMsg.style.color = "";
}

function hideLoader() {
  loadingOverlay.style.display = "none";
}

function setStatus(msg, kind) {
  statusWrap.style.display = msg ? "block" : "none";
  statusMsg.textContent = msg;
  statusMsg.style.color = kind === "err" ? "var(--red)" : kind === "ok" ? "var(--green)" : "";
}

// ── Metrics ───────────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch(`${API}/metrics`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error("no metrics");
    const d   = await res.json();
    const m   = d.metrics;
    const rClass = m.RMSE_m  < 2.5 ? "good" : m.RMSE_m  < 4.0 ? "warn" : "bad";
    const dClass = m.delta1  > 0.8 ? "good" : m.delta1  > 0.6 ? "warn" : "bad";
    metricsGrid.innerHTML = `
      <div class="mc ${rClass}">
        <div class="mc-label">RMSE</div>
        <div class="mc-val">${m.RMSE_m.toFixed(3)}<span class="mc-unit">m</span></div>
      </div>
      <div class="mc">
        <div class="mc-label">MAE</div>
        <div class="mc-val">${m.MAE_m.toFixed(3)}<span class="mc-unit">m</span></div>
      </div>
      <div class="mc ${dClass}">
        <div class="mc-label">δ₁ Acc.</div>
        <div class="mc-val">${(m.delta1*100).toFixed(1)}<span class="mc-unit">%</span></div>
      </div>
      <div class="mc">
        <div class="mc-label">RMSE σ</div>
        <div class="mc-val">${m.RMSE_std_m.toFixed(3)}<span class="mc-unit">m</span></div>
      </div>
    `;
  } catch {
    metricsGrid.innerHTML = `<p class="muted" style="grid-column:1/-1">Run <code>evaluate.py</code> to load metrics.</p>`;
  }
}

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const d = await r.json();
    deviceChip.textContent = `${d.device === "cuda" ? "GPU" : "CPU"} · ${d.device}`;
    deviceChip.classList.add("chip-green");
  } catch {
    deviceChip.textContent = "API offline";
    deviceChip.classList.remove("chip-green");
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkHealth();
loadMetrics();
