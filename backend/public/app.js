/**
 * DepthWizard — app.js (clean rebuild)
 * 4-panel grid: RGB | DEM (turbo) | Hillshade | Surface Normals
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const API = window.location.origin;

// DOM refs
const dropzone       = document.getElementById("dropzone");
const dzInner        = document.getElementById("dzInner");
const fileInput      = document.getElementById("fileInput");
const previewCanvas  = document.getElementById("previewCanvas");
const runBtn         = document.getElementById("runBtn");
const statusBar      = document.getElementById("statusBar");
const statusText     = document.getElementById("statusText");
const progressFill   = document.getElementById("progressFill");
const statsCard      = document.getElementById("statsCard");
const statsList      = document.getElementById("statsList");
const emptyState     = document.getElementById("emptyState");
const panels         = document.getElementById("panels");
const panelGrid      = document.getElementById("panelGrid");
const canvas3d       = document.getElementById("canvas3d");
const threeMnt       = document.getElementById("three-mount");
const hud3d          = document.getElementById("hud3d");
const loadingOverlay = document.getElementById("loadingOverlay");
const loaderMsg      = document.getElementById("loaderMsg");
const deviceBadge    = document.getElementById("deviceBadge");
const vbtnGrid       = document.getElementById("vbtnGrid");
const vbtn3D         = document.getElementById("vbtn3D");
const cRgb           = document.getElementById("cRgb");
const cDem           = document.getElementById("cDem");
const cHs            = document.getElementById("cHs");
const cNorm          = document.getElementById("cNorm");
const demMin         = document.getElementById("demMin");
const demMax         = document.getElementById("demMax");

let selectedFile = null;
let heightsArr   = null;
let gridW = 518, gridH = 518;
let predMaxM = 30;

// ── Upload UX ─────────────────────────────────────────────────────
dropzone.addEventListener("click",   () => fileInput.click());
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
  selectedFile = file;
  runBtn.disabled = false;
  setStatus("Image loaded — click Generate Height Map");
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
}

// ── Inference ─────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  showLoader("Uploading image…");
  setProgress(20);

  try {
    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name);

    showLoader("Running HeightNet inference on GPU…");
    setProgress(40);

    const res = await fetch(`${API}/api/infer-form`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(120000),
    });

    if (!res.ok) throw new Error(`Inference failed (${res.status}): ${await res.text()}`);

    showLoader("Rendering panels…");
    setProgress(70);

    const data = await res.json();
    heightsArr = new Float32Array(data.heights);
    gridW = data.width;
    gridH = data.height;
    predMaxM = data.pred_max_m;

    // Render all 4 canvases in parallel
    await Promise.all([
      drawB64ToCanvas(cRgb,  data.rgb_b64),
      drawB64ToCanvas(cDem,  data.dem_b64),
      drawB64ToCanvas(cHs,   data.hillshade_b64),
      drawB64ToCanvas(cNorm, data.normals_b64),
    ]);

    setProgress(90);
    showLoader("Building 3D terrain…");
    await buildTerrain(data);
    setProgress(100);

    // Update UI
    hideLoader();
    setStatus(`✓ Done — height range ${data.pred_min_m.toFixed(1)}–${data.pred_max_m.toFixed(1)} m`, "ok");

    demMin.textContent = `${data.pred_min_m.toFixed(1)} m`;
    demMax.textContent = `${data.pred_max_m.toFixed(1)} m`;

    emptyState.style.display = "none";
    panels.style.display     = "flex";
    setViewMode("grid");

    statsCard.style.display = "block";
    statsList.innerHTML = `
      <div class="info-row"><span>Height range</span><span class="mono">${data.pred_min_m.toFixed(2)} – ${data.pred_max_m.toFixed(2)} m</span></div>
      <div class="info-row"><span>Mean height</span><span class="mono">${data.pred_mean_m.toFixed(2)} m</span></div>
      <div class="info-row"><span>Grid size</span><span class="mono">${gridW} × ${gridH}</span></div>
      <div class="info-row"><span>Model</span><span class="mono">DepthAnythingV2</span></div>
    `;

  } catch(err) {
    console.error(err);
    hideLoader();
    setStatus(`Error: ${err.message}`, "err");
  } finally {
    runBtn.disabled = false;
  }
});

// ── Canvas drawing ─────────────────────────────────────────────────
function drawB64ToCanvas(canvas, b64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = `data:image/png;base64,${b64}`;
  });
}

// ── WebGL detection ──────────────────────────────────────────────
function isWebGLAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return !!(window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")));
  } catch { return false; }
}

if (!isWebGLAvailable()) {
  vbtn3D.disabled = true;
  vbtn3D.title    = "WebGL not supported in this browser";
  vbtn3D.style.opacity = "0.4";
  vbtn3D.style.cursor  = "not-allowed";
  vbtn3D.textContent   = "3D (Unavailable)";
}

// ── View mode ─────────────────────────────────────────────────────
function setViewMode(mode) {
  if (mode === "3d" && !isWebGLAvailable()) return;
  vbtnGrid.classList.toggle("active", mode === "grid");
  vbtn3D.classList.toggle("active",   mode === "3d");
  panelGrid.style.display = mode === "grid" ? "grid"  : "none";
  canvas3d.style.display  = mode === "3d"   ? "block" : "none";
}

vbtnGrid.addEventListener("click", () => setViewMode("grid"));
vbtn3D.addEventListener("click",   () => setViewMode("3d"));

// ── Three.js 3D ────────────────────────────────────────────────────
let renderer, scene, camera, controls, terrainMesh;
let webglFailed = false;

function initThree() {
  if (renderer || webglFailed) return;
  try {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111a22);
    scene.fog = new THREE.FogExp2(0x111a22, 0.005);

    const w = threeMnt.clientWidth  || 800;
    const h = threeMnt.clientHeight || 600;

    camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 2000);
    camera.position.set(0, 90, 150);

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.shadowMap.enabled = true;
    threeMnt.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance   = 10;
    controls.maxDistance   = 500;

    const hemi = new THREE.HemisphereLight(0x334455, 0x112233, 0.7);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffeedd, 2.0);
    sun.position.set(80, 150, 60);
    sun.castShadow = true;
    scene.add(sun);

    window.addEventListener("resize", () => {
    if (!renderer) return;
    const w2 = threeMnt.clientWidth, h2 = threeMnt.clientHeight;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
  });

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  } catch (err) {
    webglFailed = true;
    console.warn("WebGL init failed:", err);
    // Disable the 3D button gracefully
    vbtn3D.disabled = true;
    vbtn3D.textContent = "3D (Unavailable)";
    vbtn3D.style.opacity = "0.4";
    vbtn3D.style.cursor  = "not-allowed";
    vbtn3D.title = "WebGL not available on this system";
    // Show canvas3d error message
    threeMnt.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;flex-direction:column;gap:12px;font-family:sans-serif">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        <p style="font-size:14px;font-weight:600">WebGL not available</p>
        <p style="font-size:12px;text-align:center;max-width:260px">3D terrain is unavailable on this system. Use the 4-Panel Grid view instead.</p>
      </div>`;
  }
}

async function buildTerrain(data) {
  if (webglFailed || !isWebGLAvailable()) return;  // skip silently — 4-panel grid still works
  initThree();
  if (!renderer) return;
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
  }

  // Use the hillshade canvas as texture
  const tex = new THREE.CanvasTexture(cHs);
  tex.colorSpace = THREE.SRGBColorSpace;

  const seg     = gridW - 1;
  const meshSz  = 200;
  const geo     = new THREE.PlaneGeometry(meshSz, meshSz, seg, seg);
  geo.rotateX(-Math.PI / 2);

  const pos = geo.attributes.position;
  const vs  = (meshSz / Math.max(predMaxM, 1)) * 4.0;  // 4x exaggeration

  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, heightsArr[i] * vs);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85, metalness: 0.05 });
  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

  camera.position.set(0, 100, 160);
  controls.target.set(0, 20, 0);
  controls.update();

  hud3d.textContent = `Range: ${data.pred_min_m.toFixed(1)} – ${data.pred_max_m.toFixed(1)} m  ·  Orbit to explore`;
}

// ── Helpers ───────────────────────────────────────────────────────
function showLoader(msg) {
  loaderMsg.textContent = msg;
  loadingOverlay.style.display = "flex";
}
function hideLoader() {
  loadingOverlay.style.display = "none";
}
function setProgress(pct) {
  statusBar.style.display = "block";
  progressFill.style.width = `${pct}%`;
}
function setStatus(msg, kind) {
  statusBar.style.display = "block";
  statusText.textContent  = msg;
  statusText.style.color  = kind === "err" ? "#dc2626" : kind === "ok" ? "#16a34a" : "";
}

// ── Health check ──────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    deviceBadge.textContent = d.device === "cuda" ? `GPU · ${d.device}` : `CPU`;
    if (!d.checkpoint_exists) {
      deviceBadge.textContent += " · No checkpoint";
      deviceBadge.style.background = "#fef3c7";
      deviceBadge.style.color      = "#92400e";
    }
  } catch {
    deviceBadge.textContent = "API offline";
    deviceBadge.style.background = "#fee2e2";
    deviceBadge.style.color      = "#991b1b";
  }
}

checkHealth();
