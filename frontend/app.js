import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const API_URL = "/api/process";
const HEIGHT_SCALE = 26; // visual exaggeration for a readable terrain

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewThumb = document.getElementById("previewThumb");
const runBtn = document.getElementById("runBtn");
const statusText = document.getElementById("statusText");
const emptyState = document.getElementById("emptyState");
const metaHint = document.getElementById("metaHint");
const inspectPanel = document.getElementById("inspect-panel");
const canvasHolder = document.getElementById("canvas-holder");

let selectedFile = null;
let heightData = null; // Float32Array, normalized 0..1, GRID x GRID
let gridSize = 0;

// ---------------------------------------------------------------
// Upload UX
// ---------------------------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        setStatus("Please choose an image file.", "err");
        return;
    }
    selectedFile = file;
    previewThumb.src = URL.createObjectURL(file);
    previewThumb.style.display = "block";
    runBtn.disabled = false;
    setStatus("");
}

runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    runBtn.disabled = true;
    setStatus("Processing on backend…");

    try {
        const form = new FormData();
        form.append("image", selectedFile);
        const res = await fetch(API_URL, { method: "POST", body: form });
        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        const data = await res.json();

        setStatus("Building 3D terrain…");
        await buildScene(data);
        setStatus("Done — orbit, zoom, or click the terrain.", "ok");
        metaHint.textContent = `${data.width}×${data.height} grid`;
    } catch (err) {
        console.error(err);
        setStatus(`Error: ${err.message}`, "err");
    } finally {
        runBtn.disabled = false;
    }
});

function setStatus(msg, kind) {
    statusText.textContent = msg;
    statusText.className = "status" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------
// Three.js scene (created once, reused across uploads)
// ---------------------------------------------------------------
let renderer, scene, camera, controls, terrainMesh, raycaster;

function initThree() {
    if (renderer) return;

    emptyState.style.display = "none";

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe4ebf5);

    const w = canvasHolder.clientWidth;
    const h = canvasHolder.clientHeight;

    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000);
    camera.position.set(0, 140, 200);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    canvasHolder.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.1);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(120, 220, 80);
    scene.add(dir);

    raycaster = new THREE.Raycaster();
    renderer.domElement.addEventListener("click", onTerrainClick);
    window.addEventListener("resize", onResize);

    animate();
}

function onResize() {
    const w = canvasHolder.clientWidth;
    const h = canvasHolder.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// ---------------------------------------------------------------
// Build the terrain mesh from the backend's heightmap + source image
// ---------------------------------------------------------------
async function buildScene(data) {
    initThree();

    const [colorImg, heightImg] = await Promise.all([
        loadImage(data.image),
        loadImage(data.heightmap),
    ]);

    gridSize = data.width;
    heightData = sampleHeightmap(heightImg, gridSize);

    if (terrainMesh) {
        scene.remove(terrainMesh);
        terrainMesh.geometry.dispose();
        terrainMesh.material.dispose();
    }

    const size = 180;
    const geometry = new THREE.PlaneGeometry(size, size, gridSize - 1, gridSize - 1);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const h = heightData[i];
        pos.setY(i, h * HEIGHT_SCALE);
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();

    const texture = new THREE.Texture(colorImg);
    texture.needsUpdate = true;
    texture.colorSpace = THREE.SRGBColorSpace;

    const material = new THREE.MeshStandardMaterial({
        map: texture,
        roughness: 0.9,
        metalness: 0.0,
        side: THREE.DoubleSide,
    });

    terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.userData.gridSize = gridSize;
    terrainMesh.userData.planeSize = size;
    scene.add(terrainMesh);

    camera.position.set(0, 140, 200);
    controls.target.set(0, 10, 0);
    controls.update();
}

function loadImage(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = dataUrl;
    });
}

// Draw the heightmap PNG to an offscreen canvas and read back a
// normalized [0,1] float per grid cell.
function sampleHeightmap(img, grid) {
    const c = document.createElement("canvas");
    c.width = grid;
    c.height = grid;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, grid, grid);
    const px = ctx.getImageData(0, 0, grid, grid).data;

    const out = new Float32Array(grid * grid);
    for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
            const srcIdx = (y * grid + x) * 4;
            // PlaneGeometry rows run top-to-bottom same as image rows here
            const destIdx = y * grid + x;
            out[destIdx] = px[srcIdx] / 255;
        }
    }
    return out;
}

// ---------------------------------------------------------------
// Click-to-inspect: report the normalized predicted height at a
// clicked point. This is the hook where real ground-truth comparison
// (DFC2019 AGL labels) plugs in once wired to real model output.
// ---------------------------------------------------------------
function onTerrainClick(event) {
    if (!terrainMesh) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(terrainMesh);
    if (!hits.length) {
        inspectPanel.style.display = "none";
        return;
    }
    const p = hits[0].point;
    const normalizedHeight = p.y / HEIGHT_SCALE;

    inspectPanel.style.display = "block";
    inspectPanel.innerHTML =
        `<b>Predicted height (normalized):</b> ${normalizedHeight.toFixed(3)}<br>` +
        `<b>Position:</b> x=${p.x.toFixed(1)}, z=${p.z.toFixed(1)}<br>` +
        `<span style="color:#6b7280">Ground-truth comparison plugs in here once wired to DFC2019 labels.</span>`;
}