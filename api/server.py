"""
DepthWizard — Integrated Server (FastAPI + Static)
===================================================
Single server on port 8000 that:
  1. Serves the frontend from ./public/
  2. Runs HeightNet GPU inference
  3. Returns raw float32 heightmap as JSON (no 16-bit PNG tricks)

Endpoints:
  GET  /              → index.html
  POST /api/infer     → { heights: float[], width, height, pred_min_m, pred_max_m, pred_mean_m }
  GET  /api/metrics   → eval_results.json
  GET  /api/health    → { status, device }
  GET  /static/*      → public/ static files

Start:
  cd c:/Users/ronit/OneDrive/Desktop/DepthWizard_Training
  src\\dsm_model\\.venv\\Scripts\\python.exe api\\server.py
  → open http://localhost:8000
"""

import os
import sys
import base64
import json
import io
import logging
from pathlib import Path
from typing import Optional, List

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR   = REPO_ROOT / "src" / "dsm_model"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from model import Prompt2DEM

# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("depthwizard")

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
CKPT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "prompt2dem_urban_best.pt"
if not CKPT_PATH.exists():
    CKPT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "prompt2dem_urban_final.pt"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[Prompt2DEM] = None

def get_model() -> Prompt2DEM:
    global _model
    if _model is None:
        log.info(f"Loading Prompt2DEM checkpoint: {CKPT_PATH} → {DEVICE}")
        m = Prompt2DEM(freeze_backbone=False)
        try:
            state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
            m.load_state_dict(state)
        except Exception as e:
            log.warning(f"Could not load Prompt2DEM weights (train first). Error: {e}")
        m.to(DEVICE).eval()
        _model = m
        log.info("Prompt2DEM ready.")
    return _model

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_inference(pil_img: Image.Image) -> np.ndarray:
    """RGB PIL Image -> float32 ndarray (224, 224) in meters."""
    model = get_model()
    inp   = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    
    # Prompt2DEM requires a coarse DEM prompt. If the user doesn't upload one,
    # we provide a zero-tensor (uninformative prior), forcing the model to rely 
    # entirely on the RGB imagery.
    coarse_dem = torch.zeros((1, 1, 224, 224), device=DEVICE, dtype=torch.float32)
    
    with torch.no_grad():
        pred_and_var = model(inp, coarse_dem)
    
    # Channel 0 is DEM, Channel 1 is LogVariance (Uncertainty)
    pred_m = pred_and_var[0, 0].cpu().numpy().astype(np.float32)  # (224, 224)
    pred_m = np.maximum(pred_m, 0.0)
    return pred_m


def compute_hillshade(dem: np.ndarray, azimuth_deg=315.0, altitude_deg=45.0) -> np.ndarray:
    """
    Compute hillshade from a DEM.
    Returns normalized [0,1] float array.
    """
    az  = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)

    # Sobel-style gradients
    zy = np.gradient(dem, axis=0)
    zx = np.gradient(dem, axis=1)

    slope   = np.arctan(np.sqrt(zx**2 + zy**2))
    aspect  = np.arctan2(zy, -zx)

    hs = (np.cos(alt) * np.cos(slope) +
          np.sin(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    return hs


def apply_green_colormap(dem_norm: np.ndarray, hillshade: np.ndarray) -> np.ndarray:
    """
    IM2ELEVATION-style green colormap:
    - Low areas: deep forest green (#0d3320)
    - Mid areas: olive/medium green (#4a8c3f)
    - High areas: bright lime/yellow-green (#c2f03c)
    Multiplied by hillshade for 3D relief effect.
    Returns (H, W, 3) uint8.
    """
    # Green colormap keypoints (dark→bright green)
    keys = np.array([
        [0.051, 0.200, 0.125],   # 0.0 – deep forest green
        [0.102, 0.392, 0.196],   # 0.2 – dark green
        [0.184, 0.549, 0.216],   # 0.4 – medium green
        [0.392, 0.706, 0.196],   # 0.6 – light green
        [0.647, 0.875, 0.196],   # 0.8 – yellow-green
        [0.824, 0.969, 0.200],   # 1.0 – bright lime
    ], dtype=np.float32)

    N = 512
    t_keys = np.linspace(0, 1, len(keys))
    t_lut  = np.linspace(0, 1, N)
    lut_r  = np.interp(t_lut, t_keys, keys[:, 0])
    lut_g  = np.interp(t_lut, t_keys, keys[:, 1])
    lut_b  = np.interp(t_lut, t_keys, keys[:, 2])

    idx  = (np.clip(dem_norm, 0, 1) * (N - 1)).astype(np.int32)
    rgb  = np.stack([lut_r[idx], lut_g[idx], lut_b[idx]], axis=-1)

    # Blend: apply hillshade (multiply, keep some ambient)
    ambient = 0.35
    shaded  = rgb * (ambient + (1.0 - ambient) * hillshade[:, :, None])
    return (np.clip(shaded, 0, 1) * 255).astype(np.uint8)


def encode_png(arr: np.ndarray) -> str:
    """numpy uint8 array → base64 PNG string (no data: prefix)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def resize_for_display(pil_img: Image.Image, size=224) -> np.ndarray:
    return np.array(pil_img.convert("RGB").resize((size, size), Image.LANCZOS))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="DepthWizard", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from backend/public/
PUBLIC_DIR = REPO_ROOT / "backend" / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


@app.get("/")
def index():
    p = PUBLIC_DIR / "index.html"
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse({"status": "DepthWizard API running — no frontend found at backend/public/"})


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": _model is not None}


@app.get("/metrics")
def metrics():
    p = REPO_ROOT / "outputs" / "eval_results.json"
    if not p.exists():
        raise HTTPException(404, "Run evaluate.py first")
    with open(p) as f:
        return json.load(f)


class InferRequest(BaseModel):
    image_b64: str

class InferResponse(BaseModel):
    # Raw float heights as list (224*224 values), meters
    heights:       List[float]
    # Visualization PNGs (base64, no data: prefix — JS adds it)
    rgb_b64:       str   # resized input RGB
    hillshade_b64: str   # IM2ELEVATION green hillshade
    width:         int
    height:        int
    pred_min_m:    float
    pred_max_m:    float
    pred_mean_m:   float


@app.post("/api/infer", response_model=InferResponse)
async def infer_json(req: InferRequest):
    """JSON body: { image_b64: str } → full inference result."""
    try:
        img_bytes = base64.b64decode(req.image_b64)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred_m = run_inference(pil)

    # Resize input for side-by-side display
    rgb_display = resize_for_display(pil)
    rgb_b64     = encode_png(rgb_display)

    # Normalize DEM for colormap
    p_min = float(pred_m.min())
    p_max = float(pred_m.max())
    dem_norm = (pred_m - p_min) / max(p_max - p_min, 1e-4)

    # Hillshade + green colormap
    # Scale meters → pixel units for gradient (assume ~0.5m/pixel at JAX resolution)
    hs  = compute_hillshade(pred_m * 2.0)
    hs_colored = apply_green_colormap(dem_norm, hs)
    hs_b64 = encode_png(hs_colored)

    return InferResponse(
        heights       = pred_m.flatten().tolist(),
        rgb_b64       = rgb_b64,
        hillshade_b64 = hs_b64,
        width         = 224,
        height        = 224,
        pred_min_m    = p_min,
        pred_max_m    = p_max,
        pred_mean_m   = float(pred_m.mean()),
    )


@app.post("/api/infer-form")
async def infer_form(image: UploadFile = File(...)):
    """Multipart form upload → same response as /api/infer."""
    img_bytes = await image.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred_m   = run_inference(pil)
    rgb_disp = resize_for_display(pil)
    rgb_b64  = encode_png(rgb_disp)

    p_min    = float(pred_m.min())
    p_max    = float(pred_m.max())
    dem_norm = (pred_m - p_min) / max(p_max - p_min, 1e-4)
    hs       = compute_hillshade(pred_m * 2.0)
    hs_col   = apply_green_colormap(dem_norm, hs)
    hs_b64   = encode_png(hs_col)

    return {
        "heights":       pred_m.flatten().tolist(),
        "rgb_b64":       rgb_b64,
        "hillshade_b64": hs_b64,
        "width":  224, "height": 224,
        "pred_min_m":  p_min,
        "pred_max_m":  p_max,
        "pred_mean_m": float(pred_m.mean()),
    }


if __name__ == "__main__":
    get_model()  # Pre-load before accepting requests
    log.info(f"Serving frontend from: {PUBLIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
