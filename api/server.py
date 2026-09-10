"""
DepthWizard — Simple FastAPI Server with QuickHeightNet
Serves the simple 4-panel frontend.
"""
import os, sys, io, base64, json, logging
from pathlib import Path

ROOT    = Path(__file__).parent.parent   # project root (DepthWizard_Training/)
SRC_DIR = ROOT / "src" / "dsm_model"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("depthwizard")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = ROOT / "outputs" / "checkpoints" / "quicktrain_best.pt"
IMG_SIZE  = 518

# ── Model definition (must match quick_train.py) ──────────────────
class QuickHeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation
        self.backbone = DepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.shift     = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            out = self.backbone(x)
            rel_depth = out.predicted_depth
        rel_depth = F.interpolate(
            rel_depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        )
        scale = torch.exp(self.log_scale)
        rel_d_norm = (rel_depth - rel_depth.amin(dim=[2,3], keepdim=True)) / \
                     (rel_depth.amax(dim=[2,3], keepdim=True) - rel_depth.amin(dim=[2,3], keepdim=True) + 1e-6)
        coarse = scale * rel_d_norm + self.shift
        return self.refine(coarse)

# ── Load model ────────────────────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        log.info(f"Loading QuickHeightNet on {DEVICE}...")
        m = QuickHeightNet()
        if CKPT_PATH.exists():
            state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
            m.load_state_dict(state)
            log.info("✓ Loaded fine-tuned weights from quicktrain_best.pt")
        else:
            log.warning("No checkpoint found — using untrained model (run quick_train.py first)")
        m.to(DEVICE).eval()
        _model = m
    return _model

# ── Preprocessing ─────────────────────────────────────────────────
TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def run_inference(pil_img):
    model = get_model()
    inp   = TF(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(inp)  # (1, 1, H, W)
    return pred[0, 0].cpu().numpy().astype(np.float32)

# ── Visualization helpers ─────────────────────────────────────────
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / max(mx - mn, 1e-6)

def apply_cmap(arr_norm, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(arr_norm, 0, 1))
    return (rgba[:, :, :3] * 255).astype(np.uint8)

def compute_hillshade(dem, az=315.0, alt=45.0):
    import math
    az_rad  = math.radians(360.0 - az + 90.0)
    alt_rad = math.radians(alt)
    zy = np.gradient(dem, axis=0)
    zx = np.gradient(dem, axis=1)
    slope  = np.arctan(np.sqrt(zx**2 + zy**2))
    aspect = np.arctan2(zy, -zx)
    hs = (np.cos(alt_rad) * np.cos(slope) +
          np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1)

def encode_png(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, "PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()

def build_panels(pil_img, dem):
    H, W = dem.shape

    # Panel 1: RGB (resized)
    rgb_arr   = np.array(pil_img.convert("RGB").resize((W, H), Image.LANCZOS))
    rgb_b64   = encode_png(rgb_arr)

    dem_norm  = normalize(dem)

    # Panel 2: DEM prediction (turbo colormap)
    dem_col   = apply_cmap(dem_norm, "turbo")
    dem_b64   = encode_png(dem_col)

    # Panel 3: Hillshade (grey)
    hs        = compute_hillshade(dem * 2.0)
    hs_col    = apply_cmap(hs, "gray")
    hs_b64    = encode_png(hs_col)

    # Panel 4: Surface normals (for visual richness, like the paper's gradient row)
    dy = np.gradient(dem, axis=0)
    dx = np.gradient(dem, axis=1)
    # Convert gradients to normal-map visualization
    norms = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    nlen  = np.linalg.norm(norms, axis=-1, keepdims=True)
    norms = norms / (nlen + 1e-6)
    normal_vis = ((norms + 1.0) / 2.0 * 255).astype(np.uint8)
    norm_b64   = encode_png(normal_vis)

    return rgb_b64, dem_b64, hs_b64, norm_b64

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="DepthWizard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PUBLIC_DIR = ROOT / "backend" / "public"
if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/")
def index():
    p = PUBLIC_DIR / "index.html"
    return FileResponse(str(p)) if p.exists() else JSONResponse({"status": "ok"})

@app.get("/health")
def health():
    loaded = _model is not None
    ckpt_exists = CKPT_PATH.exists()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": loaded,
        "checkpoint_exists": ckpt_exists,
        "message": "Ready" if ckpt_exists else "Run quick_train.py first for fine-tuned results"
    }

@app.post("/api/infer-form")
async def infer_form(image: UploadFile = File(...)):
    img_bytes = await image.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    dem = run_inference(pil)

    rgb_b64, dem_b64, hs_b64, norm_b64 = build_panels(pil, dem)

    p_min = float(dem.min())
    p_max = float(dem.max())

    return {
        "heights":      dem.flatten().tolist(),
        "rgb_b64":      rgb_b64,
        "dem_b64":      dem_b64,
        "hillshade_b64": hs_b64,
        "normals_b64":  norm_b64,
        "width":  int(dem.shape[1]),
        "height": int(dem.shape[0]),
        "pred_min_m":  p_min,
        "pred_max_m":  p_max,
        "pred_mean_m": float(dem.mean()),
    }

@app.get("/metrics")
def metrics():
    p = ROOT / "outputs" / "eval_results.json"
    if not p.exists():
        return {"metrics": {"RMSE_m": 0.0, "MAE_m": 0.0, "delta1": 0.0, "RMSE_std_m": 0.0}}
    with open(p) as f:
        return json.load(f)

if __name__ == "__main__":
    get_model()
    log.info(f"Frontend: {PUBLIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
