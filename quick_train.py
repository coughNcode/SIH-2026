"""
DepthWizard — Quick Hackathon Training
=======================================
Uses Depth-Anything-V2-Small as frozen backbone + tiny conv decoder.
Trains on raw DFC2019 TIF pairs (RGB + AGL) directly.
No patching needed — just resizes to 518x518.
Target: ~15 minutes on RTX 4060, good enough demo output.
"""

import os, sys, glob, random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
RGB_DIR  = ROOT / "data" / "dfc2019" / "raw" / "rgb"
AGL_DIR  = ROOT / "data" / "dfc2019" / "raw" / "reference"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_OUT = CKPT_DIR / "quicktrain_best.pt"

# ── Config ─────────────────────────────────────────────────────────
NUM_TRAIN = 120      # number of training images
NUM_VAL   = 20       # number of val images
EPOCHS    = 15
BATCH     = 4
LR        = 2e-4
IMG_SIZE  = 518      # DepthAnythingV2 native size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────
class DFCDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs   = pairs
        self.augment = augment
        self.img_tf  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, agl_path = self.pairs[idx]
        try:
            # Load RGB
            from PIL import Image as PILImage
            img = PILImage.open(rgb_path).convert("RGB")

            # Load AGL (height ground truth)
            import tifffile
            agl = tifffile.imread(str(agl_path)).astype(np.float32)
            # Replace nodata values
            agl = np.where(np.isnan(agl) | (agl < 0), 0.0, agl)
            agl = np.clip(agl, 0, 100)  # Clip to 0-100m AGL

            # Resize AGL to IMG_SIZE
            agl_pil = Image.fromarray(agl)
            agl_pil = agl_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            agl_t   = torch.tensor(np.array(agl_pil), dtype=torch.float32).unsqueeze(0)

            img_t = self.img_tf(img)

            if self.augment and random.random() > 0.5:
                img_t = torch.flip(img_t, [-1])
                agl_t = torch.flip(agl_t, [-1])

            return img_t, agl_t

        except Exception as e:
            # Return zeros if file is corrupt
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE)


def get_pairs(rgb_dir, agl_dir, limit=None):
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.tif")))
    pairs = []
    for rgb_path in rgb_files:
        stem = Path(rgb_path).stem.replace("_RGB", "")
        agl_path = agl_dir / f"{stem}_AGL.tif"
        if agl_path.exists():
            pairs.append((rgb_path, str(agl_path)))
    if limit:
        pairs = pairs[:limit]
    return pairs


# ── Model ──────────────────────────────────────────────────────────
class QuickHeightNet(nn.Module):
    """
    Frozen DepthAnythingV2 backbone + lightweight conv decoder.
    Outputs single-channel height map (meters).
    """
    def __init__(self):
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation

        print("Loading DepthAnything-V2-Small backbone...")
        self.backbone = DepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Lightweight decoder: takes the backbone's predicted_depth (H x W) + backbone features
        # We use the backbone's direct depth output as a relative depth prior
        # Then learn a scale+shift + refinement CNN
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.ReLU(inplace=True),  # heights must be >= 0
        )
        # Learned scale/shift to convert relative depth → absolute meters
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.shift     = nn.Parameter(torch.tensor(5.0))   # mean urban AGL ~5m

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            out = self.backbone(x)
            rel_depth = out.predicted_depth  # (B, H', W') — relative depth

        # Resize to input size
        rel_depth = F.interpolate(
            rel_depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        )  # (B, 1, H, W)

        # Scale + shift: convert relative inverse depth → absolute AGL
        scale = torch.exp(self.log_scale)
        rel_d_norm = (rel_depth - rel_depth.amin(dim=[2,3], keepdim=True)) / \
                     (rel_depth.amax(dim=[2,3], keepdim=True) - rel_depth.amin(dim=[2,3], keepdim=True) + 1e-6)

        coarse = scale * rel_d_norm + self.shift

        # Refinement CNN
        pred = self.refine(coarse)
        return pred  # (B, 1, H, W) in meters


# ── Loss ───────────────────────────────────────────────────────────
def loss_fn(pred, gt):
    mask = gt > 0
    if mask.sum() < 10:
        return pred.mean() * 0.0

    # L1 loss on valid pixels
    l1 = F.l1_loss(pred[mask], gt[mask])

    # Scale-invariant gradient loss
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_g = gt[:, :, 1:, :]   - gt[:, :, :-1, :]
    dx_g = gt[:, :, :, 1:]   - gt[:, :, :, :-1]
    grad_loss = F.l1_loss(dy_p, dy_g) + F.l1_loss(dx_p, dx_g)

    return l1 + 0.5 * grad_loss


# ── Training ───────────────────────────────────────────────────────
def train():
    # Check for tifffile
    try:
        import tifffile
    except ImportError:
        os.system(f"{sys.executable} -m pip install tifffile -q")
        import tifffile

    all_pairs = get_pairs(RGB_DIR, AGL_DIR)
    print(f"Found {len(all_pairs)} RGB-AGL pairs")

    random.shuffle(all_pairs)
    train_pairs = all_pairs[:NUM_TRAIN]
    val_pairs   = all_pairs[NUM_TRAIN:NUM_TRAIN + NUM_VAL]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = DFCDataset(train_pairs, augment=True)
    val_ds   = DFCDataset(val_pairs,   augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    model = QuickHeightNet().to(DEVICE)

    # Only train the refinement head + scale/shift
    params = list(model.refine.parameters()) + [model.log_scale, model.shift]
    opt    = torch.optim.AdamW(params, lr=LR)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for i, (imgs, agls) in enumerate(train_dl):
            imgs, agls = imgs.to(DEVICE), agls.to(DEVICE)
            opt.zero_grad()
            pred = model(imgs)
            loss = loss_fn(pred, agls)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            train_loss += loss.item()
            if (i+1) % 5 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} step {i+1}/{len(train_dl)} loss={loss.item():.4f}", flush=True)

        sched.step()

        # Val
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        n = 0
        with torch.no_grad():
            for imgs, agls in val_dl:
                imgs, agls = imgs.to(DEVICE), agls.to(DEVICE)
                pred = model(imgs)
                loss = loss_fn(pred, agls)
                val_loss += loss.item()
                mask = agls > 0
                if mask.sum() > 0:
                    val_mae += F.l1_loss(pred[mask], agls[mask]).item()
                    n += 1

        avg_train = train_loss / max(len(train_dl), 1)
        avg_val   = val_loss / max(len(val_dl), 1)
        avg_mae   = val_mae / max(n, 1)

        print(f"Epoch {epoch}/{EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f} | MAE={avg_mae:.2f}m | scale={model.log_scale.item():.3f} shift={model.shift.item():.2f}", flush=True)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), str(CKPT_OUT))
            print(f"  [BEST] Saved best checkpoint -> {CKPT_OUT}", flush=True)

    print(f"\n[DONE] Training done. Best val loss: {best_val:.4f}")
    print(f"[OK] Checkpoint: {CKPT_OUT}")


if __name__ == "__main__":
    train()
