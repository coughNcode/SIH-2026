"""
DFC2019 Track 1 dataset — RGB and Reference matched by tile ID.
Prompt2DEM-style: Simulates a Coarse DEM prompt on-the-fly to train
the cross-modal fusion mechanics perfectly without needing perfectly
paired low-res SRTM data on disk.
"""

import os
import glob
import re
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

URBAN_SITE_PREFIX = "JAX"
PATCH_SIZE = 224


def tile_id_from_filename(path: str):
    name = os.path.basename(path)
    match = re.match(r"([A-Za-z]+_\d+)", name)
    return match.group(1) if match else None


def site_code(tile_id: str):
    return tile_id.split("_")[0] if tile_id else "UNKNOWN"


def find_pairs(rgb_dir: str, ref_dir: str, urban_only: bool = True):
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.tif")))
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.tif")))

    agl_by_tile = {}
    for f in ref_files:
        if "AGL" in os.path.basename(f).upper():
            tid = tile_id_from_filename(f)
            if tid:
                agl_by_tile[tid] = f

    if urban_only and URBAN_SITE_PREFIX:
        rgb_files = [f for f in rgb_files if site_code(tile_id_from_filename(f)) == URBAN_SITE_PREFIX]

    pairs = []
    for rgb_path in rgb_files:
        tid = tile_id_from_filename(rgb_path)
        if tid in agl_by_tile:
            pairs.append((rgb_path, agl_by_tile[tid]))

    print(f"Found {len(pairs)} matched RGB/AGL pairs")
    return pairs


def prepare_patches(rgb_dir: str, ref_dir: str, out_dir: str,
                     patch_size: int = PATCH_SIZE, stride: int = 112,
                     urban_only: bool = True):
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("pip install rasterio to read the DFC2019 .tif files") from e

    rgb_out = os.path.join(out_dir, "rgb")
    agl_out = os.path.join(out_dir, "agl")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(agl_out, exist_ok=True)

    pairs = find_pairs(rgb_dir, ref_dir, urban_only=urban_only)
    if not pairs:
        raise FileNotFoundError("No usable pairs found.")

    patch_id = 0
    for rgb_path, agl_path in pairs:
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        with rasterio.open(agl_path) as src:
            agl = src.read(1).astype(np.float32)

        agl = np.nan_to_num(agl, nan=0.0, posinf=0.0, neginf=0.0)
        agl[agl < -1000] = 0.0  # DFC2019 no-data sentinel (~-9999)
        agl = np.maximum(agl, 0.0)

        h, w = agl.shape
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                rgb_patch = rgb[y:y + patch_size, x:x + patch_size]
                agl_patch = agl[y:y + patch_size, x:x + patch_size]

                if rgb_patch.shape[:2] != (patch_size, patch_size):
                    continue

                Image.fromarray(rgb_patch).save(
                    os.path.join(rgb_out, f"patch_{patch_id:05d}.png")
                )
                np.save(os.path.join(agl_out, f"patch_{patch_id:05d}.npy"), agl_patch)
                patch_id += 1

    print(f"Generated {patch_id} urban patches ({patch_size}x{patch_size}) into {out_dir}")


def simulate_coarse_dem(hr_dem: np.ndarray, scale_factor=32) -> np.ndarray:
    """
    Simulates a generic global DEM (like SRTM) by degrading the high-res DEM.
    - Aggressive downsample
    - Gaussian noise
    - Upsample
    - Gaussian Blur to remove sharp cubic artifacts
    """
    h, w = hr_dem.shape
    new_w, new_h = max(2, w // scale_factor), max(2, h // scale_factor)
    
    small = cv2.resize(hr_dem, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ~2m standard deviation noise is typical for 30m SRTM in urban areas
    noise = np.random.normal(0, 2.0, small.shape).astype(np.float32)
    small = np.clip(small + noise, 0, None)
    
    coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    coarse = cv2.GaussianBlur(coarse, (15, 15), 0)
    
    return coarse


class DFC2019UrbanDataset(Dataset):
    def __init__(self, patches_dir: str, augment: bool = True):
        self.rgb_dir = os.path.join(patches_dir, "rgb")
        self.agl_dir = os.path.join(patches_dir, "agl")
        self.filenames = sorted(
            f for f in os.listdir(self.rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.augment = augment

        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        stem = os.path.splitext(name)[0]

        rgb = Image.open(os.path.join(self.rgb_dir, name)).convert("RGB")
        agl = np.load(os.path.join(self.agl_dir, stem + ".npy"))
        agl = np.nan_to_num(agl, nan=0.0, posinf=0.0, neginf=0.0)
        agl = np.maximum(agl, 0.0)

        # Simulate coarse DEM before augmentation so both get augmented identically
        coarse = simulate_coarse_dem(agl, scale_factor=32)

        if self.augment and np.random.rand() > 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            agl = np.fliplr(agl).copy()
            coarse = np.fliplr(coarse).copy()

        rgb_tensor = self.rgb_transform(rgb)
        coarse_tensor = torch.from_numpy(coarse).unsqueeze(0).float()
        agl_tensor = torch.from_numpy(agl).unsqueeze(0).float()

        return rgb_tensor, coarse_tensor, agl_tensor


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ds = DFC2019UrbanDataset("../../data/dfc2019/patches", augment=False)
        rgb, coarse, agl = ds[0]
        print("RGB:", rgb.shape, "Coarse:", coarse.shape, "AGL:", agl.shape)
        print("Coarse max/min:", coarse.max(), coarse.min())
        print("AGL max/min:", agl.max(), agl.min())