"""
Data Analysis — run this FIRST, before dataset.py or train.py.

Works with RGB and Reference files kept in SEPARATE folders, matched
by a shared tile ID (e.g. "JAX_004") extracted from each filename
regardless of suffix.

Usage:
    python analyze_data.py --rgb-dir ../../data/dfc2019/raw/rgb \
                            --ref-dir ../../data/dfc2019/raw/reference
"""

import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
from PIL import Image

URBAN_SITE_PREFIX = "JAX"  # Jacksonville = our urban subset; Omaha ("OMA") is excluded


def tile_id_from_filename(path: str):
    """Extracts a shared tile ID like 'JAX_004' from any filename variant."""
    name = os.path.basename(path)
    match = re.match(r"([A-Za-z]+_\d+)", name)
    return match.group(1) if match else None


def site_code(tile_id: str):
    return tile_id.split("_")[0] if tile_id else "UNKNOWN"


def analyze(rgb_dir: str, ref_dir: str):
    print(f"RGB folder:       {rgb_dir}")
    print(f"Reference folder: {ref_dir}\n")

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.tif")))
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.tif")))

    if not rgb_files:
        print(f"!! No .tif files found in {rgb_dir} — did you copy your RGB download here?")
        return
    if not ref_files:
        print(f"!! No .tif files found in {ref_dir} — did you copy your Reference download here?")
        return

    print(f"Found {len(rgb_files)} RGB files, {len(ref_files)} reference files.\n")

    agl_files = [f for f in ref_files if "AGL" in os.path.basename(f).upper()]
    cls_files = [f for f in ref_files if "CLS" in os.path.basename(f).upper()]
    other_ref = [f for f in ref_files if f not in agl_files and f not in cls_files]

    print(f"  Reference breakdown: {len(agl_files)} AGL (height) files, "
          f"{len(cls_files)} CLS (semantic) files, {len(other_ref)} unrecognized")
    if other_ref:
        print(f"  Unrecognized example: {os.path.basename(other_ref[0])}")
    print()

    rgb_tile_ids = [tile_id_from_filename(f) for f in rgb_files]
    site_counts = Counter(site_code(t) for t in rgb_tile_ids if t)
    print("Site breakdown (from RGB filenames):")
    for site, count in site_counts.items():
        tag = "  <- URBAN, we're using this" if site == URBAN_SITE_PREFIX else ""
        print(f"   {site}: {count} tiles{tag}")
    print()

    agl_by_tile = {}
    for f in agl_files:
        tid = tile_id_from_filename(f)
        if tid:
            agl_by_tile[tid] = f

    urban_rgb = [f for f in rgb_files if site_code(tile_id_from_filename(f)) == URBAN_SITE_PREFIX]
    print(f"Urban ({URBAN_SITE_PREFIX}) RGB tiles: {len(urban_rgb)}")

    paired = []
    missing = []
    for rgb_path in urban_rgb:
        tid = tile_id_from_filename(rgb_path)
        if tid in agl_by_tile:
            paired.append((rgb_path, agl_by_tile[tid]))
        else:
            missing.append(rgb_path)

    print(f"Urban tiles with a matching AGL file: {len(paired)}")
    if missing:
        print(f"  WARNING: {len(missing)} urban RGB tiles have no matching AGL file.")
        print(f"  Example: {os.path.basename(missing[0])}")
    print()

    if not paired:
        print("!! No usable urban RGB/AGL pairs found. Stopping here.")
        return

    try:
        import rasterio
    except ImportError:
        print("Install rasterio to inspect AGL value ranges: pip install rasterio")
        return

    print(f"Inspecting AGL value ranges across {min(10, len(paired))} sample tiles...")
    all_mins, all_maxs, sentinel_counts = [], [], []
    for rgb_path, agl_path in paired[:10]:
        with rasterio.open(agl_path) as src:
            agl = src.read(1).astype(np.float32)
        n_sentinel = int(np.sum(agl < -1000))
        valid = agl[agl > -1000]
        if valid.size > 0:
            all_mins.append(float(valid.min()))
            all_maxs.append(float(valid.max()))
        sentinel_counts.append(n_sentinel)

    if all_mins:
        print(f"   Valid AGL range: {min(all_mins):.2f}m to {max(all_maxs):.2f}m")
    print(f"   No-data sentinel pixels in samples: {sum(sentinel_counts)}")
    print()

    sample_img = Image.open(paired[0][0])
    print(f"Sample RGB tile size: {sample_img.size} (width x height)")
    print()

    print("=" * 60)
    print(f"SUMMARY: {len(paired)} usable urban (JAX) tile pairs ready for training.")
    print("Next step: python dataset.py prepare")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", default="../../data/dfc2019/raw/rgb")
    parser.add_argument("--ref-dir", default="../../data/dfc2019/raw/reference")
    args = parser.parse_args()
    analyze(args.rgb_dir, args.ref_dir)