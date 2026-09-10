"""
Evaluates the Prompt2DEM network.
Generates metrics (RMSE, MAE, δ1, Slope MAE) and 5-panel visual comparisons.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from model import Prompt2DEM
from dataset import DFC2019UrbanDataset, simulate_coarse_dem

import matplotlib.pyplot as plt


def compute_slope(dem: np.ndarray) -> np.ndarray:
    """Compute slope map (gradients) for a given DEM."""
    zy, zx = np.gradient(dem)
    slope = np.sqrt(zx**2 + zy**2)
    return slope


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We will use the best checkpoint if it exists, otherwise final
    ckpt_path = "../../outputs/checkpoints/prompt2dem_urban_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "../../outputs/checkpoints/prompt2dem_urban_final.pt"
        if not os.path.exists(ckpt_path):
            print("No checkpoint found. Skipping evaluation.")
            return

    print(f"Loading {ckpt_path} to {device}...")
    model = Prompt2DEM(freeze_backbone=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    ds = DFC2019UrbanDataset("../../data/dfc2019/patches", augment=False)
    
    # Evaluate on the first 300 patches for speed
    n_eval = min(300, len(ds))
    loader = DataLoader(Subset(ds, range(n_eval)), batch_size=1, shuffle=False)

    out_dir = "../../outputs/eval_images"
    os.makedirs(out_dir, exist_ok=True)

    errors = []
    slope_errors = []
    metrics = {"RMSE_m": 0, "MAE_m": 0, "delta1": 0, "Slope_MAE": 0}

    print(f"Evaluating {n_eval} patches...")

    with torch.no_grad():
        for i, (rgb, coarse, gt) in enumerate(loader):
            rgb, coarse = rgb.to(device), coarse.to(device)
            
            # Forward pass
            pred_and_var = model(rgb, coarse)
            
            # (B, 1, 224, 224)
            pred_dem = pred_and_var[0, 0].cpu().numpy()
            pred_unc = pred_and_var[0, 1].cpu().numpy()  # log-variance
            pred_unc_std = np.exp(pred_unc / 2.0)        # std deviation
            
            gt_dem = gt[0, 0].cpu().numpy()
            coarse_dem = coarse[0, 0].cpu().numpy()

            # Fix arbitrary negatives
            pred_dem = np.maximum(pred_dem, 0.0)

            # Metrics
            diff = pred_dem - gt_dem
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.mean(diff**2))
            
            # Slope metric
            pred_slope = compute_slope(pred_dem)
            gt_slope = compute_slope(gt_dem)
            slope_mae = np.mean(np.abs(pred_slope - gt_slope))

            # delta 1
            thresh = np.maximum((pred_dem / (gt_dem + 1e-3)), (gt_dem / (pred_dem + 1e-3)))
            d1 = (thresh < 1.25).mean()

            errors.append(rmse)
            slope_errors.append(slope_mae)
            
            metrics["RMSE_m"] += rmse
            metrics["MAE_m"] += mae
            metrics["delta1"] += d1
            metrics["Slope_MAE"] += slope_mae

            # Save visuals for the first 6
            if i < 6:
                # Denormalize RGB
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                rgb_img = rgb[0].cpu().numpy() * std + mean
                rgb_img = np.clip(rgb_img, 0, 1).transpose(1, 2, 0)
                
                # 5-panel plot
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                axes[0].imshow(rgb_img)
                axes[0].set_title("RGB Input")
                axes[0].axis('off')
                
                vmin, vmax = 0, max(gt_dem.max(), pred_dem.max())
                
                cax1 = axes[1].imshow(coarse_dem, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title("Coarse DEM Prompt")
                axes[1].axis('off')
                
                cax2 = axes[2].imshow(pred_dem, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title(f"Predicted DEM (RMSE: {rmse:.2f}m)")
                axes[2].axis('off')
                
                cax3 = axes[3].imshow(gt_dem, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[3].set_title("Ground Truth LiDAR")
                axes[3].axis('off')
                
                cax4 = axes[4].imshow(pred_unc_std, cmap='magma')
                axes[4].set_title("Predicted Uncertainty (Std)")
                axes[4].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"tile_{i:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

    metrics["RMSE_m"] /= n_eval
    metrics["MAE_m"] /= n_eval
    metrics["delta1"] /= n_eval
    metrics["Slope_MAE"] /= n_eval
    metrics["RMSE_std_m"] = float(np.std(errors))

    print("\nEvaluation Results:")
    print(f"  RMSE:      {metrics['RMSE_m']:.4f} m (± {metrics['RMSE_std_m']:.4f})")
    print(f"  MAE:       {metrics['MAE_m']:.4f} m")
    print(f"  Slope MAE: {metrics['Slope_MAE']:.4f}")
    print(f"  δ1 Acc:    {metrics['delta1']:.4%}")

    with open("../../outputs/eval_results.json", "w") as f:
        json.dump({"metrics": metrics, "n_samples": n_eval}, f, indent=2)


if __name__ == "__main__":
    evaluate()
