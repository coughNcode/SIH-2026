"""
Prompt2DEM Training Curriculum.
Fine-tunes Prompt2DEM on urban DFC2019 patches using a simulated coarse DEM.

Two-phase schedule:
  Phase 1 (--warmup-epochs): Vision Backbone (DepthAnythingV2) is frozen. 
                             Trains the DEM Encoder, Fusion, and Decoder.
  Phase 2 (remaining epochs): Vision Backbone unfrozen, full end-to-end fine-tuning.

Usage:
    python train.py --epochs 30 --warmup-epochs 5 --batch-size 8
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from model import Prompt2DEM, DEFAULT_CHECKPOINT
from losses import Prompt2DEMLoss
from dataset import DFC2019UrbanDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available, falling back to CPU/MPS. "
          "This model is a transformer backbone and will be very slow without a GPU.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    
    total_loss = 0.0
    total_nll = 0.0
    total_normal = 0.0
    total_unc = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for rgb, coarse, agl in loader:
            rgb, coarse, agl = rgb.to(device), coarse.to(device), agl.to(device)

            if train:
                optimizer.zero_grad()

            if device.type == "cuda":
                with autocast():
                    pred_and_var = model(rgb, coarse)
                    loss, parts = criterion(pred_and_var, agl)
            else:
                pred_and_var = model(rgb, coarse)
                loss, parts = criterion(pred_and_var, agl)

            if train:
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = rgb.size(0)
            total_loss += loss.item() * bs
            total_nll += parts["nll_loss"] * bs
            total_normal += parts["normal_loss"] * bs
            total_unc += parts["mean_uncertainty"] * bs
            n_samples += bs

    n = max(n_samples, 1)
    return {
        "loss": total_loss / n,
        "nll": total_nll / n,
        "normal": total_normal / n,
        "unc": total_unc / n
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches-dir", default="../../data/dfc2019/patches")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)  # Reduced for dual-encoder
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--checkpoint-dir", default="../../outputs/checkpoints")
    parser.add_argument("--model-checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = get_device()
    print(f"Training on device: {device}")

    full_dataset = DFC2019UrbanDataset(args.patches_dir, augment=True)
    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No patches found in {args.patches_dir}. Run analyze_data.py, "
            f"then dataset.py prepare, before training."
        )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"Train patches: {train_size}, Val patches: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loading Prompt2DEM architecture (Backbone: {args.model_checkpoint})")
    model = Prompt2DEM(checkpoint=args.model_checkpoint, freeze_backbone=True).to(device)
    criterion = Prompt2DEMLoss(gradient_weight=1.0, normal_weight=1.0)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            print(f"\n>>> Phase 2: Unfreezing Vision Backbone for end-to-end refinement at lr={args.lr_finetune}\n")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
            epochs_without_improvement = 0
        elif epoch == 1:
            print(f"\n>>> Phase 1: Training DEM Encoder, Fusion, and Decoder (Vision Backbone frozen) at lr={args.lr_head}\n")
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr_head
            )

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

        phase = "Stage1" if epoch <= args.warmup_epochs else "Stage2"
        
        train_l, val_l = train_metrics["loss"], val_metrics["loss"]
        val_unc = val_metrics["unc"]
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{phase}] "
              f"Train={train_l:.4f} | Val={val_l:.4f} (NLL={val_metrics['nll']:.3f}, Norm={val_metrics['normal']:.3f}, Unc={val_unc:.3f})")

        if val_l < best_val_loss:
            best_val_loss = val_l
            epochs_without_improvement = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "prompt2dem_urban_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best checkpoint to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s) "
                  f"(patience: {args.early_stopping_patience})")
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nStopping early at epoch {epoch}. Best checkpoint is already saved.")
                break

    final_path = os.path.join(args.checkpoint_dir, "prompt2dem_urban_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training done. Final: {final_path}, best: prompt2dem_urban_best.pt")


if __name__ == "__main__":
    main()