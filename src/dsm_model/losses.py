"""
Advanced Loss functions for Prompt2DEM architecture.
Includes Aleatoric Uncertainty Loss, Gradient Loss, and Surface Normal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_gradients(x: torch.Tensor):
    """Computes spatial gradients (dx, dy)."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy

def compute_normals(x: torch.Tensor):
    """
    Computes simple surface normals from a DEM (B, 1, H, W).
    Returns normals (B, 3, H-1, W-1).
    """
    dx = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]
    dy = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
    
    # Normal is cross product of (1, 0, dx) and (0, 1, dy)
    # -> (-dx, -dy, 1)
    normals = torch.cat([-dx, -dy, torch.ones_like(dx)], dim=1)
    normals = F.normalize(normals, p=2, dim=1)
    return normals


class Prompt2DEMLoss(nn.Module):
    def __init__(self, gradient_weight: float = 1.0, normal_weight: float = 1.0):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.normal_weight = normal_weight

    def forward(self, pred_and_var: torch.Tensor, target: torch.Tensor):
        """
        pred_and_var: (B, 2, H, W) -> [pred_dem, log_variance]
        target: (B, 1, H, W)
        """
        pred = pred_and_var[:, 0:1, :, :]
        log_var = pred_and_var[:, 1:2, :, :]
        
        # 1. Aleatoric Uncertainty Loss (Negative Log Likelihood)
        # Using L1 distance robust formulation: exp(-log_var) * L1 + log_var
        # We clamp log_var for stability.
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        l1_dist = torch.abs(pred - target)
        nll_loss = (torch.exp(-log_var) * l1_dist + log_var).mean()
        
        # 2. Gradient Loss
        pred_dx, pred_dy = compute_gradients(pred)
        target_dx, target_dy = compute_gradients(target)
        grad_loss = (torch.abs(pred_dx - target_dx).mean() + 
                     torch.abs(pred_dy - target_dy).mean())
        
        # 3. Surface Normal Loss
        pred_norm = compute_normals(pred)
        target_norm = compute_normals(target)
        # Cosine similarity loss = 1 - cos(theta)
        normal_loss = (1 - F.cosine_similarity(pred_norm, target_norm, dim=1)).mean()
        
        total = nll_loss + self.gradient_weight * grad_loss + self.normal_weight * normal_loss
        
        return total, {
            "nll_loss": nll_loss.item(),
            "gradient_loss": grad_loss.item(),
            "normal_loss": normal_loss.item(),
            "total_loss": total.item(),
            "mean_uncertainty": torch.exp(log_var).mean().item()
        }


if __name__ == "__main__":
    criterion = Prompt2DEMLoss()
    pred_and_var = torch.rand(2, 2, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    loss, parts = criterion(pred_and_var, target)
    print("Loss:", loss.item(), parts)