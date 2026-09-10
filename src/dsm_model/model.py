"""
State-of-the-art Single-View DEM Reconstruction Network.
Implements a Prompt2DEM-style architecture:
1. Vision Encoder: Pretrained DepthAnythingV2-Small backbone (ViT).
2. DEM Encoder: Lightweight CNN for the coarse elevation prompt.
3. Multi-scale Gated Fusion: Cross-modal feature fusion.
4. Decoder: High-resolution upsampling to (DEM, Uncertainty).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)


class DEMEncoder(nn.Module):
    """Encodes 1-channel Coarse DEM into 4 feature scales to match ViT outputs."""
    def __init__(self, embed_dim=384):
        super().__init__()
        # Input (B, 1, 224, 224) -> (B, 64, 112, 112)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # (B, 64, 56, 56)
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(64, 64)
        )
        # (B, 128, 28, 28)
        self.layer2 = ResBlock(64, 128, stride=2)
        # (B, 256, 14, 14)
        self.layer3 = ResBlock(128, 256, stride=2)
        
        # Projections to match ViT embed_dim (384)
        self.proj1 = nn.Conv2d(64, embed_dim, 1)   # 56x56 -> we'll pool it later or align
        self.proj2 = nn.Conv2d(128, embed_dim, 1)  # 28x28
        self.proj3 = nn.Conv2d(256, embed_dim, 1)  # 14x14
        
    def forward(self, x):
        x = self.stem(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        
        # We need 4 features matching ViT 14x14 grid for small model
        # Actually ViT patch size is 14. 224 / 14 = 16x16 grid!
        # So ViT gives 16x16.
        # Let's just output matching spatial sizes using interpolation if needed
        return [self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)]


class FeatureFusion(nn.Module):
    """Gated fusion of RGB and DEM features."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, rgb_f, dem_f):
        if rgb_f.shape != dem_f.shape:
            dem_f = nn.functional.interpolate(dem_f, size=rgb_f.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([rgb_f, dem_f], dim=1)
        g = self.gate(cat)
        fused = rgb_f * (1 - g) + dem_f * g
        return self.out_proj(fused)


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim=384, out_channels=2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, 2, stride=2)
        self.conv1 = ResBlock(256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = ResBlock(128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = ResBlock(64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = ResBlock(32, 32)
        
        self.head = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        # x is (B, 384, 14, 14) or (B, 384, 16, 16)
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        
        # Up to original size
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # or use up4 depending on exact sizing. Let's force target size at the end.
        out = self.head(x)
        return out


class Prompt2DEM(nn.Module):
    """
    Inputs:
        rgb: (B, 3, H, W)
        coarse_dem: (B, 1, H, W)
    Outputs:
        pred: (B, 2, H, W) -> [DEM, LogVariance]
    """
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, freeze_backbone=True):
        super().__init__()
        # Load vision backbone
        hf_model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
        self.vision_encoder = hf_model.backbone
        
        self._frozen = False
        if freeze_backbone:
            self.freeze_backbone()
            
        self.embed_dim = 384 # Hardcoded for Small
        
        self.dem_encoder = DEMEncoder(self.embed_dim)
        
        # We'll extract from layers 2, 5, 8, 11
        self.fusion = FeatureFusion(self.embed_dim)
        
        self.decoder = SimpleDecoder(self.embed_dim, out_channels=2)
        
    def freeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        self._frozen = False

    def forward(self, rgb, coarse_dem):
        B, _, H, W = rgb.shape
        
        # 1. Vision Features
        # ViT returns hidden states at each block
        vision_outputs = self.vision_encoder(rgb, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        
        # Grab the last hidden state
        last_hidden = hidden_states[-1] # (B, 257, 384)
        
        # Strip CLS token and reshape to spatial grid
        # For 224x224 and patch_size=14, grid is 16x16
        seq_len = last_hidden.shape[1]
        grid_size = int((seq_len - 1) ** 0.5)
        
        rgb_feat = last_hidden[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, grid_size, grid_size)
        
        # 2. DEM Features
        dem_feats = self.dem_encoder(coarse_dem)
        # Grab the deepest DEM feature for fusion
        dem_feat_deep = dem_feats[-1]
        
        # 3. Fusion
        fused = self.fusion(rgb_feat, dem_feat_deep)
        
        # 4. Decoder
        out = self.decoder(fused)
        
        if out.shape[-2:] != (H, W):
            out = nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            
        return out


if __name__ == "__main__":
    model = Prompt2DEM(freeze_backbone=True)
    
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 (frozen backbone): {frozen_trainable:,} / {total_params:,} params trainable")
    
    model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 (unfrozen):        {unfrozen_trainable:,} / {total_params:,} params trainable")
    
    dummy_rgb = torch.randn(2, 3, 224, 224)
    dummy_dem = torch.randn(2, 1, 224, 224)
    out = model(dummy_rgb, dummy_dem)
    print("Output shape:", out.shape)
    assert out.shape == (2, 2, 224, 224), "Output shape mismatch!"
    print("Shape check passed.")