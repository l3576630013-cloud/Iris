"""
FAD-Net SFE — Structural Feature Extractor (Pretrained Encoder)
=================================================================
Paper: FAD-Net, IEEE TIM 2026, Section III-B, Fig.3

Architecture:
    ResNet-18 front layers (first conv modified for grayscale input),
    classification head removed, outputting multi-scale feature maps.

    CPC pre-training (Stage 1, separate script):
        Patches (target/context/similar, P=32) → encoder → feature vectors
        Triplet InfoNCE loss (Eq.4-5) forces structural feature learning.

    Frozen usage in FAD-Net (Stage 2):
        Full OCT image → SFE → multi-scale feature maps
        → 1×1 Conv projection to match U-Net encoder channels
        → Cross-Attention (Q=U-Net, K/V=SFE_projected)

Input:  [B, 1, 512, 512]   grayscale OCT image
Output: f1: [B,  64, 128, 128]     after layer1
        f2: [B, 128,  64,  64]     after layer2
        f3: [B, 256,  32,  32]     after layer3
        f4: [B, 512,  16,  16]     after layer4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ResNet-18 BasicBlock (standard, from torchvision)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ============================================================
# SFE Encoder Backbone
# ============================================================
class SFE(nn.Module):
    """
    Structural Feature Extractor — ResNet-18 backbone without classification head.

    Paper Section III-B:
        "We design a SFE based on contrastive predictive coding."
        Encoder architecture is NOT specified in the paper.
        Using ResNet-18 as standard CPC backbone (common in the field).

    Input:  [B, 1, 512, 512]   grayscale OCT image
    Output: dict of 4 multi-scale feature maps matching MHFB scales
            'f1': [B,  64, 128, 128]
            'f2': [B, 128,  64,  64]
            'f3': [B, 256,  32,  32]
            'f4': [B, 512,  16,  16]
    """

    def __init__(self, in_channels: int = 1):
        """
        Args:
            in_channels: 1 for grayscale OCT, 3 for RGB.
                         Paper default: 1 (OCT grayscale, Sec IV-A2).
        """
        super().__init__()

        # ---- Stem (replaces standard ResNet conv1) ----
        # Standard ResNet: Conv2d(3, 64, 7, stride=2, padding=3)
        # FAD-Net:     Conv2d(1, 64, 7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- layer1 (64→64, stride=1, 2 blocks) → 128² ----
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)

        # ---- layer2 (64→128, stride=2, 2 blocks) → 64² ----
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)

        # ---- layer3 (128→256, stride=2, 2 blocks) → 32² ----
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        # ---- layer4 (256→512, stride=2, 2 blocks) → 16² ----
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # No avgpool, no fc — keep spatial features for cross-attention

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion),
            )

        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, 512, 512] grayscale OCT image

        Returns:
            dict with keys 'f1'..'f4', each a feature map tensor.
        """
        # Stem: 512 → 256 → 128
        x = self.relu(self.bn1(self.conv1(x)))   # [B, 64, 256, 256]
        x = self.maxpool(x)                       # [B, 64, 128, 128]

        # ResNet layers
        f1 = self.layer1(x)   # [B,  64, 128, 128]
        f2 = self.layer2(f1)  # [B, 128,  64,  64]
        f3 = self.layer3(f2)  # [B, 256,  32,  32]
        f4 = self.layer4(f3)  # [B, 512,  16,  16]

        return {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}


# ============================================================
# Projection Heads → match U-Net encoder channel dimensions
# ============================================================
class SFEProjection(nn.Module):
    """
    Projects SFE multi-scale features to match U-Net encoder channel dimensions
    for cross-attention.

    Paper Section III-C3 (paraphrased):
        "F_enc ... obtained by performing cross-attention between the encoder
        feature F_unet of U-Net and the spatially projected feature from SFE."

    Each projection is a 1×1 Conv, preserving spatial dimensions.

    Args:
        sfe_channels: list of SFE output channels at each scale.
                      Default matches ResNet-18 layer outputs.
        unet_channels: list of U-Net encoder output channels at each scale.
                       PAPER NOT SPECIFIED — using standard DDPM U-Net defaults.
    """

    def __init__(self,
                 sfe_channels: list = None,
                 unet_channels: list = None):
        super().__init__()
        if sfe_channels is None:
            sfe_channels = [64, 128, 256, 512]   # ResNet-18 layers
        if unet_channels is None:
            unet_channels = [128, 256, 256, 256]  # DDPM U-Net standard (speculative)

        self.projections = nn.ModuleList([
            nn.Conv2d(sc, uc, kernel_size=1, bias=False)
            for sc, uc in zip(sfe_channels, unet_channels)
        ])

    def forward(self, sfe_features: dict):
        """
        Args:
            sfe_features: dict with 'f1'..'f4' from SFE.forward()

        Returns:
            dict with 'p1'..'p4': projected features matching U-Net encoder dims.
        """
        return {
            f'p{i+1}': self.projections[i](sfe_features[f'f{i+1}'])
            for i in range(4)
        }


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAD-Net SFE Module — Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Test 1: SFE backbone ----
    print(f"\n{'─' * 60}")
    print("Test 1: SFE Backbone (ResNet-18, no classification head)")
    print(f"{'─' * 60}")

    sfe = SFE(in_channels=1).to(device)

    # Simulate OCT image batch (B=2, grayscale, 512×512)
    x = torch.randn(2, 1, 512, 512).to(device)
    print(f"  Input:  {list(x.shape)}")

    with torch.no_grad():
        feats = sfe(x)

    scale_names = [
        ('f1', 128, 'after layer1 (stride=1)'),
        ('f2',  64, 'after layer2 (stride=2)'),
        ('f3',  32, 'after layer3 (stride=2)'),
        ('f4',  16, 'after layer4 (stride=2)'),
    ]
    for key, res, desc in scale_names:
        t = feats[key]
        print(f"  {key}: {list(t.shape)}  ({res}x{res})  {desc}")

    n_params = sum(p.numel() for p in sfe.parameters())
    print(f"\n  Total parameters: {n_params:,}")

    # ---- Test 2: SFE Projection ----
    print(f"\n{'─' * 60}")
    print("Test 2: SFE Projection (1x1 Conv to match U-Net channels)")
    print(f"{'─' * 60}")

    proj = SFEProjection().to(device)

    with torch.no_grad():
        projected = proj(feats)

    for i, (sf, pf) in enumerate(zip(feats.values(), projected.values())):
        print(f"  SFE f{i+1}: {list(sf.shape)}  ->  proj p{i+1}: {list(pf.shape)}")

    proj_params = sum(p.numel() for p in proj.parameters())
    print(f"\n  Projection parameters: {proj_params:,}")

    # ---- Test 3: Verify grad flow ----
    print(f"\n{'─' * 60}")
    print("Test 3: Gradient Flow")
    print(f"{'─' * 60}")

    x_grad = torch.randn(2, 1, 512, 512).to(device).requires_grad_(True)
    sfe_grad = SFE(in_channels=1).to(device)
    proj_grad = SFEProjection().to(device)

    feats_grad = sfe_grad(x_grad)
    proj_grad_out = proj_grad(feats_grad)
    loss = sum(p.sum() for p in proj_grad_out.values())
    loss.backward()

    print(f"  Input requires_grad: {x_grad.requires_grad}")
    print(f"  x.grad shape: {list(x_grad.grad.shape)}")
    print(f"  x.grad max:   {x_grad.grad.abs().max().item():.6f}")
    print(f"  Grad flow OK: {x_grad.grad is not None and x_grad.grad.abs().max() > 0}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("SFE Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    SFE Backbone:       ResNet-18 (first 4 layers, no avgpool/fc)
    Input:              [B, 1, 512, 512]  grayscale OCT image
    Output scales:      f1 [B,  64, 128, 128]
                        f2 [B, 128,  64,  64]
                        f3 [B, 256,  32,  32]
                        f4 [B, 512,  16,  16]

    SFE Projection:     1x1 Conv per scale
    Output (to U-Net):  p1 [B, 128, 128, 128]
                        p2 [B, 256,  64,  64]
                        p3 [B, 256,  32,  32]
                        p4 [B, 256,  16,  16]

    CPC Pre-training (Stage 1):
        Patch size P=32  (user specified)
        Triplet InfoNCE loss on feature vectors after global pooling
        Paper Section III-B, Eq.(4)-(5)

    Frozen in FAD-Net (Stage 2):
        Full image → SFE → SFEProjection → Cross-Attention with U-Net

    Speculative (paper not specified):
        Encoder architecture  →  ResNet-18 (standard CPC choice)
        U-Net channels        →  [128, 256, 256, 256] (DDPM standard)
        Projection method     →  1x1 Conv (standard in attention-based fusion)
    """)
