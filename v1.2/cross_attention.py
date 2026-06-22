"""
FAD-Net Multi-Head Cross-Attention Module
===========================================
Paper: FAD-Net, IEEE TIM 2026, Section III-C3

    "F_enc represents the feature obtained by performing cross-attention
     between the encoder feature F_unet of U-Net and the spatially
     projected feature from SFE."

Architecture (per scale):
    Q  = F_unet                   [B, Cq,   H, W]   diffusion latent
    KV = F_sfe (projected)        [B, Ckv,  H, W]   structural prior

    → Multi-Head Cross-Attention →
    → Add residual (Q + attention_out) →
    → F_enc                       [B, Cq,   H, W]   fused feature

The module operates on a SINGLE scale. The caller iterates over the 4
encoder levels (128², 64², 32², 16²), each with its own CrossAttention instance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention for fusing diffusion latent (Q) with
    structural prior (K, V) at a single spatial scale.

    Paper: FAD-Net Section III-C3
        Q = F_unet (U-Net encoder feature)
        K = F_sfe  (projected SFE structural feature)
        V = F_sfe  (projected SFE structural feature, same as K)

    Args:
        q_dim:    channel dimension of Q (U-Net encoder feature)
        kv_dim:   channel dimension of K/V (projected SFE feature)
        n_heads:  number of attention heads. Default 4 (user specified).
        qk_dim:   per-head dimension for Q, K projection.
                  PAPER NOT SPECIFIED — default kv_dim // n_heads.
    """

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4,
                 qk_dim: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.qk_dim = qk_dim if qk_dim is not None else kv_dim // n_heads
        self.scale = self.qk_dim ** -0.5

        inner_dim = n_heads * self.qk_dim

        # Q projection: from U-Net feature space
        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)

        # K, V projections: from SFE feature space
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)

        # Output projection: back to Q's channel dimension
        self.to_out = nn.Linear(inner_dim, q_dim, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, Cq,  H, W]   U-Net encoder feature (query)
            kv: [B, Ckv, H, W]   SFE projected feature (key & value)

        Returns:
            out: [B, Cq, H, W]   fused feature (F_enc in paper)
        """
        B, Cq, H, W = q.shape
        _, Ckv, _, _ = kv.shape

        # ---- Flatten spatial dims → [B, N, C] where N = H * W ----
        q_flat = q.reshape(B, Cq, H * W).permute(0, 2, 1)    # [B, N, Cq]
        kv_flat = kv.reshape(B, Ckv, H * W).permute(0, 2, 1)  # [B, N, Ckv]

        # ---- Linear projections ----
        Q = self.to_q(q_flat)    # [B, N, n_heads * qk_dim]
        K = self.to_k(kv_flat)   # [B, N, n_heads * qk_dim]
        V = self.to_v(kv_flat)   # [B, N, n_heads * qk_dim]

        # ---- Reshape for multi-head: [B, n_heads, N, qk_dim] ----
        Q = Q.view(B, -1, self.n_heads, self.qk_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.qk_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.qk_dim).transpose(1, 2)

        # ---- Scaled dot-product attention ----
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, nh, N, N]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, nh, N, qk_dim]

        # ---- Merge heads → [B, N, inner_dim] → [B, N, Cq] ----
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.qk_dim)
        out = self.to_out(out)  # [B, N, Cq]

        # ---- Residual connection + reshape back to spatial ----
        out = out.permute(0, 2, 1).view(B, Cq, H, W)
        out = out + q  # residual: preserve diffusion features
        return out


class MultiScaleCrossAttention(nn.Module):
    """
    Applies CrossAttention independently at each of the 4 U-Net encoder scales.

    This wraps 4 CrossAttention modules, one per scale, matching the
    MHFB output scales (128², 64², 32², 16²).

    Args:
        unet_channels: list of U-Net encoder channel dims at each scale.
                       Default [128, 256, 256, 256] (DDPM standard, speculative).
        sfe_channels:  list of projected SFE channel dims at each scale.
                       Default [128, 256, 256, 256] (from SFEProjection).
        n_heads:       number of attention heads. Default 4.
    """

    def __init__(self, unet_channels: list = None, sfe_channels: list = None,
                 n_heads: int = 4):
        super().__init__()
        if unet_channels is None:
            unet_channels = [128, 256, 256, 256]   # speculative: DDPM standard
        if sfe_channels is None:
            sfe_channels = [128, 256, 256, 256]     # from SFEProjection default

        assert len(unet_channels) == len(sfe_channels) == 4

        self.attentions = nn.ModuleList([
            CrossAttention(q_dim=qc, kv_dim=sc, n_heads=n_heads)
            for qc, sc in zip(unet_channels, sfe_channels)
        ])

    def forward(self, unet_features: dict, sfe_features: dict) -> dict:
        """
        Args:
            unet_features: dict with keys 'e1'..'e4'
                U-Net encoder outputs at each scale.
                e1: [B, 128, 128, 128]
                e2: [B, 256,  64,  64]
                e3: [B, 256,  32,  32]
                e4: [B, 256,  16,  16]
            sfe_features: dict with keys 'p1'..'p4'
                Projected SFE features at each scale (from SFEProjection).
                p1: [B, 128, 128, 128]
                p2: [B, 256,  64,  64]
                p3: [B, 256,  32,  32]
                p4: [B, 256,  16,  16]

        Returns:
            dict with keys 'f1'..'f4'
                Fused features F_enc at each scale (same shapes as U-Net inputs).
        """
        return {
            f'f{i}': self.attentions[i - 1](
                unet_features[f'e{i}'], sfe_features[f'p{i}']
            )
            for i in range(1, 5)
        }


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAD-Net Cross-Attention Module — Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Test 1: Single-scale CrossAttention ----
    print(f"\n{'─' * 60}")
    print("Test 1: Single-Scale Cross-Attention")
    print(f"{'─' * 60}")

    B, Cq, Ckv, H, W = 2, 128, 128, 64, 64
    q_test = torch.randn(B, Cq, H, W, device=device)
    kv_test = torch.randn(B, Ckv, H, W, device=device)

    ca = CrossAttention(q_dim=Cq, kv_dim=Ckv, n_heads=4).to(device)
    print(f"  n_heads: {ca.n_heads}")
    print(f"  qk_dim per head: {ca.qk_dim}")
    print(f"  Q shape:  {list(q_test.shape)}")
    print(f"  KV shape: {list(kv_test.shape)}")

    with torch.no_grad():
        out = ca(q_test, kv_test)
    print(f"  Output:   {list(out.shape)}")
    print(f"  Shape preserved: {out.shape == q_test.shape}")

    n_params = sum(p.numel() for p in ca.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Test 2: Multi-Scale CrossAttention ----
    print(f"\n{'─' * 60}")
    print("Test 2: Multi-Scale Cross-Attention (4 encoder levels)")
    print(f"{'─' * 60}")

    msca = MultiScaleCrossAttention(n_heads=4).to(device)

    # Simulate U-Net encoder features at 4 scales
    unet_feats = {
        'e1': torch.randn(B, 128, 128, 128, device=device),  # level 1
        'e2': torch.randn(B, 256,  64,  64, device=device),  # level 2
        'e3': torch.randn(B, 256,  32,  32, device=device),  # level 3
        'e4': torch.randn(B, 256,  16,  16, device=device),  # level 4
    }

    # Simulate SFE projected features at 4 scales
    sfe_feats = {
        'p1': torch.randn(B, 128, 128, 128, device=device),
        'p2': torch.randn(B, 256,  64,  64, device=device),
        'p3': torch.randn(B, 256,  32,  32, device=device),
        'p4': torch.randn(B, 256,  16,  16, device=device),
    }

    print(f"  Scale 1: Q={list(unet_feats['e1'].shape)}  KV={list(sfe_feats['p1'].shape)}")
    print(f"  Scale 2: Q={list(unet_feats['e2'].shape)}  KV={list(sfe_feats['p2'].shape)}")
    print(f"  Scale 3: Q={list(unet_feats['e3'].shape)}  KV={list(sfe_feats['p3'].shape)}")
    print(f"  Scale 4: Q={list(unet_feats['e4'].shape)}  KV={list(sfe_feats['p4'].shape)}")

    with torch.no_grad():
        fused = msca(unet_feats, sfe_feats)

    print(f"\n  Fused outputs (F_enc per scale):")
    for k in ['f1', 'f2', 'f3', 'f4']:
        print(f"    {k}: {list(fused[k].shape)}")

    total_params = sum(p.numel() for p in msca.parameters())
    print(f"\n  Total parameters (4 scales): {total_params:,}")

    # ---- Test 3: Gradient flow ----
    print(f"\n{'─' * 60}")
    print("Test 3: Gradient Flow")
    print(f"{'─' * 60}")

    ca_grad = CrossAttention(q_dim=128, kv_dim=128, n_heads=4).to(device)
    q_grad = torch.randn(2, 128, 32, 32, device=device, requires_grad=True)
    kv_grad = torch.randn(2, 128, 32, 32, device=device, requires_grad=True)

    out_grad = ca_grad(q_grad, kv_grad)
    loss = out_grad.sum()
    loss.backward()

    print(f"  Q grad max: {q_grad.grad.abs().max().item():.6f}")
    print(f"  KV grad max: {kv_grad.grad.abs().max().item():.6f}")
    print(f"  Grad flow OK: {q_grad.grad is not None and kv_grad.grad is not None}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("Cross-Attention Module Summary")
    print(f"{'=' * 60}")
    print("""
    CrossAttention (per scale):
        Input:  Q  [B, Cq,  H, W]   U-Net encoder feature (diffusion latent)
                KV [B, Ckv, H, W]   SFE projected feature (structural prior)
        Heads:  4
        Output: F_enc [B, Cq, H, W] fused feature (with residual)

    MultiScaleCrossAttention:
        Applies CrossAttention independently at 4 scales:
          Scale 1: 128x128   Q[128] x KV[128]
          Scale 2:  64x64    Q[256] x KV[256]
          Scale 3:  32x32    Q[256] x KV[256]
          Scale 4:  16x16    Q[256] x KV[256]

    Integration in FAD-Net (Section III-C3):
        F_unet (U-Net encoder) → Q
        SFE_projected           → K, V
        CrossAttention          → F_enc
        TAWG fusion: F_fuse = (1-W_t)*F_enc + W_t*F_h

    Paper NOT specified (reasonable defaults used):
        - n_heads = 4            (user specified)
        - qk_dim = kv_dim // n_heads
        - Residual connection    (standard attention practice)
    """)
