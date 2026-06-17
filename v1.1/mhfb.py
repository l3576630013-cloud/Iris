"""
FAD-Net MHFB — Multi-scale High-Frequency Feature Extraction Block
====================================================================
Paper: FAD-Net, IEEE TIM 2026, Section III-C2, Eq.(7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BConv(nn.Module):
    """3x3 Conv → BatchNorm → ReLU (paper Section III-C2, Eq.7 notation)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MHFB(nn.Module):
    """
    Multi-scale High-Frequency Feature Extraction Block.

    Strictly follows Eq.(7) of the paper.

    Pyramid layout (i=layer depth 1..4, j=scale level):

           j=1     j=2     j=3     j=4
    i=1   H11  →  H12  →  H13  →  H14     (128x128)
    i=2   H21  →  H22  →  H23            (64x64)
    i=3   H31  →  H32                     (32x32)
    i=4   H41                              (16x16)

    Eq.(7) — three cases:
      Case A (i+j <= 4):
        H^i_j = BConv(Concat(H^i_{j-1}, Up(H^{i+1}_j)))
      Case B (i=4, j=1):
        H^4_1 = BConv(Concat(H^4_0, Z))        ← Z = zero matrix
      Case C (i+j=5, i != 4):
        H^i_j = BConv(Concat(H^i_{j-1}, Up(H^{i+1}_{j-1})))

    Output (diagonal): F_h = {H^1_4, H^2_3, H^3_2, H^4_1}

    Backbone (paper: "three cascaded convolutional blocks with channel shrinkage"):
      Input HF subbands are processed through strided convs to produce
      the virtual column-0 features H^i_0 at each pyramid level.

    Input:  HF concat [B, 3*C_in, H/2, W/2]   → typically [B, 3, 256, 256]
    Output: dict with keys 'h14','h23','h32','h41' → F_h for TAWG fusion
    """

    def __init__(self, in_channels: int = 3, base_channels: list = None,
                 subband_c1: int = 24, subband_c2: int = 16,
                 subband_c3: int = 12):
        """
        Args:
            in_channels:   C_in * 3 — the 3 HF subbands concatenated.
            base_channels: channel counts for pyramid levels i=1..4.
                           [ch1, ch2, ch3, ch4] with shrinkage.
                           Default: [64, 48, 32, 16].
            subband_c1..c3: per-subband cascade channels.
                            Paper: "three cascaded convolutional blocks
                            with channel shrinkage". Each block reduces
                            channels.  Default: 1→24→16→12.
                            PAPER NOT SPECIFIED — progressive compression
                            principle applied.
        """
        super().__init__()
        if base_channels is None:
            base_channels = [64, 48, 32, 16]

        ch1, ch2, ch3, ch4 = base_channels
        self.base_channels = base_channels
        C_in_per_subband = in_channels // 3  # typically 1

        # ================================================================
        # Per-subband cascaded blocks (paper Section III-C2)
        #   "For each high frequency subband S ∈ {I_LH, I_HL, I_HH},
        #    MHFB employs three cascaded convolutional blocks with
        #    channel shrinkage."
        #
        # Each block REDUCES channels (gradual compression):
        #   1 → 24 → 16 → 12   (ratios: 24x, 1.5x, 1.33x)
        # Shared weights across subbands.
        # ================================================================
        self.subband_block1 = BConv(C_in_per_subband, subband_c1)  # 1→24
        self.subband_block2 = BConv(subband_c1, subband_c2)        # 24→16
        self.subband_block3 = BConv(subband_c2, subband_c3)        # 16→12

        # Concat of 3 subbands after cascades: 3 * subband_c3 channels
        backbone_in = subband_c3 * 3  # 36

        # ================================================================
        # Pyramid backbone — produces H^i_0 (virtual column 0 of Eq.7)
        #   Strided convs generate the 4 resolution levels used by the
        #   pyramid.  Channel shrinkage follows the pyramid pattern.
        #   Paper NOT specified: downsampling method → stride-2 Conv
        #   (standard practice).
        # ================================================================
        # H^1_0: backbone_in → ch1, 256→128 (stride 2)
        self.down1 = nn.Sequential(
            nn.Conv2d(backbone_in, ch1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        # H^2_0: ch1 → ch2, 128→64 (stride 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        # H^3_0: ch2 → ch3, 64→32 (stride 2)
        self.down3 = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )
        # H^4_0: ch3 → ch4, 32→16 (stride 2)
        self.down4 = nn.Sequential(
            nn.Conv2d(ch3, ch4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),
        )

        # ---- Pyramid BConv blocks (one per node in the triangle) ----
        # Each BConv takes concat(ch_i + ch_{i+1}) → ch_i (shrinkage to row's channel count)

        # --- Column 1 (j=1) ---
        self.bconv_41 = BConv(ch4 + ch4, ch4)  # case B: concat(H^4_0, Z), Z same shape as H^4_0
        self.bconv_31 = BConv(ch3 + ch4, ch3)  # case A: concat(H^3_0, Up(H^4_1))
        self.bconv_21 = BConv(ch2 + ch3, ch2)  # case A: concat(H^2_0, Up(H^3_1))
        self.bconv_11 = BConv(ch1 + ch2, ch1)  # case A: concat(H^1_0, Up(H^2_1))

        # --- Column 2 (j=2) ---
        self.bconv_32 = BConv(ch3 + ch4, ch3)  # case C: concat(H^3_1, Up(H^4_1))
        self.bconv_22 = BConv(ch2 + ch3, ch2)  # case A: concat(H^2_1, Up(H^3_2))
        self.bconv_12 = BConv(ch1 + ch2, ch1)  # case A: concat(H^1_1, Up(H^2_2))

        # --- Column 3 (j=3) ---
        self.bconv_23 = BConv(ch2 + ch3, ch2)  # case C: concat(H^2_2, Up(H^3_2))
        self.bconv_13 = BConv(ch1 + ch2, ch1)  # case A: concat(H^1_2, Up(H^2_3))

        # --- Column 4 (j=4) ---
        self.bconv_14 = BConv(ch1 + ch2, ch1)  # case C: concat(H^1_3, Up(H^2_3))

    def forward(self, hf_concat: torch.Tensor):
        """
        Args:
            hf_concat: concatenated HF subbands [B, 3*C_in, 256, 256]
                       = cat([I_LH, I_HL, I_HH], dim=1)

        Returns:
            dict with keys:
                'h14': [B, ch1, 128, 128]   ← highest resolution
                'h23': [B, ch2, 64, 64]
                'h32': [B, ch3, 32, 32]
                'h41': [B, ch4, 16, 16]    ← deepest / smallest
        """
        B = hf_concat.shape[0]
        C_per = hf_concat.shape[1] // 3  # channels per subband (typically 1)

        # ---- Stage 1: per-subband cascaded blocks (paper: 3 each) ----
        # Split concatenated input → 3 independent subbands
        I_LH = hf_concat[:, 0:C_per, :, :]                    # [B, C_per, 256, 256]
        I_HL = hf_concat[:, C_per:2*C_per, :, :]
        I_HH = hf_concat[:, 2*C_per:3*C_per, :, :]

        # Apply same 3 cascaded BConv blocks to each subband independently
        subband_feats = []
        for subband in [I_LH, I_HL, I_HH]:
            f = self.subband_block1(subband)                  # [B, mid, 256, 256]
            f = self.subband_block2(f)                        # [B, mid, 256, 256]
            f = self.subband_block3(f)                        # [B, out, 256, 256]
            subband_feats.append(f)

        # Concatenate per-subband features
        hf_feat = torch.cat(subband_feats, dim=1)             # [B, 3*out, 256, 256]

        # ---- Stage 2: pyramid backbone → H^i_0 (virtual column 0) ----
        H10 = self.down1(hf_feat)                             # [B, ch1, 128, 128]
        H20 = self.down2(H10)                                 # [B, ch2, 64, 64]
        H30 = self.down3(H20)                                 # [B, ch3, 32, 32]
        H40 = self.down4(H30)                                 # [B, ch4, 16, 16]

        # ---- Pyramid: compute column by column, bottom to top ----

        # --- Column 1 (j=1) ---
        # Case B: i=4, j=1
        Z = torch.zeros_like(H40)                               # [B, ch4, 16, 16]
        H41 = self.bconv_41(torch.cat([H40, Z], dim=1))        # [B, ch4, 16, 16]

        # Case A: i=3, j=1  (3+1=4 <= 4)
        H41_up_32 = F.interpolate(H41, scale_factor=2.0, mode='bilinear',
                                   align_corners=False)        # [B, ch4, 32, 32]
        H31 = self.bconv_31(torch.cat([H30, H41_up_32], dim=1))  # [B, ch3, 32, 32]

        # Case A: i=2, j=1  (2+1=3 <= 4)
        H31_up_64 = F.interpolate(H31, scale_factor=2.0, mode='bilinear',
                                   align_corners=False)        # [B, ch3, 64, 64]
        H21 = self.bconv_21(torch.cat([H20, H31_up_64], dim=1))  # [B, ch2, 64, 64]

        # Case A: i=1, j=1  (1+1=2 <= 4)
        H21_up_128 = F.interpolate(H21, scale_factor=2.0, mode='bilinear',
                                    align_corners=False)       # [B, ch2, 128, 128]
        H11 = self.bconv_11(torch.cat([H10, H21_up_128], dim=1))  # [B, ch1, 128, 128]

        # --- Column 2 (j=2) ---
        # Case C: i=3, j=2  (3+2=5, i=3≠4)
        H41_up_32_v2 = F.interpolate(H41, scale_factor=2.0, mode='bilinear',
                                      align_corners=False)     # [B, ch4, 32, 32]
        H32 = self.bconv_32(torch.cat([H31, H41_up_32_v2], dim=1))  # [B, ch3, 32, 32]

        # Case A: i=2, j=2  (2+2=4 <= 4)
        H32_up_64 = F.interpolate(H32, scale_factor=2.0, mode='bilinear',
                                   align_corners=False)        # [B, ch3, 64, 64]
        H22 = self.bconv_22(torch.cat([H21, H32_up_64], dim=1))  # [B, ch2, 64, 64]

        # Case A: i=1, j=2  (1+2=3 <= 4)
        H22_up_128 = F.interpolate(H22, scale_factor=2.0, mode='bilinear',
                                    align_corners=False)       # [B, ch2, 128, 128]
        H12 = self.bconv_12(torch.cat([H11, H22_up_128], dim=1))  # [B, ch1, 128, 128]

        # --- Column 3 (j=3) ---
        # Case C: i=2, j=3  (2+3=5, i=2≠4)
        H32_up_64_v2 = F.interpolate(H32, scale_factor=2.0, mode='bilinear',
                                      align_corners=False)     # [B, ch3, 64, 64]
        H23 = self.bconv_23(torch.cat([H22, H32_up_64_v2], dim=1))  # [B, ch2, 64, 64]

        # Case A: i=1, j=3  (1+3=4 <= 4)
        H23_up_128 = F.interpolate(H23, scale_factor=2.0, mode='bilinear',
                                    align_corners=False)       # [B, ch2, 128, 128]
        H13 = self.bconv_13(torch.cat([H12, H23_up_128], dim=1))  # [B, ch1, 128, 128]

        # --- Column 4 (j=4) ---
        # Case C: i=1, j=4  (1+4=5, i=1≠4)
        H23_up_128_v2 = F.interpolate(H23, scale_factor=2.0, mode='bilinear',
                                       align_corners=False)    # [B, ch2, 128, 128]
        H14 = self.bconv_14(torch.cat([H13, H23_up_128_v2], dim=1))  # [B, ch1, 128, 128]

        F_h = {
            'h14': H14,  # [B, ch1, 128, 128]
            'h23': H23,  # [B, ch2, 64, 64]
            'h32': H32,  # [B, ch3, 32, 32]
            'h41': H41,  # [B, ch4, 16, 16]
        }
        return F_h


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAD-Net MHFB Module — Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Simulate output from DWT: HF concat of 3 subbands
    B, C, H_half, W_half = 2, 1, 256, 256
    hf_input = torch.randn(B, 3 * C, H_half, W_half).to(device)
    print(f"\nInput (from DWT HF concat):  {list(hf_input.shape)}")
    print(f"  = cat([I_LH, I_HL, I_HH], dim=1)")

    # MHFB forward
    mhfb = MHFB(in_channels=3 * C, base_channels=[64, 48, 32, 16]).to(device)
    print(f"\nMHFB channels (speculative, paper not specified):")
    print(f"  per-subband cascades: {C}→{mhfb.subband_block1.conv.out_channels}→"
          f"{mhfb.subband_block2.conv.out_channels}→{mhfb.subband_block3.conv.out_channels}")
    print(f"  pyramid: ch1={mhfb.base_channels[0]}, ch2={mhfb.base_channels[1]}, "
          f"ch3={mhfb.base_channels[2]}, ch4={mhfb.base_channels[3]}")

    F_h = mhfb(hf_input)

    print(f"\n{'─' * 60}")
    print("MHFB Output: F_h = {H^i_j | i+j = 5}")
    print(f"{'─' * 60}")
    for key in ['h14', 'h23', 'h32', 'h41']:
        t = F_h[key]
        print(f"  {key}: {list(t.shape)}  "
              f"range=[{t.min().item():.4f}, {t.max().item():.4f}]")

    # Verify resolutions
    expected_res = {'h14': 128, 'h23': 64, 'h32': 32, 'h41': 16}
    print(f"\n{'─' * 60}")
    print("Resolution Verification")
    print(f"{'─' * 60}")
    all_ok = True
    for key, exp_hw in expected_res.items():
        actual = F_h[key].shape[-2:]
        ok = actual == (exp_hw, exp_hw)
        print(f"  {key}: expected ({exp_hw},{exp_hw}), got ({actual[0]},{actual[1]}) {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    print(f"\n  All resolutions correct: {all_ok}")

    # Parameter count
    n_params = sum(p.numel() for p in mhfb.parameters())
    print(f"\n  Total parameters: {n_params:,}")

    # Summary
    print(f"\n{'=' * 60}")
    print("MHFB Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    Input:  HF = cat([I_LH, I_HL, I_HH], dim=1)  ->  [B, 3*C, 256, 256]
    Output: F_h dict:
              h14  [B, ch1, 128, 128]  - row i=1, col j=4
              h23  [B, ch2,  64,  64]  - row i=2, col j=3
              h32  [B, ch3,  32,  32]  - row i=3, col j=2
              h41  [B, ch4,  16,  16]  - row i=4, col j=1

    Architecture (paper Section III-C2):
      Stage 1 — per-subband 3-cascade (shared weights):
        I_LH, I_HL, I_HH each processed independently:
          BConv(1->32) -> BConv(32->32) -> BConv(32->16)
        -> concat 3x16ch -> [B, 48, 256, 256]

      Stage 2 — pyramid backbone (strided convs):
        48->64(s2)->128^2  64->48(s2)->64^2
        48->32(s2)->32^2   32->16(s2)->16^2

      Stage 3 — Eq.(7) pyramid (bottom-up + lateral):
        i+j <= 4:    H^ij = BConv(Concat(H^i_{{j-1}}, Up(H^{{i+1}}_j)))
        i=4, j=1:    H^41 = BConv(Concat(H^4_0, Z))
        i+j=5,i!=4:  H^ij = BConv(Concat(H^i_{{j-1}}, Up(H^{{i+1}}_{{j-1}})))

    Speculative (paper not specified):
      Pyramid channels:  [64, 48, 32, 16]  - channel shrinkage
      Subband cascade:   1->32->32->16     - channel shrinkage
      Subband weights:   shared (parsimonious, paper ambiguous)
      Up(.):             bilinear, factor 2
      Down(.):           stride-2 Conv (backbone)
    """)
