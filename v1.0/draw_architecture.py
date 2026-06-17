"""
FAD-Net Architecture Diagram — Module names only, no data/dimensions
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.dpi': 200})

C = {
    'bg':'#FAFAFA','input':'#E3F2FD','dwt':'#BBDEFB','mhfb':'#64B5F6',
    'fh':'#1565C0','sfe':'#C8E6C9','sfe_proj':'#81C784','struct':'#2E7D32',
    'tawg':'#FFB74D','wt':'#E65100','x_t':'#F48FB1','unet_enc':'#CE93D8',
    'unet_dec':'#AB47BC','cross_attn':'#BA68C8','fuse':'#FF8A65',
    'loss':'#EF5350','output':'#66BB6A','skip':'#BDBDBD','note':'#FFF9C4',
    'text':'#212121','arrow':'#546E7A',
}
WHITE = {'fh','struct','wt','loss','output'}

fig = plt.figure(figsize=(24, 17))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,24); ax.set_ylim(0,17)
ax.axis('off'); fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])

def box(ax, x, y, w, h, s, fc, fs=8.5, fw='normal'):
    tc = 'white' if fc in WHITE else C['text']
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                        facecolor=fc, edgecolor='#90A4AE', linewidth=0.8, zorder=3)
    ax.add_patch(p)
    lines = s.split('\n')
    for i, line in enumerate(lines):
        ax.text(x+w/2, y+h-(i+0.7)*h/len(lines), line,
                ha='center',va='center',fontsize=fs,fontweight=fw,color=tc,zorder=4)

def group(ax, x, y, w, h, label):
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.2",
                        facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=1.0,
                        linestyle='--', alpha=0.3, zorder=1)
    ax.add_patch(p)
    ax.text(x+0.3, y+h-0.2, label, ha='left',va='top',fontsize=8,
            fontweight='bold',color='#455A64',zorder=4)

def arr(ax, x1,y1,x2,y2, c=None, lw=1.0, ls='-', rad=0.0):
    if c is None: c = C['arrow']
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'), zorder=2)

# ═══════════ TITLE ═══════════
ax.text(12, 16.3, 'FAD-Net: Frequency-Aware Diffusion Network — Architecture Diagram',
        ha='center',va='center',fontsize=14,fontweight='bold',color='#0D47A1')

# ═══════════ ROW 0: INPUT ═══════════
box(ax, 0.2, 14.5, 3.5, 1.0, 'I_noisy\n(Noisy OCT)', C['input'], fs=9, fw='bold')
box(ax, 0.2, 13.3, 3.5, 0.6, 'Data Augmentation', C['note'], fs=7)

group(ax, 4.5, 13.0, 5.0, 2.0, 'Forward Diffusion')
box(ax, 4.7, 14.3, 1.3, 0.5, 'Timestep t', C['tawg'], fs=7)
box(ax, 4.7, 13.3, 1.3, 0.5, 'Noise ε', C['x_t'], fs=7)
box(ax, 6.4, 13.5, 2.8, 1.5, 'x_t\n(Noised Image)', C['x_t'], fs=9, fw='bold')
arr(ax, 6.0, 14.55, 6.4, 14.55)
arr(ax, 6.0, 13.55, 6.4, 13.9)
arr(ax, 3.7, 15.0, 4.7, 14.55)

group(ax, 10.2, 13.0, 3.5, 2.0, 'Time Embedding')
box(ax, 10.4, 13.5, 3.1, 1.1, 'Sinusoidal\nEmbedding', C['tawg'], fs=8)
arr(ax, 7.8, 14.55, 10.4, 14.55)

# ═══════════ ROW 1: THREE PRIOR PATHS ═══════════
YP = 7.2
HP = 5.5

# PATH A
group(ax, 0.15, YP, 7.5, HP, 'Path A: Frequency Prior')
box(ax, 0.3, YP+3.8, 7.2, 1.2, 'Haar DWT\n→ I_LL, I_LH, I_HL, I_HH', C['dwt'], fs=8.5)
box(ax, 0.3, YP+2.3, 7.2, 1.2, 'Per-Subband\nCascaded BConv', C['mhfb'], fs=8.5)
box(ax, 0.3, YP+0.8, 7.2, 1.2, 'MHFB Pyramid\n(Bottom-Up + Lateral Fusion)', C['mhfb'], fs=8.5, fw='bold')
box(ax, 0.3, YP+0.1, 7.2, 0.5, 'F_h (High-Freq Pyramid)', C['fh'], fs=8, fw='bold')
arr(ax, 3.9, YP+3.8, 3.9, YP+3.5)
arr(ax, 3.9, YP+2.3, 3.9, YP+2.0)
arr(ax, 2.0, 14.5, 3.9, YP+5.0, C['arrow'], rad=0.2)

# PATH B
group(ax, 8.1, YP, 7.7, HP, 'Path B: Structure Prior')
box(ax, 8.2, YP+3.5, 7.5, 1.7, 'SFE (ResNet-18)\n[FROZEN — Stage 0]\n→ f1, f2, f3, f4', C['sfe'], fs=8.5)
box(ax, 8.2, YP+2.2, 7.5, 1.0, 'SFEProjection\n(1×1 Conv ×4) [FROZEN]', C['sfe_proj'], fs=8.5, fw='bold')
box(ax, 8.2, YP+0.1, 7.5, 1.8, 'sfe_proj = {p1, p2, p3, p4}\nCross-Attention K/V\n(4 spatial scales)', C['struct'], fs=8.5, fw='bold')
arr(ax, 11.9, YP+3.5, 11.9, YP+3.2)
arr(ax, 3.7, 14.5, 11.9, YP+5.2, C['arrow'])

# PATH C
group(ax, 16.3, YP+1.5, 7.5, 4.0, 'Path C: TAWG')
box(ax, 16.5, YP+4.0, 7.1, 1.1, 'TAWG\nLinear → GELU → Linear → Sigmoid', C['tawg'], fs=8.5)
box(ax, 16.5, YP+2.0, 7.1, 1.7, 'W_t = [w₁, w₂, w₃, w₄]\nPer-Scale Fusion Weights\n(High t → rely on F_h)', C['wt'], fs=8.5, fw='bold')
arr(ax, 13.7, 14.5, 20.0, YP+4.8, C['arrow'])

# ═══════════ ROW 2: U-NET ENCODER ═══════════
YE = 2.8
HE = 4.0

group(ax, 0.15, YE, 12.0, HE, 'U-Net Encoder')
box(ax, 0.3, YE+3.0, 4.5, 0.7, 'conv_in', C['unet_enc'], fs=8.5)

enc = [
    (0.3, YE+2.2, 'Enc L0: ResBlocks + Time', 'skip0', 'Down'),
    (0.3, YE+1.4, 'Enc L1: ResBlocks + Time', 'skip1', 'Down'),
    (0.3, YE+0.6, 'Enc L2: ResBlocks + Time', 'skip2 → Q₁', 'Down'),
    (0.3, YE-0.2, 'Enc L3: ResBlocks + Time', 'skip3 → Q₂', 'Down → Bottleneck'),
]
for ex, ey, el, sl, dl in enc:
    box(ax, ex, ey, 6.0, 0.6, el, C['unet_enc'], fs=7.5)
    box(ax, 6.5, ey, 2.5, 0.6, sl, C['skip'], fs=7)
    box(ax, 9.3, ey, 2.5, 0.6, dl, '#9C27B0', fs=7)

box(ax, 6.5, YE-0.6, 5.3, 0.45, 'Bottleneck → Q₃ → Q₄', '#7B1FA2', fs=7.5)

arr(ax, 4.8, YE+3.0, 4.8, YE+2.8)
for i in range(4):
    arr(ax, 9.1, YE+2.25-i*0.8, 9.3, YE+2.25-i*0.8, '#9C27B0')
arr(ax, 9.0, 13.4, 2.8, YE+3.4, C['arrow'], rad=0.3)

# ═══════════ ROW 3: CROSS-ATTN + FUSION ═══════════
YF = 2.8
HF = 4.0

group(ax, 12.5, YF, 11.3, HF, 'Cross-Attention + TAWG Fusion (4 Scales)')

scales = [
    (YF+3.0, 'Scale 1', 'Q₁ × K/V₁', 'F_h¹⁴', 'w₁'),
    (YF+2.2, 'Scale 2', 'Q₂ × K/V₂', 'F_h²³', 'w₂'),
    (YF+1.4, 'Scale 3', 'Q₃ × K/V₃', 'F_h³²', 'w₃'),
    (YF+0.6, 'Scale 4', 'Q₄ × K/V₄', 'F_h⁴¹', 'w₄'),
]

for sy, sn, qs, fhs, wi in scales:
    box(ax, 12.7, sy, 5.0, 0.6,
        f'{sn}: CrossAttention  →  F_enc',
        C['cross_attn'], fs=7.5)
    box(ax, 18.0, sy, 5.5, 0.6,
        f'F_fuse = (1-{wi})·F_enc + {wi}·F_h',
        C['fuse'], fs=7.5)
    arr(ax, 17.7, sy+0.3, 18.0, sy+0.3)

# Q → cross-attn
arr(ax, 9.3, YE+0.65, 12.7, YF+3.3, C['arrow'], lw=0.7)
arr(ax, 9.3, YE-0.15, 12.7, YF+2.5, C['arrow'], lw=0.7)
arr(ax, 11.8, YE-0.35, 12.7, YF+1.7, C['arrow'], lw=0.7)
arr(ax, 11.8, YE-0.5,  12.7, YF+0.9, C['arrow'], lw=0.7)

# sfe_proj → cross-attn KV
arr(ax, 15.9, YP+1.5, 14.5, YF+3.5, C['struct'], lw=0.8, rad=-0.15)
# F_h → fuse
arr(ax, 7.5, YP+0.5, 18.0, YF+3.3, C['fh'], lw=0.8, rad=0.12)
# W_t → fuse
arr(ax, 20.0, YP+3.0, 20.7, YF+3.5, C['wt'], lw=0.8, rad=-0.05)

# ═══════════ ROW 4: U-NET DECODER ═══════════
YD = 0.1
HD = 2.3

group(ax, 0.15, YD, 23.7, HD, 'U-Net Decoder + F_fuse Injection → ε_pred')

decs = [
    (0.3,  YD+0.2, 4.0, 1.8, 'Mid Block\nResBlock → SelfAttn\n+ F_fuse₈ → ResBlock'),
    (4.6,  YD+0.2, 4.5, 1.8, 'Dec L3\nUp → +F_fuse₁₆\ncat(skip3) → ResBlocks'),
    (9.4,  YD+0.2, 4.5, 1.8, 'Dec L2\nUp → +F_fuse₃₂\ncat(skip2) → ResBlocks'),
    (14.2, YD+0.2, 4.5, 1.8, 'Dec L1\nUp → +F_fuse₆₄\ncat(skip1) → ResBlocks'),
    (19.0, YD+0.2, 4.5, 1.8, 'Dec L0\nUp → cat(skip0)\n→ ResBlocks → ε_pred'),
]

for dx, dy, dw, dh, dl in decs:
    fc = C['output'] if 'ε_pred' in dl else C['unet_dec']
    box(ax, dx, dy, dw, dh, dl, fc, fs=7.5)

for i in range(4):
    arr(ax, decs[i][0]+decs[i][2], YD+1.1, decs[i+1][0], YD+1.1, C['arrow'])

# F_fuse injection
arr(ax, 23.5, YF+3.3, 4.3, YD+2.0, C['fuse'], lw=0.6, rad=-0.35)
arr(ax, 23.5, YF+2.5, 6.9, YD+2.0, C['fuse'], lw=0.6, rad=-0.20)
arr(ax, 23.5, YF+1.7, 11.7, YD+2.0, C['fuse'], lw=0.6, rad=-0.10)
arr(ax, 23.5, YF+0.9, 16.5, YD+2.0, C['fuse'], lw=0.6, rad=-0.05)

# Skip connections
for i, sy in enumerate([YE+2.25, YE+1.45, YE+0.65, YE-0.15]):
    arr(ax, 9.0, sy, decs[i+1][0]+2.2, YD+1.8, C['skip'], lw=0.6, ls='--')

# ═══════════ LOSS ═══════════
box(ax, 6.5, YD-0.12, 11.0, 0.45,
    'Loss: MSE(ε_pred, ε)  →  Backprop  (SFE frozen)',
    C['loss'], fs=8.5, fw='bold')
arr(ax, 21.2, YD+0.2, 12.0, YD+0.1, C['arrow'])

# ═══════════ LEGEND ═══════════
LY = 16.3
items = [('F_h',C['fh']),('sfe_proj',C['struct']),('W_t',C['wt']),
         ('x_t',C['x_t']),('UNet Enc',C['unet_enc']),('UNet Dec',C['unet_dec']),
         ('CrossAttn+Fuse',C['fuse']),('Skip',C['skip']),('SFE (Frozen)',C['sfe']),
         ('Loss',C['loss'])]
for i,(lb,co) in enumerate(items):
    lx = 14.0 + (i%5)*2.0
    ly_i = LY - (i//5)*0.45
    box(ax, lx, ly_i, 1.7, 0.32, '', co, fs=4)
    ax.text(lx+1.8, ly_i+0.16, lb, fontsize=6.5, va='center', color=C['text'])

box(ax, 0.2, 15.8, 13.0, 0.6,
    'Inference (Stage 2): ① Extract F_h, sfe_proj once  ② DDIM denoising (50 steps)  →  clean image',
    '#E8F5E9', fs=7.5)

# ═══════════ SAVE ═══════════
out = 'd:/lyx/fadnet/fadnet_architecture.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=C['bg'],
            edgecolor='none', pad_inches=0.2)
print(f'Saved: {out}  ({fig.get_size_inches()[0]:.0f}×{fig.get_size_inches()[1]:.0f} in)')
plt.close()
