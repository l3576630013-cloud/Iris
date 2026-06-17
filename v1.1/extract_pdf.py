import sys
sys.stdout.reconfigure(encoding='utf-8')

from pypdf import PdfReader

reader = PdfReader(r'C:\Users\YZ\Desktop\资料\比赛\FAD-Net_Unsupervised_Frequency-Aware_Diffusion_Network_for_Optical_Coherence_Tomography_Speckle_Reduction.pdf')
print(f'Total pages: {len(reader.pages)}')
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        print(f'\n===== PAGE {i+1} =====')
        print(text)
