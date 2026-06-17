"""
FAD-Net OCT Dataset Loader
============================
Paper: FAD-Net, IEEE TIM 2026, Section IV-A2

    "The images in both the training and testing sets have been resized to 512x512."

Dataset specifications:
    - Grayscale (single channel)
    - Resized to 512x512 (bilinear interpolation)
    - Normalized to [-1, 1]
    - Supports: png, jpg, jpeg, bmp, tif, tiff

Usage:
    ds = OCTDataset('path/to/images/')
    loader = DataLoader(ds, batch_size=2, shuffle=True)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path


class OCTDataset(Dataset):
    """
    PyTorch Dataset for OCT/OCTA grayscale images.

    Recursively finds all supported image files under data_dir,
    loads as grayscale, resizes to 512x512, normalizes to [-1, 1].

    Args:
        data_dir:    path to directory containing OCT images
        target_size: target spatial size (default 512, paper Section IV-A2)
        exts:        set of supported file extensions
    """

    def __init__(self, data_dir: str, target_size: int = 512,
                 exts: tuple = None):
        self.data_dir = Path(data_dir)
        self.target_size = target_size

        if exts is None:
            exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

        # Collect all image paths
        self.paths = []
        for pattern in exts:
            self.paths.extend(sorted(self.data_dir.rglob(pattern)))

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir} "
                f"with extensions: {exts}"
            )

        # Transform pipeline:
        #   PIL (L mode) -> Resize(512) -> ToTensor  [0,1] -> Normalize [-1,1]
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                                 # [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5]),           # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            tensor of shape [1, 512, 512], dtype float32, range [-1, 1]
        """
        # Load as grayscale (L mode = single channel)
        img = PILImage.open(self.paths[idx]).convert('L')

        # Transform: [H,W] PIL -> [1, H, W] tensor in [-1, 1]
        tensor = self.transform(img)  # [1, 512, 512]

        return tensor

    def get_path(self, idx: int) -> str:
        """Return the file path for the given index."""
        return str(self.paths[idx])


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    import os

    print("=" * 60)
    print("FAD-Net OCTDataset — Test")
    print("=" * 60)

    # ---- Locate data directory ----
    # Use the fad-net folder as default (contains OCTA images)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir  # Images are in same folder as this script

    print(f"\nData directory: {data_dir}")

    # ---- Create dataset ----
    try:
        ds = OCTDataset(data_dir, target_size=512)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"  Place your OCTA images (.png/.jpg/.bmp) in: {data_dir}")
        exit(1)

    print(f"  Images found: {len(ds)}")
    if len(ds) > 0:
        print(f"  Example paths:")
        for i in range(min(3, len(ds))):
            print(f"    [{i}] {ds.get_path(i)}")

    # ---- Test 1: Single sample ----
    print(f"\n{'─' * 60}")
    print("Test 1: Single Sample")
    print(f"{'─' * 60}")

    sample = ds[0]
    print(f"  Shape:      {list(sample.shape)}")
    print(f"  dtype:      {sample.dtype}")
    print(f"  device:     {sample.device}")
    print(f"  min:        {sample.min().item():.4f}")
    print(f"  max:        {sample.max().item():.4f}")
    print(f"  mean:       {sample.mean().item():.4f}")
    print(f"  std:        {sample.std().item():.4f}")
    print(f"  channels:   {sample.shape[0]} (single-channel grayscale)")

    # ---- Test 2: Batch loading with DataLoader ----
    print(f"\n{'─' * 60}")
    print("Test 2: DataLoader (batch_size=2)")
    print(f"{'─' * 60}")

    batch_size = min(2, len(ds))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    batch = next(iter(loader))
    print(f"  Batch shape:    {list(batch.shape)}")
    print(f"  Expected:       [{batch_size}, 1, 512, 512]")
    print(f"  Shape correct:  {list(batch.shape) == [batch_size, 1, 512, 512]}")
    print(f"  Batch min:      {batch.min().item():.4f}")
    print(f"  Batch max:      {batch.max().item():.4f}")
    print(f"  Batch mean:     {batch.mean().item():.4f}")
    print(f"  All in [-1,1]:  {batch.min() >= -1.0 and batch.max() <= 1.0}")

    # ---- Test 3: Iteration ----
    print(f"\n{'─' * 60}")
    print("Test 3: Full Iteration")
    print(f"{'─' * 60}")

    from tqdm import tqdm
    total = 0
    for batch in tqdm(loader, desc="  Iterating"):
        assert batch.shape == (batch_size, 1, 512, 512), \
            f"Unexpected shape: {batch.shape}"
        total += batch.size(0)
    print(f"  Total samples iterated: {total}")
    print(f"  All batches correct shape: True")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("OCTDataset Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    Dataset:
        OCTDataset(data_dir, target_size=512)

    Transform pipeline:
        PIL (L mode) -> Resize(512, bilinear) -> ToTensor [0,1] -> Normalize [-1,1]

    Output:
        [1, 512, 512]  float32  range [-1, 1]

    DataLoader example:
        ds = OCTDataset('/path/to/oct/images/')
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4)

    Supported formats:
        .png  .jpg  .jpeg  .bmp  .tif  .tiff

    Paper reference (Section IV-A2):
        "The images in both the training and testing sets have been
         resized to 512x512."
    """)
