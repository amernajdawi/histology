#!/usr/bin/env python3
"""
FastGlioma Grad-CAM Pipeline - Final Correct Implementation

Generates smooth, accurate Grad-CAM heatmaps for tumor infiltration detection
using the official FastGlioma foundation model.

Usage:
    python run_fastglioma_gradcam_final.py \
        --strips-root MUV_0635-2 \
        --histo-root MUV_0635 \
        --out outputs_final

References:
- FastGlioma paper: https://www.nature.com/articles/s41586-024-08169-3
- Grad-CAM paper: https://arxiv.org/abs/1610.02391  
- Model: https://huggingface.co/mlinslab/fastglioma
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grad_cam_correct import GradCAMForMIL, assemble_heatmap, overlay_heatmap
from src.model_loader import load_official_fastglioma


def load_dicom_image(path: str) -> np.ndarray:
    """Load DICOM and normalize to [0, 1]."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def image_to_tensor(img: np.ndarray, num_channels: int = 3) -> torch.Tensor:
    """Convert grayscale image to tensor [1, C, H, W]."""
    if img.ndim == 2:
        if num_channels == 3:
            img = np.stack([img, img, img], axis=0)  # [3, H, W]
        else:
            img = img[np.newaxis, :, :]  # [1, H, W]
    
    return torch.from_numpy(img).unsqueeze(0)  # [1, C, H, W]


def extract_patches(
    image_tensor: torch.Tensor,
    patch_size: int = 224
) -> tuple:
    """
    Extract non-overlapping patches from image.
    
    Returns:
        patches: List of [C, H, W] tensors
        coords: List of (row, col) coordinates
        shape: (H, W) original shape
    """
    _, C, H, W = image_tensor.shape
    patches = []
    coords = []
    
    for row in range(0, H, patch_size):
        for col in range(0, W, patch_size):
            # Calculate patch bounds
            row_end = min(row + patch_size, H)
            col_end = min(col + patch_size, W)
            
            # Extract patch
            patch = image_tensor[0, :, row:row_end, col:col_end]  # [C, h, w]
            
            # Pad to patch_size if needed
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch = torch.nn.functional.pad(
                    patch, (0, pad_w, 0, pad_h), mode='constant', value=0.0
                )
            
            patches.append(patch)  # [C, 224, 224]
            coords.append((row, col))
    
    # Handle edge case
    if len(patches) == 0:
        patch = torch.nn.functional.interpolate(
            image_tensor, size=(patch_size, patch_size),
            mode='bilinear', align_corners=False
        )[0]
        patches = [patch]
        coords = [(0, 0)]
    
    return patches, coords, (H, W)


def find_histology_image(histo_root: str) -> str:
    """Find first DICOM file in histology root."""
    candidates = glob.glob(os.path.join(histo_root, "**", "*.dcm"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No DICOM found in {histo_root}")
    return sorted(candidates)[0]


def iter_strip_dirs(strips_root: str):
    """Iterate over strip directories."""
    for subdir in sorted(os.listdir(strips_root)):
        strips_dir = os.path.join(strips_root, subdir, "strips")
        if os.path.isdir(strips_dir):
            yield strips_dir, subdir


def main():
    parser = argparse.ArgumentParser(
        description="FastGlioma Grad-CAM - Tumor Infiltration Visualization"
    )
    parser.add_argument("--strips-root", required=True,
                        help="Root dir with strip subdirectories")
    parser.add_argument("--histo-root", required=True,
                        help="Root dir with histology images")
    parser.add_argument("--out", required=True,
                        help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, default=224)
    
    args = parser.parse_args()
    
    # Setup output dirs
    os.makedirs(args.out, exist_ok=True)
    heatmaps_dir = os.path.join(args.out, "heatmaps")
    overlays_dir = os.path.join(args.out, "overlays")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    print("=" * 70)
    print("FastGlioma Grad-CAM Pipeline")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/4] Loading FastGlioma model...")
    model = load_official_fastglioma()
    model = model.to(args.device)
    model.eval()
    
    # Get target layer
    if not (hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4')):
        raise RuntimeError("Cannot find backbone.layer4 in model")
    
    target_layer = model.backbone.layer4[-1]
    print(f"      Target layer: {target_layer.__class__.__name__}")
    
    # Initialize Grad-CAM
    gradcam = GradCAMForMIL(model, target_layer)
    
    # Load histology for overlay reference
    print(f"\n[2/4] Loading histology image from {args.histo_root}...")
    histo_path = find_histology_image(args.histo_root)
    histo_img = load_dicom_image(histo_path)
    histo_rgb = np.stack([histo_img, histo_img, histo_img], axis=-1)
    print(f"      Histology shape: {histo_img.shape}")
    
    # Process all strips
    print(f"\n[3/4] Processing strips...")
    total_processed = 0
    total_errors = 0
    
    for strips_dir, subdir_name in iter_strip_dirs(args.strips_root):
        strip_files = sorted([f for f in os.listdir(strips_dir) 
                              if f.lower().endswith('.dcm')])
        
        print(f"\n    Subdirectory: {subdir_name}")
        print(f"    Strips found: {len(strip_files)}")
        
        for strip_file in tqdm(strip_files, desc=f"    {subdir_name}", ncols=70):
            strip_path = os.path.join(strips_dir, strip_file)
            
            try:
                # Load strip
                strip_img = load_dicom_image(strip_path)
                strip_tensor = image_to_tensor(strip_img, num_channels=3)
                strip_tensor = strip_tensor.to(args.device)
                
                # Extract patches
                patches, coords, (H, W) = extract_patches(
                    strip_tensor, patch_size=args.patch_size
                )
                
                # Prepare coordinates
                coords_tensor = torch.tensor(coords, dtype=torch.float32)
                
                # Generate Grad-CAM
                with torch.set_grad_enabled(True):
                    patch_cams, score = gradcam.generate(
                        patches, coords_tensor, device=args.device
                    )
                
                # Assemble heatmap
                heatmap = assemble_heatmap(
                    patch_cams, coords, (H, W), patch_size=args.patch_size
                )
                
                # Create overlays
                strip_rgb = np.stack([strip_img, strip_img, strip_img], axis=-1)
                overlay_on_strip = overlay_heatmap(strip_rgb, heatmap, alpha=0.4)
                
                # Save
                base_name = strip_file.replace('.dcm', '')
                heatmap_file = f"{base_name}_cam.png"
                overlay_file = f"{base_name}_overlay.png"
                
                heatmap_path = os.path.join(heatmaps_dir, heatmap_file)
                overlay_path = os.path.join(overlays_dir, overlay_file)
                
                cv2.imwrite(heatmap_path, 
                           cv2.cvtColor(overlay_on_strip, cv2.COLOR_RGB2BGR))
                
                # Also save overlay on histology (for reference)
                heatmap_resized = cv2.resize(heatmap, 
                                            (histo_rgb.shape[1], histo_rgb.shape[0]))
                overlay_on_histo = overlay_heatmap(histo_rgb, heatmap_resized, alpha=0.3)
                cv2.imwrite(overlay_path,
                           cv2.cvtColor(overlay_on_histo, cv2.COLOR_RGB2BGR))
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n      Error: {strip_file}: {e}")
                total_errors += 1
                continue
    
    # Summary
    print("\n" + "=" * 70)
    print("[4/4] Summary")
    print("=" * 70)
    print(f"  Successfully processed: {total_processed} strips")
    print(f"  Errors: {total_errors} strips")
    print(f"\n  Output directories:")
    print(f"    Heatmaps: {heatmaps_dir}")
    print(f"    Overlays: {overlays_dir}")
    print("=" * 70)
    print("\n✓ Pipeline completed successfully!")
    

if __name__ == "__main__":
    main()

