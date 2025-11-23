#!/usr/bin/env python3
"""
Overlay Grad-CAM Heatmaps on H&E Histopathology Image

This script overlays heatmaps directly on a single H&E image (like the one Thomas sent).
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grad_cam_correct import overlay_heatmap


def parse_strip_number(filename: str) -> int:
    """Extract strip number from filename.
    
    Example: 'img1_7_*.png' -> 7
    Example: 'img2_10_*.png' -> 10
    """
    match = re.search(r'img\d+_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def load_he_image(he_path: str) -> np.ndarray:
    """Load H&E image and normalize to [0, 1]."""
    img = cv2.imread(he_path)
    if img is None:
        raise FileNotFoundError(f"Could not load H&E image: {he_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def overlay_all_heatmaps_on_he(
    he_image: np.ndarray,
    heatmaps_dir: str,
    output_path: str,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay all Grad-CAM heatmaps on H&E image.
    
    Args:
        he_image: H&E image [H, W, 3] in [0, 1]
        heatmaps_dir: Directory containing heatmap PNG files
        output_path: Path to save output
        alpha: Blending weight for heatmap (default 0.4)
    
    Returns:
        Overlayed image [H, W, 3]
    """
    heatmap_files = glob.glob(os.path.join(heatmaps_dir, "*_cam.png"))
    
    if not heatmap_files:
        raise FileNotFoundError(f"No heatmap files found in {heatmaps_dir}")
    
    print(f"Found {len(heatmap_files)} heatmap files")
    
    print("Combining heatmaps...")
    combined_heatmap = None
    count = 0
    
    for heatmap_path in tqdm(sorted(heatmap_files), desc="Loading heatmaps"):
        # Load heatmap
        heatmap_img = cv2.imread(heatmap_path, cv2.IMREAD_COLOR)
        if heatmap_img is None:
            print(f"  Warning: Could not load {heatmap_path}")
            continue
        
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        
        heatmap_resized = cv2.resize(heatmap_img, (he_image.shape[1], he_image.shape[0]))
        
        heatmap_normalized = heatmap_resized.astype(np.float32) / 255.0
        
        heatmap_gray = np.mean(heatmap_normalized, axis=2)
        
        if combined_heatmap is None:
            combined_heatmap = heatmap_gray.copy()
        else:
            combined_heatmap = combined_heatmap + heatmap_gray
        count += 1
    
    if combined_heatmap is None:
        raise ValueError("No valid heatmaps found")
    
    combined_heatmap = combined_heatmap / count
    
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
    
    print("Overlaying combined heatmap...")
    result = overlay_heatmap(he_image, combined_heatmap, alpha=alpha)
    
    result_uint8 = (result * 255).astype(np.uint8)
    
    # Save
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   Dimensions: {result_uint8.shape[1]} x {result_uint8.shape[0]}")
    
    return result_uint8


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Grad-CAM heatmaps on H&E histopathology image"
    )
    parser.add_argument("--he-image", required=True,
                        help="Path to H&E histopathology image (PNG)")
    parser.add_argument("--heatmaps-dir", required=True,
                        help="Directory containing Grad-CAM heatmap PNG files")
    parser.add_argument("--out", required=True,
                        help="Output path for overlay image")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Blending weight for heatmap (0.0-1.0, default: 0.1 = 10% heatmap, 90% image)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Overlaying Grad-CAM Heatmaps on H&E Histopathology Image")
    print("=" * 70)
    
    # Load H&E image
    print(f"\nLoading H&E image: {args.he_image}")
    he_image = load_he_image(args.he_image)
    print(f"   Dimensions: {he_image.shape[1]} x {he_image.shape[0]}")
    
    # Overlay heatmaps
    print(f"\nOverlaying heatmaps from: {args.heatmaps_dir}")
    overlay_image = overlay_all_heatmaps_on_he(
        he_image, args.heatmaps_dir, args.out, alpha=args.alpha
    )
    
    print("\n" + "=" * 70)
    print("✅ Overlay complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

