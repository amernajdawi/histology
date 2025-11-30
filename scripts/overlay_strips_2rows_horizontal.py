#!/usr/bin/env python3
"""
Overlay Grad-CAM Heatmap Strips on H&E Image in 2 Rows (Horizontal Layout)

Arranges all heatmap strips horizontally in 2 rows:
- Divides all strips evenly between row 1 and row 2
- Resizes strips to fit perfectly within the image dimensions
- Overlays strips on the histopathology image
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grad_cam_correct import overlay_heatmap


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


def overlay_strip_on_region(
    he_image: np.ndarray,
    heatmap_path: str,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a single heatmap strip on a specific region of the H&E image.
    
    Args:
        he_image: H&E image [H, W, 3] in [0, 1]
        heatmap_path: Path to heatmap PNG file
        x_start, x_end: Horizontal region bounds
        y_start, y_end: Vertical region bounds
        alpha: Blending weight for heatmap
    
    Returns:
        Modified H&E image with overlay
    """
    # Load heatmap
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img is None:
        print(f"  Warning: Could not load {heatmap_path}")
        return he_image
    
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    # Calculate region dimensions
    region_h = y_end - y_start
    region_w = x_end - x_start
    
    # Resize heatmap to fit the region perfectly
    heatmap_resized = cv2.resize(heatmap_img, (region_w, region_h))
    heatmap_norm = heatmap_resized.astype(np.float32) / 255.0
    
    # Extract grayscale heatmap
    heatmap_gray = np.mean(heatmap_norm, axis=2)
    
    # Normalize heatmap
    if heatmap_gray.max() > 1.0:
        heatmap_gray = heatmap_gray / heatmap_gray.max()
    
    # Apply threshold to show only high-activation areas
    threshold = np.percentile(heatmap_gray, 95)
    heatmap_thresholded = np.where(heatmap_gray > threshold, heatmap_gray, 0.0)
    
    # Normalize thresholded heatmap
    if heatmap_thresholded.max() > 0:
        heatmap_thresholded = (heatmap_thresholded - heatmap_thresholded.min()) / (heatmap_thresholded.max() - heatmap_thresholded.min() + 1e-8)
    else:
        return he_image
    
    # Apply HOT colormap (black -> red -> yellow)
    heatmap_uint8 = (heatmap_thresholded * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0
    
    # Create mask for high-activation areas
    mask = (heatmap_gray > threshold).astype(np.float32)[:, :, np.newaxis]
    
    # Extract region from H&E image
    region = he_image[y_start:y_end, x_start:x_end].copy()
    
    # Blend: keep H&E everywhere, add heatmap only in high-activation areas
    blended_region = region * (1 - mask * alpha) + heatmap_color * (mask * alpha)
    
    # Place back into H&E image
    he_image[y_start:y_end, x_start:x_end] = blended_region
    
    return he_image


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Grad-CAM heatmap strips on H&E image in 2 horizontal rows"
    )
    parser.add_argument("--he-image", required=True,
                        help="Path to H&E histopathology image (PNG)")
    parser.add_argument("--heatmaps-dir", required=True,
                        help="Directory containing Grad-CAM heatmap PNG files")
    parser.add_argument("--out", required=True,
                        help="Output path for overlay image")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Blending weight for heatmap (0.0-1.0, default: 0.4)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Overlaying Grad-CAM Heatmap Strips on H&E Image (2 Rows)")
    print("=" * 70)
    
    # Load H&E image
    print(f"\nLoading H&E image: {args.he_image}")
    he_image = load_he_image(args.he_image)
    he_h, he_w = he_image.shape[:2]
    print(f"   Dimensions: {he_w} x {he_h} (width x height)")
    
    # Find all heatmap files
    print(f"\nFinding heatmap files in: {args.heatmaps_dir}")
    heatmap_files = sorted(glob.glob(os.path.join(args.heatmaps_dir, "*_cam.png")))
    
    if not heatmap_files:
        raise FileNotFoundError(f"No heatmap files found in {args.heatmaps_dir}")
    
    print(f"Found {len(heatmap_files)} heatmap files")
    
    # Divide strips into 2 rows
    num_rows = 2
    total_strips = len(heatmap_files)
    strips_per_row = (total_strips + 1) // 2  # Round up for first row
    
    print(f"\nLayout:")
    print(f"  Total strips: {total_strips}")
    print(f"  Row 1: {strips_per_row} strips")
    print(f"  Row 2: {total_strips - strips_per_row} strips")
    
    # Calculate dimensions for each strip
    row_height = he_h // num_rows
    strip_width = he_w // strips_per_row
    
    print(f"  Row height: {row_height}px")
    print(f"  Strip width: {strip_width}px (per strip in row 1)")
    
    # Create result image
    result = he_image.copy()
    
    # Process Row 1 (first half of strips)
    print(f"\nProcessing Row 1 ({strips_per_row} strips)...")
    for idx in tqdm(range(strips_per_row), desc="Row 1"):
        heatmap_path = heatmap_files[idx]
        
        y_start = 0
        y_end = row_height
        x_start = idx * strip_width
        x_end = min((idx + 1) * strip_width, he_w)
        
        result = overlay_strip_on_region(
            result, heatmap_path, x_start, x_end, y_start, y_end, alpha=args.alpha
        )
    
    # Process Row 2 (second half of strips)
    strips_row2 = total_strips - strips_per_row
    if strips_row2 > 0:
        strip_width_row2 = he_w // strips_row2
        print(f"\nProcessing Row 2 ({strips_row2} strips)...")
        for idx in tqdm(range(strips_row2), desc="Row 2"):
            heatmap_path = heatmap_files[strips_per_row + idx]
            
            y_start = row_height
            y_end = he_h
            x_start = idx * strip_width_row2
            x_end = min((idx + 1) * strip_width_row2, he_w)
            
            result = overlay_strip_on_region(
                result, heatmap_path, x_start, x_end, y_start, y_end, alpha=args.alpha
            )
    
    # Convert back to uint8 and save
    result_uint8 = (result * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, result_bgr)
    
    print("\n" + "=" * 70)
    print(f"✅ Saved: {args.out}")
    print(f"   Dimensions: {result_uint8.shape[1]} x {result_uint8.shape[0]}")
    print(f"   Total strips overlaid: {total_strips} in 2 rows")
    print("=" * 70)


if __name__ == "__main__":
    main()

