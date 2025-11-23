#!/usr/bin/env python3
"""
Stitch Histopathology Images and Overlay Grad-CAM Heatmaps

Based on supervisor feedback:
- Histopathology image is a stitching of all strips
- 50 pixel overlap between consecutive strips
- Strip location is in the filename (img1_7, img1_8, etc.)
- Overlay all heatmaps on the stitched histopathology image
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grad_cam_correct import overlay_heatmap


def parse_strip_number(filename: str) -> int:
    """Extract strip number from filename.
    
    Example: 'img1_7_*.dcm' -> 7
    Example: 'img2_10_*.dcm' -> 10
    """
    match = re.search(r'img\d+_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_image(path: str) -> np.ndarray:
    """Load DICOM and normalize to [0, 1]."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def stitch_histopathology_strips(
    histo_dir: str,
    overlap: int = 50,
    required_strip_nums: set = None
) -> tuple:
    """
    Stitch all histopathology strips together.
    
    Args:
        histo_dir: Directory containing histopathology DICOM files
        overlap: Pixel overlap between consecutive strips (default 50)
        required_strip_nums: Set of strip numbers to include (if None, include all)
    
    Returns:
        stitched_image: Stitched histopathology image [H, W]
        strip_positions: Dict mapping strip_number -> (y_start, y_end, x_start, x_end)
    """
    # Find all DICOM files
    dcm_files = glob.glob(os.path.join(histo_dir, "**", "*.dcm"), recursive=True)
    
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {histo_dir}")
    
    # Sort by strip number
    files_with_numbers = [(f, parse_strip_number(os.path.basename(f))) for f in dcm_files]
    
    # Filter to only required strips if specified
    if required_strip_nums is not None:
        files_with_numbers = [(f, n) for f, n in files_with_numbers if n in required_strip_nums]
        print(f"Filtering to {len(required_strip_nums)} required strips: {sorted(required_strip_nums)}")
    
    files_with_numbers.sort(key=lambda x: x[1])  # Sort by strip number
    
    print(f"Found {len(files_with_numbers)} histopathology strips to stitch")
    
    # Load first strip to get dimensions
    first_img = load_dicom_image(files_with_numbers[0][0])
    if first_img.ndim == 2:
        H_strip, W_strip = first_img.shape
    elif first_img.ndim == 3:
        H_strip, W_strip = first_img.shape[:2]
    else:
        raise ValueError(f"Unexpected image shape: {first_img.shape}")
    
    print(f"Strip dimensions: {H_strip} x {W_strip}")
    
    # Calculate stitched image dimensions
    # Strips are tall and narrow (4000x1000), so stitch HORIZONTALLY (side by side)
    num_strips = len(files_with_numbers)
    # Each strip contributes W_strip, but we subtract overlap except for first
    total_width = W_strip + (num_strips - 1) * (W_strip - overlap)
    
    # Height should be the same (assuming strips are aligned)
    total_height = H_strip
    
    print(f"Stitched dimensions: {total_height} x {total_width}")
    
    # Create stitched image (handle RGB if needed)
    if first_img.ndim == 3:
        stitched = np.zeros((total_height, total_width, 3), dtype=np.float32)
    else:
        stitched = np.zeros((total_height, total_width), dtype=np.float32)
    
    strip_positions = {}
    
    # Place each strip HORIZONTALLY (side by side)
    current_x = 0
    for filepath, strip_num in tqdm(files_with_numbers, desc="Stitching strips"):
        img = load_dicom_image(filepath)
        
        # Get actual dimensions
        if img.ndim == 3:
            img_h, img_w = img.shape[:2]
            is_rgb = True
        else:
            img_h, img_w = img.shape
            is_rgb = False
        
        # Resize to match expected height (H_strip), keep aspect ratio or use W_strip
        if img_h != H_strip:
            scale = H_strip / img_h
            new_w = int(img_w * scale)
            if is_rgb:
                img = cv2.resize(img, (new_w, H_strip), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (new_w, H_strip), interpolation=cv2.INTER_LINEAR)
            img_w = new_w
            img_h = H_strip
        
        # Use W_strip as target width (or actual width if smaller)
        target_w = min(W_strip, img_w)
        
        # Ensure grayscale/RGB format matches stitched image
        if is_rgb and stitched.ndim == 2:
            img = img.mean(axis=2)  # Convert RGB to grayscale
            is_rgb = False
        elif not is_rgb and stitched.ndim == 3:
            img = np.stack([img, img, img], axis=2)  # Convert grayscale to RGB
            is_rgb = True
        
        # Crop or pad to target width
        if img_w > target_w:
            img = img[:, :target_w]
        elif img_w < target_w:
            if is_rgb:
                pad = np.zeros((H_strip, target_w - img_w, 3), dtype=img.dtype)
            else:
                pad = np.zeros((H_strip, target_w - img_w), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=1)
        
        # Calculate position (horizontal stitching)
        x_start = current_x
        x_end = current_x + target_w
        
        # Ensure we don't go out of bounds
        if x_end > total_width:
            x_end = total_width
            img = img[:, :x_end - x_start]
        
        # Place strip (handle overlap by averaging)
        if current_x > 0 and x_start + overlap <= total_width:
            # Overlap region: average with existing content
            overlap_end = min(x_start + overlap, total_width)
            overlap_width = overlap_end - x_start
            
            if overlap_width > 0:
                overlap_region = stitched[:, x_start:overlap_end]
                new_overlap = img[:, :overlap_width]
                
                # Ensure shapes match
                if overlap_region.shape == new_overlap.shape:
                    stitched[:, x_start:overlap_end] = (overlap_region + new_overlap) / 2.0
                else:
                    # Resize new_overlap to match if needed
                    if new_overlap.shape != overlap_region.shape:
                        if is_rgb:
                            new_overlap = cv2.resize(new_overlap, 
                                                    (overlap_width, overlap_region.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                        else:
                            new_overlap = cv2.resize(new_overlap, 
                                                    (overlap_width, overlap_region.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                    stitched[:, x_start:overlap_end] = (overlap_region + new_overlap) / 2.0
        
        # Place non-overlapping part
        if x_start + overlap < total_width:
            end_col = min(x_end, total_width)
            start_col = x_start + overlap
            img_start = overlap
            img_end = overlap + (end_col - start_col)
            
            if end_col > start_col and img_end <= img.shape[1]:
                stitched[:, start_col:end_col] = img[:, img_start:img_end]
        
        strip_positions[strip_num] = (0, total_height, x_start, x_end)
        
        # Move to next position
        current_x += target_w - overlap
    
    return stitched, strip_positions


def overlay_heatmaps_on_stitched(
    stitched_histo: np.ndarray,
    heatmaps_dir: str,
    strip_positions: dict,
    overlap: int = 50
) -> np.ndarray:
    """
    Overlay all Grad-CAM heatmaps on stitched histopathology.
    
    Args:
        stitched_histo: Stitched histopathology image [H, W] or [H, W, C]
        heatmaps_dir: Directory containing heatmap PNG files
        strip_positions: Dict mapping strip_number -> (y_start, y_end, x_start, x_end)
        overlap: Pixel overlap between strips
    
    Returns:
        Overlayed image [H, W, 3]
    """
    # Convert to RGB
    if stitched_histo.ndim == 2:
        stitched_rgb = np.stack([stitched_histo, stitched_histo, stitched_histo], axis=-1)
    elif stitched_histo.ndim == 3:
        if stitched_histo.shape[2] == 3:
            stitched_rgb = stitched_histo.copy()
        else:
            stitched_rgb = np.stack([stitched_histo[:,:,0], stitched_histo[:,:,0], stitched_histo[:,:,0]], axis=-1)
    else:
        raise ValueError(f"Unexpected stitched_histo shape: {stitched_histo.shape}")
    
    # Ensure it's float32 and 3D
    stitched_rgb = stitched_rgb.astype(np.float32)
    if stitched_rgb.ndim != 3:
        raise ValueError(f"stitched_rgb should be 3D, got {stitched_rgb.ndim}D with shape {stitched_rgb.shape}")
    
    # Find all heatmap files
    heatmap_files = glob.glob(os.path.join(heatmaps_dir, "*_cam.png"))
    
    print(f"Found {len(heatmap_files)} heatmap files")
    
    # Get available strip numbers from heatmaps
    available_strip_nums = set()
    for heatmap_path in heatmap_files:
        filename = os.path.basename(heatmap_path)
        strip_num = parse_strip_number(filename)
        available_strip_nums.add(strip_num)
    
    print(f"Available heatmaps for strips: {sorted(available_strip_nums)}")
    
    # Process each heatmap
    for heatmap_path in tqdm(sorted(heatmap_files), desc="Overlaying heatmaps"):
        filename = os.path.basename(heatmap_path)
        strip_num = parse_strip_number(filename)
        
        if strip_num not in strip_positions:
            print(f"  Warning: Strip {strip_num} not found in positions, skipping")
            continue
        
        # Load heatmap
        heatmap_img = cv2.imread(heatmap_path, cv2.IMREAD_COLOR)
        if heatmap_img is None:
            print(f"  Warning: Could not load {heatmap_path}")
            continue
        
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        
        # Get strip position (horizontal stitching: x_start to x_end)
        y_start, y_end, x_start, x_end = strip_positions[strip_num]
        strip_width = x_end - x_start
        
        # Resize heatmap to match strip dimensions
        # Get original strip dimensions from heatmap (assuming it matches strip)
        heatmap_h, heatmap_w = heatmap_img.shape[:2]
        
        # Resize to match strip width (accounting for overlap)
        target_width = strip_width
        target_height = int(heatmap_h * (target_width / heatmap_w))
        
        heatmap_resized = cv2.resize(heatmap_img, (target_width, target_height))
        
        # Ensure heatmap is RGB (3 channels)
        if heatmap_resized.ndim == 2:
            heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_GRAY2RGB)
        elif heatmap_resized.ndim == 4:
            heatmap_resized = heatmap_resized.squeeze()
        if heatmap_resized.ndim != 3:
            print(f"  Warning: Unexpected heatmap shape {heatmap_resized.shape}, skipping")
            continue
        
        # Ensure we don't exceed bounds
        if x_start + target_width > stitched_rgb.shape[1]:
            target_width = stitched_rgb.shape[1] - x_start
            heatmap_resized = heatmap_resized[:, :target_width]
        
        if y_start + target_height > stitched_rgb.shape[0]:
            target_height = stitched_rgb.shape[0] - y_start
            heatmap_resized = heatmap_resized[:target_height, :]
        
        # Ensure dimensions match
        actual_h, actual_w = heatmap_resized.shape[:2]
        region_h = min(actual_h, stitched_rgb.shape[0] - y_start)
        region_w = min(actual_w, stitched_rgb.shape[1] - x_start)
        
        # Extract region and heatmap
        region = stitched_rgb[y_start:y_start + region_h, 
                              x_start:x_start + region_w].copy()
        heatmap_region = heatmap_resized[:region_h, :region_w].copy()
        
        # Ensure both are 3D (H, W, C)
        if region.ndim == 4:
            region = region.squeeze()
        if heatmap_region.ndim == 4:
            heatmap_region = heatmap_region.squeeze()
        if region.ndim == 2:
            region = np.stack([region, region, region], axis=2)
        if heatmap_region.ndim == 2:
            heatmap_region = np.stack([heatmap_region, heatmap_region, heatmap_region], axis=2)
        
        # Ensure shapes match exactly
        if region.shape != heatmap_region.shape:
            print(f"  Warning: Shape mismatch - region {region.shape} vs heatmap {heatmap_region.shape}, resizing heatmap")
            heatmap_region = cv2.resize(heatmap_region, (region.shape[1], region.shape[0]))
        
        # Blend
        alpha = 0.4
        blended = (alpha * heatmap_region.astype(np.float32) + 
                   (1 - alpha) * region.astype(np.float32))
        
        # Place back
        stitched_rgb[y_start:y_start + region_h, 
                     x_start:x_start + region_w] = blended.astype(np.uint8)
    
    return stitched_rgb.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch histopathology and overlay Grad-CAM heatmaps"
    )
    parser.add_argument("--histo-root", required=True,
                        help="Root directory with histopathology images")
    parser.add_argument("--heatmaps-dir", required=True,
                        help="Directory with Grad-CAM heatmap PNG files")
    parser.add_argument("--out", required=True,
                        help="Output path for stitched overlay")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Pixel overlap between strips (default: 50)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Stitching Histopathology and Overlaying Grad-CAM Heatmaps")
    print("=" * 70)
    
    # First, find available heatmap strips
    heatmap_files = glob.glob(os.path.join(args.heatmaps_dir, "*_cam.png"))
    available_strip_nums = set()
    for heatmap_path in heatmap_files:
        filename = os.path.basename(heatmap_path)
        strip_num = parse_strip_number(filename)
        available_strip_nums.add(strip_num)
    
    if not available_strip_nums:
        raise FileNotFoundError(f"No heatmap files found in {args.heatmaps_dir}")
    
    print(f"\nAvailable heatmaps for strips: {sorted(available_strip_nums)}")
    print(f"Will stitch only histopathology strips that have corresponding heatmaps")
    
    # Find histopathology directory
    # Check if histo_root contains DICOMs directly
    dcm_files_direct = glob.glob(os.path.join(args.histo_root, "*.dcm"))
    if dcm_files_direct:
        histo_dirs = [args.histo_root]
    else:
        # Check subdirectories
        histo_dirs = []
        for subdir in sorted(os.listdir(args.histo_root)):
            histo_path = os.path.join(args.histo_root, subdir)
            if os.path.isdir(histo_path):
                dcm_files = glob.glob(os.path.join(histo_path, "*.dcm"))
                if dcm_files:
                    histo_dirs.append(histo_path)
    
    if not histo_dirs:
        raise FileNotFoundError(f"No histopathology DICOM files found in {args.histo_root}")
    
    print(f"\nFound {len(histo_dirs)} histopathology subdirectories")
    
    # Process each subdirectory
    for histo_dir in histo_dirs:
        subdir_name = os.path.basename(histo_dir)
        print(f"\nProcessing: {subdir_name}")
        
        # Stitch histopathology strips (only those with heatmaps)
        print("  Stitching histopathology strips...")
        stitched_histo, strip_positions = stitch_histopathology_strips(
            histo_dir, overlap=args.overlap, required_strip_nums=available_strip_nums
        )
        
        # Overlay heatmaps
        print("  Overlaying Grad-CAM heatmaps...")
        stitched_overlay = overlay_heatmaps_on_stitched(
            stitched_histo, args.heatmaps_dir, strip_positions, overlap=args.overlap
        )
        
        # Save
        output_path = args.out.replace(".png", f"_{subdir_name}.png")
        cv2.imwrite(output_path, cv2.cvtColor(stitched_overlay, cv2.COLOR_RGB2BGR))
        
        print(f"  ✅ Saved: {output_path}")
        print(f"  Dimensions: {stitched_overlay.shape[1]} x {stitched_overlay.shape[0]}")
    
    print("\n" + "=" * 70)
    print("✅ Stitching complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

