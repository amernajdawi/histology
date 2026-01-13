#!/usr/bin/env python3
"""
Step 4: Reassemble Grad-CAM patches into full heatmap image

This script takes Grad-CAM patches (already computed) and reassembles them
to match the stitched image dimensions (3600 x 3900).

Usage:
    python step4_reassemble_gradcam.py [patches_dir] [output_dir] [--target-size WIDTH HEIGHT]

Arguments:
    patches_dir: Directory containing Grad-CAM patch files (default: MUV_0635-2/01/patches)
    output_dir: Output directory for reassembled heatmaps (default: gradcam_heatmaps)
    --target-size: Target image dimensions WIDTH HEIGHT (default: 3600 3900)
"""

import sys
import argparse
from pathlib import Path
import re
from PIL import Image
import numpy as np


def parse_patch_coordinates(filename):
    """Parse coordinates from patch filename.
    
    Format: MUV_0635-ALA-01-<strip_x>-<strip_y>-<patch_x>-<patch_y>.tif
    Example: MUV_0635-ALA-01-0-1000-300-600.tif
      - strip_x = 0 (horizontal strip offset)
      - strip_y = 1000 (vertical strip offset)
      - patch_x = 300 (x coordinate within strip)
      - patch_y = 600 (y coordinate within strip)
    
    Absolute position = (strip_x + patch_x, strip_y + patch_y)
    
    Returns: (absolute_x, absolute_y)
    """
    parts = filename.stem.split('-')
    if len(parts) >= 7:
        try:
            strip_x = int(parts[3])  # Horizontal strip offset (0, 1000, 2000, 3000)
            strip_y = int(parts[4])  # Vertical strip offset (0, 1000, 2000, 3000)
            patch_x = int(parts[5])  # X coordinate within strip (0, 300, 600)
            patch_y = int(parts[6])  # Y coordinate within strip (0, 300, 600)
            
            # Calculate absolute position
            absolute_x = strip_x + patch_x
            absolute_y = strip_y + patch_y
            
            return absolute_x, absolute_y
        except ValueError:
            return None, None
    return None, None


def reassemble_patches_from_strips(patches_dir, target_width=3600, target_height=3900, patch_size=300):
    """Reassemble Grad-CAM patches into full heatmap image.
    
    Patches are organized by strip offsets (0, 1000, 2000, 3000, etc.)
    Each strip contains patches with local coordinates (x, y) within that strip.
    """
    patches_dir = Path(patches_dir)
    
    if not patches_dir.exists():
        print(f"✗ ERROR: Patches directory not found: {patches_dir}")
        return None
    
    # Find all patch files - use ALA patches for Grad-CAM (SRH are raw images)
    patch_files = sorted(patches_dir.glob("*ALA*.tif"))
    if not patch_files:
        # Fallback to all patches if no ALA patches found
        patch_files = sorted(patches_dir.glob("*.tif"))
    
    if not patch_files:
        print(f"✗ ERROR: No patch files found in {patches_dir}")
        return None
    
    print(f"Found {len(patch_files)} patch files")
    
    # Parse all patches and get their absolute positions
    patches_by_position = {}
    all_x_coords = set()
    all_y_coords = set()
    
    for patch_file in patch_files:
        abs_x, abs_y = parse_patch_coordinates(patch_file)
        if abs_x is not None and abs_y is not None:
            patches_by_position[(abs_x, abs_y)] = patch_file
            all_x_coords.add(abs_x)
            all_y_coords.add(abs_y)
    
    if not patches_by_position:
        print("✗ ERROR: Could not parse patch coordinates")
        return None
    
    print(f"Found {len(patches_by_position)} patches")
    print(f"X range: {min(all_x_coords)} to {max(all_x_coords)}")
    print(f"Y range: {min(all_y_coords)} to {max(all_y_coords)}")
    
    # Load a sample patch to get dimensions
    sample_patch = Image.open(patch_files[0])
    patch_w, patch_h = sample_patch.size
    print(f"Patch size: {patch_w} x {patch_h} pixels")
    
    # Calculate image dimensions from patch positions
    max_x = max(all_x_coords)
    max_y = max(all_y_coords)
    calculated_width = max_x + patch_w
    calculated_height = max_y + patch_h
    
    print(f"Calculated image size: {calculated_width} x {calculated_height}")
    print(f"Target size: {target_width} x {target_height}")
    
    # Create canvas for full image
    full_image = np.zeros((calculated_height, calculated_width, 3), dtype=np.uint8)
    
    # Place each patch at its absolute position
    print(f"\nPlacing {len(patches_by_position)} patches...")
    
    # Normalize each patch individually to preserve relative intensities
    # This is better for sparse heatmaps where each patch has its own intensity range
    patches_placed = 0
    patches_with_data = 0
    
    for (abs_x, abs_y), patch_file in sorted(patches_by_position.items()):
        patch_img = Image.open(patch_file)
        if patch_img.mode != 'RGB':
            patch_img = patch_img.convert('RGB')
        patch_arr = np.array(patch_img)
        
        # Convert to grayscale
        if len(patch_arr.shape) == 3:
            patch_gray = patch_arr.mean(axis=2)
        else:
            patch_gray = patch_arr
        
        # Normalize each patch individually: scale non-zero values to full range
        patch_normalized = np.zeros_like(patch_gray, dtype=np.float32)
        mask = patch_gray > 0
        
        if mask.any():
            patches_with_data += 1
            patch_min = patch_gray[mask].min()
            patch_max = patch_gray[mask].max()
            
            if patch_max > patch_min:
                # Normalize non-zero values to 0-255 range
                patch_normalized[mask] = ((patch_gray[mask] - patch_min) / (patch_max - patch_min) * 255)
            else:
                # All non-zero values are the same, set them to a mid-range value
                patch_normalized[mask] = 128
            
            patch_normalized = np.clip(patch_normalized, 0, 255).astype(np.uint8)
        else:
            # All zeros, keep as is
            patch_normalized = patch_normalized.astype(np.uint8)
        
        # Convert back to RGB for consistency
        patch_rgb = np.stack([patch_normalized, patch_normalized, patch_normalized], axis=2)
        
        # Place patch at absolute position
        full_image[abs_y:abs_y+patch_h, abs_x:abs_x+patch_w] = patch_rgb
        patches_placed += 1
    
    print(f"  ✓ Placed {patches_placed} patches ({patches_with_data} with data)")
    
    print(f"  ✓ Assembled full image: {calculated_width} x {calculated_height}")
    
    # Crop/resize to target dimensions
    current_h, current_w = full_image.shape[:2]
    
    if current_w != target_width or current_h != target_height:
        print(f"\nAdjusting dimensions from {current_w} x {current_h} to {target_width} x {target_height}...")
        
        # Crop or resize to match target
        if current_w > target_width:
            # Crop from center
            crop_left = (current_w - target_width) // 2
            full_image = full_image[:, crop_left:crop_left+target_width]
        elif current_w < target_width:
            # Pad with zeros
            pad_left = (target_width - current_w) // 2
            pad_right = target_width - current_w - pad_left
            full_image = np.pad(full_image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        
        if current_h > target_height:
            # Crop from center
            crop_top = (current_h - target_height) // 2
            full_image = full_image[crop_top:crop_top+target_height, :]
        elif current_h < target_height:
            # Pad with zeros
            pad_top = (target_height - current_h) // 2
            pad_bottom = target_height - current_h - pad_top
            full_image = np.pad(full_image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    final_h, final_w = full_image.shape[:2]
    print(f"  ✓ Final dimensions: {final_w} x {final_h}")
    
    return full_image


def main():
    parser = argparse.ArgumentParser(description='Reassemble Grad-CAM patches into full heatmap')
    parser.add_argument('--patches-dir', type=str, default=None,
                       help='Directory containing patch files (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default='gradcam_heatmaps',
                       help='Output directory for reassembled heatmaps (default: gradcam_heatmaps)')
    parser.add_argument('--target-width', type=int, default=3600,
                       help='Target image width (default: 3600)')
    parser.add_argument('--target-height', type=int, default=3900,
                       help='Target image height (default: 3900)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 4: Reassemble Grad-CAM Patches (FLEXIBLE)")
    print("="*80)
    
    # Auto-detect patches directory if not provided
    if args.patches_dir is None:
        # Look for patches directories in common locations
        base_dir = Path('.')
        possible_dirs = []
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                for series_dir in dataset_dir.iterdir():
                    if series_dir.is_dir() and series_dir.name.isdigit():
                        patches_dir = series_dir / "patches"
                        if patches_dir.exists() and any(patches_dir.glob("*.tif")):
                            possible_dirs.append((dataset_dir.name, series_dir.name, patches_dir))
        
        if not possible_dirs:
            print("\n✗ ERROR: Could not auto-detect patches directory")
            print("  Please specify --patches-dir")
            sys.exit(1)
        
        print(f"\nFound {len(possible_dirs)} dataset/series with patches:")
        for dataset, series, patches_path in possible_dirs:
            print(f"  - {dataset}/series_{series}: {patches_path}")
        
        # Process all found directories
        patches_dirs = [(d, s, p) for d, s, p in possible_dirs]
    else:
        patches_dir = Path(args.patches_dir)
        if not patches_dir.exists():
            print(f"\n✗ ERROR: Patches directory not found: {patches_dir}")
            sys.exit(1)
        
        # Try to extract dataset and series from path
        # Path format: dataset_name/series_num/patches
        parts = patches_dir.parts
        dataset_name = "unknown"
        series_num = "01"
        
        for i, part in enumerate(parts):
            if part == "patches" and i > 0:
                series_num = parts[i-1]
                if i > 1:
                    dataset_name = parts[i-2]
                break
        
        patches_dirs = [(dataset_name, series_num, patches_dir)]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each patches directory
    for dataset_name, series_num, patches_dir in patches_dirs:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name} / Series {series_num}")
        print(f"{'='*80}")
        print(f"Patches directory: {patches_dir}")
        
        # Reassemble patches
        full_heatmap = reassemble_patches_from_strips(
            patches_dir,
            target_width=args.target_width,
            target_height=args.target_height
        )
        
        if full_heatmap is None:
            print("  ✗ Failed to reassemble patches")
            continue
        
        # Save result
        output_name = f"{dataset_name}_series_{series_num}_gradcam_heatmap.png"
        output_path = output_dir / output_name
        
        heatmap_img = Image.fromarray(full_heatmap, mode='RGB')
        heatmap_img.save(output_path)
        
        print(f"\n  ✓ Saved: {output_name}")
        print(f"    Dimensions: {heatmap_img.size[0]} x {heatmap_img.size[1]} (W x H)")
        print(f"    Target: {args.target_width} x {args.target_height} (W x H)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

