#!/usr/bin/env python3
"""
Step 6: Overlay Grad-CAM heatmap on original image

This script reassembles Grad-CAM patches directly onto the original image
at their actual coordinates, ensuring patches fit the original image correctly.

Usage:
    python step6_combine_with_original.py [--original-image PATH] [--patches-dir PATH] [--output-path PATH] [--alpha ALPHA]

Arguments:
    --original-image: Path to original image (default: unnamed.png)
    --patches-dir: Directory containing Grad-CAM patch files (default: auto-detect)
    --output-path: Output path (default: auto-generated)
    --alpha: Transparency for overlay (0.0-1.0, default: 0.6)
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.cm as cm


def parse_patch_coordinates(filename):
    """Parse coordinates from patch filename.
    
    Format: MUV_0635-ALA-01-<strip_x>-<strip_y>-<patch_x>-<patch_y>.tif
    Returns: (absolute_x, absolute_y)
    """
    parts = filename.stem.split('-')
    if len(parts) >= 7:
        try:
            strip_x = int(parts[3])
            strip_y = int(parts[4])
            patch_x = int(parts[5])
            patch_y = int(parts[6])
            absolute_x = strip_x + patch_x
            absolute_y = strip_y + patch_y
            return absolute_x, absolute_y
        except ValueError:
            return None, None
    return None, None


def reassemble_patches_on_original(original_path, patches_dir, output_path, alpha=0.6, target_size=(3600, 3900)):
    """Reassemble Grad-CAM patches directly onto original image at their actual coordinates."""
    
    print("="*80)
    print("Overlay Grad-CAM Patches on Original Image")
    print("="*80)
    
    # Load original image
    print(f"\nLoading original image...")
    original = Image.open(original_path)
    original_rgb = original.convert('RGB')
    orig_w, orig_h = original_rgb.size
    
    print(f"  Original: {orig_w} x {orig_h} (W x H), mode: {original_rgb.mode}")
    print(f"  Target size: {target_size[0]} x {target_size[1]} (W x H)")
    
    # Find all ALA patches (Grad-CAM heatmaps)
    patches_dir = Path(patches_dir)
    patch_files = sorted(patches_dir.glob("*ALA*.tif"))
    
    if not patch_files:
        print(f"✗ ERROR: No ALA patch files found in {patches_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(patch_files)} patch files")
    
    # Load a sample patch to get dimensions
    sample_patch = Image.open(patch_files[0])
    patch_w, patch_h = sample_patch.size
    print(f"  Patch size: {patch_w} x {patch_h} pixels")
    
    # Create heatmap canvas matching original image size
    # We'll place patches at their absolute coordinates
    heatmap_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    patch_count = 0
    
    print(f"\nPlacing patches on original image...")
    
    for patch_file in patch_files:
        abs_x, abs_y = parse_patch_coordinates(patch_file)
        if abs_x is None or abs_y is None:
            continue
        
        # Load patch
        patch_img = Image.open(patch_file)
        patch_arr = np.array(patch_img)
        
        # Convert to grayscale if needed
        if len(patch_arr.shape) == 3:
            patch_gray = patch_arr.mean(axis=2)
        else:
            patch_gray = patch_arr
        
        # Normalize patch to 0-1
        if patch_gray.max() > 0:
            patch_normalized = patch_gray.astype(np.float32) / 255.0
        else:
            patch_normalized = patch_gray.astype(np.float32)
        
        # Determine where to place patch on original image
        # Calculate overlap region
        x_start = max(0, abs_x)
        y_start = max(0, abs_y)
        x_end = min(orig_w, abs_x + patch_w)
        y_end = min(orig_h, abs_y + patch_h)
        
        # Skip if patch is completely outside original image bounds
        if x_start >= x_end or y_start >= y_end:
            continue
        
        # Calculate corresponding region in patch
        patch_x_start = x_start - abs_x
        patch_y_start = y_start - abs_y
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_end = patch_y_start + (y_end - y_start)
        
        # Place patch on canvas (use max to handle overlapping patches)
        heatmap_canvas[y_start:y_end, x_start:x_end] = np.maximum(
            heatmap_canvas[y_start:y_end, x_start:x_end],
            patch_normalized[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        )
        
        patch_count += 1
    
    print(f"  Placed {patch_count} patches")
    print(f"  Heatmap coverage: {(heatmap_canvas > 0).sum()} / {heatmap_canvas.size} pixels ({100*(heatmap_canvas > 0).sum()/heatmap_canvas.size:.2f}%)")
    
    # Normalize heatmap (only non-zero values)
    heatmap_normalized = np.zeros_like(heatmap_canvas, dtype=np.float32)
    mask = heatmap_canvas > 0
    
    if mask.any():
        non_zero_values = heatmap_canvas[mask]
        if non_zero_values.max() > non_zero_values.min():
            heatmap_normalized[mask] = (non_zero_values - non_zero_values.min()) / (non_zero_values.max() - non_zero_values.min())
        else:
            heatmap_normalized[mask] = 0.5
    
    # Apply colormap
    gradcam_colored = cm.jet(heatmap_normalized)[:, :, :3]
    
    # Convert original to numpy array
    original_arr = np.array(original_rgb).astype(np.float32) / 255.0
    
    # Blend images
    print(f"\n  Blending with alpha={alpha}...")
    mask_3d = mask[:, :, np.newaxis]
    
    # Start with original image
    blended = original_arr.copy()
    
    # Only blend where heatmap has values
    blended = np.where(mask_3d, 
                      (1 - alpha) * original_arr + alpha * gradcam_colored,
                      original_arr)
    blended = np.clip(blended, 0, 1)
    
    # Convert to uint8
    blended_uint8 = (blended * 255).astype(np.uint8)
    result = Image.fromarray(blended_uint8)
    
    # Resize to target size
    if result.size != target_size:
        print(f"\n  Resizing to target size: {result.size} -> {target_size}")
        result = result.resize(target_size, Image.Resampling.LANCZOS)
    
    # Save
    result.save(output_path, optimize=True, compress_level=9)
    
    print(f"\n✓ Saved overlay image:")
    print(f"  {output_path}")
    print(f"  Dimensions: {result.size[0]} x {result.size[1]} (W x H)")
    print(f"  Alpha: {alpha} ({(1-alpha)*100:.0f}% original, {alpha*100:.0f}% Grad-CAM)")
    
    # Verify dimensions
    if result.size == target_size:
        print(f"  ✓ Size matches target: {target_size[0]} x {target_size[1]}")
    else:
        print(f"  ✗ WARNING: Size mismatch! Expected {target_size}, got {result.size}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Overlay Grad-CAM patches on original image')
    parser.add_argument('--original-image', type=str, default='unnamed.png',
                       help='Path to original image (default: unnamed.png)')
    parser.add_argument('--patches-dir', type=str, default=None,
                       help='Directory containing Grad-CAM patch files (default: auto-detect)')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output path (default: auto-generated)')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Transparency for overlay (0.0-1.0, default: 0.6)')
    parser.add_argument('--target-width', type=int, default=3600,
                       help='Target width (default: 3600)')
    parser.add_argument('--target-height', type=int, default=3900,
                       help='Target height (default: 3900)')
    
    args = parser.parse_args()
    
    target_size = (args.target_width, args.target_height)
    
    # Check original image
    original_path = Path(args.original_image)
    if not original_path.exists():
        print(f"✗ ERROR: Original image not found: {original_path}")
        sys.exit(1)
    
    # Auto-detect patches directory if not provided
    if args.patches_dir is None:
        # Look for patches directory
        possible_dirs = [
            Path('MUV_0635-2/01/patches'),
            Path('patches'),
        ]
        
        patches_dir = None
        for pd in possible_dirs:
            if pd.exists() and list(pd.glob("*ALA*.tif")):
                patches_dir = pd
                break
        
        if patches_dir is None:
            print(f"✗ ERROR: Could not find patches directory")
            print(f"  Tried: {[str(p) for p in possible_dirs]}")
            sys.exit(1)
        
        print(f"Auto-detected patches directory: {patches_dir}")
    else:
        patches_dir = Path(args.patches_dir)
    
    if not patches_dir.exists():
        print(f"✗ ERROR: Patches directory not found: {patches_dir}")
        sys.exit(1)
    
    # Determine output path
    if args.output_path is None:
        output_dir = Path('stitched_images')
        output_dir.mkdir(exist_ok=True)
        output_name = original_path.stem + '_with_gradcam.png'
        output_path = output_dir / output_name
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    reassemble_patches_on_original(original_path, patches_dir, output_path, args.alpha, target_size)


if __name__ == "__main__":
    main()
