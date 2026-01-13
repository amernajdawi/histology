#!/usr/bin/env python3
"""
Step 5: Overlay Grad-CAM heatmap on stitched image

This script overlays the reassembled Grad-CAM heatmap on the stitched image.

Usage:
    python step5_overlay_gradcam.py [stitched_image] [gradcam_heatmap] [output_path] [--alpha ALPHA]

Arguments:
    stitched_image: Path to stitched image (default: auto-detect)
    gradcam_heatmap: Path to Grad-CAM heatmap (default: auto-detect)
    output_path: Output path (default: auto-generated)
    --alpha: Transparency for overlay (0.0-1.0, default: 0.6)
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.cm as cm


def overlay_gradcam_on_image(stitched_path, gradcam_path, output_path, alpha=0.6):
    """Overlay Grad-CAM heatmap on stitched image."""
    
    print("="*80)
    print("Overlay Grad-CAM Heatmap on Stitched Image")
    print("="*80)
    
    # Load images
    print(f"\nLoading images...")
    stitched = Image.open(stitched_path)
    gradcam = Image.open(gradcam_path)
    
    print(f"  Stitched: {stitched.size[0]} x {stitched.size[1]} (W x H), mode: {stitched.mode}")
    print(f"  Grad-CAM: {gradcam.size[0]} x {gradcam.size[1]} (W x H), mode: {gradcam.mode}")
    
    # Ensure same size
    if stitched.size != gradcam.size:
        print(f"\n  Resizing Grad-CAM from {gradcam.size} to {stitched.size}...")
        gradcam = gradcam.resize(stitched.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    stitched_arr = np.array(stitched.convert('RGB')).astype(np.float32) / 255.0
    gradcam_arr = np.array(gradcam.convert('RGB')).astype(np.float32) / 255.0
    
    # Convert gradcam to single channel (use grayscale)
    if len(gradcam_arr.shape) == 3:
        # Use luminance: 0.299*R + 0.587*G + 0.114*B
        gradcam_gray = 0.299 * gradcam_arr[:, :, 0] + 0.587 * gradcam_arr[:, :, 1] + 0.114 * gradcam_arr[:, :, 2]
    else:
        gradcam_gray = gradcam_arr
    
    # Normalize heatmap to 0-1 range, but handle sparse data better
    # Only normalize non-zero values to preserve the sparse structure
    gradcam_normalized = np.zeros_like(gradcam_gray, dtype=np.float32)
    mask = gradcam_gray > 0
    
    if mask.any():
        non_zero_values = gradcam_gray[mask]
        if non_zero_values.max() > non_zero_values.min():
            # Normalize only non-zero values to 0-1 range
            gradcam_normalized[mask] = (non_zero_values - non_zero_values.min()) / (non_zero_values.max() - non_zero_values.min())
        else:
            # All non-zero values are the same
            gradcam_normalized[mask] = 0.5
    
    # Apply colormap (jet/rainbow) - zeros will map to dark blue/black
    gradcam_colored = cm.jet(gradcam_normalized)[:, :, :3]  # Remove alpha channel
    
    # For sparse heatmaps, make zero areas transparent (use original image)
    # Instead of blending zeros, only blend non-zero heatmap regions
    heatmap_mask = mask.astype(np.float32)[:, :, np.newaxis]
    
    # Blend images
    print(f"\n  Blending with alpha={alpha}...")
    print(f"  Heatmap coverage: {mask.sum()} / {mask.size} pixels ({100*mask.sum()/mask.size:.2f}%)")
    
    # For sparse heatmaps: only overlay where heatmap has values
    # Create mask for RGB blending
    mask_3d = mask[:, :, np.newaxis]
    
    # Start with original image
    blended = stitched_arr.copy()
    
    # Only blend where heatmap has values
    # Where mask is True: blend heatmap with original
    # Where mask is False: keep original image
    blended = np.where(mask_3d, 
                      (1 - alpha) * stitched_arr + alpha * gradcam_colored,
                      stitched_arr)
    blended = np.clip(blended, 0, 1)
    
    # Convert to uint8
    blended_uint8 = (blended * 255).astype(np.uint8)
    result = Image.fromarray(blended_uint8, mode='RGB')
    
    # Save
    result.save(output_path)
    
    print(f"\n✓ Saved overlay image:")
    print(f"  {output_path}")
    print(f"  Dimensions: {result.size[0]} x {result.size[1]} (W x H)")
    print(f"  Alpha: {alpha} ({(1-alpha)*100:.0f}% stitched, {alpha*100:.0f}% Grad-CAM)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Overlay Grad-CAM heatmap on stitched image')
    parser.add_argument('--stitched-image', type=str, default=None,
                       help='Path to stitched image (default: auto-detect)')
    parser.add_argument('--gradcam-heatmap', type=str, default=None,
                       help='Path to Grad-CAM heatmap (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default='stitched_images',
                       help='Output directory (default: stitched_images)')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Transparency for overlay (0.0-1.0, default: 0.6)')
    
    args = parser.parse_args()
    
    # Auto-detect files if not provided
    if args.stitched_image is None or args.gradcam_heatmap is None:
        # Look for stitched images and gradcam heatmaps
        stitched_dir = Path('stitched_images')
        gradcam_dir = Path('gradcam_heatmaps')
        
        if not stitched_dir.exists():
            print(f"✗ ERROR: Stitched images directory not found: {stitched_dir}")
            sys.exit(1)
        
        if not gradcam_dir.exists():
            print(f"✗ ERROR: Grad-CAM heatmaps directory not found: {gradcam_dir}")
            sys.exit(1)
        
        # Find matching pairs
        stitched_files = sorted(stitched_dir.glob("*_overlaid_strips_stitched.png"))
        gradcam_files = sorted(gradcam_dir.glob("*_gradcam_heatmap.png"))
        
        if not stitched_files:
            print(f"✗ ERROR: No stitched images found in {stitched_dir}")
            sys.exit(1)
        
        if not gradcam_files:
            print(f"✗ ERROR: No Grad-CAM heatmaps found in {gradcam_dir}")
            sys.exit(1)
        
        # Match files by dataset/series
        pairs = []
        for stitched_file in stitched_files:
            # Extract dataset_series from filename
            name_parts = stitched_file.stem.split('_')
            if 'series' in name_parts:
                series_idx = name_parts.index('series')
                dataset_series = '_'.join(name_parts[:series_idx+2])
                
                # Find matching gradcam file
                for gradcam_file in gradcam_files:
                    if dataset_series in gradcam_file.stem:
                        pairs.append((stitched_file, gradcam_file))
                        break
        
        if not pairs:
            print(f"✗ ERROR: Could not match stitched images with Grad-CAM heatmaps")
            print(f"  Stitched files: {[f.name for f in stitched_files]}")
            print(f"  Grad-CAM files: {[f.name for f in gradcam_files]}")
            sys.exit(1)
        
        print(f"Found {len(pairs)} matching pair(s):")
        for stitched_file, gradcam_file in pairs:
            print(f"  - {stitched_file.name} <-> {gradcam_file.name}")
        
        # Process all pairs
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for stitched_file, gradcam_file in pairs:
            # Generate output filename
            output_name = stitched_file.stem.replace('_overlaid_strips_stitched', '_with_gradcam') + '.png'
            output_path = output_dir / output_name
            
            overlay_gradcam_on_image(stitched_file, gradcam_file, output_path, args.alpha)
    else:
        # Use provided paths
        stitched_path = Path(args.stitched_image)
        gradcam_path = Path(args.gradcam_heatmap)
        
        if not stitched_path.exists():
            print(f"✗ ERROR: Stitched image not found: {stitched_path}")
            sys.exit(1)
        
        if not gradcam_path.exists():
            print(f"✗ ERROR: Grad-CAM heatmap not found: {gradcam_path}")
            sys.exit(1)
        
        # Determine output path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        output_name = stitched_path.stem + '_with_gradcam.png'
        output_path = output_dir / output_name
        
        overlay_gradcam_on_image(stitched_path, gradcam_path, output_path, args.alpha)


if __name__ == "__main__":
    main()

