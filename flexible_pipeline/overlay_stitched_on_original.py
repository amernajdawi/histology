#!/usr/bin/env python3
"""
Overlay stitched image on original histopathology image

Usage:
    python overlay_stitched_on_original.py <stitched_image> <original_image> [output_path] [--alpha ALPHA]

Arguments:
    stitched_image: Path to stitched/combined image
    original_image: Path to original histopathology image
    output_path: Output path (default: <original_image>_with_overlay.png)
    --alpha: Transparency for overlay (0.0-1.0, default: 0.5)
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def overlay_images(original_path, stitched_path, output_path, alpha=0.5):
    """Overlay stitched image on original image."""
    
    print("="*80)
    print("Overlay Stitched Image on Original")
    print("="*80)
    
    # Load images
    print(f"\nLoading images...")
    original = Image.open(original_path)
    stitched = Image.open(stitched_path)
    
    print(f"  Original: {original.size[0]} x {original.size[1]} (W x H), mode: {original.mode}")
    print(f"  Stitched: {stitched.size[0]} x {stitched.size[1]} (W x H), mode: {stitched.mode}")
    
    # Convert to RGB if needed
    if original.mode != 'RGB':
        original = original.convert('RGB')
    if stitched.mode != 'RGB':
        stitched = stitched.convert('RGB')
    
    # Resize stitched to match original if needed
    if stitched.size != original.size:
        print(f"\n  Resizing stitched image from {stitched.size} to {original.size}...")
        stitched = stitched.resize(original.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    original_arr = np.array(original).astype(np.float32)
    stitched_arr = np.array(stitched).astype(np.float32)
    
    # Blend images
    print(f"\n  Blending with alpha={alpha}...")
    blended = (1 - alpha) * original_arr + alpha * stitched_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    result = Image.fromarray(blended)
    
    # Save
    result.save(output_path)
    
    print(f"\n✓ Saved overlay image:")
    print(f"  {output_path}")
    print(f"  Dimensions: {result.size[0]} x {result.size[1]} (W x H)")
    print(f"  Alpha: {alpha} ({(1-alpha)*100:.0f}% original, {alpha*100:.0f}% stitched)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Overlay stitched image on original')
    parser.add_argument('stitched_image', type=str,
                       help='Path to stitched/combined image')
    parser.add_argument('original_image', type=str,
                       help='Path to original histopathology image')
    parser.add_argument('output_path', type=str, nargs='?', default=None,
                       help='Output path (default: <original>_with_overlay.png)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency for overlay (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    stitched_path = Path(args.stitched_image)
    original_path = Path(args.original_image)
    
    if not stitched_path.exists():
        print(f"✗ ERROR: Stitched image not found: {stitched_path}")
        sys.exit(1)
    
    if not original_path.exists():
        print(f"✗ ERROR: Original image not found: {original_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = original_path.parent / f"{original_path.stem}_with_overlay.png"
    
    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        print(f"✗ ERROR: Alpha must be between 0.0 and 1.0")
        sys.exit(1)
    
    overlay_images(original_path, stitched_path, output_path, args.alpha)


if __name__ == "__main__":
    main()

