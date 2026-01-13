#!/usr/bin/env python3
"""
Step 2: Group similar/duplicate strips and overlay them (FLEXIBLE VERSION)

This script automatically detects all datasets and series, groups similar strips,
and saves each overlaid group as a separate image.

Usage:
    python step2_overlay_duplicates.py [input_dir] [output_dir] [--manual-pairs JSON]

Arguments:
    input_dir: Directory containing *_original.png files (default: strips_heatmaps)
    output_dir: Output directory for overlaid groups (default: overlaid_strips)
    --manual-pairs: JSON file with manual pairs per dataset/series (optional)
"""

import sys
import argparse
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine


def is_black_strip(img_arr, threshold=5.0):
    """Check if strip is mostly black/empty."""
    mean_intensity = img_arr.mean()
    return mean_intensity < threshold


def calculate_similarity(img1_arr, img2_arr):
    """Calculate similarity between two images using multiple methods."""
    # Method 1: Histogram comparison
    hist1, _ = np.histogram(img1_arr.flatten(), bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(img2_arr.flatten(), bins=256, range=(0, 256), density=True)
    hist_sim = 1 - cosine(hist1, hist2)
    
    # Method 2: Structural similarity (if same size)
    if img1_arr.shape == img2_arr.shape:
        img1_norm = img1_arr.astype(float) / max(img1_arr.max(), 1)
        img2_norm = img2_arr.astype(float) / max(img2_arr.max(), 1)
        mse = np.mean((img1_norm - img2_norm) ** 2)
        struct_sim = 1 / (1 + mse)
        similarity = (hist_sim * 0.6 + struct_sim * 0.4)
    else:
        similarity = hist_sim
    
    return similarity


def group_similar_strips(strip_files, similarity_threshold=0.75, manual_pairs=None):
    """Group strips that are similar to each other."""
    print(f"  Analyzing {len(strip_files)} strips for similarity...")
    
    # Load all strips
    all_strips = []
    black_count = 0
    for strip_file in strip_files:
        img = Image.open(strip_file)
        img_arr = np.array(img)
        is_black = is_black_strip(img_arr)
        if is_black:
            black_count += 1
        all_strips.append((strip_file, img_arr, is_black))
    
    print(f"    Total strips: {len(all_strips)}")
    print(f"    Black/empty strips: {black_count}")
    print(f"    Non-black strips: {len(all_strips) - black_count}")
    
    if len(all_strips) == 0:
        return []
    
    # Group similar strips
    groups = []
    used = set()
    
    # Handle manual pairs first
    if manual_pairs:
        for pair in manual_pairs:
            idx1 = None
            idx2 = None
            for i, (f, _, _) in enumerate(all_strips):
                if pair[0] in f.name and idx1 is None and i not in used:
                    idx1 = i
                if pair[1] in f.name and idx2 is None and i not in used:
                    idx2 = i
            if idx1 is not None and idx2 is not None and idx1 != idx2:
                groups.append([all_strips[idx1][0], all_strips[idx2][0]])
                used.add(idx1)
                used.add(idx2)
                print(f"    Manual pair: {Path(all_strips[idx1][0]).name} <-> {Path(all_strips[idx2][0]).name}")
    
    # Group remaining strips
    for i, (file1, arr1, black1) in enumerate(all_strips):
        if i in used:
            continue
        
        group = [file1]
        used.add(i)
        
        for j, (file2, arr2, black2) in enumerate(all_strips[i+1:], start=i+1):
            if j in used:
                continue
            
            # Black strips: only match with other black strips
            if black1 and black2:
                similarity = calculate_similarity(arr1, arr2)
                if similarity >= 0.98:
                    group.append(file2)
                    used.add(j)
            elif not black1 and not black2:
                similarity = calculate_similarity(arr1, arr2)
                if arr1.shape == arr2.shape:
                    arr1_norm = (arr1 - arr1.mean()) / (arr1.std() + 1e-8)
                    arr2_norm = (arr2 - arr2.mean()) / (arr2.std() + 1e-8)
                    correlation = np.corrcoef(arr1_norm.flatten(), arr2_norm.flatten())[0, 1]
                    combined_sim = (similarity * 0.3 + (correlation + 1) / 2 * 0.7)
                else:
                    combined_sim = similarity
                
                if combined_sim >= 0.65 or correlation >= 0.5:
                    group.append(file2)
                    used.add(j)
        
        groups.append(group)
    
    return groups


def overlay_strips(strip_files):
    """Overlay multiple similar strips by averaging them."""
    if not strip_files:
        return None
    
    images = []
    for strip_file in strip_files:
        img = Image.open(strip_file)
        if img.mode != 'L':
            img = img.convert('L')
        images.append(np.array(img).astype(np.float32))
    
    combined = np.mean(images, axis=0).astype(np.uint8)
    return Image.fromarray(combined)


def extract_dataset_series_from_filename(filename):
    """Extract dataset name and series number from filename."""
    # Pattern: <dataset>_series_<num>_strip_<num>_...
    match = re.search(r'(.+?)_series_(\d+)_strip_(\d+)', filename)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None


def find_all_datasets_and_series(input_dir):
    """Find all unique dataset/series combinations."""
    strip_files = sorted([f for f in input_dir.glob("*_original.png")])
    
    datasets_series = {}
    for strip_file in strip_files:
        dataset, series, strip_num = extract_dataset_series_from_filename(strip_file.name)
        if dataset and series:
            key = f"{dataset}_series_{series}"
            if key not in datasets_series:
                datasets_series[key] = []
            datasets_series[key].append(strip_file)
    
    return datasets_series


def main():
    parser = argparse.ArgumentParser(description='Group and overlay similar strips')
    parser.add_argument('--input-dir', type=str, default='strips_heatmaps',
                       help='Input directory with *_original.png files (default: strips_heatmaps)')
    parser.add_argument('--output-dir', type=str, default='overlaid_strips',
                       help='Output directory for overlaid groups (default: overlaid_strips)')
    parser.add_argument('--manual-pairs', type=str, default=None,
                       help='JSON file with manual pairs: {"dataset_series_01": [["strip_03", "strip_04"]]}')
    parser.add_argument('--similarity-threshold', type=float, default=0.75,
                       help='Similarity threshold for grouping (default: 0.75)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 2: Overlay Duplicate Strips (FLEXIBLE)")
    print("="*80)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"\n✗ ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load manual pairs if provided
    manual_pairs_dict = {}
    if args.manual_pairs:
        with open(args.manual_pairs, 'r') as f:
            full_config = json.load(f)
            # Handle both formats: direct manual_pairs or nested in "manual_pairs"
            if 'manual_pairs' in full_config:
                manual_pairs_dict = full_config['manual_pairs']
            else:
                manual_pairs_dict = full_config
    
    # Find all datasets and series
    datasets_series = find_all_datasets_and_series(input_dir)
    
    if not datasets_series:
        print(f"\n✗ ERROR: No *_original.png files found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(datasets_series)} dataset/series combination(s):")
    for key in sorted(datasets_series.keys()):
        print(f"  - {key}: {len(datasets_series[key])} strips")
    
    total_groups = 0
    
    # Process each dataset/series
    for key in sorted(datasets_series.keys()):
        print(f"\n{'='*80}")
        print(f"Processing: {key}")
        print(f"{'='*80}")
        
        strip_files = datasets_series[key]
        
        # Get manual pairs for this dataset/series if available
        manual_pairs = manual_pairs_dict.get(key, None)
        if manual_pairs:
            print(f"  Using {len(manual_pairs)} manual pair(s)")
        
        # Group similar strips
        groups = group_similar_strips(strip_files, args.similarity_threshold, manual_pairs)
        print(f"  Grouped into {len(groups)} groups")
        
        # Overlay and save each group
        for i, group in enumerate(groups, 1):
            overlaid = overlay_strips(group)
            if overlaid is not None:
                # Extract strip numbers for naming
                strip_nums = []
                for f in group:
                    match = re.search(r'strip_(\d+)', Path(f).stem)
                    if match:
                        strip_nums.append(match.group(1))
                
                if len(group) == 1:
                    output_name = f"{key}_group_{i:02d}_strip_{strip_nums[0]}_single.png"
                else:
                    strip_nums_str = '_'.join(sorted(strip_nums))
                    output_name = f"{key}_group_{i:02d}_overlaid_strips_{strip_nums_str}.png"
                
                output_path = output_dir / output_name
                overlaid.save(output_path)
                print(f"  ✓ Saved group {i}: {output_name} ({len(group)} strip(s))")
                total_groups += 1
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total groups created: {total_groups}")


if __name__ == "__main__":
    main()

