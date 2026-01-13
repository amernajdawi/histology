#!/usr/bin/env python3
"""
Save each overlaid group (after combining duplicates) as a separate image file
"""

from PIL import Image
import numpy as np
from pathlib import Path
import re
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
        # Normalize for comparison
        img1_norm = img1_arr.astype(float) / max(img1_arr.max(), 1)
        img2_norm = img2_arr.astype(float) / max(img2_arr.max(), 1)
        
        # Mean squared error (inverted to similarity)
        mse = np.mean((img1_norm - img2_norm) ** 2)
        struct_sim = 1 / (1 + mse)
        
        # Combine both methods
        similarity = (hist_sim * 0.6 + struct_sim * 0.4)
    else:
        similarity = hist_sim
    
    return similarity

def group_similar_strips(strip_files, similarity_threshold=0.75, manual_pairs=None):
    """Group strips that are similar to each other."""
    print(f"\nAnalyzing {len(strip_files)} strips for similarity...")
    
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
    
    print(f"  Total strips: {len(all_strips)}")
    print(f"  Black/empty strips: {black_count}")
    print(f"  Non-black strips: {len(all_strips) - black_count}")
    
    if len(all_strips) == 0:
        return []
    
    # Group similar strips
    groups = []
    used = set()
    
    # First, handle manual pairs
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
                print(f"  Manual pair grouped: {Path(all_strips[idx1][0]).name} <-> {Path(all_strips[idx2][0]).name}")
    
    for i, (file1, arr1, black1) in enumerate(all_strips):
        if i in used:
            continue
        
        group = [file1]
        used.add(i)
        
        # For black strips, only match with other black strips
        # For non-black strips, match with similar ones
        for j, (file2, arr2, black2) in enumerate(all_strips[i+1:], start=i+1):
            if j in used:
                continue
            
            # Black strips: only match if both are black
            if black1 and black2:
                similarity = calculate_similarity(arr1, arr2)
                if similarity >= 0.98:
                    group.append(file2)
                    used.add(j)
                    print(f"  Similar (both black): {Path(file1).name} <-> {Path(file2).name} (sim={similarity:.3f})")
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
                    print(f"  Similar: {Path(file1).name} <-> {Path(file2).name} (sim={similarity:.3f}, corr={correlation:.3f}, combined={combined_sim:.3f})")
        
        groups.append(group)
    
    return groups

def overlay_strips(strip_files):
    """Overlay multiple similar strips by averaging them."""
    if not strip_files:
        return None
    
    # Load all strips
    images = []
    for strip_file in strip_files:
        img = Image.open(strip_file)
        if img.mode != 'L':
            img = img.convert('L')
        images.append(np.array(img).astype(np.float32))
    
    # Average them
    combined = np.mean(images, axis=0).astype(np.uint8)
    return Image.fromarray(combined)

def save_overlaid_strips_by_series():
    """Save each overlaid group as a separate image file."""
    
    print("="*80)
    print("Save Overlaid Strips (After Combining Duplicates)")
    print("="*80)
    
    base_dir = Path("/Users/ameralnajdawi/Desktop/new_his")
    strips_dir = base_dir / "strips_heatmaps"
    output_dir = base_dir / "overlaid_strips"
    output_dir.mkdir(exist_ok=True)
    
    # Process Series 01
    print("\n" + "="*80)
    print("SERIES 01")
    print("="*80)
    
    series_01_files = sorted([f for f in strips_dir.glob("series_01_*_original.png")])
    print(f"Found {len(series_01_files)} strips from series 01")
    
    if series_01_files:
        # Manual pairs based on user's observation
        manual_pairs_01 = [
            ('strip_03', 'strip_04'),  # User said these are the same
        ]
        groups = group_similar_strips(series_01_files, similarity_threshold=0.75, manual_pairs=manual_pairs_01)
        print(f"\nGrouped into {len(groups)} groups")
        
        # Overlay and save each group
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i}: {len(group)} strip(s)")
            
            # Show which strips are in this group
            strip_names = [Path(f).name for f in group]
            print(f"  Strips: {', '.join(strip_names)}")
            
            # Overlay the strips
            overlaid = overlay_strips(group)
            if overlaid is not None:
                # Save with descriptive name
                if len(group) == 1:
                    # Single strip (no duplicates) - extract just the strip number
                    strip_name = Path(group[0]).stem
                    # Extract strip number (e.g., "strip_01" from "series_01_strip_01_...")
                    match = re.search(r'strip_(\d+)', strip_name)
                    strip_num = match.group(1) if match else "unknown"
                    output_name = f"series_01_group_{i:02d}_strip_{strip_num}_single.png"
                else:
                    # Multiple strips overlaid - extract just the strip numbers
                    strip_nums = []
                    for f in group:
                        strip_name = Path(f).stem
                        match = re.search(r'strip_(\d+)', strip_name)
                        if match:
                            strip_nums.append(match.group(1))
                    strip_nums_str = '_'.join(sorted(strip_nums))
                    output_name = f"series_01_group_{i:02d}_overlaid_strips_{strip_nums_str}.png"
                
                output_path = output_dir / output_name
                overlaid.save(output_path)
                print(f"  ✓ Saved: {output_name}")
                print(f"    Dimensions: {overlaid.size[0]} x {overlaid.size[1]}")
    
    # Process Series 02
    print("\n" + "="*80)
    print("SERIES 02")
    print("="*80)
    
    series_02_files = sorted([f for f in strips_dir.glob("series_02_*_original.png")])
    print(f"Found {len(series_02_files)} strips from series 02")
    
    if series_02_files:
        groups = group_similar_strips(series_02_files, similarity_threshold=0.75)
        print(f"\nGrouped into {len(groups)} groups")
        
        # Overlay and save each group
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i}: {len(group)} strip(s)")
            
            # Show which strips are in this group
            strip_names = [Path(f).name for f in group]
            print(f"  Strips: {', '.join(strip_names)}")
            
            # Overlay the strips
            overlaid = overlay_strips(group)
            if overlaid is not None:
                # Save with descriptive name
                if len(group) == 1:
                    # Single strip (no duplicates) - extract just the strip number
                    strip_name = Path(group[0]).stem
                    # Extract strip number (e.g., "strip_01" from "series_02_strip_01_...")
                    match = re.search(r'strip_(\d+)', strip_name)
                    strip_num = match.group(1) if match else "unknown"
                    output_name = f"series_02_group_{i:02d}_strip_{strip_num}_single.png"
                else:
                    # Multiple strips overlaid - extract just the strip numbers
                    strip_nums = []
                    for f in group:
                        strip_name = Path(f).stem
                        match = re.search(r'strip_(\d+)', strip_name)
                        if match:
                            strip_nums.append(match.group(1))
                    strip_nums_str = '_'.join(sorted(strip_nums))
                    output_name = f"series_02_group_{i:02d}_overlaid_strips_{strip_nums_str}.png"
                
                output_path = output_dir / output_name
                overlaid.save(output_path)
                print(f"  ✓ Saved: {output_name}")
                print(f"    Dimensions: {overlaid.size[0]} x {overlaid.size[1]}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print(f"All overlaid strips saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    save_overlaid_strips_by_series()

