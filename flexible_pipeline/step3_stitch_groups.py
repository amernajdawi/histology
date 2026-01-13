#!/usr/bin/env python3
"""
Step 3: Stitch overlaid groups together (FLEXIBLE VERSION)

This script stitches overlaid groups together in the correct order.
Supports configurable excluded strips and ordering.

Usage:
    python step3_stitch_groups.py [input_dir] [output_dir] [--config JSON]

Arguments:
    input_dir: Directory containing group images (default: overlaid_strips)
    output_dir: Output directory for stitched images (default: combined_strips)
    --config: JSON file with ordering and excluded strips (optional)
"""

import sys
import argparse
import json
from pathlib import Path
import re
from PIL import Image
import numpy as np
from skimage import exposure


def extract_strip_numbers_from_filename(filename):
    """Extract strip numbers from filename."""
    if 'overlaid_strips' in filename:
        match = re.search(r'overlaid_strips_(\d+(?:_\d+)*)', filename)
        if match:
            return [int(n) for n in match.group(1).split('_')]
    elif 'single' in filename:
        match = re.search(r'strip_(\d+)_single', filename)
        if match:
            return [int(match.group(1))]
    return []


def extract_dataset_series_from_filename(filename):
    """Extract dataset name and series number from filename."""
    # Pattern: <dataset>_series_<num>_group_...
    match = re.search(r'(.+?)_series_(\d+)_group_', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def auto_detect_black_groups(group_files):
    """Auto-detect black/empty groups by checking mean intensity."""
    from PIL import Image
    import numpy as np
    
    black_groups = []
    for group_file in group_files:
        img = Image.open(group_file)
        img_arr = np.array(img)
        mean_intensity = img_arr.mean()
        if mean_intensity < 5.0:  # Threshold for black
            strip_nums = extract_strip_numbers_from_filename(group_file.name)
            if strip_nums:
                black_groups.extend(strip_nums)
    
    return set(black_groups)


def determine_order(group_files, config=None):
    """Determine the correct order of groups."""
    # Parse config
    excluded_strips = set()
    first_strips = None
    last_strips = None
    
    if config:
        excluded_strips = set(config.get('excluded_strips', []))
        first_strips = config.get('first_strips', None)
        last_strips = config.get('last_strips', None)
    
    # Always auto-detect black groups and add to excluded_strips
    print("  Auto-detecting black/empty groups...")
    black_strips = auto_detect_black_groups(group_files)
    if black_strips:
        print(f"  Found {len(black_strips)} black strip numbers: {sorted(black_strips)}")
        # Merge with manually specified excluded strips
        excluded_strips = excluded_strips.union(black_strips)
        print(f"  Total excluded strips (manual + black): {sorted(excluded_strips)}")
    
    # Map each group to its strip numbers
    group_info = []
    for group_file in group_files:
        strip_nums = extract_strip_numbers_from_filename(group_file.name)
        if strip_nums:
            min_strip = min(strip_nums)
            group_info.append({
                'file': group_file,
                'strip_numbers': strip_nums,
                'min_strip': min_strip,
                'filename': group_file.name
            })
    
    # Filter out excluded groups
    # Only exclude groups where ALL strips are excluded
    if excluded_strips:
        filtered_groups = []
        for info in group_info:
            # Only exclude if ALL strips in the group are excluded
            if not all(strip in excluded_strips for strip in info['strip_numbers']):
                filtered_groups.append(info)
        group_info = filtered_groups
    
    # Determine ordering
    # Check if first_strips and last_strips values themselves are excluded
    first_strips_valid = []
    last_strips_valid = []
    
    if first_strips:
        first_strips_valid = [s for s in first_strips if s not in excluded_strips]
    if last_strips:
        last_strips_valid = [s for s in last_strips if s not in excluded_strips]
    
    # Only use custom ordering if we have valid first/last strips that are not excluded
    if first_strips and last_strips and (first_strips_valid or last_strips_valid):
        # Custom ordering: first_strips first, then others, then last_strips last
        first_groups = []
        last_groups = []
        other_groups = []
        
        for info in group_info:
            # Only check against valid (non-excluded) first/last strips
            has_first = any(strip in first_strips_valid for strip in info['strip_numbers']) if first_strips_valid else False
            has_last = any(strip in last_strips_valid for strip in info['strip_numbers']) if last_strips_valid else False
            
            if has_first and has_last:
                # Group contains both first and last - put in first_groups, but also track for last
                first_groups.append(info)
                # Don't add to last_groups to avoid duplication
            elif has_first:
                first_groups.append(info)
            elif has_last:
                last_groups.append(info)
            else:
                other_groups.append(info)
        
        # Sort first_groups by min_strip (position number) for consistent ordering
        # This ensures groups are sorted by their actual position, not by first_strip value
        first_groups_sorted = sorted(first_groups, key=lambda x: x['min_strip']) if first_groups else []
        last_groups_sorted = sorted(last_groups, key=lambda x: x['min_strip']) if last_groups else []
        
        # Check if any first_group also has last_strips
        last_group = None
        for fg in first_groups_sorted:
            if any(strip in last_strips for strip in fg['strip_numbers']):
                last_group = fg
                break
        
        # If no first_group has last_strips, use the first from last_groups
        if not last_group and last_groups_sorted:
            last_group = last_groups_sorted[0]
        
        # Build ordered list
        # If we have first/last groups, use custom ordering
        # Otherwise, sort all by position number
        if first_groups_sorted and last_group:
            # Remove first and last groups from other_groups
            other_groups = [g for g in other_groups if g != last_group]
            for fg in first_groups_sorted:
                other_groups = [g for g in other_groups if g != fg]
            
            if last_group in first_groups_sorted:
                # Last group is also in first - put all first_groups at beginning, then others
                # (last_group is already in first_groups_sorted, so it's at the beginning)
                ordered_groups = first_groups_sorted + sorted(other_groups, key=lambda x: x['min_strip'])
            else:
                # Last group is separate - put first_groups, then others, then last
                ordered_groups = first_groups_sorted + sorted(other_groups, key=lambda x: x['min_strip']) + [last_group]
        elif first_groups_sorted:
            # Merge first_groups with other_groups and sort all by position number
            # This ensures consistent position-based ordering
            all_groups = first_groups_sorted + other_groups
            ordered_groups = sorted(all_groups, key=lambda x: x['min_strip'])
        elif last_group:
            other_groups = [g for g in other_groups if g != last_group]
            ordered_groups = sorted(other_groups, key=lambda x: x['min_strip']) + [last_group]
        else:
            # No first/last groups - sort all by position number
            ordered_groups = sorted(group_info, key=lambda x: x['min_strip'])
    else:
        # Default: sort by minimum strip number
        ordered_groups = sorted(group_info, key=lambda x: x['min_strip'])
    
    return ordered_groups


def find_all_datasets_and_series(input_dir):
    """Find all unique dataset/series combinations."""
    group_files = sorted([f for f in input_dir.glob("*_group_*.png")])
    
    datasets_series = {}
    for group_file in group_files:
        dataset, series = extract_dataset_series_from_filename(group_file.name)
        if dataset and series:
            key = f"{dataset}_series_{series}"
            if key not in datasets_series:
                datasets_series[key] = []
            datasets_series[key].append(group_file)
    
    return datasets_series


def main():
    parser = argparse.ArgumentParser(description='Stitch overlaid groups together')
    parser.add_argument('--input-dir', type=str, default='overlaid_strips',
                       help='Input directory with group images (default: overlaid_strips)')
    parser.add_argument('--output-dir', type=str, default='combined_strips',
                       help='Output directory for stitched images (default: combined_strips)')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON config file with ordering: {"dataset_series_01": {"excluded_strips": [1,2,11], "first_strips": [18,19], "last_strips": [14]}}')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 3: Stitch Overlaid Groups (FLEXIBLE)")
    print("="*80)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"\n✗ ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load config if provided
    config_dict = {}
    if args.config:
        with open(args.config, 'r') as f:
            full_config = json.load(f)
            # Handle both formats: direct config or nested in "stitch_config"
            if 'stitch_config' in full_config:
                config_dict = full_config['stitch_config']
            else:
                config_dict = full_config
    
    # Find all datasets and series
    datasets_series = find_all_datasets_and_series(input_dir)
    
    if not datasets_series:
        print(f"\n✗ ERROR: No group images found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(datasets_series)} dataset/series combination(s):")
    for key in sorted(datasets_series.keys()):
        print(f"  - {key}: {len(datasets_series[key])} groups")
    
    # Process each dataset/series
    for key in sorted(datasets_series.keys()):
        print(f"\n{'='*80}")
        print(f"Processing: {key}")
        print(f"{'='*80}")
        
        group_files = datasets_series[key]
        
        # Get config for this dataset/series
        config = config_dict.get(key, None)
        if config:
            print(f"  Using custom config")
        
        # Determine order
        ordered_groups = determine_order(group_files, config)
        
        print(f"\n  Found {len(ordered_groups)} groups to stitch:")
        for i, info in enumerate(ordered_groups, 1):
            print(f"    {i}. {info['filename']} (strips: {info['strip_numbers']})")
        
        if not ordered_groups:
            print("  ✗ No groups to stitch!")
            continue
        
        # Target resolution from scanner
        target_width = 3600
        target_height = 3900
        crop_pixels = 50  # Crop 50px from all sides of each strip
        
        # Load all images, crop, and apply histogram equalization
        images = []
        for info in ordered_groups:
            img = Image.open(info['file'])
            if img.mode != 'L':
                img = img.convert('L')
            
            # Crop 50px from all sides
            img_array = np.array(img, dtype=np.float32)
            h, w = img_array.shape
            # Crop: remove crop_pixels from top, bottom, left, right
            img_array = img_array[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]
            
            # Apply histogram equalization for brightness
            img_array_normalized = img_array / 255.0  # Normalize to 0-1
            img_array_normalized = exposure.equalize_adapthist(img_array_normalized, clip_limit=0.03)
            img_array = (img_array_normalized * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            images.append(img)
        
        # Check dimensions after cropping
        heights = [img.size[1] for img in images]
        widths = [img.size[0] for img in images]
        
        # Use target height, or max height if target is larger
        final_height = min(target_height, max(heights))
        
        # Calculate overlap needed to achieve target width
        num_overlaps = len(images) - 1 if len(images) > 1 else 0
        if num_overlaps > 0:
            # Solve: target_width = sum(widths) - num_overlaps * overlap_pixels
            # overlap_pixels = (sum(widths) - target_width) / num_overlaps
            total_width_without_overlap = sum(widths)
            overlap_pixels = int((total_width_without_overlap - target_width) / num_overlaps)
            overlap_pixels = max(0, overlap_pixels)  # Ensure non-negative
        else:
            overlap_pixels = 0
        
        total_width = sum(widths) - (num_overlaps * overlap_pixels)
        
        print(f"\n  Cropping {crop_pixels}px from all sides of each strip")
        print(f"  Combining {len(images)} images with {overlap_pixels}-pixel overlap:")
        print(f"    Target resolution: {target_width} x {target_height}")
        print(f"    Calculated width (with overlap averaging): {total_width} pixels")
        print(f"    Final height: {final_height} pixels")
        print(f"    Overlap regions: {num_overlaps}")
        
        # Create combined image using accumulation array for averaging overlaps
        combined_array = np.zeros((final_height, total_width), dtype=np.float32)
        overlap_count = np.zeros((final_height, total_width), dtype=np.float32)
        
        x_offset = 0
        for i, (img, info) in enumerate(zip(images, ordered_groups)):
            img_array = np.array(img, dtype=np.float32)
            img_w, img_h = img.size
            
            # Crop height to match target if needed
            if img_h > final_height:
                # Crop from top and bottom equally
                crop_top = (img_h - final_height) // 2
                img_array = img_array[crop_top:crop_top+final_height, :]
                img_h = final_height
            
            # Calculate y offset for centering (if image is smaller than target)
            y_offset = (final_height - img_h) // 2 if img_h < final_height else 0
            
            if i == 0:
                # First image: use all pixels
                start_x = 0
                end_x = img_w
                region_to_add = img_array
                region_w = img_w
            elif i == len(images) - 1:
                # Last image: skip first overlap_pixels (already covered by previous image)
                start_x = overlap_pixels
                end_x = img_w
                region_to_add = img_array[:, start_x:end_x]
                region_w = region_to_add.shape[1]
            else:
                # Middle images: include overlap region for averaging
                # We'll add the full image, but the overlap region will be averaged
                start_x = 0
                end_x = img_w
                region_to_add = img_array
                region_w = img_w
            
            # Add to combined array
            x_end = min(x_offset + region_w, total_width)
            actual_region_w = x_end - x_offset
            
            if actual_region_w > 0:
                # Handle case where region might be larger than remaining space
                if region_to_add.shape[1] > actual_region_w:
                    region_to_add = region_to_add[:, :actual_region_w]
                
                combined_array[y_offset:y_offset+img_h, x_offset:x_end] += region_to_add
                overlap_count[y_offset:y_offset+img_h, x_offset:x_end] += 1.0
            
            # Update x_offset for next image (accounting for overlap)
            if i < len(images) - 1:
                x_offset += (img_w - overlap_pixels)
        
        # Average overlapping regions
        overlap_count = np.maximum(overlap_count, 1.0)  # Avoid division by zero
        combined_array = combined_array / overlap_count
        
        # Crop/resize to exact target dimensions if needed
        if combined_array.shape[1] != target_width or combined_array.shape[0] != target_height:
            # Crop or pad to match target dimensions
            current_h, current_w = combined_array.shape
            
            # Handle width
            if current_w > target_width:
                # Crop from center
                crop_left = (current_w - target_width) // 2
                combined_array = combined_array[:, crop_left:crop_left+target_width]
            elif current_w < target_width:
                # Pad with zeros (shouldn't happen with correct overlap calculation)
                pad_left = (target_width - current_w) // 2
                pad_right = target_width - current_w - pad_left
                combined_array = np.pad(combined_array, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # Handle height
            if current_h > target_height:
                # Crop from center
                crop_top = (current_h - target_height) // 2
                combined_array = combined_array[crop_top:crop_top+target_height, :]
            elif current_h < target_height:
                # Pad with zeros
                pad_top = (target_height - current_h) // 2
                pad_bottom = target_height - current_h - pad_top
                combined_array = np.pad(combined_array, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        
        # Convert to uint8 and create final image
        combined_array = np.clip(combined_array, 0, 255).astype(np.uint8)
        combined = Image.fromarray(combined_array, mode='L')
        
        # Verify final dimensions
        if combined.size[0] != target_width or combined.size[1] != target_height:
            print(f"  ⚠ Warning: Final size {combined.size} doesn't match target {target_width}x{target_height}, resizing...")
            combined = combined.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Save result
        output_name = f"{key}_overlaid_strips_stitched.png"
        output_path = output_dir / output_name
        combined.save(output_path)
        
        print(f"\n  ✓ Saved: {output_name}")
        print(f"    Dimensions: {combined.size[0]} x {combined.size[1]} (W x H)")
        print(f"    Target: {target_width} x {target_height} (W x H)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

