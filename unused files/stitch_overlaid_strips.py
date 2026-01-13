#!/usr/bin/env python3
"""
Stitch overlaid strips together in the correct order
Group 7 (strips 18, 19) is FIRST
Group 6 (strip 14) is LAST
"""

from PIL import Image
from pathlib import Path
import re

def extract_strip_numbers_from_filename(filename):
    """Extract strip numbers from filename."""
    # For overlaid: series_01_group_01_overlaid_strips_03_04.png
    # For single: series_01_group_05_strip_12_single.png
    
    if 'overlaid_strips' in filename:
        # Extract numbers after "overlaid_strips_"
        match = re.search(r'overlaid_strips_(\d+(?:_\d+)*)', filename)
        if match:
            numbers = [int(n) for n in match.group(1).split('_')]
            return numbers
    elif 'single' in filename:
        # Extract number after "strip_"
        match = re.search(r'strip_(\d+)_single', filename)
        if match:
            return [int(match.group(1))]
    
    return []

def determine_order():
    """Determine the correct order of groups based on strip positions."""
    
    base_dir = Path("/Users/ameralnajdawi/Desktop/new_his")
    overlaid_dir = base_dir / "overlaid_strips"
    
    # Get all series 01 group files
    group_files = sorted([f for f in overlaid_dir.glob("series_01_group_*.png")])
    
    # Map each group to its strip numbers
    group_info = []
    for group_file in group_files:
        strip_nums = extract_strip_numbers_from_filename(group_file.name)
        if strip_nums:
            # Use the minimum strip number as the position for ordering
            min_strip = min(strip_nums)
            group_info.append({
                'file': group_file,
                'strip_numbers': strip_nums,
                'min_strip': min_strip,
                'filename': group_file.name
            })
    
    # Sort by minimum strip number
    group_info.sort(key=lambda x: x['min_strip'])
    
    # But user said group 7 (18,19) is FIRST and group 6 (14) is LAST
    # This suggests a circular or specific ordering
    # Let me check: if 18,19 is first, maybe we need to start from there
    
    # Exclude groups: Group 2 (1,2,11), Group 3 (5,6,7,10,15,16,17,20), Group 5 (12)
    excluded_strips = {1, 2, 11, 5, 6, 7, 10, 15, 16, 17, 20, 12}
    
    # Filter out excluded groups
    filtered_groups = []
    for info in group_info:
        # Check if this group contains any excluded strips
        if not any(strip in excluded_strips for strip in info['strip_numbers']):
            filtered_groups.append(info)
    
    # Find group 7 (strips 18,19) and group 6 (strip 14)
    group_7 = None
    group_6 = None
    other_groups = []
    
    for info in filtered_groups:
        if 18 in info['strip_numbers'] or 19 in info['strip_numbers']:
            group_7 = info
        elif 14 in info['strip_numbers']:
            group_6 = info
        else:
            other_groups.append(info)
    
    # Order: group 7 first, then others in order, then group 6 last
    if group_7 and group_6:
        ordered_groups = [group_7] + sorted(other_groups, key=lambda x: x['min_strip']) + [group_6]
    else:
        # Fallback: just sort by min_strip
        ordered_groups = sorted(filtered_groups, key=lambda x: x['min_strip'])
    
    return ordered_groups

def stitch_overlaid_strips():
    """Stitch all overlaid strips together in correct order."""
    
    print("="*80)
    print("Stitch Overlaid Strips Together")
    print("="*80)
    
    base_dir = Path("/Users/ameralnajdawi/Desktop/new_his")
    output_dir = base_dir / "combined_strips"
    output_dir.mkdir(exist_ok=True)
    
    # Determine order
    ordered_groups = determine_order()
    
    print(f"\nFound {len(ordered_groups)} groups to stitch:")
    for i, info in enumerate(ordered_groups, 1):
        print(f"  {i}. {info['filename']} (strips: {info['strip_numbers']})")
    
    # Load all images
    images = []
    for info in ordered_groups:
        img = Image.open(info['file'])
        if img.mode != 'L':
            img = img.convert('L')
        images.append(img)
        print(f"\n  Loaded: {info['filename']}")
        print(f"    Dimensions: {img.size[0]} x {img.size[1]}")
    
    if not images:
        print("\n✗ No images to stitch!")
        return
    
    # Check if all images have the same height
    heights = [img.size[1] for img in images]
    widths = [img.size[0] for img in images]
    
    if len(set(heights)) > 1:
        print(f"\n⚠ Warning: Images have different heights: {heights}")
        print("  Using maximum height and centering images")
        max_height = max(heights)
    else:
        max_height = heights[0]
    
    # Calculate total width
    total_width = sum(widths)
    
    print(f"\nCombining {len(images)} images:")
    print(f"  Total width: {total_width} pixels")
    print(f"  Height: {max_height} pixels")
    
    # Create combined image
    combined = Image.new('L', (total_width, max_height))
    
    x_offset = 0
    for i, (img, info) in enumerate(zip(images, ordered_groups), 1):
        # Center vertically if heights differ
        if img.size[1] < max_height:
            y_offset = (max_height - img.size[1]) // 2
        else:
            y_offset = 0
        
        combined.paste(img, (x_offset, y_offset))
        print(f"  ✓ Placed {i}/{len(images)}: {info['filename']} at x={x_offset}")
        x_offset += img.size[0]
    
    # Save result
    output_path = output_dir / "series_01_overlaid_strips_stitched.png"
    combined.save(output_path)
    
    print(f"\n✓ Saved stitched image:")
    print(f"  {output_path}")
    print(f"  Dimensions: {combined.size[0]} x {combined.size[1]} (W x H)")
    print(f"  Total groups: {len(ordered_groups)}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    stitch_overlaid_strips()

