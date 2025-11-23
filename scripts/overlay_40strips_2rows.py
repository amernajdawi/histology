#!/usr/bin/env python3
"""
Overlay 40 GradCAM heatmap strips on H&E image
Arranges 40 strips horizontally in 2 rows (20 strips per row)
- Row 1: img1 strips (7-26)
- Row 2: img2 strips (7-26)
"""

import cv2
import numpy as np
import sys
import glob
import os
import re
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_strip_info(filename: str) -> tuple:

    match = re.search(r'(img\d+)_(\d+)', filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return (None, 0)


def overlay_strip_on_region(he_image, heatmap_path, x_start, x_end, y_start, y_end, alpha=0.4):

    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img is None:
        return he_image
    
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    

    region_h = y_end - y_start
    region_w = x_end - x_start
    

    heatmap_resized = cv2.resize(heatmap_img, (region_w, region_h))
    heatmap_norm = heatmap_resized.astype(np.float32) / 255.0
    

    heatmap_gray = np.mean(heatmap_norm, axis=2)
    

    if heatmap_gray.max() > 1.0:
        heatmap_gray = heatmap_gray / heatmap_gray.max()
    

    threshold = np.percentile(heatmap_gray, 95)
    

    mask = (heatmap_gray > threshold).astype(np.float32)[:, :, np.newaxis]
    

    heatmap_thresholded = np.where(heatmap_gray > threshold, heatmap_gray, 0.0)
    

    if heatmap_thresholded.max() > 0:
        heatmap_thresholded = (heatmap_thresholded - heatmap_thresholded.min()) / (heatmap_thresholded.max() - heatmap_thresholded.min() + 1e-8)
    else:
  
        return he_image
    
    heatmap_uint8 = (heatmap_thresholded * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype(np.float32) / 255.0
    

    region = he_image[y_start:y_end, x_start:x_end].copy()
    

    blended_region = region * (1 - mask * alpha) + heatmap_color * (mask * alpha)
    

    he_image[y_start:y_end, x_start:x_end] = blended_region
    
    return he_image


def main():
    # Load H&E image
    he_path = "unnamed.png"
    he_img = cv2.imread(he_path)
    if he_img is None:
        print(f"❌ Could not load {he_path}")
        return
    
    he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
    he_img_norm = he_img.astype(np.float32) / 255.0
    
    he_h, he_w = he_img_norm.shape[:2]
    print(f"H&E image: {he_img_norm.shape} (H={he_h}, W={he_w})")
    
    # Find all heatmap files
    heatmap_dir = "outputs_success/heatmaps"
    heatmap_files = glob.glob(os.path.join(heatmap_dir, "*_cam.png"))
    
    if not heatmap_files:
        print(f"❌ No heatmap files found in {heatmap_dir}")
        return
    
    print(f"Found {len(heatmap_files)} heatmap files")
    
    heatmaps_with_info = []
    for heatmap_path in heatmap_files:
        filename = os.path.basename(heatmap_path)
        img_group, strip_num = parse_strip_info(filename)
        if img_group and strip_num:
            heatmaps_with_info.append((img_group, strip_num, heatmap_path))
    
    heatmaps_with_info.sort(key=lambda x: (x[0], x[1]))
    
    print(f"Sorted {len(heatmaps_with_info)} heatmaps")
    print(f"  Row 1: {len([h for h in heatmaps_with_info if h[0] == 'img1'])} strips (img1)")
    print(f"  Row 2: {len([h for h in heatmaps_with_info if h[0] == 'img2'])} strips (img2)")
    
    num_rows = 2
    strips_per_row = len(heatmaps_with_info) // num_rows
    
    row_height = he_h // num_rows
    strip_width = he_w // strips_per_row
    
    print(f"\nLayout:")
    print(f"  Rows: {num_rows}, Strips per row: {strips_per_row}")
    print(f"  Row height: {row_height}px, Strip width: {strip_width}px")
    
    result = he_img_norm.copy()
    
    for idx, (img_group, strip_num, heatmap_path) in enumerate(tqdm(heatmaps_with_info, desc="Overlaying strips")):
        row_idx = 0 if img_group == 'img1' else 1
        
        if img_group == 'img1':
            col_idx = idx
        else:
            col_idx = idx - strips_per_row
        
        y_start = row_idx * row_height
        y_end = (row_idx + 1) * row_height
        x_start = col_idx * strip_width
        x_end = (col_idx + 1) * strip_width
        
        y_end = min(y_end, he_h)
        x_end = min(x_end, he_w)
        

        result = overlay_strip_on_region(result, heatmap_path, x_start, x_end, y_start, y_end, alpha=0.4)
    
    # Convert back to uint8 and save
    result_uint8 = (result * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
    
    output_path = "outputs_success/he_overlay_40strips_2rows.png"
    cv2.imwrite(output_path, result_bgr)
    print(f"\n✅ Saved: {output_path}")
    print(f"   Overlaid {len(heatmaps_with_info)} strips in 2 rows")


if __name__ == "__main__":
    main()

