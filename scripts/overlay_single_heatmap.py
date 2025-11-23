#!/usr/bin/env python3
"""
Simple overlay: Single heatmap on H&E image
"""

import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grad_cam_correct import overlay_heatmap


def main():
    # Load H&E image
    he_path = "unnamed.png"
    he_img = cv2.imread(he_path)
    if he_img is None:
        print(f"❌ Could not load {he_path}")
        return
    
    he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
    he_img_norm = he_img.astype(np.float32) / 255.0
    
    print(f"H&E image: {he_img_norm.shape}")
    
    # Load one heatmap
    heatmap_path = "outputs_success/heatmaps/img1_10_08874c947682d42af01faab3a9940e6ffa016f74_cam.png"
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img is None:
        print(f"❌ Could not load {heatmap_path}")
        return
    
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_img, (he_img_norm.shape[1], he_img_norm.shape[0]))
    heatmap_norm = heatmap_resized.astype(np.float32) / 255.0
    
    heatmap_gray = np.mean(heatmap_norm, axis=2)
    
    print(f"Heatmap: {heatmap_gray.shape}, min={heatmap_gray.min():.3f}, max={heatmap_gray.max():.3f}")
    

    for threshold in [0.3, 0.4, 0.5]:
        for alpha in [0.3, 0.4]:
            result = overlay_heatmap(he_img_norm, heatmap_gray, alpha=alpha, threshold=threshold)
            output_path = f"outputs_success/he_overlay_alpha{alpha}_th{threshold}.png"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            print(f"✅ Saved: {output_path} (alpha={alpha}, threshold={threshold})")


if __name__ == "__main__":
    main()

