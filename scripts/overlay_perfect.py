#!/usr/bin/env python3
"""
Perfect overlay: H&E + GradCAM with warm colors (red/yellow)
"""

import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def overlay_heatmap_perfect(he_image, heatmap, alpha=0.3):
    """
    Perfect overlay: H&E background + warm heatmap (red/yellow) for high activations.
    """
    if heatmap.shape != he_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (he_image.shape[1], he_image.shape[0]))
    
    if heatmap.max() > 1.0:
        heatmap = heatmap / heatmap.max()
    
    threshold = np.percentile(heatmap, 90)
    heatmap_thresholded = np.where(heatmap > threshold, heatmap, 0.0)
    
    if heatmap_thresholded.max() > 0:
        heatmap_thresholded = (heatmap_thresholded - heatmap_thresholded.min()) / (heatmap_thresholded.max() - heatmap_thresholded.min() + 1e-8)
    
    heatmap_uint8 = (heatmap_thresholded * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    mask = (heatmap > threshold).astype(np.float32)[:, :, np.newaxis]
    
    he_float = he_image.astype(np.float32) if he_image.dtype != np.float32 else he_image
    heatmap_float = heatmap_color.astype(np.float32) / 255.0
    
    overlay = he_float * (1 - mask * alpha) + heatmap_float * (mask * alpha)
    
    return (overlay * 255).astype(np.uint8)


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
    
    # Extract grayscale
    heatmap_gray = np.mean(heatmap_norm, axis=2)
    
    print(f"Heatmap: {heatmap_gray.shape}, min={heatmap_gray.min():.3f}, max={heatmap_gray.max():.3f}")
    threshold = np.percentile(heatmap_gray, 90)
    print(f"Threshold (90th percentile): {threshold:.3f}")
    coverage = np.sum(heatmap_gray > threshold) / heatmap_gray.size * 100
    print(f"Coverage: {coverage:.1f}%")
    
    # Create perfect overlay
    result = overlay_heatmap_perfect(he_img_norm, heatmap_gray, alpha=0.3)
    output_path = "outputs_success/he_overlay_perfect.png"
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"\n✅ Saved: {output_path}")
    print(f"   → H&E tissue واضح (pink/purple)")
    print(f"   → Heatmap بألوان دافئة (red/yellow) - HOT colormap")
    print(f"   → يظهر فقط في المناطق عالية التنشيط (top 10%)")
    print(f"   → جاهزة للإرسال إلى Thomas")


if __name__ == "__main__":
    main()

