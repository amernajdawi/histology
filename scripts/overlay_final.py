#!/usr/bin/env python3
"""
Final optimized overlay: H&E + GradCAM heatmap
"""

import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def overlay_heatmap_final(he_image, heatmap, alpha=0.25):
    """
    Overlay heatmap on H&E image with optimized settings.
    Shows red/yellow only in high-activation areas.
    """
    # Resize heatmap to match H&E
    if heatmap.shape != he_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (he_image.shape[1], he_image.shape[0]))
    
    # Normalize heatmap to [0, 1]
    if heatmap.max() > 1.0:
        heatmap = heatmap / heatmap.max()
    
    # Apply threshold: only show top activations (top 5% = 95th percentile)
    # This ensures we only show red/yellow for the most important regions
    threshold = np.percentile(heatmap, 95)
    heatmap_thresholded = np.where(heatmap > threshold, heatmap, 0.0)
    
    # Normalize thresholded heatmap
    if heatmap_thresholded.max() > 0:
        heatmap_thresholded = (heatmap_thresholded - heatmap_thresholded.min()) / (heatmap_thresholded.max() - heatmap_thresholded.min() + 1e-8)
    
    # Apply JET colormap (blue -> cyan -> green -> yellow -> red)
    heatmap_uint8 = (heatmap_thresholded * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Create mask for high-activation areas
    mask = (heatmap > threshold).astype(np.float32)[:, :, np.newaxis]
    
    # Blend: keep H&E everywhere, add heatmap only in high-activation areas
    he_float = he_image.astype(np.float32) if he_image.dtype != np.float32 else he_image
    heatmap_float = heatmap_color.astype(np.float32) / 255.0
    
    # Blend formula: result = (1 - mask*alpha) * H&E + (mask*alpha) * heatmap
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
    print(f"Percentile 70: {np.percentile(heatmap_gray, 70):.3f}")
    

    for alpha in [0.3, 0.4, 0.5]:
        result = overlay_heatmap_final(he_img_norm, heatmap_gray, alpha=alpha)
        output_path = f"outputs_success/he_overlay_warm_alpha{alpha}.png"
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        print(f"✅ Saved: {output_path} (alpha={alpha}, HOT colormap, top 5%)")


if __name__ == "__main__":
    main()

