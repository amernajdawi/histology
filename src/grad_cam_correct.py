"""
Correct Grad-CAM implementation for FastGlioma MIL architecture.

The FastGlioma model has 3 components:
1. Backbone (ResNet34): Processes each patch individually [1, 3, 224, 224] -> [128]
2. MIL Transformer: Takes list of embeddings + coords -> [512]
3. Head (MLP): Takes [512] -> [1] (tumor score)

For Grad-CAM on MIL models, we compute:
- Gradients of the final tumor score with respect to backbone features
- Apply Grad-CAM separately for each patch
- Assemble smoothly into whole-slide heatmap

References:
- FastGlioma: https://www.nature.com/articles/s41586-024-08169-3
- Grad-CAM: https://arxiv.org/abs/1610.02391
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GradCAMForMIL:
    """Grad-CAM for Multiple Instance Learning models."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: FastGlioma MIL_Classifier
            target_layer: Layer to hook for Grad-CAM (e.g., backbone.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        
    def _clear(self):
        self.activations = []
        self.gradients = []
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        fh = self.target_layer.register_forward_hook(forward_hook)
        # Use register_full_backward_hook to avoid issues with multi-node graphs
        bh = self.target_layer.register_full_backward_hook(backward_hook)
        
        return fh, bh
    
    def generate(
        self,
        patches: List[torch.Tensor],
        coords: torch.Tensor,
        device: str = "cuda"
    ) -> Tuple[List[np.ndarray], float]:
        """
        Generate Grad-CAM for a bag of patches.
        
        Args:
            patches: List of patch tensors [C, H, W]
            coords: Coordinates [num_patches, 2] where each row is [row, col]
            device: Device to run on
            
        Returns:
            patch_cams: List of CAM arrays [H', W'] for each patch
            score: Prediction score (logit)
        """
        self._clear()
        self.model.zero_grad()
        
        # Register hooks
        fh, bh = self._register_hooks()
        
        try:
            # FastGlioma expects:
            # bag: list with ONE element, which is a tensor [num_patches, C, H, W]
            # coords: list with ONE element, which is a tensor [num_patches, 2]
            
            # Stack all patches into a single tensor
            patches_stacked = []
            for p in patches:
                if p.ndim == 3:
                    patches_stacked.append(p)  # [C, H, W]
                elif p.ndim == 4:
                    patches_stacked.append(p[0])  # [C, H, W]
                else:
                    raise ValueError(f"Invalid patch shape: {p.shape}")
            
            # Stack to [num_patches, C, H, W]
            patches_tensor = torch.stack(patches_stacked, dim=0).to(device)
            
            # Wrap in lists as expected by model
            bag = [patches_tensor]  # List with one tensor [num_patches, C, H, W]
            coords_list = [coords.to(device)]  # List with one tensor [num_patches, 2]
            
            # Forward pass
            output = self.model(bag, coords=coords_list)
            
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output
            
            # Backward pass
            score = logits.squeeze()
            if score.ndim > 0:
                score = score.sum()
            
            score.backward()
            
            # The backbone processes all patches in one batch, so we get ONE activation/gradient
            # with shape [num_patches, C, H', W']
            if len(self.activations) != 1 or len(self.gradients) != 1:
                raise RuntimeError(
                    f"Expected 1 activation and 1 gradient, got {len(self.activations)} "
                    f"activations and {len(self.gradients)} gradients"
                )
            
            A = self.activations[0]  # [num_patches, C, H', W']
            G = self.gradients[0]     # [num_patches, C, H', W']
            
            # Verify batch dimension matches number of patches
            if A.shape[0] != len(patches):
                raise RuntimeError(
                    f"Activation batch size {A.shape[0]} doesn't match "
                    f"number of patches {len(patches)}"
                )
            
            # Compute CAM for each patch
            patch_cams = []
            for i in range(len(patches)):
                # Extract activation and gradient for this patch
                A_i = A[i:i+1]  # [1, C, H', W']
                G_i = G[i:i+1]  # [1, C, H', W']
                
                # Compute Grad-CAM weights
                weights = G_i.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                cam = (weights * A_i).sum(dim=1)  # [1, H', W']
                cam = F.relu(cam)
                cam = cam[0].cpu().numpy()  # [H', W']
                patch_cams.append(cam)
            
            score_value = logits.detach().cpu().item()
            
        finally:
            fh.remove()
            bh.remove()
        
        return patch_cams, score_value


def assemble_heatmap(
    patch_cams: List[np.ndarray],
    coords: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    patch_size: int = 224
) -> np.ndarray:
    """
    Assemble patch CAMs into smooth whole-image heatmap.
    
    Args:
        patch_cams: List of [H', W'] CAM arrays from backbone
        coords: List of (row, col) patch coordinates
        image_shape: (H, W) original image shape
        patch_size: Patch size in pixels
        
    Returns:
        heatmap: Normalized heatmap [H, W] in range [0, 1]
    """
    H, W = image_shape
    
    if len(patch_cams) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    # Get backbone feature map size
    cam_h, cam_w = patch_cams[0].shape
    
    # Calculate grid dimensions
    num_patches_h = (H + patch_size - 1) // patch_size
    num_patches_w = (W + patch_size - 1) // patch_size
    
    # Create low-res grid
    grid_h = num_patches_h * cam_h
    grid_w = num_patches_w * cam_w
    cam_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    count_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    # Place each CAM at its grid position
    for cam, (row, col) in zip(patch_cams, coords):
        patch_idx_h = row // patch_size
        patch_idx_w = col // patch_size
        
        grid_row = patch_idx_h * cam_h
        grid_col = patch_idx_w * cam_w
        
        if grid_row + cam_h <= grid_h and grid_col + cam_w <= grid_w:
            cam_grid[grid_row:grid_row+cam_h, grid_col:grid_col+cam_w] += cam
            count_grid[grid_row:grid_row+cam_h, grid_col:grid_col+cam_w] += 1.0
    
    # Average overlaps
    count_grid = np.maximum(count_grid, 1.0)
    cam_grid = cam_grid / count_grid
    
    # Upscale to full resolution
    heatmap = cv2.resize(cam_grid, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Global normalization
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin + 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)
    
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap on image using JET colormap, showing only high-activation areas.
    
    Args:
        image: RGB [H, W, 3] in [0, 1]
        heatmap: Heatmap [H, W] in [0, 1]
        alpha: Blending weight for heatmap (default 0.3 = 30% heatmap, 70% image)
        threshold: Only show heatmap above this threshold (default 0.5)
        
    Returns:
        overlay: RGB uint8 [H, W, 3]
    """
    # Resize heatmap if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to [0, 1] if needed
    if heatmap.max() > 1.0:
        heatmap = heatmap / heatmap.max()
    
    # Apply threshold - only show high-activation areas
    # Create mask for high-activation regions
    heatmap_thresholded = np.where(heatmap > threshold, heatmap, 0.0)
    
    # Normalize thresholded heatmap to [0, 1] for colormap
    if heatmap_thresholded.max() > 0:
        heatmap_thresholded = (heatmap_thresholded - heatmap_thresholded.min()) / (heatmap_thresholded.max() - heatmap_thresholded.min() + 1e-8)
    
    # Apply HOT colormap for warm colors (black -> red -> yellow)
    # HOT is perfect for Grad-CAM: shows red/yellow for high activations
    heatmap_for_colormap = (heatmap_thresholded * 255).astype(np.uint8)
    
    # Use HOT colormap (black -> red -> orange -> yellow)
    # This gives warm colors (red/yellow) for high activations
    heatmap_color = cv2.applyColorMap(heatmap_for_colormap, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Create mask: only apply heatmap where activation is above threshold
    mask = (heatmap > threshold).astype(np.float32)[:, :, np.newaxis]
    
    # Normalize to [0, 1] for blending
    heatmap_color_norm = heatmap_color.astype(np.float32) / 255.0
    image_float = image.astype(np.float32) if image.dtype != np.float32 else image
    
    # Blend only in high-activation areas
    # Outside mask: keep original image
    # Inside mask: blend with heatmap
    overlay = image_float.copy()
    overlay = overlay * (1 - mask * alpha) + heatmap_color_norm * (mask * alpha)
    
    overlay = (overlay * 255).astype(np.uint8)
    
    return overlay

