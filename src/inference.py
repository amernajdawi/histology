import os
from typing import Dict, List, Tuple

import numpy as np
import pydicom
import torch
from tqdm import tqdm


def load_strip_dicom(path: str) -> torch.Tensor:
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # ensure 3 channels
    img = np.stack([img, img, img], axis=0)
    tensor = torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]
    return tensor


def extract_patches(img_tensor: torch.Tensor, patch_size: int = 224, stride: int = 224) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], Tuple[int, int]]:
    """Extract patches from strip image for FastGlioma MIL format.
    
    Returns:
        patches: List of patch tensors [C, H, W] (all resized to patch_size x patch_size)
        coords: List of (row, col) coordinates for each patch's top-left corner
                Note: row=y (height dimension), col=x (width dimension)
        img_shape: (H, W) - original image dimensions for proper reassembly
    """
    _, C, H, W = img_tensor.shape
    patches = []
    coords = []
    
    # Extract patches with the given stride (can be overlapping)
    for row in range(0, H, stride):
        for col in range(0, W, stride):
            # Get actual patch bounds in original image
            row_end = min(row + patch_size, H)
            col_end = min(col + patch_size, W)
            
            # Extract patch from original image
            patch = img_tensor[0, :, row:row_end, col:col_end]
            
            # If patch is smaller than patch_size, pad to the bottom/right
            _, ph, pw = patch.shape
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch = torch.nn.functional.pad(
                    patch, (0, pad_w, 0, pad_h), mode='constant', value=0.0
                )
            
            patches.append(patch)
            # Store coordinates as (row, col) where row=y (height), col=x (width)
            coords.append((row, col))
    
    # If no patches extracted (image too small), take whole image
    if len(patches) == 0:
        patch = torch.nn.functional.interpolate(
            img_tensor, size=(patch_size, patch_size), 
            mode='bilinear', align_corners=False)[0]
        patches.append(patch)
        coords.append((0, 0))
    
    return patches, coords, (H, W)


@torch.no_grad()
def run_inference_on_dir(model: torch.nn.Module, strips_dir: str) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    files = [f for f in os.listdir(strips_dir) if f.lower().endswith('.dcm')]
    for f in tqdm(sorted(files), desc=f"Predicting {os.path.basename(strips_dir)}"):
        path = os.path.join(strips_dir, f)
        x = load_strip_dicom(path)
        x = x.to(next(model.parameters()).device) if any(p.requires_grad for p in model.parameters()) else x
        y = model(x)
        results[f] = y.squeeze().detach().cpu().numpy()
    return results


