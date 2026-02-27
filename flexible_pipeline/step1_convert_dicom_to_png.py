#!/usr/bin/env python3
"""
Step 1: Convert DICOM strips to PNG images (FLEXIBLE VERSION)

This script automatically detects all datasets and series, and converts DICOM files to PNG.
Works with any dataset structure.

Usage:
    python step1_convert_dicom_to_png.py [base_dir] [checkpoint_path] [output_dir]

Arguments:
    base_dir: Base directory containing datasets (default: current directory)
    checkpoint_path: Path to FastGlioma checkpoint (default: fastglioma_ckpts/fastglioma_highres_model.ckpt)
    output_dir: Output directory for PNG strips (default: strips_heatmaps)
"""

import sys
import argparse
import re
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torchvision.transforms import Compose
from PIL import Image

# Add pipeline root so "fastglioma" package is found (fastglioma/ is under pipeline root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastglioma.datasets.improc import get_srh_base_aug
from fastglioma.models.resnet import resnet_backbone
from fastglioma.models.cnn import MLP
from fastglioma.models.mil import MIL_forward, MIL_Classifier, TransformerMIL
from functools import partial

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom not installed. Install with: pip install pydicom")
    sys.exit(1)


class StripReconstructor:
    """Process strips and convert to PNG images."""
    
    def __init__(self, checkpoint_path: str, output_dir: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: Model and transform are loaded but not used for PNG conversion
        # They're kept for potential future use (e.g., heatmap generation)
        # The preprocessing is done manually to match fastglioma pipeline exactly
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = Compose(get_srh_base_aug(base_aug="three_channels"))
    
    def _load_model(self):
        """Load FastGlioma model."""
        config = {
            "model": {
                "patch": {
                    "backbone": {"which": "resnet34", "params": {"num_channel_in": 3}},
                    "mlp_hidden": [],
                    "num_embedding_out": 128
                },
                "slide": {
                    "mil": {
                        "which": "transformer",
                        "params": {
                            "embed_dim": 512, "depth": 2, "num_heads": 4,
                            "pos_emb_type": "FFPEG", "pos_emb_grad": True, "prefix_len": 8
                        }
                    },
                    "mlp_hidden": [512]
                }
            }
        }
        
        bb = partial(resnet_backbone, arch="resnet34", num_channel_in=3)
        mil = partial(MIL_forward, mil=partial(TransformerMIL, **config["model"]["slide"]["mil"]["params"]))
        mlp = partial(MLP, n_in=mil().num_out, hidden_layers=config["model"]["slide"]["mlp_hidden"], n_out=1)
        model = MIL_Classifier(bb, mil, mlp)
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except:
            pass
        
        return model
    
    def load_strip(self, path: Path) -> Optional[torch.Tensor]:
        """Load strip as [2, H, W] following fastglioma preprocessing.
        
        Requirements:
        1. Load as 2-channel [2, H, W]
        2. Ignore 4-ALA channel (2nd row) if 4-channel
        3. Normalize to 0-65536 range (2^16)
        4. Apply fastglioma preprocessing: divide by 2^16, augment 3rd channel, clip
        """
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array
            
            # Handle different channel configurations
            if len(arr.shape) == 2:
                # Single channel - duplicate to 2 channels
                arr = np.stack([arr, arr], axis=0)
            elif len(arr.shape) == 3 and arr.shape[0] == 4:
                # 4-channel: ignore 2nd row (4-ALA channel), use channels 0 and 2
                arr = arr[[0, 2], :, :]
            elif len(arr.shape) == 3:
                # 3-channel or other: take first 2 channels
                arr = arr[:2, :, :]
            
            # Convert to tensor and normalize to 0-65536 range (2^16)
            tensor = torch.from_numpy(arr.astype(np.float32))
            
            # Normalize to 0-65536 range (v_max = 2^16)
            if tensor.max() > 65536:
                tensor = (tensor / tensor.max()) * 65536.0
            elif tensor.max() <= 1.0:
                tensor = tensor * 65536.0
            elif tensor.max() <= 255:
                tensor = (tensor / 255.0) * 65536.0
            
            # Ensure values are in valid range
            tensor = torch.clamp(tensor, 0.0, 65536.0)
            
            return tensor
        except Exception as e:
            print(f"  Warning: Failed to load {path.name}: {e}")
            return None
    
    def process_individual_strip(self, strip_path: Path, output_name: str):
        """Process a single strip and convert to PNG following fastglioma preprocessing.
        
        Applies:
        1. Divide by 2^16 (normalize)
        2. Augment 3rd channel: ch1 = ch3 - ch2 + base (where base = 5000/65536)
        3. Clip values to [0, 1]
        4. Save as grayscale PNG (using first channel for visualization)
        """
        print(f"  Processing: {strip_path.name}")
        
        # Load strip as [2, H, W]
        strip = self.load_strip(strip_path)
        if strip is None:
            return None
        
        _, h, w = strip.shape
        
        # Apply fastglioma preprocessing pipeline
        # Step 1: Normalize by dividing by 2^16 (65536)
        strip_normalized = strip / 65536.0
        
        # Step 2: Augment 3rd channel using fastglioma method
        # ch1 = ch3 - ch2 + base, where base = 5000/65536
        ch2 = strip_normalized[0, :, :]  # First channel
        ch3 = strip_normalized[1, :, :]  # Second channel
        subtracted_base = 5000.0 / 65536.0
        ch1 = ch3 - ch2 + subtracted_base
        
        # Step 3: Clip values to [0, 1] range
        ch1 = torch.clamp(ch1, 0.0, 1.0)
        ch2 = torch.clamp(ch2, 0.0, 1.0)
        ch3 = torch.clamp(ch3, 0.0, 1.0)
        
        # For visualization, use the original channel (ch2) with proper normalization
        # The preprocessing (ch1 augmentation) is applied but we use ch2 for visualization
        # since ch1 can be very dark when ch2 and ch3 are similar
        img_vis = ch2.cpu().numpy()
        
        # Normalize for display: stretch to full dynamic range
        if img_vis.max() > img_vis.min():
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        else:
            img_vis = img_vis
        
        # Convert to uint8 and save
        original_uint8 = (np.clip(img_vis, 0, 1) * 255).astype(np.uint8)
        original_img = Image.fromarray(original_uint8, mode='L')
        original_img.save(self.output_dir / f"{output_name}_original.png")
        
        print(f"  ✓ Saved processed strip: {w} x {h} (fastglioma preprocessing applied)")
        
        return {
            'file': str(strip_path),
            'dimensions': (w, h)
        }


def find_datasets_and_series(base_dir: Path):
    """Auto-detect all datasets and series."""
    datasets = []
    
    # Look for common dataset folder patterns
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains series folders (01, 02, etc.)
            series_found = []
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name.isdigit():
                    # Check if it has a 'strips' folder
                    strips_dir = subitem / "strips"
                    if strips_dir.exists() and any(strips_dir.glob("*.dcm")):
                        series_found.append(subitem.name)
            
            if series_found:
                datasets.append({
                    'name': item.name,
                    'path': item,
                    'series': sorted(series_found)
                })
    
    return datasets


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert DICOM strips to PNG images')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory containing datasets (default: current directory)')
    parser.add_argument('--checkpoint', type=str, default='fastglioma_ckpts/fastglioma_highres_model.ckpt',
                       help='Path to FastGlioma checkpoint (default: fastglioma_ckpts/fastglioma_highres_model.ckpt)')
    parser.add_argument('--output-dir', type=str, default='strips_heatmaps',
                       help='Output directory for PNG strips (default: strips_heatmaps)')
    parser.add_argument('--max-strips', type=int, default=None,
                       help='Process only the first N strips (for quick test). Default: process all.')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 1: Convert DICOM Strips to PNG Images (FLEXIBLE)")
    print("="*80)
    
    base_dir = Path(args.base_dir).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    
    output_dir = base_dir / args.output_dir
    
    print(f"\nBase directory: {base_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    if not checkpoint_path.exists():
        print(f"\n✗ ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Auto-detect datasets and series
    datasets = find_datasets_and_series(base_dir)
    
    if not datasets:
        print("\n✗ ERROR: No datasets found!")
        print("  Expected structure: <dataset_name>/<series_number>/strips/*.dcm")
        sys.exit(1)
    
    print(f"\nFound {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  - {ds['name']}: {len(ds['series'])} series ({', '.join(ds['series'])})")
    
    # Initialize reconstructor
    reconstructor = StripReconstructor(str(checkpoint_path), str(output_dir))
    
    total_processed = 0
    
    # Process all datasets and series
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*80}")
        
        for series_num in dataset['series']:
            strips_dir = dataset['path'] / series_num / "strips"
            strip_files = sorted([f for f in strips_dir.glob("*.dcm")])
            
            if not strip_files:
                print(f"\n  Series {series_num}: No DICOM files found")
                continue
            
            print(f"\n  Series {series_num}: {len(strip_files)} strips")
            
            # Extract position numbers from filenames and sort by position
            strip_files_with_pos = []
            for strip_file in strip_files:
                # Extract position: img1_10, img2_11, etc. -> position 10, 11, etc.
                match = re.search(r'img\d+_(\d+)', strip_file.name)
                if match:
                    position = int(match.group(1))
                    strip_files_with_pos.append((position, strip_file))
                else:
                    # Fallback: use index if no position found
                    strip_files_with_pos.append((len(strip_files_with_pos) + 1, strip_file))
            
            # Sort by position number
            strip_files_with_pos.sort(key=lambda x: x[0])
            
            # Limit to max_strips total (across all series) if set (for quick test run)
            if args.max_strips is not None and total_processed >= args.max_strips:
                break
            if args.max_strips is not None:
                remaining = args.max_strips - total_processed
                strip_files_with_pos = strip_files_with_pos[: remaining]
                print(f"  (Limiting to {len(strip_files_with_pos)} strips this series; {total_processed} already done)")
            
            results = []
            for position, strip_file in strip_files_with_pos:
                if args.max_strips is not None and total_processed + len(results) >= args.max_strips:
                    break
                # Use position number from filename, not sequential index
                output_name = f"{dataset['name']}_series_{series_num}_strip_{position:02d}_{strip_file.stem}"
                result = reconstructor.process_individual_strip(strip_file, output_name)
                if result:
                    results.append(result)
            
            print(f"  ✓ Processed {len(results)}/{len(strip_files)} strips")
            total_processed += len(results)
            if args.max_strips is not None and total_processed >= args.max_strips:
                print(f"  Reached max_strips={args.max_strips}. Stopping.")
                break
        if args.max_strips is not None and total_processed >= args.max_strips:
            break
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total strips processed: {total_processed}")


if __name__ == "__main__":
    main()

