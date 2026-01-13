#!/usr/bin/env python3
"""
Convert DICOM strips to PNG images

This script processes DICOM strips and generates:
- Original strip images
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "fastglioma"))

from fastglioma.datasets.improc import get_srh_base_aug
from fastglioma.models.resnet import resnet_backbone
from fastglioma.models.cnn import MLP
from fastglioma.models.mil import MIL_forward, MIL_Classifier, TransformerMIL
from functools import partial

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom not installed")
    sys.exit(1)


class StripReconstructor:
    """Process strips and convert to PNG images."""
    
    def __init__(self, checkpoint_path: str, output_dir: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
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
        """Load strip as [2, H, W]."""
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array
            
            if len(arr.shape) == 2:
                arr = np.stack([arr, arr], axis=0)
            elif len(arr.shape) == 3 and arr.shape[0] == 4:
                arr = arr[[0, 2], :, :]
            elif len(arr.shape) == 3:
                arr = arr[:2, :, :]
            
            tensor = torch.from_numpy(arr.astype(np.float32))
            
            if tensor.max() > 65536:
                tensor = (tensor / tensor.max()) * 65536.0
            elif tensor.max() <= 1.0:
                tensor = tensor * 65536.0
            elif tensor.max() <= 255:
                tensor = (tensor / 255.0) * 65536.0
            
            return tensor
        except:
            return None
    
    def process_individual_strip(self, strip_path: Path, output_name: str):
        """Process a single strip and convert to PNG."""
        print(f"\nProcessing strip: {strip_path.name}")
        
        # Load strip
        strip = self.load_strip(strip_path)
        if strip is None:
            print(f"  ✗ Failed to load strip")
            return None
        
        _, h, w = strip.shape
        print(f"  Strip dimensions: {w} x {h} (W x H)")
        
        # Create original strip image
        img_vis = strip[0].cpu().numpy()
        img_norm = img_vis.astype(np.float32) / img_vis.max() if img_vis.max() > 0 else img_vis
        
        # Save using PIL to ensure exact dimensions
        from PIL import Image as PILImage
        
        # Save original strip only
        original_uint8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        original_img = PILImage.fromarray(original_uint8)
        original_img.save(self.output_dir / f"{output_name}_original.png")
        
        print(f"  ✓ Saved original strip: {w} x {h}")
        
        return {
            'file': str(strip_path),
            'dimensions': (w, h)
        }
    
    def predict_patch(self, patch_tensor: torch.Tensor):
        """Predict patch."""
        with torch.no_grad():
            patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
            bag = [patch_tensor]
            coords = torch.tensor([[0, 0]], dtype=torch.float32).to(self.device)
            try:
                output = self.model(bag, coords=coords)
                logits = output['logits'] if isinstance(output, dict) else output
                score = torch.sigmoid(logits).item() if isinstance(logits, torch.Tensor) else torch.sigmoid(torch.tensor(logits)).item()
                return {'score': score}
            except:
                return {'score': 0.5}


def main():
    """Main function."""
    print("="*80)
    print("Convert DICOM Strips to PNG Images")
    print("="*80)
    
    base_dir = Path("/Users/ameralnajdawi/Desktop/new_his")
    checkpoint_path = base_dir / "fastglioma_ckpts" / "fastglioma_highres_model.ckpt"
    output_dir = base_dir / "strips_heatmaps"
    
    reconstructor = StripReconstructor(str(checkpoint_path), str(output_dir))
    
    # Process all strips from series 01
    strips_dir_01 = base_dir / "MUV_0635-2" / "01" / "strips"
    if strips_dir_01.exists():
        strip_files = sorted([f for f in strips_dir_01.glob("*.dcm")])
        print(f"\nProcessing {len(strip_files)} strips from series 01...")
        
        results_01 = []
        for i, strip_file in enumerate(strip_files, 1):
            output_name = f"series_01_strip_{i:02d}_{strip_file.stem}"
            result = reconstructor.process_individual_strip(strip_file, output_name)
            if result:
                results_01.append(result)
        
        print(f"\n✓ Processed {len(results_01)} strips from series 01")
    
    # Process all strips from series 02
    strips_dir_02 = base_dir / "MUV_0635-2" / "02" / "strips"
    if strips_dir_02.exists():
        strip_files = sorted([f for f in strips_dir_02.glob("*.dcm")])
        print(f"\nProcessing {len(strip_files)} strips from series 02...")
        
        results_02 = []
        for i, strip_file in enumerate(strip_files, 1):
            output_name = f"series_02_strip_{i:02d}_{strip_file.stem}"
            result = reconstructor.process_individual_strip(strip_file, output_name)
            if result:
                results_02.append(result)
        
        print(f"\n✓ Processed {len(results_02)} strips from series 02")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total strips processed: {len(results_01) + len(results_02)}")


if __name__ == "__main__":
    main()

