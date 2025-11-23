import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

# Optional official FastGlioma imports
try:
    from huggingface_hub import hf_hub_download
    import yaml
    import pytorch_lightning as pl
    from fastglioma.inference.run_inference import FastGliomaInferenceSystem
    _HAS_OFFICIAL = True
except Exception:
    _HAS_OFFICIAL = False


class ResNet50SingleHead(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def try_load_torchscript(weights_path: str) -> Optional[nn.Module]:
    try:
        model = torch.jit.load(weights_path, map_location="cpu")
        model.eval()
        return model
    except Exception:
        return None


def try_load_state_dict(weights_path: str) -> Optional[nn.Module]:
    try:
        state = torch.load(weights_path, map_location="cpu")
        model = ResNet50SingleHead()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            # It's okay if state_dict contains extra keys
            pass
        model.eval()
        return model
    except Exception:
        return None


def load_fastglioma_model(weights_path: str) -> nn.Module:
    # Prefer the official pipeline if available via HuggingFace
    use_official = os.environ.get("FASTGLIOMA_OFFICIAL", "1") == "1" and _HAS_OFFICIAL
    if use_official:
        try:
            return load_official_fastglioma()
        except Exception:
            pass

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = try_load_torchscript(weights_path)
    if model is not None:
        return model

    model = try_load_state_dict(weights_path)
    if model is not None:
        return model

    raise RuntimeError(
        "Failed to load FastGlioma weights. If you have the official repo, "
        "replace load_fastglioma_model() to use its build function."
    )


def get_default_target_layer(model: nn.Module):
    # For official FastGlioma MIL model: target backbone's last conv layer
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        return model.backbone.layer4[-1]
    
    # For ResNet-like models
    try:
        return model.backbone.layer4[-1]
    except Exception:
        # Try to find any ResNet-style layer4
        for name, mod in model.named_modules():
            if 'layer4' in name or 'backbone.layer4' in name:
                if isinstance(mod, nn.Sequential) and len(mod) > 0:
                    return mod[-1]
    
    # Fallback: any last module with parameters
    last_mod = None
    for last_mod in model.modules():
        pass
    return last_mod


def load_official_fastglioma() -> nn.Module:
    """Load the official FastGlioma model via their LightningModule and checkpoint.

    - First tries to find local checkpoint folder
    - Falls back to HuggingFace download if local not found
    - Loads FastGliomaInferenceSystem and returns its .model (MIL_Classifier)
    """
    # Try local checkpoint folder first
    local_ckpt_dir = "/Users/ameralnajdawi/Desktop/histology/fastglioma_ckpts copy"
    local_cfg_path = os.path.join(local_ckpt_dir, "config.yaml")
    local_ckpt_path = os.path.join(local_ckpt_dir, "fastglioma_highres_model.ckpt")
    
    if os.path.exists(local_ckpt_path) and os.path.exists(local_cfg_path):
        # Use local checkpoint
        with open(local_cfg_path, "r") as f:
            cf = yaml.safe_load(f)
        ckpt_path = local_ckpt_path
    else:
        # Fall back to HuggingFace
        repo_cfg_path = \
            "/Users/ameralnajdawi/Desktop/histology/fastglioma/fastglioma/inference/config/infer.yaml"
        if not os.path.exists(repo_cfg_path):
            # Fallback: try site-packages location if installed in a different place
            try:
                import fastglioma as fg
                pkg_root = os.path.dirname(fg.__file__)
                repo_cfg_path = os.path.join(pkg_root, "inference", "config", "infer.yaml")
            except Exception:
                raise FileNotFoundError("Cannot find FastGlioma infer.yaml config.")

        with open(repo_cfg_path, "r") as f:
            cf = yaml.safe_load(f)

        repo_id = cf["infra"]["hf_repo"]
        ckpt_name = cf["eval"]["ckpt_path"]
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    
    system: pl.LightningModule = FastGliomaInferenceSystem.load_from_checkpoint(
        ckpt_path, cf=cf, num_it_per_ep=0)
    model = system.model
    model.eval()
    return model


