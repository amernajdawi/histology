# Histology Image Processing Pipeline

A flexible pipeline for processing histopathology DICOM strips, stitching them together, and overlaying Grad-CAM heatmaps.

## What It Does

This pipeline takes DICOM image strips and:
1. Converts them to PNG images with FastGlioma preprocessing
2. Groups and overlays duplicate strips
3. Stitches strips together in the correct order
4. Reassembles Grad-CAM heatmap patches
5. Overlays the heatmap on the stitched image
6. Combines everything with the original histopathology image

## Quick Start

The main pipeline is in the `flexible_pipeline/` folder. Run the steps in order:

```bash
# Step 1: Convert DICOM to PNG
python flexible_pipeline/step1_convert_dicom_to_png.py

# Step 2: Overlay duplicate strips
python flexible_pipeline/step2_overlay_duplicates.py

# Step 3: Stitch groups together
python flexible_pipeline/step3_stitch_groups.py

# Step 4: Reassemble Grad-CAM patches
python flexible_pipeline/step4_reassemble_gradcam.py

# Step 5: Overlay Grad-CAM on stitched image
python flexible_pipeline/step5_overlay_gradcam.py

# Step 6: Combine with original image
python flexible_pipeline/step6_combine_with_original.py
```

## Configuration

You can customize the pipeline using `flexible_pipeline/example_config.json`:
- Specify which strips are duplicates
- Control strip ordering and exclusions
- Set target image dimensions

See `flexible_pipeline/README.md` for detailed documentation.

## Requirements

- Python 3.7+
- PyTorch
- pydicom
- PIL/Pillow
- scikit-image
- FastGlioma model checkpoints (in `fastglioma_ckpts/`)

## Output

Final results are saved in `stitched_images/`:
- `unnamed_with_gradcam.png` - Original image with Grad-CAM overlay (3600x3900)
- `*_with_gradcam.png` - Stitched images with heatmaps

## References

- FastGlioma: https://www.nature.com/articles/s41586-024-08169-3
- Grad-CAM: https://arxiv.org/abs/1610.02391
