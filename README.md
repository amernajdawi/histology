# Histology Grad-CAM Visualization

FastGlioma + Grad-CAM heatmap visualization for tumor infiltration detection.

**References:**
- FastGlioma: https://www.nature.com/articles/s41586-024-08169-3
- Grad-CAM: https://arxiv.org/abs/1610.02391

---

## Python Scripts

### 1. `run_fastglioma_gradcam_final.py` - Generate Heatmaps
**Main pipeline** - Creates all heatmap images from SRH strips.

```bash
python scripts/run_fastglioma_gradcam_final.py \
    --strips-root MUV_0635-2 \
    --histo-root MUV_0635 \
    --out outputs_success
```

**Outputs:**
- `outputs_success/heatmaps/*_cam.png` - Individual heatmaps
- `outputs_success/overlays/*_overlay.png` - Overlays on histology

---

### 2. `overlay_strips_2rows_horizontal.py` - 2 Rows Horizontal ⭐
Arranges all heatmap strips in 2 horizontal rows on histopathology image.

```bash
python scripts/overlay_strips_2rows_horizontal.py \
    --he-image unnamed.png \
    --heatmaps-dir outputs_success/heatmaps \
    --out outputs_success/he_overlay_2rows_horizontal.png \
    --alpha 0.4
```

**Output:** `he_overlay_2rows_horizontal.png` - All strips in 2 rows

---

### 3. `overlay_heatmap_on_he.py` - Combined Overlay
Combines all heatmaps into one overlay on histopathology image.

```bash
python scripts/overlay_heatmap_on_he.py \
    --he-image unnamed.png \
    --heatmaps-dir outputs_success/heatmaps \
    --out outputs_success/he_overlay.png \
    --alpha 0.4
```

**Output:** `he_overlay.png` - Single combined heatmap overlay

---

### 4. `stitch_histopathology_overlay.py` - Stitched View
Stitches histopathology strips horizontally and overlays heatmaps.

```bash
python scripts/stitch_histopathology_overlay.py \
    --histo-root MUV_0635 \
    --heatmaps-dir outputs_success/heatmaps \
    --out outputs_success/stitched_overlay_final.png \
    --overlap 50
```

**Output:** `stitched_overlay_final_01.png` - Wide stitched view with heatmaps

---

### 5. `overlay_40strips_2rows.py` - Grouped Strips
Arranges 40 specific strips (img1/img2) in 2 rows.

```bash
python scripts/overlay_40strips_2rows.py
```

**Output:** `he_overlay_40strips_2rows.png` - 40 strips grouped by type

---

## Output Images

### Main Visualizations

**`he_overlay_2rows_horizontal.png`** ⭐
- All heatmap strips in 2 horizontal rows on histopathology image
- Created by: `overlay_strips_2rows_horizontal.py`
- **Best for:** Final presentation

**`he_overlay.png`**
- Combined heatmap overlay on histopathology image
- Created by: `overlay_heatmap_on_he.py`
- Shows overall model attention

**`stitched_overlay_final_01.png`**
- Wide stitched histopathology with heatmaps in correct positions
- Created by: `stitch_histopathology_overlay.py`
- Shows spatial relationships

**`he_overlay_40strips_2rows.png`**
- 40 strips (img1/img2) in 2 rows
- Created by: `overlay_40strips_2rows.py`

**`he_overlay_warm_alpha0.5.png`**
- Test output (intermediate file)

### Directories

**`outputs_success/heatmaps/`**
- Individual Grad-CAM heatmap images (`*_cam.png`)
- Red/yellow = high activation (tumor), dark = low activation (normal)
- Created by: `run_fastglioma_gradcam_final.py`

**`outputs_success/overlays/`**
- Heatmaps overlaid on histology reference (`*_overlay.png`)
- Created by: `run_fastglioma_gradcam_final.py`

---

## Core Code (`src/`)

### `grad_cam_correct.py`
**Grad-CAM implementation** for FastGlioma MIL architecture.
- `GradCAMForMIL` class - Generates Grad-CAM for Multiple Instance Learning models
- `assemble_heatmap()` - Assembles patch CAMs into smooth whole-image heatmap
- `overlay_heatmap()` - Overlays heatmap on image using HOT colormap
- Used by all visualization scripts

### `model_loader.py`
**Model loading utilities** for FastGlioma.
- `load_official_fastglioma()` - Loads FastGlioma model from checkpoints or HuggingFace
- `get_default_target_layer()` - Finds target layer for Grad-CAM
- Handles both local checkpoints and HuggingFace downloads

### `inference.py`
**Inference utilities** for processing DICOM strips.
- `load_strip_dicom()` - Loads and normalizes DICOM images
- `extract_patches()` - Extracts patches from strip images for model input
- `run_inference_on_dir()` - Runs inference on directory of strips

---

## Quick Start

1. **Generate heatmaps:**
```bash
python scripts/run_fastglioma_gradcam_final.py \
    --strips-root MUV_0635-2 \
    --histo-root MUV_0635 \
    --out outputs_success
```

2. **Create visualization:**
```bash
python scripts/overlay_strips_2rows_horizontal.py \
    --he-image unnamed.png \
    --heatmaps-dir outputs_success/heatmaps \
    --out outputs_success/he_overlay_2rows_horizontal.png \
    --alpha 0.4
```

---

## Color Meanings

- **Red/Yellow (HOT):** High activation = Tumor infiltration detected
- **Dark/Black:** Low activation = Normal tissue
- **Pink/Purple:** Original H&E stained tissue (background)
