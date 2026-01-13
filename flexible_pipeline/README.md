# Flexible Pipeline for Histopathology Image Processing

This pipeline processes DICOM strips, groups duplicates, and stitches them together. **Works with any dataset structure!**

## 📁 Required Structure

Your data should be organized like this:

```
your_project/
├── dataset_name/
│   ├── 01/
│   │   └── strips/
│   │       ├── strip1.dcm
│   │       ├── strip2.dcm
│   │       └── ...
│   ├── 02/
│   │   └── strips/
│   │       └── ...
│   └── ...
├── fastglioma_ckpts/
│   └── fastglioma_highres_model.ckpt
└── fastglioma/  (model library)
```

## 🚀 Usage

### Step 1: Convert DICOM to PNG

```bash
python step1_convert_dicom_to_png.py \
    --base-dir /path/to/your/project \
    --checkpoint fastglioma_ckpts/fastglioma_highres_model.ckpt \
    --output-dir strips_heatmaps
```

**What it does:**
- Auto-detects all datasets and series
- Converts each DICOM file to PNG
- Saves as `<dataset>_series_<num>_strip_<num>_*_original.png`

### Step 2: Overlay Duplicates

```bash
python step2_overlay_duplicates.py \
    --input-dir strips_heatmaps \
    --output-dir overlaid_strips \
    --similarity-threshold 0.75
```

**Optional:** Create `manual_pairs.json` for known duplicates:
```json
{
    "MUV_0635-2_series_01": [
        ["strip_03", "strip_04"]
    ]
}
```

Then run:
```bash
python step2_overlay_duplicates.py \
    --input-dir strips_heatmaps \
    --output-dir overlaid_strips \
    --manual-pairs manual_pairs.json
```

**What it does:**
- Groups similar/duplicate strips
- Overlays (averages) strips in each group
- Saves each group as a separate image

### Step 3: Stitch Groups

```bash
python step3_stitch_groups.py \
    --input-dir overlaid_strips \
    --output-dir combined_strips
```

**Optional:** Create `stitch_config.json` for custom ordering:
```json
{
    "MUV_0635-2_series_01": {
        "excluded_strips": [1, 2, 11, 5, 6, 7, 10, 15, 16, 17, 20, 12],
        "first_strips": [18, 19],
        "last_strips": [14]
    }
}
```

Then run:
```bash
python step3_stitch_groups.py \
    --input-dir overlaid_strips \
    --output-dir combined_strips \
    --config stitch_config.json
```

**What it does:**
- Stitches groups together in correct order
- Auto-detects black/empty groups (if no config)
- Saves final stitched image

## 📋 Complete Pipeline

Run all steps in sequence:

```bash
# Step 1
python step1_convert_dicom_to_png.py --base-dir . --output-dir strips_heatmaps

# Step 2
python step2_overlay_duplicates.py --input-dir strips_heatmaps --output-dir overlaid_strips

# Step 3
python step3_stitch_groups.py --input-dir overlaid_strips --output-dir combined_strips
```

## ⚙️ Configuration Options

### Manual Pairs (Step 2)
If you know certain strips are duplicates, specify them in JSON:
```json
{
    "dataset_series_01": [["strip_03", "strip_04"]],
    "dataset_series_02": [["strip_05", "strip_06", "strip_07"]]
}
```

### Stitching Config (Step 3)
Control which strips to exclude and ordering:
```json
{
    "dataset_series_01": {
        "excluded_strips": [1, 2, 11],  // Strip numbers to exclude
        "first_strips": [18, 19],        // Strips that should be first
        "last_strips": [14]              // Strips that should be last
    }
}
```

**Note:** If you don't provide a config, the script will:
- Auto-detect black/empty groups and exclude them
- Order groups by minimum strip number

## 🔍 Key Features

✅ **Auto-detection**: Finds all datasets and series automatically  
✅ **No hardcoding**: Works with any dataset name and structure  
✅ **Flexible**: Configurable via command-line or JSON files  
✅ **Same results**: Produces the same output as the original pipeline  

## 📊 Output Structure

```
strips_heatmaps/
├── dataset_series_01_strip_01_*_original.png
├── dataset_series_01_strip_02_*_original.png
└── ...

overlaid_strips/
├── dataset_series_01_group_01_overlaid_strips_03_04.png
├── dataset_series_01_group_02_*_single.png
└── ...

combined_strips/
├── dataset_series_01_overlaid_strips_stitched.png
└── ...
```

## 🆚 Differences from Original

| Feature | Original | Flexible |
|---------|----------|----------|
| Dataset name | Hardcoded (`MUV_0635-2`) | Auto-detected |
| Series numbers | Hardcoded (01, 02) | Auto-detected |
| Base directory | Hardcoded path | Command-line argument |
| Excluded strips | Hardcoded list | Configurable |
| Ordering | Hardcoded logic | Configurable or auto |

## 💡 Tips

1. **First time**: Run without config files to see auto-detection results
2. **Fine-tuning**: Create config files based on auto-detection results
3. **Multiple datasets**: Scripts process all datasets automatically
4. **Check output**: Review intermediate results before proceeding to next step

