# Project Explanation: Histology Image Processing Pipeline

## 🎯 What We Did (The Big Picture)

We built a pipeline that takes **raw DICOM medical images** (from a scanner) and creates a **final visualization** showing:
1. The original histopathology image (tissue sample)
2. A **Grad-CAM heatmap** overlaid on top (showing which areas the AI model thinks are important)

The final output is a single image that matches the scanner's exact resolution (3600 x 3900 pixels).

---

## 🔄 How We Did It (The Pipeline)

### **Step 1: Convert DICOM to PNG** 
**What it does:** Takes raw DICOM files and converts them to regular PNG images

**Main techniques:**
- **FastGlioma preprocessing**: Normalizes pixel values by dividing by 2^16 (65536)
- **Channel augmentation**: Creates a 3rd channel using the formula: `ch1 = ch3 - ch2 + base` (where base = 5000/65536)
- **Position extraction**: Reads strip position numbers from filenames (e.g., `img1_10` → strip position 10)

**Why:** DICOM files are medical format - we need regular images. The preprocessing matches what the FastGlioma AI model expects.

---

### **Step 2: Overlay Duplicate Strips**
**What it does:** Finds strips that look similar/identical and averages them together

**Main techniques:**
- **Similarity detection**: Uses histogram comparison + structural similarity to find duplicates
- **Averaging**: Overlays (averages) similar strips pixel-by-pixel to reduce noise
- **Manual pairs**: Supports config file to manually specify which strips should be grouped

**Why:** The scanner sometimes captures the same area multiple times. Averaging reduces noise and improves quality.

---

### **Step 3: Stitch Groups Together**
**What it does:** Takes all the overlaid groups and arranges them in the correct spatial order to create one full image

**Main techniques:**
- **Position-based ordering**: Uses strip position numbers from filenames to determine correct order
- **Overlap handling**: Calculates overlap between consecutive strips and averages pixels in overlapping regions
- **Black strip detection**: Automatically detects and excludes empty/black strips
- **Histogram equalization**: Uses CLAHE (adaptive histogram equalization) to balance brightness across strips
- **Dynamic cropping**: Crops 50px from all sides of each strip
- **Target resolution**: Calculates overlap dynamically to achieve exactly 3600 x 3900 pixels

**Why:** The strips need to be arranged in the correct spatial order to reconstruct the full tissue sample. Overlap averaging creates smooth transitions.

---

### **Step 4: Reassemble Grad-CAM Patches**
**What it does:** Takes small Grad-CAM heatmap patches and puts them back together into one full heatmap

**Main techniques:**
- **Coordinate parsing**: Extracts absolute coordinates from patch filenames
  - Format: `MUV_0635-ALA-01-<strip_x>-<strip_y>-<patch_x>-<patch_y>.tif`
  - Calculates: `absolute_x = strip_x + patch_x`, `absolute_y = strip_y + patch_y`
- **Patch normalization**: Normalizes each patch individually (scales non-zero values to 0-255 range)
- **Canvas assembly**: Places patches at their absolute coordinates on a blank canvas
- **Resizing**: Crops/resizes final heatmap to match target dimensions (3600 x 3900)

**Why:** Grad-CAM was computed on small patches. We need to reassemble them to match the full image size.

---

### **Step 5: Overlay Grad-CAM on Stitched Image**
**What it does:** Combines the Grad-CAM heatmap with the stitched histopathology image

**Main techniques:**
- **Sparse-aware blending**: Only applies heatmap color where it has non-zero values
- **Colormap**: Uses jet colormap (blue → green → yellow → red) to visualize heatmap intensity
- **Alpha blending**: Blends heatmap with original image using transparency (alpha = 0.5)

**Why:** Shows which parts of the tissue the AI model considers important, overlaid on the actual tissue image.

---

### **Step 6: Combine with Original Image**
**What it does:** Overlays Grad-CAM patches directly onto the original scanner image (`unnamed.png`)

**Main techniques:**
- **Direct patch placement**: Places Grad-CAM patches at their absolute coordinates on the original image
- **Coordinate scaling**: Scales patch coordinates to match resized original image
- **Image resizing**: Resizes original image to target dimensions (3600 x 3900)
- **Boundary checking**: Only places patches that fall within the original image bounds

**Why:** Creates the final visualization showing Grad-CAM on the original scanner image.

---

## 🔑 Main Techniques Used

### 1. **FastGlioma Preprocessing**
- Normalization: Divide by 2^16
- Channel augmentation: Create 3rd channel from existing channels
- Clipping: Ensure values stay in [0, 1] range

### 2. **Image Similarity Detection**
- Histogram comparison (cosine similarity)
- Structural similarity (MSE-based)
- Combined similarity score

### 3. **Spatial Ordering**
- Extract position numbers from filenames
- Sort groups by minimum strip position
- Handle first/last strip ordering

### 4. **Overlap Handling**
- Calculate overlap dynamically to achieve target dimensions
- Average pixels in overlapping regions
- Crop edges to remove artifacts

### 5. **Coordinate Parsing**
- Parse complex filenames to extract coordinates
- Calculate absolute positions from relative positions
- Handle multi-level coordinate systems (strip + patch)

### 6. **Sparse Data Normalization**
- Normalize only non-zero values
- Scale each patch individually (not globally)
- Preserve zero values (background)

### 7. **Adaptive Histogram Equalization (CLAHE)**
- Balance brightness across different strips
- Prevent over-brightening or over-darkening
- Improve visual consistency

---

## 📊 Key Design Decisions

1. **Flexible & Auto-detecting**: Works with any dataset structure, auto-detects series and datasets
2. **Position-based**: Uses strip positions from filenames (not sequential indexing)
3. **Target resolution**: Ensures final output exactly matches scanner resolution (3600 x 3900)
4. **Sparse-aware**: Handles sparse Grad-CAM data correctly (doesn't tint entire image)
5. **Configurable**: Supports JSON config for manual pairs, excluded strips, and ordering

---

## 🎓 Summary

**The Problem:** Raw DICOM strips → Need final visualization with Grad-CAM overlay

**The Solution:** 6-step pipeline that:
1. Converts DICOM → PNG (with proper preprocessing)
2. Groups and averages duplicate strips
3. Stitches strips in correct spatial order
4. Reassembles Grad-CAM patches
5. Overlays heatmap on stitched image
6. Combines with original scanner image

**The Result:** A single 3600 x 3900 pixel image showing the original tissue with Grad-CAM heatmap overlay, perfectly aligned and ready for analysis.
