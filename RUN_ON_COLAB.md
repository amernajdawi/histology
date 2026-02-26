# Run the pipeline on the Google Drive dataset (no local storage)

Run the full 6-step pipeline **in Google Colab** so the dataset and outputs never need to sit on your PC.

---

## Recommended: gdown (faster)

The notebook is set up to use **gdown** by default: it downloads the dataset straight into Colab and runs the pipeline on local disk (faster than reading from Drive).

### 1. Open the notebook in Colab

- Open **`colab/Run_Pipeline_on_Drive_Dataset.ipynb`** in [Google Colab](https://colab.research.google.com) (File → Upload notebook), or open from GitHub with “Open in Colab”.

### 2. Run the first cell (gdown)

- Run **Section 1**. It will:
  - Install **gdown**
  - Download the dataset using file ID `1G1SkbhvfJVKIbTDYqUjJ9cdkwtHTNLnU`
  - Unzip it under `/content/work`
  - Set `BASE_DIR` and `WORK_DIR` automatically

No need to “Add to My Drive” or set dataset paths. For very large files, gdown may show a browser confirmation once; follow the message.

### 3. Set checkpoint and pipeline (optional)

- **Checkpoint (Step 1):** Upload `fastglioma_highres_model.ckpt` to a Drive folder (e.g. `My Drive/colab_ckpts/`). In the checkpoint cell, set `ckpt_drive` to that path. The notebook will mount Drive and copy the file when you run that section.
- **Pipeline:** Set `REPO_URL` to your GitHub repo, or upload this repo as a ZIP to Drive and set `PIPELINE_ZIP_ON_DRIVE`.

### 4. Run the rest of the notebook

- Run the remaining cells in order (clone FastGlioma, run Steps 1–6). Results go to `/content/work/`. Optionally run the last section to copy outputs to Drive.

---

## Alternative: use Google Drive for the dataset

If you prefer to keep the dataset in Drive instead of gdown:

1. Open the dataset link and click **“Add to My Drive”**.
2. In the notebook, **skip the gdown cell** and run **Section 2 (Optional) Use Google Drive**.
3. Mount Drive and set either:
   - `DRIVE_DATASET_ZIP = "/content/drive/MyDrive/yourfile.zip"`, or  
   - `DRIVE_DATASET_PATH = "/content/drive/MyDrive"` (default; use if Meta_MasterThesis.tar is in My Drive root)
4. Continue with the rest of the notebook.

---

## Expected dataset layout

Under the base directory (`/content/work` after gdown, or your Drive folder):

- **DICOM strips:** `DATASET_NAME / SERIES_NUM / strips / *.dcm`  
  e.g. `MUV_0635-2/01/strips/*.dcm`
- **Grad-CAM patches (Steps 4–6):** `DATASET_NAME / SERIES_NUM / patches / *ALA*.tif`  
  e.g. `MUV_0635-2/01/patches/*ALA*.tif`

---

## If you don’t have the checkpoint

Step 1 (DICOM → PNG) needs the FastGlioma checkpoint. Without it, Step 1 will fail. You can still run Steps 2–6 if you already have the right inputs (e.g. from a previous run).

---

## Summary

| Step | What you need |
|------|----------------|
| 1 | Dataset with `.../strips/*.dcm`; FastGlioma checkpoint |
| 2 | Output of Step 1 (`*_original.png`) |
| 3 | Output of Step 2; optional config |
| 4 | Dataset with `.../patches/*ALA*.tif` |
| 5 | Outputs of Steps 3 and 4 |
| 6 | Original image (e.g. `unnamed.png`) and patches; optional |

By using **gdown** you keep the run faster; Drive is only needed for the checkpoint (and optionally for saving results).
