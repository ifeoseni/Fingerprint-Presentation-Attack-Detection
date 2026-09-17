# Fingerprint Presentation Attack Detection (FPAD)
### A Comparative Evaluation of SVM and KNN Classifiers with LBP Features

> **Research objective:** Determine whether fingerprint images represent *genuine (live)* captures or
> *presentation attacks (spoofed / altered)* using classical machine-learning classifiers trained
> on texture features extracted with the Local Binary Pattern (LBP) descriptor.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Questions](#research-questions)
3. [Datasets](#datasets)
4. [Project Structure](#project-structure)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Installation](#installation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Cloud Execution (Colab / Kaggle)](#cloud-execution-colab--kaggle)
9. [Results & Output Artifacts](#results--output-artifacts)
10. [Reproducibility Guarantees](#reproducibility-guarantees)

---

## Project Overview

This repository implements a complete, scientifically reproducible **Fingerprint Presentation Attack Detection (FPAD)** system evaluated on two independent datasets:

| Dataset  | Genuine Class | Fake / Spoof Class |
|----------|--------------|---------------------|
| **FVC2000** | DB1 + DB2 + DB3 (real sensor captures) | DB4 (synthetic generation) |
| **SOCOFing** | `Real/` folder (6,000 BMP images) | `Altered/Altered-Easy` + `Altered-Medium` + `Altered-Hard` |

The pipeline applies **Local Binary Pattern (LBP)** texture feature extraction and evaluates two classifiers:
- **Support Vector Machine (SVM)** with RBF kernel
- **K-Nearest Neighbor (KNN)** with Euclidean distance

Both classifiers are automatically tuned end-to-end using **GridSearchCV** over a predefined hyperparameter grid with **Stratified 5-Fold Cross-Validation**, scored by **ROC AUC**.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | How accurately can LBP texture features distinguish genuine fingerprints from spoofed/altered ones? |
| RQ2 | Which classifier (SVM vs KNN) performs better for FPAD? |
| RQ3 | How well do models generalise to the hardest alteration difficulty tier (`Altered-Hard`)? |
| RQ4 | Does performance vary between FVC2000 acquisition technologies and SOCOFing altered fingerprints? |
| RQ5 | How robust are classifiers when tested against the SOCOFing `Altered-Hard` subset? |

---

## Datasets

### FVC2000
- **Source:** Springer supplementary material (`74034_3_En_4_MOESM1_ESM`)
- **Structure:** 4 databases × 110 fingers × 8 impressions = 880 images per database
- **Set A** (fingers 1–100): **Benchmark / Test** set
- **Set B** (fingers 101–110): **Parameter tuning / Train** set

| DB | Sensor | Image Size | Resolution |
|----|--------|-----------|------------|
| DB1 | KeyTronic Secure Desktop (Optical) | 300×300 | 500 dpi |
| DB2 | ST Microelectronics TouchChip (Capacitive) | 256×364 | 500 dpi |
| DB3 | Identicator DF-90 (Optical) | 448×478 | 500 dpi |
| DB4 | Synthetic Generator | 240×320 | ~500 dpi |

### SOCOFing
- **Source:** Kaggle (`ruizgara/socofing`) — placed in `archive(2)/SOCOFing/`
- **Genuine (Real):** 6,000 fingerprint BMP images (600 African subjects × 10 fingers/subject).
- **Altered (Fake):** 49,273 synthetically altered spoof images generated via the STRANGE toolbox.
  - `Altered-Easy`: 17,934 images
  - `Altered-Medium`: 17,067 images
  - `Altered-Hard`: 14,272 images

#### SOCOFing Data Handling & Splitting Strategy
1. **Nested Duplicate Folder Prevention**: When extracted, the dataset creates a duplicate structure (`archive(2)/SOCOFing/SOCOFing`). Our pipeline explicitly reads *only* from the top-level `Real/` and `Altered/` directories to prevent double-counting data.
2. **Subject-wise Isolation (GroupShuffleSplit)**: The pipeline strictly extracts the subject ID from every filename (e.g., `100__M_Left...` $\rightarrow$ Subject 100) and performs an **80/20 split on Subjects, not images**.
3. **Training Set Balancing**: Because there are far more fake images than genuine images, models trained naively will collapse. We extract 80% of the subjects for training, keep all their genuine images (~4,800), and **randomly undersample** their fake images to match exactly ~4,800. The remaining fake images belonging to these training subjects are safely discarded.
4. **Test Set Integrity**: The test set is comprised exclusively of the remaining 20% unseen subjects (~1,200 genuine and ~3,400+ fake, depending on difficulty). The final evaluation metrics (and Confusion Matrix) reflect only this isolated test set, hence why the matrix sum is much lower than the raw total file count.

---

## Project Structure

```
fingerprint-scoofing/
├── 74034_3_En_4_MOESM1_ESM/         ← FVC2000 dataset (extracted)
│   └── FVC2000/
│       └── Dbs/
│           ├── Db1_a/  Db1_b/
│           ├── Db2_a/  Db2_b/
│           ├── Db3_a/  Db3_b/
│           └── Db4_a/  Db4_b/
├── archive(2)/                        ← SOCOFing dataset (extracted)
│   └── SOCOFing/
│       ├── Real/
│       └── Altered/
│           ├── Altered-Easy/
│           ├── Altered-Medium/
│           └── Altered-Hard/
├── src/
│   ├── config.py       ← Hyperparameters, paths, all tunable constants
│   ├── preprocess.py   ← Image loading, CLAHE, normalisation
│   ├── features.py     ← Block-LBP feature extraction (640-dim vector)
│   ├── models.py       ← SVM & KNN pipelines with GridSearchCV
│   ├── evaluate.py     ← Metrics, ROC curves, confusion matrices
│   ├── main.py         ← CLI orchestration script
│   └── lbp_verify.py   ← Full LBP pipeline verification script
├── results/            ← Model performance artifacts
│   ├── FVC2000_metrics.csv
│   ├── SOCOFing_metrics.csv
│   ├── FVC2000_SVM_roc.png
│   ├── FVC2000_KNN_roc.png
│   ├── FVC2000_SVM_confusion_matrix.png
│   └── ...
├── outputs/            ← Generated by lbp_verify.py
│   ├── features/       ← Extracted features (.npy), labels, metadata
│   └── visualizations/ ← LBP image plots, block grids, histograms
├── run_pipeline.ipynb  ← Interactive Jupyter Notebook
├── requirements.txt
└── README.md
```

---

## Pipeline Architecture

```
Raw Images (.tif / .bmp)
        │
        ▼
┌─────────────────────────────┐
│        preprocess.py        │
│  1. Load → Grayscale        │
│  2. Resize → 300×300 bicubic│
│  3. CLAHE (clip=2.0, 8×8)  │
│  4. Normalize [0, 255]      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│         features.py         │
│  LBP (R=1, P=8, uniform)   │
│  8×8 block grid → histograms│
│  Output: 640-dim vector     │
└─────────────┬───────────────┘
              │
       ┌──────┴──────┐
       ▼              ▼
┌────────────┐  ┌────────────┐
│ SVM + Grid │  │ KNN + Grid │
│ SearchCV   │  │ SearchCV   │
│ (5-fold    │  │ (5-fold    │
│  Strat-KF) │  │  Strat-KF) │
└─────┬──────┘  └──────┬─────┘
      │                │
      └──────┬─────────┘
             ▼
┌─────────────────────────────┐
│         evaluate.py         │
│  • Accuracy                 │
│  • Precision / Recall / F1  │
│  • ROC AUC                  │
│  • Confusion Matrix         │
│  • CSV + PNG artifacts      │
└─────────────────────────────┘
```

---

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencies:
| Package | Purpose |
|---------|---------|
| numpy | Array ops |
| scikit-learn | SVM, KNN, GridSearchCV |
| scikit-image | LBP extraction |
| opencv-python | Image loading, CLAHE |
| Pillow | Additional format support |
| pandas | CSV results |
| matplotlib | ROC + confusion matrix plots |
| joblib | Parallel processing |

> **Note:** Versions are unpinned in `requirements.txt` to guarantee installation success on newer environments (e.g., Python 3.13), as older pinned versions often fail to compile.

---

## Running the Pipeline

All commands must be run from the **project root directory**.

### Terminal (CLI)

```bash
# FVC2000 experiment only
python src/main.py --fvc

# SOCOFing experiment (5% sample — fast test)
python src/main.py --socofing --sample 0.05

# SOCOFing experiment (full dataset)
python src/main.py --socofing

# Both experiments back-to-back
python src/main.py --fvc --socofing
```

### Verifying Feature Extraction
A standalone script guarantees that the LBP pipeline is extracting features exactly as specified mathematically.
```bash
python src/lbp_verify.py
```
*Outputs are saved to `outputs/visualizations` and `outputs/features`.*

### Jupyter Notebook

Open `run_pipeline.ipynb` in JupyterLab, VS Code, or any notebook environment and run the cells interactively.

---

## Cloud Execution (Colab / Kaggle)

The pipeline uses **environment variables** for all dataset paths, so **no code changes are needed** when running on cloud platforms.

### Google Colab
```python
import os
os.environ["FVC2000_DIR"]  = "/content/drive/MyDrive/FVC2000"
os.environ["SOCOFING_DIR"] = "/content/drive/MyDrive/SOCOFing"
os.environ["RESULTS_DIR"]  = "/content/drive/MyDrive/fpad_results"
```

### Kaggle
```python
import os
os.environ["SOCOFING_DIR"] = "/kaggle/input/socofing/SOCOFing"
os.environ["RESULTS_DIR"]  = "/kaggle/working/results"
```

Then run normally:
```python
from src.main import load_socofing_data, run_pipeline
X_tr, X_te, y_tr, y_te = load_socofing_data(sample_fraction=1.0)
run_pipeline("SOCOFing", X_tr, X_te, y_tr, y_te)
```

> **Performance note:** `GridSearchCV` uses `n_jobs=-1` and `KNeighborsClassifier` uses `n_jobs=-1`, so the pipeline automatically scales across all available CPU cores on cloud VMs.

---

## Results Summary

Results below are from the full-dataset runs in `main.ipynb` (all images, subject-disjoint 80/20 split for SOCOFing; standard Set A/B protocol for FVC2000).

### Model Performance

| Dataset | Alteration Tier | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| FVC2000 | — | SVM | 99.6% | 99.6%ᵃ | 99.6%ᵃ | 99.6%ᵃ | 1.000 |
| FVC2000 | — | KNN | 99.9% | 99.9%ᵃ | 99.9%ᵃ | 99.9%ᵃ | 1.000 |
| SOCOFing | Easy | SVM | 80.2% | 75.2%ᵇ | 81.0%ᵇ | 76.7%ᵇ | 0.890 |
| SOCOFing | Easy | KNN | 52.1% | 61.1%ᵇ | 63.3%ᵇ | 51.7%ᵇ | 0.714 |
| SOCOFing | Medium | SVM | 94.4% | 91.6%ᵇ | 94.7%ᵇ | 93.0%ᵇ | 0.988 |
| SOCOFing | Medium | KNN | 77.4% | 75.2%ᵇ | 82.4%ᵇ | 75.3%ᵇ | 0.923 |
| SOCOFing | Hard | SVM | 97.6% | 96.6%ᵇ | 97.6%ᵇ | 97.1%ᵇ | 0.997 |
| SOCOFing | Hard | KNN | 88.7% | 85.9%ᵇ | 89.9%ᵇ | 87.3%ᵇ | 0.970 |
| SOCOFing | Combined | SVM | 86.8% | 71.4%ᵇ | 87.1%ᵇ | 75.6%ᵇ | 0.949 |
| SOCOFing | Combined | KNN | 62.0% | 59.9%ᵇ | 75.3%ᵇ | 53.9%ᵇ | 0.853 |

*ᵃ Weighted average. ᵇ Macro average (unweighted mean across genuine/altered classes), used for SOCOFing to account for test-set class imbalance.*

**Key finding:** SVM outperforms KNN on every metric and every SOCOFing alteration tier, with the gap widest on the Easy tier (28 points of accuracy) and narrowest on Hard (9 points). Both models' accuracy *increases* with alteration severity (Easy → Hard) — counter-intuitive at first glance, but consistent with what the SOCOFing severity tiers actually measure: heavier digital distortion diverges further from genuine ridge-pattern texture, making it easier — not harder — for a texture-based classifier to detect. FVC2000's near-ceiling accuracy reflects an easier, non-comparable task (separating real sensor captures from a synthetic generator), not presentation-attack-detection capability.

### Computational Cost

Measured on an Intel Core i7-10610U (4 cores / 8 threads, 1.80 GHz), 15.7 GB RAM, no GPU — single-run measurements taken directly within each experiment cell in `main.ipynb` (feature extraction, model training, and inference all timed in the same run, not stitched together from separate sessions). "Training Time" is the full `GridSearchCV` search (SVM: 40 candidates × 5-fold CV, including the one-time refit needed for `predict_proba`; KNN: 14 candidates × 5-fold CV).

| Condition | Model | Feature Extraction | Training Time | Total Pipeline | Inference (ms/sample) | Test Set Size |
|---|---|---|---|---|---|---|
| FVC2000 | SVM | 2.9 min | 0.10 min | 3.0 min | 0.339 | 3,200 |
| FVC2000 | KNN | 2.9 min | 0.07 min | 3.0 min | 0.072 | 3,200 |
| SOCOFing Easy | SVM | 24.2 min | 77.7 min | 102.4 min | 6.265 | 4,785 |
| SOCOFing Easy | KNN | 17.9 min | 1.4 min | 19.4 min | 1.727 | 4,785 |
| SOCOFing Medium | SVM | 9.4 min | 52.3 min | 62.0 min | 3.573 | 4,602 |
| SOCOFing Medium | KNN | 16.2 min | 1.1 min | 17.4 min | 1.316 | 4,602 |
| SOCOFing Hard | SVM | 8.0 min | 46.7 min | 54.9 min | 2.249 | 4,011 |
| SOCOFing Hard | KNN | 18.0 min | 2.1 min | 20.2 min | 2.342 | 4,011 |
| SOCOFing Combined | SVM | 21.1 min | 55.5 min | 77.4 min | 4.515 | 10,998 |
| SOCOFing Combined | KNN | 76.9 min | 1.8 min | 79.0 min | 1.855 | 10,998 |

**Grid-search tuning is a one-time offline cost — inference is fast for both models.** SVM's hyperparameter search (47–78 minutes per SOCOFing condition) dominates total pipeline time; KNN's training cost is under 2.1 minutes throughout, since KNN has no training phase beyond storing the data. Once trained, per-sample inference is millisecond-scale for both classifiers (KNN: 0.07–2.3 ms; SVM: 0.3–6.3 ms), well within real-time authentication requirements, and requires no GPU.

Note that feature extraction time varies between runs of the same tier — e.g. SOCOFing Combined's KNN cell measured 76.9 minutes of extraction against 21.1 minutes for the SVM cell on the same underlying images, since each experiment cell independently re-extracts features rather than sharing a cached pass. This reflects real run-to-run system variance (disk I/O, background load), not a measurement error.

---

## Results & Output Artifacts

All artifacts are written to the `results/` folder (created automatically).

| File | Description |
|------|-------------|
| `FVC2000_metrics.csv` | Accuracy, Precision, Recall, F1, ROC AUC for SVM and KNN on FVC2000 |
| `SOCOFing_metrics.csv` | Same metrics for SOCOFing |
| `*_SVM_roc.png` | Publication-quality ROC curve (150 dpi) for SVM |
| `*_KNN_roc.png` | Publication-quality ROC curve (150 dpi) for KNN |
| `*_SVM_confusion_matrix.png` | Confusion matrix plot for SVM |
| `*_KNN_confusion_matrix.png` | Confusion matrix plot for KNN |

---

## Reproducibility Guarantees

This pipeline is engineered for **exact reproducibility** across runs and platforms:

| Mechanism | Where | Effect |
|-----------|-------|--------|
| `RANDOM_STATE = 42` | `config.py` | Seeds SVC, train-test split, and CV fold assignment |
| `StratifiedKFold(shuffle=True, random_state=42)` | `models.py` | Deterministic, balanced CV folds |
| `sorted()` on all file lists | `main.py` | Alphabetical load order independent of OS filesystem ordering |
| `sample_fraction` slice from sorted list | `main.py` | Same subset selected every run |
| `matplotlib.use("Agg")` | `evaluate.py` | Headless-safe plotting on all platforms |
| Fixed `dtype=np.float32` and `np.int32` | `main.py` | Consistent numeric precision, safe cross-platform indexing |
