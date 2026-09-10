# Fingerprint Presentation Attack Detection (FPAD) ML Pipeline

This document outlines the implementation plan for building a Presentation Attack Detection system using Support Vector Machine (SVM) and K-Nearest Neighbor (KNN) classifiers, based on Local Binary Pattern (LBP) features. The system will evaluate the FVC2000 and SOCOFing datasets.

## User Review Required

> [!WARNING]
> **Data Duplication Strategy**: The provided implementation specification requests restructuring the datasets into a new `data/` folder. Given that the datasets are already extracted in `74034_3_En_4_MOESM1_ESM` and `archive(2)`, copying them will duplicate thousands of images and consume extra disk space. 
> 
> **My recommendation**: I will write the data loading scripts to directly read from the existing `74034_3_En_4_MOESM1_ESM/FVC2000` and `archive(2)/SOCOFing` folders to save space and time, mapping them internally to the required structure. Is this acceptable, or would you prefer I explicitly copy/move the files into the strict `data/fvc2000` and `data/socofing` directory tree as outlined in the plan?

## Open Questions

> [!IMPORTANT]
> 1. Should I set up a Python virtual environment automatically and install the required dependencies (`requirements.txt`), or will you be running the scripts in your own existing environment?
> 2. The FVC2000 datasets are spread across DB1-DB4. The plan specifies using DB1-3 as "Genuine" and DB4 (synthetic) as "Fake". Are there any other specific labeling rules you'd like me to apply for the `_A` vs `_B` splits?

## Proposed Changes

We will build the pipeline according to the provided `fingerprintscoofing.md` plan.

### Source Code (`src/`)

#### [NEW] src/config.py
Will act as the single source of truth for hyperparameters (LBP settings, SVM/KNN grids, paths). I will adjust the paths to point to your existing dataset folders based on your feedback above.

#### [NEW] src/preprocess.py
Will contain functions for:
- Image loading (handling `.tif` and `.bmp`)
- Grayscale conversion and resizing to 300x300 via bicubic interpolation
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Normalization [0, 255]

#### [NEW] src/features.py
Will extract Local Binary Patterns (LBP):
- Using Radius=1, Points=8, Method='uniform'
- Block-based extraction (8x8 grids) resulting in 3,776-dimensional feature vectors.

#### [NEW] src/models.py
Will define the pipeline for:
- SVM with RBF kernel and GridSearchCV tuning.
- KNN with Euclidean distance and GridSearchCV tuning.

#### [NEW] src/evaluate.py
Will contain metrics calculation:
- Accuracy, Precision, Recall, F1-score
- ROC AUC and plotting tools.
- Generation of the required result artifacts for the research paper.

#### [NEW] src/main.py
The main orchestration script that runs the data loading, preprocessing, feature extraction, training, testing, and evaluation for both FVC2000 and SOCOFing independently.

### Root Files

#### [NEW] requirements.txt
Will specify exact package versions (numpy, scikit-learn, scikit-image, opencv-python, pandas, matplotlib).

## Verification Plan

### Automated Verification
- I will run a smaller subset/dry-run of the pipeline (e.g., 5-10 images per class) to ensure the scripts execute without syntax errors, feature vectors are the correct shape (3776), and the models can train and predict successfully.

### Manual Verification
- After successful execution, the final output metrics (Accuracy, ROC AUC, etc.) will be available in the `results/` folder for your review to include in your final research report.
