# Fingerprint Presentation Attack Detection (FPAD) — Usage & Developer Guide

Welcome to the comprehensive guide for the FPAD pipeline. This document explains the architecture of the codebase, details what each file does, specifies expected inputs and outputs, and provides clear instructions on how to run the pipeline.

---

## 🏗️ System Architecture

The pipeline is designed in a highly modular fashion. Each step of the machine learning workflow—configuration, preprocessing, feature extraction, modeling, evaluation, and orchestration—is isolated in its own file within the `src/` directory.

### Directory Overview
```text
fingerprint-scoofing/
├── src/
│   ├── config.py         # Global settings and hyperparameters
│   ├── preprocess.py     # Image loading and enhancement
│   ├── features.py       # Local Binary Pattern (LBP) extraction
│   ├── models.py         # SVM and KNN classifier pipelines
│   ├── evaluate.py       # Metrics and visualization logic
│   ├── main.py           # Core execution script for the full pipeline
│   └── lbp_verify.py     # Diagnostic script to verify feature extraction
├── results/              # Metrics and ROC/Confusion Matrix plots
└── outputs/              # Verification artifacts (visuals and raw features)
```

---

## 📂 File-by-File Breakdown

### 1. `src/config.py`
**What it does:** Acts as the central nervous system of the project. It defines all constants, hyperparameters, dataset paths, and search grids in one place.
* **Input:** Environment variables (e.g., `FVC2000_DIR`, `N_JOBS`) if running on cloud platforms like Colab/Kaggle.
* **Output:** Python constants imported by all other modules.
* **How to use it:** Edit this file directly if you want to change the image resize dimensions, LBP radius/points, cross-validation folds, or SVM/KNN hyperparameter search grids.

### 2. `src/preprocess.py`
**What it does:** Prepares raw fingerprint images for feature extraction to ensure consistency and enhance texture.
* **Input:** A file path (`str`) to a raw fingerprint image (`.tif` or `.bmp`).
* **Processing:**
  1. Loads the image in grayscale.
  2. Resizes it to a standardized `300x300` resolution using bicubic interpolation.
  3. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance ridge/valley contrast.
  4. Normalizes pixel values to a strict `[0, 255]` range.
* **Output:** A preprocessed 2D NumPy array (`np.uint8`) of shape `(300, 300)`.

### 3. `src/features.py`
**What it does:** Extracts mathematical representations of the fingerprint texture using Uniform Local Binary Patterns (LBP).
* **Input:** A preprocessed 2D NumPy array (from `preprocess.py`).
* **Processing:** 
  1. Applies uniform LBP mapping with $P=8$ points and $R=1$ radius.
  2. Divides the image into an $8 \times 8$ grid (64 spatial blocks).
  3. Computes a 10-bin histogram for each block.
  4. L1-normalizes each histogram and concatenates them.
* **Output:** A dense 1D NumPy array (`np.float32`) of shape `(640,)`.

### 4. `src/models.py`
**What it does:** Constructs the machine learning pipelines for the Support Vector Machine (SVM) and K-Nearest Neighbor (KNN) classifiers.
* **Input:** Training labels (`y_train`) used to dynamically configure the number of cross-validation folds based on class representation.
* **Processing:** Combines a `StandardScaler` (for zero-mean, unit-variance normalization) with the chosen classifier. It wraps the pipeline in a `GridSearchCV` object to automatically find the best hyperparameters.
* **Output:** An un-fitted scikit-learn `GridSearchCV` pipeline object ready for `.fit(X_train, y_train)`.

### 5. `src/evaluate.py`
**What it does:** Quantifies model performance and generates publication-ready visualizations.
* **Input:** True labels (`y_true`), predicted labels (`y_pred`), predicted probabilities (`y_prob`), and dataset metadata.
* **Processing:** Computes Accuracy, weighted Precision/Recall/F1-score, and ROC AUC. It also plots the ROC curve and the Confusion Matrix using a headless matplotlib backend (`Agg`).
* **Output:** 
  1. Appends a row to `{Dataset}_metrics.csv` in the `results/` folder.
  2. Saves `{Dataset}_{Model}_roc.png` and `{Dataset}_{Model}_confusion_matrix.png` to the `results/` folder.
  3. Prints a classification report to the terminal.

### 6. `src/main.py`
**What it does:** The orchestrator. It ties all the above modules together to run the full presentation attack detection experiment.
* **Input:** Command-line arguments (`--fvc`, `--socofing`, `--sample`).
* **Processing:**
  1. Loads images from disk, passing them through `preprocess.py` and `features.py`.
  2. Performs a stratified 80/20 train-test split.
  3. Balances the training set (undersampling the majority class) to prevent biased models.
  4. Trains the SVM and KNN models (from `models.py`).
  5. Evaluates the models on the test set (via `evaluate.py`).
* **Output:** A fully populated `results/` directory containing all metrics and charts.

### 7. `src/lbp_verify.py`
**What it does:** A diagnostic and validation tool that rigorously tests the mathematical correctness of the LBP feature extraction stage.
* **Input:** Reads up to 10 sample images from the datasets.
* **Processing:** Runs the images through the preprocessing and feature extraction stages while applying strict assertions (e.g., ensuring exact output dimensions, lack of NaN/Inf values, and L1-normalization bounds).
* **Output:** Generates detailed visual evidence in the `outputs/visualizations/` folder (side-by-side LBP comparisons, block grid overlays, dominant-code heatmaps, and histogram plots) and saves raw feature matrices in `outputs/features/`.

---

## 🚀 Execution Guide

All commands must be executed from the **root directory** of the project (`fingerprint-scoofing/`).

### 1. Running the Full Experiment
Use `main.py` to train and evaluate the models.

> **Fast Dry Run:** If you want to test that the pipeline works without waiting hours for the entire dataset to process, use the `--sample` flag to load a fraction of the data (e.g., 5%).
> ```bash
> python src/main.py --socofing --sample 0.05
> ```

**Run FVC2000 Only:**
```bash
python src/main.py --fvc
```

**Run Both Datasets (The Complete Research Suite):**
```bash
python src/main.py --fvc --socofing
```
**Expected Output:** The terminal will display dataset loading progress, optimal hyperparameters found during GridSearchCV, and the final classification report. The `results/` folder will be populated with CSVs and PNGs.

### 2. Verifying Feature Extraction
If you want to inspect exactly what the LBP algorithm is doing to the fingerprints, or need to export raw `.npy` feature files for external analysis, run the verification script.

```bash
python src/lbp_verify.py
```
**Expected Output:** The terminal will print a 10-section report validating the feature geometry. The `outputs/` folder will be populated with diagnostic images, a metadata CSV, and a `.txt` extraction report summarizing the configurations used.
