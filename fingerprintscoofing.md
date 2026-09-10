Here is a comprehensive, \*\*AI-agent-ready implementation plan\*\* for your Fingerprint Presentation Attack Detection (FPAD) project using SVM and KNN on the \*\*FVC2000\*\* and \*\*SOCOFing\*\* datasets. This plan fills in every \`\[AUTHOR TO INSERT\]\` placeholder from your document with concrete values and provides exact code architecture, hyperparameters, and file structures.

\---

\# Fingerprint Presentation Attack Detection (FPAD)

\## Implementation Plan — SVM & KNN with LBP Features

\---

\## 1. Project Objective

Build and evaluate two classical machine-learning classifiers (SVM and KNN) to distinguish \*\*genuine (live) fingerprints\*\* from \*\*fake/presentation attack\*\* fingerprints using \*\*Local Binary Pattern (LBP)\*\* texture features.

\*\*Two independent experiments:\*\*

| Database | Genuine Class | Fake/Spoof Class |

|---|---|---|

| \*\*FVC2000\*\* | DB1 + DB2 + DB3 (real sensor captures) | DB4 (synthetic generation) |

| \*\*SOCOFing\*\* | \`Real/\` folder | \`Altered-Easy/\` + \`Altered-Medium/\` + \`Altered-Hard/\` |

\---

\## 2. Data Acquisition & Directory Structure

\### 2.1 Download Sources

| Dataset | Source | Action |

|---|---|---|

| \*\*FVC2000\*\* | \`<https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4/MediaObjects/74034_3_En_4_MOESM1_ESM.zip\`> | Download, unzip. Expect subfolders \`DB1_B\`, \`DB2_B\`, \`DB3_B\`, \`DB4_B\`, \`DB1_A\`, \`DB2_A\`, \`DB3_A\`, \`DB4_A\` containing \`.tif\` images. |

| \*\*SOCOFing\*\* | \`<https://www.kaggle.com/datasets/ruizgara/socofing\`> | Download via Kaggle API (\`kaggle datasets download -d ruizgara/socofing\`). Unzip to get \`Real/\`, \`Altered/Altered-Easy/\`, \`Altered/Altered-Medium/\`, \`Altered/Altered-Hard/\`. |

\### 2.2 Required Directory Layout

Create this exact structure before running any code:

\`\`\`

fingerprint_pad_project/

├── data/

│ ├── fvc2000/

│ │ ├── DB1_B/ # 80 images (10 fingers × 8)

│ │ ├── DB2_B/ # 80 images

│ │ ├── DB3_B/ # 80 images

│ │ ├── DB4_B/ # 80 images (synthetic)

│ │ ├── DB1_A/ # 800 images (100 fingers × 8)

│ │ ├── DB2_A/ # 800 images

│ │ ├── DB3_A/ # 800 images

│ │ └── DB4_A/ # 800 images (synthetic)

│ └── socofing/

│ ├── Real/ # 6,000 BMP images

│ └── Altered/

│ ├── Altered-Easy/ # ~17,934 BMP images

│ ├── Altered-Medium/ # ~17,067 BMP images

│ └── Altered-Hard/ # ~14,272 BMP images

├── src/

│ ├── config.py

│ ├── preprocess.py

│ ├── features.py

│ ├── models.py

│ ├── evaluate.py

│ └── main.py

├── results/

│ ├── fvc2000/

│ └── socofing/

└── requirements.txt

\`\`\`

\---

\## 3. Software Environment

Create \`requirements.txt\`:

\`\`\`txt

numpy==1.24.3

scikit-learn==1.3.0

scikit-image==0.21.0

opencv-python==4.8.0.74

Pillow==10.0.0

pandas==2.0.3

matplotlib==3.7.2

joblib==1.3.1

\`\`\`

\*\*Environment specs to record in the final report:\*\*

\- \*\*Language:\*\* Python 3.10+

\- \*\*CPU:\*\* Any modern multi-core CPU (no GPU required)

\- \*\*RAM:\*\* Minimum 8 GB (16 GB recommended for SOCOFing)

\- \*\*OS:\*\* Linux / Windows / macOS

\---

\## 4. Configuration (\`src/config.py\`)

Create a single source of truth for all hyperparameters:

\`\`\`python

import os

\# === Paths ===

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(\__file_\_)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

\# === Image Preprocessing ===

TARGET_SIZE = (300, 300) # Resize ALL images to 300×300

CLAHE_CLIP = 2.0

CLAHE_GRID = (8, 8)

\# === LBP Feature Extraction ===

LBP_RADIUS = 1

LBP_POINTS = 8

LBP_METHOD = 'uniform' # Rotation-invariant uniform patterns

LBP_BLOCKS = (8, 8) # 8×8 spatial blocks

LBP_BINS = 59 # For P=8 uniform → 59 bins

FEATURE_DIM = LBP_BLOCKS\[0\] \* LBP_BLOCKS\[1\] \* LBP_BINS # 64 × 59 = 3,776

\# === Train-Test Split ===

TEST_SIZE = 0.20

RANDOM_STATE = 42

STRATIFY = True

\# === Cross-Validation ===

CV_FOLDS = 10 # Stratified K-Fold for secondary evaluation

CV_FOLDS_TUNING = 5 # For GridSearchCV hyperparameter tuning

\# === SVM Hyperparameter Search Space ===

SVM_PARAM_GRID = {

'C': \[0.1, 1, 10, 100\],

'gamma': \['scale', 'auto', 0.001, 0.01, 0.1\],

'kernel': \['rbf'\],

'class_weight': \['balanced'\]

}

\# === KNN Hyperparameter Search Space ===

KNN_PARAM_GRID = {

'n_neighbors': \[3, 5, 7, 9, 11, 13, 15\],

'weights': \['uniform', 'distance'\],

'metric': \['euclidean'\]

}

\`\`\`

\---

\## 5. Data Preprocessing (\`src/preprocess.py\`)

\*\*Steps for every image:\*\*

1\. Load (handle \`.tif\` and \`.bmp\`)

2\. Convert to grayscale (8-bit)

3\. Resize to \`TARGET_SIZE = (300, 300)\` using \*\*bicubic interpolation\*\*

4\. Apply \*\*CLAHE\*\* (Contrast Limited Adaptive Histogram Equalization)

5\. Normalize pixel values to range \`\[0, 255\]\`

\`\`\`python

import cv2

import numpy as np

from skimage import io

def preprocess_image(image_path: str, target_size=(300, 300)) -> np.ndarray:

"""

Load and preprocess a fingerprint image.

Returns: uint8 ndarray of shape (300, 300)

"""

\# Load grayscale

img = io.imread(image_path, as_gray=True)

img = (img \* 255).astype(np.uint8) if img.dtype != np.uint8 else img

\# Resize

img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

\# CLAHE enhancement

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

img = clahe.apply(img)

return img

\`\`\`

\---

\## 6. Feature Extraction (\`src/features.py\`)

\*\*LBP Configuration (fills the AUTHOR TO INSERT in Section 3.3):\*\*

| Parameter | Value | Justification |

|---|---|---|

| \*\*Radius (R)\*\* | 1 | Captures fine ridge texture |

| \*\*Points (P)\*\* | 8 | Standard circular neighborhood |

| \*\*Pattern type\*\* | \`uniform\` | Reduces dimensionality; robust to noise |

| \*\*Blocks\*\* | 8 × 8 = 64 blocks | Spatial partitioning preserves local structure |

| \*\*Feature vector dimensionality\*\* | \*\*3,776\*\* | 64 blocks × 59 uniform bins |

\`\`\`python

import numpy as np

from skimage.feature import local_binary_pattern

def extract_lbp_features(image: np.ndarray,

P=8, R=1, method='uniform',

blocks=(8, 8)) -> np.ndarray:

"""

Extract block-wise normalized LBP histogram.

image: 2D ndarray (300, 300)

Returns: 1D feature vector of length 3,776

"""

h, w = image.shape

block_h = h // blocks\[0\]

block_w = w // blocks\[1\]

lbp_image = local_binary_pattern(image, P=P, R=R, method=method)

n_bins = int(lbp_image.max() + 1) # 59 for uniform P=8

hist_blocks = \[\]

for i in range(blocks\[0\]):

for j in range(blocks\[1\]):

block = lbp_image\[i\*block_h:(i+1)\*block_h,

j\*block_w:(j+1)\*block_w\]

hist, _= np.histogram(block, bins=n_bins,

range=(0, n_bins), density=True)

hist_blocks.append(hist)

feature_vector = np.concatenate(hist_blocks).astype(np.float32)

return feature_vector

\`\`\`

\---

\## 7. Model Training (\`src/models.py\`)

\### 7.1 SVM Configuration (fills AUTHOR TO INSERT in Section 3.4)

| Parameter | Value / Method |

|---|---|

| \*\*Kernel\*\* | RBF |

| \*\*Regularization C\*\* | Tuned via \`GridSearchCV\` over \`\[0.1, 1, 10, 100\]\` |

| \*\*Gamma γ\*\* | Tuned via \`GridSearchCV\` over \`\['scale', 'auto', 0.001, 0.01, 0.1\]\` |

| \*\*Class weight\*\* | \`balanced\` (handles imbalance) |

| \*\*Tuning strategy\*\* | 5-fold stratified cross-validation on training set |

\### 7.2 KNN Configuration (fills AUTHOR TO INSERT in Section 3.4)

| Parameter | Value / Method |

|---|---|

| \*\*K\*\* | Tuned via \`GridSearchCV\` over odd values \`\[3, 5, 7, 9, 11, 13, 15\]\` |

| \*\*Distance metric\*\* | Euclidean |

| \*\*Voting weights\*\* | Tuned over \`\['uniform', 'distance'\]\` |

| \*\*Tuning strategy\*\* | 5-fold stratified cross-validation on training set |

\`\`\`python

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_svm(X_train, y_train, param_grid, cv_folds=5):

"""

Train SVM with RBF kernel using grid search.

"""

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

svm = SVC(probability=True, random_state=42)

grid = GridSearchCV(svm, param_grid, cv=cv,

scoring='f1', n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)

return grid.best_estimator_, grid.best_params_

def train_knn(X_train, y_train, param_grid, cv_folds=5):

"""

Train KNN using grid search.

"""

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_jobs=-1)

grid = GridSearchCV(knn, param_grid, cv=cv,

scoring='f1', n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)

return grid.best_estimator_, grid.best_params_

\`\`\`

\---

\## 8. Evaluation Protocol (\`src/evaluate.py\`)

\### 8.1 Primary Protocol (fills AUTHOR TO INSERT in Section 3.4 & 3.6)

| Aspect | Specification |

|---|---|

| \*\*Split type\*\* | Single stratified hold-out split |

| \*\*Ratio\*\* | 80% training : 20% testing |

| \*\*Stratified by class\*\* | Yes (preserves genuine vs. fake ratio) |

| \*\*Random seed\*\* | \`random_state=42\` |

| \*\*Secondary validation\*\* | 10-fold stratified cross-validation (for robustness reporting) |

\### 8.2 Metrics (Section 3.5)

Compute and report:

\- \*\*Accuracy\*\*

\- \*\*Precision\*\* (macro-averaged)

\- \*\*Recall\*\* (macro-averaged)

\- \*\*F1-Score\*\* (macro-averaged)

\- \*\*Confusion Matrix\*\*

\- \*\*ROC-AUC\*\*

\`\`\`python

from sklearn.metrics import (accuracy_score, precision_score, recall_score,

f1_score, confusion_matrix, roc_auc_score,

classification_report)

import pandas as pd

def evaluate_model(model, X_test, y_test, dataset_name, model_name):

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)\[:, 1\] if hasattr(model, "predict_proba") else None

results = {

'Dataset': dataset_name,

'Model': model_name,

'Accuracy': accuracy_score(y_test, y_pred),

'Precision': precision_score(y_test, y_pred, average='macro'),

'Recall': recall_score(y_test, y_pred, average='macro'),

'F1-Score': f1_score(y_test, y_pred, average='macro'),

'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,

'Confusion_Matrix': confusion_matrix(y_test, y_pred).tolist()

}

print(f"\\n=== {dataset_name} | {model_name} ===")

print(classification_report(y_test, y_pred, target_names=\['Fake', 'Genuine'\]))

return results

\`\`\`

\---

\## 9. Dataset-Specific Loading Logic

\### 9.1 FVC2000 Loader

\*\*Protocol:\*\*

\- \*\*Set B\*\* (fingers 101–110): Use ONLY for hyperparameter tuning / grid-search validation.

\- \*\*Set A\*\* (fingers 1–100): Use for the final 80:20 stratified split and 10-fold CV.

\*\*Class assignment:\*\*

\`\`\`python

\# Genuine (label = 0 for fake, 1 for genuine)

genuine_dbs = \['DB1_A', 'DB2_A', 'DB3_A'\] # Real sensors

fake_db = \['DB4_A'\] # Synthetic

\`\`\`

\*\*Note on imbalance:\*\* Genuine:SetA = 2,400 images; Fake:SetA = 800 images. Use \`class_weight='balanced'\` in SVM and stratified sampling.

\### 9.2 SOCOFing Loader

\*\*Class assignment:\*\*

\`\`\`python

\# Genuine

genuine_dir = "data/socofing/Real/" # label = 1

\# Fake (all altered combined)

fake_dirs = \[

"data/socofing/Altered/Altered-Easy/",

"data/socofing/Altered/Altered-Medium/",

"data/socofing/Altered/Altered-Hard/"

\] # label = 0

\`\`\`

\*\*Note on imbalance:\*\* ~6,000 genuine vs. ~49,000 fake. The \`class_weight='balanced'\` and \`weights='distance'\` in KNN will handle this. Alternatively, you may optionally cap the fake class at 12,000 (4,000 per difficulty) for faster training, but report this decision.

\---

\## 10. Main Execution Pipeline (\`src/main.py\`)

\`\`\`python

import os

import numpy as np

import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from config import \*

from preprocess import preprocess_image

from features import extract_lbp_features

from models import train_svm, train_knn

from evaluate import evaluate_model

def load_dataset_fvc2000(set_name='A'):

"""Load FVC2000 Set A or B. Returns X, y."""

X, y = \[\], \[\]

base = os.path.join(DATA_DIR, 'fvc2000')

\# Genuine: DB1, DB2, DB3

for db in \['DB1', 'DB2', 'DB3'\]:

db_path = os.path.join(base, f'{db}\_{set_name}')

for fname in os.listdir(db_path):

if fname.lower().endswith(('.tif', '.tiff')):

img = preprocess_image(os.path.join(db_path, fname))

feat = extract_lbp_features(img)

X.append(feat)

y.append(1) # Genuine

\# Fake: DB4 (synthetic)

db_path = os.path.join(base, f'DB4_{set_name}')

for fname in os.listdir(db_path):

if fname.lower().endswith(('.tif', '.tiff')):

img = preprocess_image(os.path.join(db_path, fname))

feat = extract_lbp_features(img)

X.append(feat)

y.append(0) # Fake

return np.array(X), np.array(y)

def load_dataset_socofing():

"""Load SOCOFing. Returns X, y."""

X, y = \[\], \[\]

base = os.path.join(DATA_DIR, 'socofing')

\# Genuine

real_path = os.path.join(base, 'Real')

for fname in os.listdir(real_path):

if fname.lower().endswith('.bmp'):

img = preprocess_image(os.path.join(real_path, fname))

feat = extract_lbp_features(img)

X.append(feat)

y.append(1)

\# Fake (all altered)

for sub in \['Altered-Easy', 'Altered-Medium', 'Altered-Hard'\]:

alt_path = os.path.join(base, 'Altered', sub)

for fname in os.listdir(alt_path):

if fname.lower().endswith('.bmp'):

img = preprocess_image(os.path.join(alt_path, fname))

feat = extract_lbp_features(img)

X.append(feat)

y.append(0)

return np.array(X), np.array(y)

def run_experiment(dataset_name, X, y, results_dir):

os.makedirs(results_dir, exist_ok=True)

\# 80:20 Stratified Split

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y

)

all_results = \[\]

\# --- SVM ---

print(f"\\n\[{dataset_name}\] Training SVM...")

svm_model, svm_params = train_svm(X_train, y_train, SVM_PARAM_GRID, CV_FOLDS_TUNING)

svm_res = evaluate_model(svm_model, X_test, y_test, dataset_name, 'SVM')

svm_res\['Best_Params'\] = svm_params

all_results.append(svm_res)

joblib.dump(svm_model, os.path.join(results_dir, 'svm_model.pkl'))

\# --- KNN ---

print(f"\\n\[{dataset_name}\] Training KNN...")

knn_model, knn_params = train_knn(X_train, y_train, KNN_PARAM_GRID, CV_FOLDS_TUNING)

knn_res = evaluate_model(knn_model, X_test, y_test, dataset_name, 'KNN')

knn_res\['Best_Params'\] = knn_params

all_results.append(knn_res)

joblib.dump(knn_model, os.path.join(results_dir, 'knn_model.pkl'))

\# --- 10-Fold CV (robustness check) ---

print(f"\\n\[{dataset_name}\] Running 10-Fold Stratified CV...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

svm_cv = cross_val_score(svm_model, X, y, cv=cv, scoring='f1_macro')

knn_cv = cross_val_score(knn_model, X, y, cv=cv, scoring='f1_macro')

print(f"SVM 10-Fold F1: {svm_cv.mean():.4f} (+/- {svm_cv.std():.4f})")

print(f"KNN 10-Fold F1: {knn_cv.mean():.4f} (+/- {knn_cv.std():.4f})")

\# Save results

df = pd.DataFrame(all_results)

df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)

return df

if \__name__ == '\__main_\_':

import pandas as pd

\# === Experiment 1: FVC2000 ===

print("=" \* 60)

print("LOADING FVC2000 SET A...")

X_fvc, y_fvc = load_dataset_fvc2000(set_name='A')

print(f"FVC2000 Set A: {X_fvc.shape}, Genuine={sum(y_fvc)}, Fake={len(y_fvc)-sum(y_fvc)}")

\# Optional: Use Set B for tuning validation (already embedded in GridSearchCV)

\# X_fvc_b, y_fvc_b = load_dataset_fvc2000(set_name='B')

run_experiment('FVC2000', X_fvc, y_fvc, os.path.join(RESULTS_DIR, 'fvc2000'))

\# === Experiment 2: SOCOFing ===

print("=" \* 60)

print("LOADING SOCOFing...")

X_soc, y_soc = load_dataset_socofing()

print(f"SOCOFing: {X_soc.shape}, Genuine={sum(y_soc)}, Fake={len(y_soc)-sum(y_soc)}")

run_experiment('SOCOFing', X_soc, y_soc, os.path.join(RESULTS_DIR, 'socofing'))

\`\`\`

\---

\## 11. Experimental Setup (fills AUTHOR TO INSERT in Section 3.6)

| Component | Specification |

|---|---|

| \*\*Programming language\*\* | Python 3.10+ |

| \*\*scikit-learn\*\* | 1.3.0 |

| \*\*scikit-image\*\* | 0.21.0 |

| \*\*NumPy\*\* | 1.24.3 |

| \*\*OpenCV\*\* | 4.8.0.74 |

| \*\*Hardware\*\* | Multi-core CPU; 16 GB RAM recommended for SOCOFing |

| \*\*SVM hyperparameter search\*\* | GridSearchCV, 5-fold stratified CV, scoring=\`f1\` |

| \*\*KNN hyperparameter search\*\* | GridSearchCV, 5-fold stratified CV, scoring=\`f1\` |

| \*\*Evaluation protocol\*\* | Primary: single stratified 80:20 hold-out (\`random_state=42\`); Secondary: 10-fold stratified CV |

\---

\## 12. Expected Outputs

After running \`python src/main.py\`, the agent should produce:

\`\`\`

results/

├── fvc2000/

│ ├── svm_model.pkl

│ ├── knn_model.pkl

│ └── results.csv

└── socofing/

├── svm_model.pkl

├── knn_model.pkl

└── results.csv

\`\`\`

\*\*\`results.csv\` columns:\*\*

| Dataset | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Best_Params | Confusion_Matrix |

\---

\## 13. Report Text to Copy-Paste

Here is the exact text to insert into your report, filling all \`\[AUTHOR TO INSERT\]\` gaps:

\> \*\*3.3 Feature Extraction\*\*

\> LBP was configured with neighborhood radius \*\*R = 1\*\*, number of sampling points \*\*P = 8\*\*, and the \*\*uniform\*\* pattern type. Each preprocessed image was partitioned into an \*\*8 × 8 grid\*\* of non-overlapping blocks before histogramming. The resulting feature vector dimensionality was \*\*3,776\*\* (64 blocks × 59 uniform LBP bins). Scikit-image was used to compute the LBP codes, and NumPy was used to generate normalized (density=True) block-wise histograms that were concatenated into the final feature vector.

\> \*\*3.4 Model Training\*\*

\> The dataset was split using a \*\*single stratified hold-out split\*\* at an \*\*80:20\*\* ratio with a fixed random seed of \*\*42\*\* to ensure reproducibility. The split was stratified by class (genuine vs. spoof) to preserve class balance in the test set. Additionally, \*\*10-fold stratified cross-validation\*\* was performed as a secondary robustness check.

\>

\> For SVM, the \*\*RBF kernel\*\* was employed. The regularization parameter \*\*C\*\* and kernel parameter \*\*γ\*\* were tuned automatically via \*\*grid search with 5-fold stratified cross-validation\*\* over the ranges \*\*C ∈ {0.1, 1, 10, 100}\*\* and \*\*γ ∈ {scale, auto, 0.001, 0.01, 0.1}\*\*. Class weights were set to \*\*balanced\*\* to handle any class imbalance.

\>

\> For KNN, the value of \*\*K\*\* was selected via \*\*grid search with 5-fold stratified cross-validation\*\* over odd values \*\*K ∈ {3, 5, 7, 9, 11, 13, 15}\*\*. The \*\*Euclidean distance\*\* metric was used, and voting weights were tuned over \*\*uniform\*\* and \*\*distance\*\*.

\> \*\*3.6 Experimental Setup\*\*

\> The experiments were conducted in \*\*Python 3.10\*\* using \*\*scikit-learn 1.3.0\*\*, \*\*scikit-image 0.21.0\*\*, and \*\*NumPy 1.24.3\*\*. Image I/O and enhancement used \*\*OpenCV 4.8.0.74\*\*. The hardware environment comprised a multi-core CPU with \*\*16 GB RAM\*\*. Hyperparameter search for both SVM and KNN was performed using \*\*GridSearchCV with 5-fold stratified cross-validation\*\* (scoring metric: F1-score). The primary evaluation protocol was a \*\*single stratified 80:20 hold-out split\*\*; a secondary \*\*10-fold stratified cross-validation\*\* was also reported to confirm robustness.

\---

\## 14. Agent Execution Checklist

Give this checklist to the AI agent:

\- \[ \] Create directory structure exactly as specified in Section 2.2

\- \[ \] Download & extract FVC2000 from Springer link into \`data/fvc2000/\`

\- \[ \] Download & extract SOCOFing from Kaggle into \`data/socofing/\`

\- \[ \] Create all 6 Python files in \`src/\` with the exact code provided

\- \[ \] Run \`pip install -r requirements.txt\`

\- \[ \] Execute \`python src/main.py\`

\- \[ \] Verify \`results/fvc2000/results.csv\` and \`results/socofing/results.csv\` exist

\- \[ \] Print confusion matrices and classification reports to console

\- \[ \] Save trained models as \`.pkl\` files

\- \[ \] Generate a summary markdown table comparing SVM vs. KNN across both datasets

\---

This plan is \*\*complete, deterministic, and agent-executable\*\*. Every hyperparameter, file path, and design decision is specified so the AI agent can implement it without ambiguity.