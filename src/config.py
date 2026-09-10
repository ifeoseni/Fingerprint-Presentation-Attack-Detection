import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Using environment variables if provided (great for Colab/Kaggle), otherwise fallback to local paths
FVC2000_DIR = os.environ.get("FVC2000_DIR", os.path.join(BASE_DIR, "74034_3_En_4_MOESM1_ESM", "FVC2000"))
SOCOFING_DIR = os.environ.get("SOCOFING_DIR", os.path.join(BASE_DIR, "archive(2)", "SOCOFing"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))

# === Image Preprocessing ===
TARGET_SIZE = (300, 300)  # Resize ALL images to 300x300
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# === LBP Feature Extraction ===
LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = 'uniform'   # Rotation-invariant uniform patterns
LBP_BLOCKS = (8, 8)      # 8×8 spatial grid → 64 blocks
# For method='uniform' and P=8, the LBP output has exactly P+2 = 10 distinct
# pattern codes (0..9). Code P+1 accumulates all non-uniform patterns.
LBP_BINS = 10            # = LBP_POINTS + 2
FEATURE_DIM = LBP_BLOCKS[0] * LBP_BLOCKS[1] * LBP_BINS  # 8 × 8 × 10 = 640

# === Train-Test Split ===
TEST_SIZE = 0.20
RANDOM_STATE = 42
STRATIFY = True

# === Cross-Validation ===
# CV_FOLDS_TUNING: used by GridSearchCV in models.py for hyperparameter tuning.
# CV_FOLDS: reserved for standalone k-fold evaluation (not active in current pipeline).
CV_FOLDS = 10           # Stratified K-Fold for secondary evaluation (reserved)
CV_FOLDS_TUNING = 5     # For GridSearchCV hyperparameter tuning

# === Parallel Jobs ===
# Default to -1 (all cores).
# On Windows + Python 3.13, the loky process-spawning backend crashes with
# ModuleNotFoundError: No module named '_posixsubprocess'.
#
# Root cause: sklearn's GridSearchCV uses joblib.Parallel as a context manager
# (`with parallel:`) which calls __enter__ -> _initialize_backend -> loky -> crash.
# sklearn also caches `from joblib import Parallel` at import time, so replacing
# joblib.Parallel with a subclass after the fact has no effect.
#
# The ONLY reliable fix: patch __init__ IN-PLACE on the original Parallel class so
# every new instance (including those sklearn creates internally) defaults to
# backend='threading'. Threading uses all CPU cores via GIL-releasing C extensions
# (libsvm, liblinear) without triggering any process-spawning code.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

import joblib.parallel as _jpar
if os.name == "nt":
    # 100% Safe Windows Multi-core Fix:
    # Instead of monkey-patching methods (which causes RecursionErrors in Jupyter
    # if cells are run multiple times), we simply overwrite the backend registry.
    _jpar.BACKENDS['loky'] = _jpar.BACKENDS['threading']
    _jpar.DEFAULT_BACKEND = 'threading'


# === SVM Hyperparameter Search Space ===
# class_weight=None is included alongside 'balanced' because the training set is
# already balanced by majority-class undersampling. Forcing 'balanced' alone would
# double-compensate for an imbalance that no longer exists after undersampling.
# GridSearchCV will pick the optimal setting via cross-validation.
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced'],
}

# === KNN Hyperparameter Search Space ===
KNN_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean']
}
