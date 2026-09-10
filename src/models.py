import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import SVM_PARAM_GRID, KNN_PARAM_GRID, CV_FOLDS_TUNING, RANDOM_STATE, N_JOBS


def _make_cv(y=None) -> StratifiedKFold:
    """
    Create a deterministic stratified K-Fold cross-validator.
    Dynamically adjusts n_splits if the training dataset or minority class is smaller
    than the default CV_FOLDS_TUNING (e.g. during micro-sample dry runs).
    """
    n_splits = CV_FOLDS_TUNING
    if y is not None:
        # BUGFIX: np.bincount requires int/int64; int8 arrays can cause TypeError on
        # some NumPy versions. Cast to int explicitly.
        counts = np.bincount(y.astype(int))
        counts = counts[counts > 0]
        min_class_count = int(np.min(counts)) if len(counts) > 0 else len(y)
        if min_class_count < 2:
            raise ValueError(
                f"Cannot cross-validate: the smallest class has only {min_class_count} "
                "training sample(s), but StratifiedKFold needs at least 2 per class. "
                "Increase --sample (or sample_fraction) so each class has >= 2 training examples."
            )
        n_splits = min(CV_FOLDS_TUNING, min_class_count)

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def get_svm_pipeline(y_train=None) -> GridSearchCV:
    """
    Build a reproducible, fully-tuned SVM pipeline.

    Pipeline steps:
      1. StandardScaler  — zero-mean, unit-variance normalisation
      2. SVC(rbf)        — support vector classifier with RBF kernel
                           class_weight='balanced' is mandatory to handle
                           any remaining class imbalance after undersampling.

    Tuning:
      GridSearchCV over SVM_PARAM_GRID with stratified cross-validation,
      scored by ROC AUC.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=True, random_state=RANDOM_STATE, cache_size=500)),
    ])

    param_grid = {f"svm__{k}": v for k, v in SVM_PARAM_GRID.items()}

    return GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = _make_cv(y_train),
        scoring    = "roc_auc",
        refit      = True,
        n_jobs     = N_JOBS,
        verbose    = 1,
    )


def get_knn_pipeline(y_train=None) -> GridSearchCV:
    """
    Build a reproducible, fully-tuned KNN pipeline.

    Pipeline steps:
      1. StandardScaler       — zero-mean, unit-variance normalisation
      2. KNeighborsClassifier — distance-based classifier

    Tuning:
      GridSearchCV over KNN_PARAM_GRID with stratified cross-validation,
      scored by ROC AUC.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_jobs=N_JOBS)),
    ])

    param_grid = {f"knn__{k}": v for k, v in KNN_PARAM_GRID.items()}

    return GridSearchCV(
        estimator  = pipeline,
        param_grid = param_grid,
        cv         = _make_cv(y_train),
        scoring    = "roc_auc",
        refit      = True,
        n_jobs     = N_JOBS,
        verbose    = 1,
    )
