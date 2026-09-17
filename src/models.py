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

    NOTE ON probability=False: sklearn's "roc_auc" scorer prefers
    decision_function over predict_proba when both are available, so
    GridSearchCV never needs predict_proba during the search itself.
    SVC(probability=True) triggers an expensive internal 5-fold Platt-scaling
    calibration on every single .fit() call (measured ~5x slower per fit) —
    paying that cost on all (candidates x folds) fits during search is pure
    waste. Search with probability=False here; call refit_svm_with_probability()
    on the result to get a predict_proba-capable model for final evaluation,
    paying the calibration cost exactly once instead of on every search fit.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=False, random_state=RANDOM_STATE, cache_size=500)),
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


def refit_svm_with_probability(svm_grid: GridSearchCV, X_train, y_train) -> Pipeline:
    """
    Refit the winning SVM configuration from an already-fitted GridSearchCV
    (searched with probability=False) as a single probability=True model,
    for use in final evaluation (predict_proba / ROC curves).

    This pays the ~5x Platt-scaling calibration cost exactly once, instead of
    on every candidate x fold fit during the grid search.
    """
    best_svm_params = {
        k.replace("svm__", ""): v
        for k, v in svm_grid.best_params_.items()
    }
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=True, random_state=RANDOM_STATE, cache_size=500, **best_svm_params)),
    ])
    final_pipeline.fit(X_train, y_train)
    return final_pipeline


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
