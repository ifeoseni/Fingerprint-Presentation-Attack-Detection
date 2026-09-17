import os
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from config import FVC2000_DIR, SOCOFING_DIR, RANDOM_STATE, STRATIFY, TEST_SIZE
from preprocess import preprocess_image
from features import extract_lbp_features
from models import get_svm_pipeline, get_knn_pipeline, refit_svm_with_probability
from evaluate import evaluate_model, plot_roc_curve


def record_actual_timing(dataset_name: str, model_name: str, fit_seconds: float,
                          predict_seconds: float, n_test: int,
                          feature_extraction_seconds: float = None,
                          total_cell_seconds: float = None) -> None:
    """
    Append timing for a single (dataset, model) run — measured on the actual
    fit/predict calls that produced the reported results, not a separate
    re-fit benchmark — to RESULTS_DIR/actual_run_timing.json.

    feature_extraction_seconds and total_cell_seconds, when provided, are
    measured in THIS SAME run (not stitched together from a separate session/
    benchmark) and are the authoritative numbers for reporting total pipeline
    cost — do not reconstruct "total" by adding numbers from different runs,
    since system load, disk cache state, etc. make that unreliable.
    """
    from config import RESULTS_DIR
    timing_path = os.path.join(RESULTS_DIR, "actual_run_timing.json")

    all_results = {}
    if os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            all_results = json.load(f)

    per_sample_ms = (predict_seconds / n_test) * 1000 if n_test else None
    entry = {
        "fit_seconds": round(fit_seconds, 3),
        "fit_minutes": round(fit_seconds / 60, 2),
        "predict_seconds_total": round(predict_seconds, 4),
        "predict_ms_per_sample": round(per_sample_ms, 5) if per_sample_ms else None,
        "n_test": n_test,
    }
    if feature_extraction_seconds is not None:
        entry["feature_extraction_seconds"] = round(feature_extraction_seconds, 2)
        entry["feature_extraction_minutes"] = round(feature_extraction_seconds / 60, 2)
    if total_cell_seconds is not None:
        entry["total_cell_seconds"] = round(total_cell_seconds, 2)
        entry["total_cell_minutes"] = round(total_cell_seconds / 60, 2)

    all_results.setdefault(dataset_name, {})[model_name] = entry

    with open(timing_path, "w") as f:
        json.dump(all_results, f, indent=2)

    msg = (f"  [TIMING] Fit: {fit_seconds:.2f}s ({fit_seconds/60:.2f} min) | "
           f"Predict: {predict_seconds:.4f}s total ({per_sample_ms:.5f} ms/sample)")
    if total_cell_seconds is not None:
        msg += f" | TOTAL CELL: {total_cell_seconds:.1f}s ({total_cell_seconds/60:.2f} min)"
    print(msg + f" | saved to {timing_path}")


def _load_images_from_dir(directory: str, label: int, extension: str,
                           X: list, y: list, tag: str = "", sample_fraction: float = 1.0) -> None:
    """
    Helper: load all images with a given extension from a directory,
    extract LBP features, and append to X and y.
    Files are sorted alphabetically to guarantee deterministic ordering.
    """
    if not os.path.exists(directory):
        print(f"[WARNING] Directory not found, skipping: {directory}")
        return

    filenames = sorted([f for f in os.listdir(directory) if f.lower().endswith(extension)])
    if not filenames:
        print(f"[WARNING] No {extension} images found in {directory}")
        return

    if 0.0 < sample_fraction < 1.0:
        filenames = filenames[: max(1, int(len(filenames) * sample_fraction))]

    print(f"  [{tag}] Loading {len(filenames)} images from {os.path.basename(directory)} (label={label})...")
    for fname in filenames:
        fpath = os.path.join(directory, fname)
        try:
            img = preprocess_image(fpath)
            features = extract_lbp_features(img)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"  [ERROR] Skipping {fpath}: {e}")


def _balance_training_set(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Balance the training set by random undersampling of the majority class.
    The minority class is kept in full; the majority class is randomly downsampled
    to match, using RANDOM_STATE for reproducibility.

    This is applied only to the TRAINING set. The test set always reflects the
    real-world (imbalanced) class distribution for honest evaluation.

    Returns:
        (X_balanced, y_balanced) as shuffled np.ndarray pairs.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        balanced_indices.append(chosen)

    all_indices = np.concatenate(balanced_indices)
    all_indices = rng.permutation(all_indices)  # shuffle combined set

    print(f"  [BALANCE] Undersampled to {min_count} samples per class "
          f"(total training: {len(all_indices)})")

    return X_train[all_indices], y_train[all_indices]


def load_fvc2000_data(sample_fraction: float = 1.0):
    """
    Load FVC2000 dataset.

    Labelling strategy (per the FVC2000 specification):
      - Genuine (Class 1): DB1, DB2, DB3  (real sensor captures)
      - Fake    (Class 0): DB4            (synthetic generation)

    Split strategy (per the FVC2000 benchmark protocol):
      - Set B  (Db*_b): fingers 101-110 → TRAINING / parameter tuning
      - Set A  (Db*_a): fingers   1-100 → TESTING  / benchmark
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    dbs_path = os.path.join(FVC2000_DIR, "Dbs")
    if not os.path.exists(dbs_path):
        raise FileNotFoundError(
            f"FVC2000 Dbs directory not found at: {dbs_path}\n"
            "Please set the FVC2000_DIR environment variable to the correct path."
        )

    # (folder_prefix, label)  1=genuine, 0=fake
    db_mapping = [("Db1", 1), ("Db2", 1), ("Db3", 1), ("Db4", 0)]

    print("Loading FVC2000 data...")
    for db_prefix, label in db_mapping:
        _load_images_from_dir(
            os.path.join(dbs_path, f"{db_prefix}_b"), label, ".tif",
            X_train, y_train, tag="TRAIN", sample_fraction=sample_fraction
        )
        _load_images_from_dir(
            os.path.join(dbs_path, f"{db_prefix}_a"), label, ".tif",
            X_test, y_test, tag="TEST", sample_fraction=sample_fraction
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("FVC2000 loader produced empty train or test arrays. "
                           "Check the dataset path and folder structure.")

    X_train_arr = np.array(X_train, dtype=np.float32)
    X_test_arr  = np.array(X_test,  dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    y_test_arr  = np.array(y_test,  dtype=np.int32)

    # Balance training set if class sizes differ
    genuine_train = (y_train_arr == 1).sum()
    fake_train    = (y_train_arr == 0).sum()
    if genuine_train != fake_train:
        X_train_arr, y_train_arr = _balance_training_set(X_train_arr, y_train_arr)

    print(f"\nFVC2000 — Train: {X_train_arr.shape[0]} samples | "
          f"Test: {X_test_arr.shape[0]} samples | "
          f"Feature dim: {X_train_arr.shape[1]}")
    print(f"  Train class distribution — Genuine: {(y_train_arr==1).sum()} | Fake: {(y_train_arr==0).sum()}")
    print(f"  Test  class distribution — Genuine: {(y_test_arr==1).sum()}  | Fake: {(y_test_arr==0).sum()}")

    return X_train_arr, X_test_arr, y_train_arr, y_test_arr


def load_socofing_data(sample_fraction: float = 1.0):
    """
    Load SOCOFing dataset.

    Labelling strategy:
      - Genuine (Class 1): Real/
      - Fake    (Class 0): Altered/Altered-Easy + Altered-Medium + Altered-Hard

    Class balance:
      The TRAINING set is balanced by undersampling the majority (Fake) class to
      match the minority (Genuine) count, preventing classifiers from trivially
      predicting the majority class. The TEST set retains the real-world distribution.

    Split strategy: stratified random 80/20 train-test split (RANDOM_STATE=42).
    Files are sorted alphabetically before sampling to guarantee reproducibility.
    """
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    if not os.path.exists(SOCOFING_DIR):
        raise FileNotFoundError(
            f"SOCOFing directory not found at: {SOCOFING_DIR}\n"
            "Please set the SOCOFING_DIR environment variable to the correct path."
        )

    X, y = [], []
    print("Loading SOCOFing data...")

    # ── Genuine ────────────────────────────────────────────────────────────────
    real_dir = os.path.join(SOCOFING_DIR, "Real")
    real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(".bmp")])
    real_files = real_files[: max(1, int(len(real_files) * sample_fraction))]
    print(f"  [REAL] Using {len(real_files)} genuine images.")
    for fname in real_files:
        fpath = os.path.join(real_dir, fname)
        try:
            X.append(extract_lbp_features(preprocess_image(fpath)))
            y.append(1)
        except Exception as e:
            print(f"  [ERROR] Skipping {fpath}: {e}")

    # ── Fake (three difficulty tiers) ─────────────────────────────────────────
    altered_base = os.path.join(SOCOFING_DIR, "Altered")
    for difficulty in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        diff_dir = os.path.join(altered_base, difficulty)
        if not os.path.exists(diff_dir):
            print(f"  [WARNING] Sub-folder not found: {diff_dir}")
            continue
        alt_files = sorted([f for f in os.listdir(diff_dir) if f.lower().endswith(".bmp")])
        alt_files = alt_files[: max(1, int(len(alt_files) * sample_fraction))]
        print(f"  [{difficulty}] Using {len(alt_files)} spoof images.")
        for fname in alt_files:
            fpath = os.path.join(diff_dir, fname)
            try:
                X.append(extract_lbp_features(preprocess_image(fpath)))
                y.append(0)
            except Exception as e:
                print(f"  [ERROR] Skipping {fpath}: {e}")

    if len(X) == 0:
        raise RuntimeError("SOCOFing loader produced no samples. "
                           "Check the dataset path and folder structure.")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    # Stratified split — test set keeps real-world distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_arr if STRATIFY else None,
    )

    # Balance only the TRAINING set
    genuine_train = (y_train == 1).sum()
    fake_train    = (y_train == 0).sum()
    if genuine_train != fake_train:
        X_train, y_train = _balance_training_set(X_train, y_train)

    print(f"\nSOCOFing — Train: {X_train.shape[0]} samples | "
          f"Test: {X_test.shape[0]} samples | "
          f"Feature dim: {X_train.shape[1]}")
    print(f"  Train class distribution — Genuine: {(y_train==1).sum()} | Fake: {(y_train==0).sum()}")
    print(f"  Test  class distribution — Genuine: {(y_test==1).sum()}  | Fake: {(y_test==0).sum()}")

    return X_train, X_test, y_train, y_test


def run_pipeline(dataset_name: str, X_train, X_test, y_train, y_test,
                  feature_extraction_seconds: float = None) -> None:
    """
    Train SVM and KNN classifiers on (X_train, y_train) and evaluate on (X_test, y_test).

    BUGFIX: The results CSV is cleared at the start of each dataset's pipeline run
    to prevent duplicate rows from accumulating across multiple invocations of the
    same experiment. Each full run produces exactly 2 rows: one SVM, one KNN.

    feature_extraction_seconds: if the caller timed the data-loading/feature-
    extraction step (e.g. load_fvc2000_data()) before calling this function,
    pass that duration here so record_actual_timing can report a true
    per-model total (feature extraction + this model's own fit/predict/eval),
    measured in this same run rather than stitched from a separate session.
    """
    from config import RESULTS_DIR
    import os

    print(f"\n{'='*60}")
    print(f"  Running Pipeline — {dataset_name}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Clear CSV for this dataset at start of each run (prevents row accumulation)
    csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_metrics.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"  [INFO] Previous results cleared: {csv_path}")

    import joblib
    for model_name, get_pipeline in [('SVM', get_svm_pipeline), ('KNN', get_knn_pipeline)]:
        model_block_start = time.perf_counter()
        print(f"\n--- Training {model_name} (GridSearchCV) ---")
        clf = get_pipeline(y_train)
        t0 = time.perf_counter()
        with joblib.parallel_backend('threading'):
            clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        print(f"  Best params: {clf.best_params_}")
        print(f"  Best CV ROC-AUC: {clf.best_score_:.4f}")

        if model_name == 'SVM':
            # Search ran with probability=False (fast); refit the winning
            # configuration once with probability=True for predict_proba.
            # This refit is a real, necessary cost of producing a usable final
            # model, so it counts toward total fit time reported in 4.3.
            t0 = time.perf_counter()
            eval_model = refit_svm_with_probability(clf, X_train, y_train)
            fit_seconds += time.perf_counter() - t0
        else:
            eval_model = clf

        t0 = time.perf_counter()
        y_pred = eval_model.predict(X_test)
        y_prob = eval_model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - t0

        evaluate_model(y_test, y_pred, y_prob, model_name, dataset_name)
        plot_roc_curve(y_test, y_prob, model_name, dataset_name)

        model_block_seconds = time.perf_counter() - model_block_start
        total_cell_seconds = (
            feature_extraction_seconds + model_block_seconds
            if feature_extraction_seconds is not None else None
        )
        record_actual_timing(dataset_name, model_name, fit_seconds, predict_seconds, len(X_test),
                              feature_extraction_seconds=feature_extraction_seconds,
                              total_cell_seconds=total_cell_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fingerprint Presentation Attack Detection Pipeline — SVM & KNN with LBP features"
    )
    parser.add_argument("--fvc",      action="store_true",  help="Run FVC2000 experiment")
    parser.add_argument("--socofing", action="store_true",  help="Run SOCOFing experiment")
    parser.add_argument("--sample",   type=float, default=1.0,
                        help="Fraction of images to use per sub-folder (0 < sample <= 1.0, default: 1.0)")
    args = parser.parse_args()

    if not args.fvc and not args.socofing:
        parser.error("Please specify at least one dataset: --fvc and/or --socofing")

    if args.fvc:
        X_tr, X_te, y_tr, y_te = load_fvc2000_data(sample_fraction=args.sample)
        run_pipeline("FVC2000", X_tr, X_te, y_tr, y_te)

    if args.socofing:
        X_tr, X_te, y_tr, y_te = load_socofing_data(sample_fraction=args.sample)
        run_pipeline("SOCOFing", X_tr, X_te, y_tr, y_te)
