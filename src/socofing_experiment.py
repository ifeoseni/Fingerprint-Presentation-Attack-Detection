import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from preprocess import preprocess_image
from features import extract_lbp_features

def load_socofing_image_paths(socofing_dir, difficulty="All"):
    """
    Returns lists of paths for genuine and fake images, ignoring the duplicate nested folder.
    difficulty: "Easy", "Medium", "Hard", or "All"
    """
    real_dir = os.path.join(socofing_dir, "Real")
    altered_base = os.path.join(socofing_dir, "Altered")
    
    genuine_paths = []
    if os.path.exists(real_dir):
        genuine_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(".bmp")])
        
    fake_paths = []
    if difficulty == "All":
        difficulties = ["Altered-Easy", "Altered-Medium", "Altered-Hard"]
    else:
        difficulties = [f"Altered-{difficulty}"]
        
    for diff in difficulties:
        diff_dir = os.path.join(altered_base, diff)
        if os.path.exists(diff_dir):
            paths = sorted([os.path.join(diff_dir, f) for f in os.listdir(diff_dir) if f.lower().endswith(".bmp")])
            fake_paths.extend(paths)
            
    return genuine_paths, fake_paths

def extract_subject_id(filename):
    """
    Extracts the subject ID from a SOCOFing filename.
    Format: 100__M_Left_index_finger.BMP -> 100
    """
    basename = os.path.basename(filename)
    if "__" in basename:
        try:
            return int(basename.split("__")[0])
        except ValueError:
            pass
    return -1

def balance_training_set(X_train, y_train, groups_train, random_state=42):
    """
    Balance the training set by undersampling the majority class.
    We don't need to preserve group structure strictly during undersampling because 
    the train/test split (GroupShuffleSplit) already isolated the identities.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        balanced_indices.append(chosen)

    all_indices = np.concatenate(balanced_indices)
    all_indices = rng.permutation(all_indices)

    return X_train[all_indices], y_train[all_indices]

def prepare_experiment_data(genuine_paths, fake_paths, sample_fraction=1.0, random_state=42):
    """
    1. Parse subject IDs for all images.
    2. GroupShuffleSplit into train/test (80/20) based on subject ID.
    3. Extract features for train and test separately.
    4. Balance the training set.

    Safety:
      GroupShuffleSplit requires at least ~5 unique subjects (each subject has ~5
      genuine images) to perform a stable 80/20 split. If the resulting genuine
      sample count is below MIN_GENUINE_SAMPLES, the fraction is auto-clamped
      upward and a warning is printed.
    """
    MIN_GENUINE_SAMPLES = 30  # need ~6 subjects minimum for GroupShuffleSplit

    if 0.0 < sample_fraction < 1.0:
        gen_count = max(1, int(len(genuine_paths) * sample_fraction))
        fake_count = max(1, int(len(fake_paths) * sample_fraction))

        # Safety clamp: if too few genuine images, auto-raise to minimum
        if gen_count < MIN_GENUINE_SAMPLES:
            clamped_frac = MIN_GENUINE_SAMPLES / len(genuine_paths)
            print(
                f"  [WARNING] SAMPLE_FRACTION={sample_fraction} yields only {gen_count} genuine images "
                f"(need >= {MIN_GENUINE_SAMPLES} for a stable GroupShuffleSplit). "
                f"Auto-clamped to fraction={clamped_frac:.4f} ({MIN_GENUINE_SAMPLES} genuine)."
            )
            gen_count = MIN_GENUINE_SAMPLES
            fake_count = max(MIN_GENUINE_SAMPLES, int(len(fake_paths) * clamped_frac))

        genuine_paths = genuine_paths[:gen_count]
        fake_paths = fake_paths[:fake_count]

    all_paths = genuine_paths + fake_paths

    labels = np.array([1] * len(genuine_paths) + [0] * len(fake_paths))
    groups = np.array([extract_subject_id(p) for p in all_paths])
    
    # Use GroupShuffleSplit to split by subject
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(all_paths, labels, groups=groups))
    
    print(f"  Extracting LBP features for {len(train_idx)} training images and {len(test_idx)} testing images...")
    
    # Extract training features
    X_train_list, y_train_list, groups_train_list = [], [], []
    for idx in train_idx:
        try:
            img = preprocess_image(all_paths[idx])
            features = extract_lbp_features(img)
            X_train_list.append(features)
            y_train_list.append(labels[idx])
            groups_train_list.append(groups[idx])
        except Exception:
            continue
            
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list)
    groups_train = np.array(groups_train_list)
    
    # Extract testing features
    X_test_list, y_test_list = [], []
    for idx in test_idx:
        try:
            img = preprocess_image(all_paths[idx])
            features = extract_lbp_features(img)
            X_test_list.append(features)
            y_test_list.append(labels[idx])
        except Exception:
            continue
            
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list)
    
    # Balance training data
    X_train_bal, y_train_bal = balance_training_set(X_train, y_train, groups_train, random_state=random_state)
    
    return X_train_bal, X_test, y_train_bal, y_test
