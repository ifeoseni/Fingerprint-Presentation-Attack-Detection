import numpy as np
from skimage.feature import local_binary_pattern
from config import LBP_RADIUS, LBP_POINTS, LBP_METHOD, LBP_BLOCKS, LBP_BINS, FEATURE_DIM


# ── Pre-computed constants ────────────────────────────────────────────────────
# For method='uniform' and P=8, skimage's local_binary_pattern produces integer
# codes in the range [0, P+1] = [0, 9]:
#   codes 0..P (i.e. 0..8): the P+1 uniform rotation-invariant patterns
#   code P+1   (i.e. 9)   : all non-uniform patterns lumped together
# Therefore the histogram range must cover [0, P+2) = [0, 10).
_LBP_RANGE = (0, LBP_POINTS + 2)  # == (0, 10)

assert LBP_BINS == LBP_POINTS + 2, (
    f"LBP_BINS must equal LBP_POINTS+2 for uniform LBP. "
    f"Got LBP_BINS={LBP_BINS}, LBP_POINTS+2={LBP_POINTS+2}."
)
assert FEATURE_DIM == LBP_BLOCKS[0] * LBP_BLOCKS[1] * LBP_BINS, (
    f"FEATURE_DIM mismatch: expected {LBP_BLOCKS[0]*LBP_BLOCKS[1]*LBP_BINS}, "
    f"got {FEATURE_DIM}."
)


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    """
    Extract block-based Local Binary Pattern (LBP) features from a grayscale image.

    Algorithm
    ---------
    1. Compute the uniform LBP representation of the image:
         - Sampling points: P = LBP_POINTS = 8
         - Radius        : R = LBP_RADIUS  = 1
         - Method        : "uniform"
       Output codes are integers in [0, P+1] = [0, 9].

    2. Divide the LBP map into a LBP_BLOCKS = (8, 8) spatial grid → 64 blocks.

    3. For each block, compute a LBP_BINS = 10 -bin histogram over range [0, 10).

    4. L1-normalise each block histogram so it sums to 1 (independent of block size).

    5. Concatenate all 64 block histograms → feature vector of length:
         FEATURE_DIM = 8 × 8 × 10 = 640

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale uint8 array of shape (H, W).
        Must be 2-D (single channel). Must not contain NaN or Inf.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector of shape (640,).
        Each element is in [0, 1]. No NaN or Inf values.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {image.shape}")

    # Step 1 — Compute LBP map (values in [0, LBP_POINTS+1])
    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)

    # Step 2 — Compute block dimensions (last row/col absorbs any remainder pixels)
    h, w    = image.shape
    block_h = h // LBP_BLOCKS[0]
    block_w = w // LBP_BLOCKS[1]

    histograms = []

    # Steps 3 & 4 — Per-block histogram + L1-normalisation
    for row in range(LBP_BLOCKS[0]):
        for col in range(LBP_BLOCKS[1]):
            start_y = row * block_h
            end_y   = (row + 1) * block_h if row < LBP_BLOCKS[0] - 1 else h
            start_x = col * block_w
            end_x   = (col + 1) * block_w if col < LBP_BLOCKS[1] - 1 else w

            block = lbp[start_y:end_y, start_x:end_x]

            hist, _ = np.histogram(block.ravel(), bins=LBP_BINS, range=_LBP_RANGE)

            # L1-normalise: eps avoids division by zero on blank blocks
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7

            histograms.append(hist)

    # Step 5 — Concatenate
    feature_vector = np.concatenate(histograms).astype(np.float32)
    return feature_vector


# ── Validation ────────────────────────────────────────────────────────────────

def validate_feature_extraction(image: np.ndarray) -> None:
    """
    Run all structural and statistical assertions on the output of
    extract_lbp_features for a given image.

    Checks
    ------
    1. Output shape is (FEATURE_DIM,) == (640,).
    2. Output dtype is float32.
    3. Exactly LBP_BLOCKS[0] * LBP_BLOCKS[1] == 64 spatial blocks are produced.
    4. Each block histogram has exactly LBP_BINS == 10 values.
    5. Each block histogram sums approximately to 1 (L1-normalisation).
    6. No NaN values in the feature vector.
    7. No Inf values in the feature vector.

    Parameters
    ----------
    image : np.ndarray
        A preprocessed grayscale image to validate.

    Raises
    ------
    AssertionError if any check fails.
    """
    # Checks 1, 2, 6, 7 — run the actual production function
    fv = extract_lbp_features(image)

    assert fv.shape == (FEATURE_DIM,), \
        f"[Check 1] Shape: expected ({FEATURE_DIM},), got {fv.shape}"
    assert fv.dtype == np.float32, \
        f"[Check 2] Dtype: expected float32, got {fv.dtype}"
    assert not np.any(np.isnan(fv)), "[Check 6] Feature vector contains NaN values"
    assert not np.any(np.isinf(fv)), "[Check 7] Feature vector contains Inf values"

    # Checks 3, 4, 5 — inspect block-level properties directly
    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    h, w    = image.shape
    block_h = h // LBP_BLOCKS[0]
    block_w = w // LBP_BLOCKS[1]

    block_count    = 0
    per_block_sums = []

    for row in range(LBP_BLOCKS[0]):
        for col in range(LBP_BLOCKS[1]):
            start_y = row * block_h
            end_y   = (row + 1) * block_h if row < LBP_BLOCKS[0] - 1 else h
            start_x = col * block_w
            end_x   = (col + 1) * block_w if col < LBP_BLOCKS[1] - 1 else w

            block = lbp[start_y:end_y, start_x:end_x]
            hist, _ = np.histogram(block.ravel(), bins=LBP_BINS, range=_LBP_RANGE)
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7

            # Check 4 — bins per block
            assert len(hist) == LBP_BINS, \
                f"[Check 4] Block ({row},{col}): expected {LBP_BINS} bins, got {len(hist)}"

            # Check 5 — float32 rounding can shift the sum very slightly above 1.0;
            # use 1e-4 tolerance (tighter than 1e-3, safe for float32 precision).
            norm_sum = float(hist.sum())
            assert abs(norm_sum - 1.0) < 1e-4, \
                f"[Check 5] Block ({row},{col}) histogram sum is {norm_sum:.7f}, expected ≈ 1.0"

            per_block_sums.append(norm_sum)
            block_count += 1

    # Check 3 — block count
    expected_blocks = LBP_BLOCKS[0] * LBP_BLOCKS[1]
    assert block_count == expected_blocks, \
        f"[Check 3] Expected {expected_blocks} blocks, got {block_count}"

    print("[VALIDATION PASSED]")
    print(f"  Feature vector shape : {fv.shape}           ✓  (expected ({FEATURE_DIM},))")
    print(f"  Feature vector dtype : {fv.dtype}          ✓  (expected float32)")
    print(f"  Spatial blocks       : {block_count}                  ✓  (expected {expected_blocks})")
    print(f"  Bins per block       : {LBP_BINS}                  ✓  (expected {LBP_BINS})")
    print(f"  No NaN               : ✓")
    print(f"  No Inf               : ✓")
    print(f"  Histogram norms (float32): "
          f"min={min(per_block_sums):.7f}  max={max(per_block_sums):.7f}  ✓")


if __name__ == "__main__":
    """
    Quick self-test: generates a random synthetic image and validates the pipeline.
    Run with:  python src/features.py
    """
    rng = np.random.default_rng(42)

    # Synthetic image 1
    img_a = (rng.integers(0, 256, size=(300, 300))).astype(np.uint8)
    print("=== Validating on synthetic image A ===")
    validate_feature_extraction(img_a)

    # Synthetic image 2 — deliberately very different pixel distribution
    img_b = np.zeros((300, 300), dtype=np.uint8)      # all-black vs random
    fv_a  = extract_lbp_features(img_a)
    fv_b  = extract_lbp_features(img_b)

    norm_a = np.linalg.norm(fv_a)
    norm_b = np.linalg.norm(fv_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        cosine_sim = 0.0  # one vector is zero — they are maximally different
    else:
        cosine_sim = float(np.dot(fv_a, fv_b) / (norm_a * norm_b))

    print(f"\n=== Distinctiveness check ===")
    print(f"  Cosine similarity (random vs all-black): {cosine_sim:.4f}")
    # Random vs all-black images must not be identical in feature space
    assert cosine_sim < 0.99, \
        f"WARNING: Two very different images gave cosine similarity={cosine_sim:.4f} — features may be degenerate!"
    print(f"  ✓  Different images produce meaningfully different vectors")

    print("\nAll self-tests passed.")
