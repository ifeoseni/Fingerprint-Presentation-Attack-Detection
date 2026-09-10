import cv2
import numpy as np
import os
from config import TARGET_SIZE, CLAHE_CLIP, CLAHE_GRID

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a fingerprint image.
    Steps: Grayscale -> Resize -> CLAHE -> Normalize
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # 2. Resize using bicubic interpolation
    # clip to [0, 255] guards against bicubic ringing artefacts at sharp edges
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)

    # 3. Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    img_clahe = clahe.apply(img_resized)

    # 4. Normalize to [0, 255] (already uint8, but just to be safe)
    img_normalized = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_normalized
