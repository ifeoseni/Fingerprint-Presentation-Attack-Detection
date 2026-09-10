import os
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from preprocess import preprocess_image
from features import extract_lbp_features, LBP_POINTS, LBP_RADIUS, LBP_METHOD, LBP_BLOCKS

def generate_lbp_examples(socofing_dir):
    """
    Save one fingerprint from each category (Genuine, Easy, Medium, Hard)
    as original and LBP images to results/lbp_examples/.
    Also save a CSV containing an example feature vector to mathematically prove the 640-dimension mapping.
    """
    out_dir = os.path.join(RESULTS_DIR, "lbp_examples")
    os.makedirs(out_dir, exist_ok=True)
    
    categories = {
        "genuine": os.path.join(socofing_dir, "Real"),
        "easy": os.path.join(socofing_dir, "Altered", "Altered-Easy"),
        "medium": os.path.join(socofing_dir, "Altered", "Altered-Medium"),
        "hard": os.path.join(socofing_dir, "Altered", "Altered-Hard")
    }
    
    from skimage.feature import local_binary_pattern
    
    feature_csv_rows = []
    
    for name, path in categories.items():
        if not os.path.exists(path):
            continue
            
        files = [f for f in os.listdir(path) if f.lower().endswith(".bmp")]
        if not files:
            continue
            
        sample_file = os.path.join(path, files[0])
        img = preprocess_image(sample_file)
        
        # Save original (preprocessed)
        cv2.imwrite(os.path.join(out_dir, f"{name}_original.png"), img)
        
        # Save LBP image representation
        lbp_img = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        # Normalize to 0-255 for visualization
        lbp_vis = (255 * (lbp_img - lbp_img.min()) / (lbp_img.max() - lbp_img.min())).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{name}_lbp.png"), lbp_vis)
        
        # Extract the actual 640-dim feature vector
        features = extract_lbp_features(img)
        assert features.shape == (640,), f"Expected 640 dimensions, got {features.shape}"
        
        # Prepare for CSV
        row = {"image": f"{name}_{files[0]}", "label": 1 if name == "genuine" else 0}
        for i, val in enumerate(features, start=1):
            row[f"feature_{i:03d}"] = val
        feature_csv_rows.append(row)
        
    if feature_csv_rows:
        df = pd.DataFrame(feature_csv_rows)
        csv_path = os.path.join(out_dir, "sample_lbp_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved LBP example images and 640-dim feature CSV to {out_dir}/")
