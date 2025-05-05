import os
import sys
# add project root (parent folder) to path so config.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import local_binary_pattern
from multiprocessing import Pool

from config import TRAIN_IMG_DIR, TEST_IMG_DIR, TRAIN_FEATURES, TEST_FEATURES

def extract_features(image_path):
    """Compute per-channel stats, HSV means, and LBP histogram features."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    r, g, b = cv2.split(image)
    
    feats = {
        "R_mean": np.mean(r),    "R_var": np.var(r),    "R_skew": skew(r.flatten()),
        "G_mean": np.mean(g),    "G_var": np.var(g),    "G_skew": skew(g.flatten()),
        "B_mean": np.mean(b),    "B_var": np.var(b),    "B_skew": skew(b.flatten()),
        "Gray_mean": np.mean(gray), "Gray_var": np.var(gray), "Gray_skew": skew(gray.flatten())
    }

    # HSV channel means
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    feats.update({
        "H_mean": np.mean(h),
        "S_mean": np.mean(s),
        "V_mean": np.mean(v),
    })

    # LBP histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    for i, val in enumerate(hist):
        feats[f"LBP_{i}"] = val

    return feats

def process_image(item):
    """
    item = (image_path, label)
    returns a dict of features + 'Image_Path' + 'label'
    """
    path, label = item
    feats = extract_features(path)
    feats["Image_Path"] = path
    feats["label"] = label
    return feats

def load_dataset_parallel(base_path):
    # gather (path, label) pairs
    items = []
    for label, category in enumerate(["fake", "real"]):
        cat_dir = os.path.join(base_path, category)
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(cat_dir, fname), label))

    # parallel feature extraction
    with Pool(8) as pool:
        feature_dicts = pool.map(process_image, items)

    # one DataFrame including path & label
    return pd.DataFrame(feature_dicts)

def run_feature_extraction():
    print("[INFO] Starting feature extraction...")
    os.makedirs(os.path.dirname(TRAIN_FEATURES), exist_ok=True)

    # train
    df_train = load_dataset_parallel(TRAIN_IMG_DIR)
    df_train.to_csv(TRAIN_FEATURES, index=False)
    print(f"[✓] Saved train features to: {TRAIN_FEATURES}")

    # test
    df_test = load_dataset_parallel(TEST_IMG_DIR)
    df_test.to_csv(TEST_FEATURES, index=False)
    print(f"[✓] Saved test features to: {TEST_FEATURES}")

    print("[✅] All features extracted and saved.")

if __name__ == "__main__":
    run_feature_extraction()
