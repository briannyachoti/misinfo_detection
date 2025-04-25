import os

# === Base Directories ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../image_pipeline
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cifake_dataset"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# === Image Paths ===
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

# === Feature CSV Paths (Defined Before Creating Directory)
FEATURE_DIR = os.path.join(DATA_DIR, "features")
TRAIN_FEATURES = os.path.join(FEATURE_DIR, "train_features.csv")
TEST_FEATURES = os.path.join(FEATURE_DIR, "test_features.csv")

# === Output Subfolders ===
EFFNET_OUTPUT = os.path.join(BASE_DIR, "efficientnet", "outputs")
FEATURE_OUTPUT = os.path.join(BASE_DIR, "feature_based", "outputs")
HYBRID_OUTPUT = os.path.join(BASE_DIR, "hybrid", "outputs")

# === Ensure output directories exist ===
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(EFFNET_OUTPUT, exist_ok=True)
os.makedirs(FEATURE_OUTPUT, exist_ok=True)
os.makedirs(HYBRID_OUTPUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === EfficientNet Outputs ===
EFFNET_MODEL_PATH = os.path.join(MODEL_DIR, "effnet_model.pth")
EFFNET_PRED_CSV = os.path.join(EFFNET_OUTPUT, "effnet_prob_predictions.csv")
EFFNET_CM_PNG = os.path.join(EFFNET_OUTPUT, "confusion_matrix.png")
EFFNET_ROC_PNG = os.path.join(EFFNET_OUTPUT, "roc_curve.png")
EFFNET_TSNE_PNG = os.path.join(EFFNET_OUTPUT, "tsne.png")
EFFNET_UMAP_PNG = os.path.join(EFFNET_OUTPUT, "umap.png")

# === Feature-Based Outputs ===
FEATURE_MODEL_PATH = os.path.join(MODEL_DIR, "feature_model.pkl")
FEATURE_PRED_CSV = os.path.join(FEATURE_OUTPUT, "feature_prob_predictions.csv")
FEATURE_CM_PNG = os.path.join(FEATURE_OUTPUT, "confusion_matrix.png")
FEATURE_NPZ_PATH = os.path.join(FEATURE_OUTPUT, "feature_best_model_outputs.npz")


# === Hybrid Voting Outputs ===
HYBRID_PRED_CSV = os.path.join(HYBRID_OUTPUT, "hybrid_predictions.csv")
HYBRID_LOG = os.path.join(HYBRID_OUTPUT, "voting_summary.txt")
