import os

# === Base Directory ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /home/bna36/misinfo_detection

# === Data Directories ===
DATA_DIR = os.path.join(BASE_DIR, "data", "cifake_dataset")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

# === Output Directories ===
MODEL_DIR = os.path.join(BASE_DIR, "models")
EFFNET_OUTPUT = os.path.join(BASE_DIR, "image_pipeline", "efficientnet", "outputs")
FEATURE_OUTPUT = os.path.join(BASE_DIR, "image_pipeline", "feature_based", "outputs")
HYBRID_OUTPUT = os.path.join(BASE_DIR, "image_pipeline", "hybrid", "outputs")

# === Ensure All Output Directories Exist ===
for directory in [FEATURE_DIR, MODEL_DIR, EFFNET_OUTPUT, FEATURE_OUTPUT, HYBRID_OUTPUT]:
    os.makedirs(directory, exist_ok=True)

# === Feature CSVs ===
TRAIN_FEATURES = os.path.join(FEATURE_DIR, "train_features.csv")
TEST_FEATURES = os.path.join(FEATURE_DIR, "test_features.csv")

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

# === Hybrid Outputs ===
HYBRID_PRED_CSV = os.path.join(HYBRID_OUTPUT, "hybrid_predictions.csv")
HYBRID_LOG = os.path.join(HYBRID_OUTPUT, "voting_summary.txt")
HYBRID_CM_PNG = os.path.join(HYBRID_OUTPUT, "hybrid_confusion_matrix.png")
HYBRID_ROC_PNG = os.path.join(HYBRID_OUTPUT, "hybrid_roc_curve.png")
