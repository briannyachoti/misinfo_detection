import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "liar_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input data
TRAIN_TEXT = os.path.join(DATA_DIR, "train_cleaned.csv")
VALID_TEXT = os.path.join(DATA_DIR, "valid_cleaned.csv")
TEST_TEXT  = os.path.join(DATA_DIR, "test_cleaned.csv")


TRAIN_EMB = os.path.join(DATA_DIR, "train_roberta_embeddings.csv")
VALID_EMB = os.path.join(DATA_DIR, "valid_roberta_embeddings.csv")
TEST_EMB  = os.path.join(DATA_DIR, "test_roberta_embeddings.csv")

# Outputs
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "rf_text_model.pkl")
PRED_PATH = os.path.join(OUTPUT_DIR, "rf_predictions.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "mae_by_bin.png")
TEST_PRED_COMPARISON = os.path.join(OUTPUT_DIR, "rf_test_predictions_comparison.csv")


