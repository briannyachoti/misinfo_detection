from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TRAIN_FEATURES, TEST_FEATURES, FEATURE_MODEL_PATH, FEATURE_PRED_CSV

def run_xgboost_training():
    print("[INFO] Loading feature data...")
    df_train = pd.read_csv(TRAIN_FEATURES)
    df_test = pd.read_csv(TEST_FEATURES)

    if "Image_Path" in df_test.columns:
        image_paths = df_test["Image_Path"].values
    else:
        raise ValueError("Image_Path column missing in test set!")

    x_train = df_train.drop(columns=["label", "Image_Path"], errors="ignore")
    y_train = df_train["label"]
    x_test = df_test.drop(columns=["label", "Image_Path"], errors="ignore")
    y_test = df_test["label"]

    print("[INFO] Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ XGBoost Test Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], zero_division=0)
    print("\nClassification Report:\n", report)

    result_df = pd.DataFrame({
        "Image_Path": image_paths,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Prob_Fake": y_proba[:, 0],
        "Prob_Real": y_proba[:, 1]
        "Correct": y_test == y_pred
    })
    result_df.to_csv(FEATURE_PRED_CSV, index=False)
    print(f"📄 Saved predictions to: {FEATURE_PRED_CSV}")

    joblib.dump(model, FEATURE_MODEL_PATH)
    print(f"💾 Saved model to: {FEATURE_MODEL_PATH}")

    np.savez(
        os.path.join(os.path.dirname(FEATURE_PRED_CSV), "feature_best_model_outputs.npz"),
        predictions=y_pred,
        probabilities=y_proba,
        labels=y_test.to_numpy(),
        model_name="XGBoost"
    )
    print(f"🗂️ Saved evaluation outputs.")

if __name__ == "__main__":
    run_xgboost_training()
