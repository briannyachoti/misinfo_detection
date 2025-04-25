import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TEST_FEATURES, TRAIN_FEATURES,
    FEATURE_MODEL_PATH, FEATURE_PRED_CSV, FEATURE_CM_PNG
)

# Get path to model outputs saved from train_model.py
OUTPUTS_FILE = FEATURE_PRED_CSV.replace("prob_predictions.csv", "best_model_outputs.npz")
CLASSIFICATION_METRICS_CSV = FEATURE_PRED_CSV.replace("prob_predictions.csv", "classification_metrics.csv")
MISCLASSIFIED_CSV = FEATURE_PRED_CSV.replace("prob_predictions.csv", "misclassified_samples.csv")
PAIRPLOT_PNG = FEATURE_PRED_CSV.replace("prob_predictions.csv", "feature_pairplot.png")
TOP_IMPORTANCE_PNG = FEATURE_PRED_CSV.replace("prob_predictions.csv", "top_20_feature_importance.png")

def run_evaluation():
    print("[INFO] Starting model evaluation...")

    # Load saved prediction arrays
    outputs = np.load(OUTPUTS_FILE, allow_pickle=True)
    y_pred = outputs["predictions"]
    y_proba = outputs["probabilities"]
    y_test = outputs["labels"]
    model_name = str(outputs["model_name"])

    df_train = pd.read_csv(TRAIN_FEATURES)
    x_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]

    df_test = pd.read_csv(TEST_FEATURES)
    test_paths = df_test.get("image_path", [f"image_{i}" for i in range(len(y_test))])

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_METRICS_CSV)
    print(f"📊 Classification report saved to: {CLASSIFICATION_METRICS_CSV}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(FEATURE_CM_PNG)
    plt.close()
    print(f"🧾 Confusion matrix saved to: {FEATURE_CM_PNG}")

    # Save Probabilities + Misclassified
    df_probs = pd.DataFrame({
        "Image_Path": test_paths,
        "Actual": y_test,
        "Predicted": y_pred,
        "Prob_Fake": y_proba[:, 0],
        "Prob_Real": y_proba[:, 1],
        "Correct": y_test == y_pred
    })

    df_probs.to_csv(FEATURE_PRED_CSV, index=False)
    df_probs[~df_probs["Correct"]].to_csv(MISCLASSIFIED_CSV, index=False)
    print(f"📁 Prediction details saved to: {FEATURE_PRED_CSV}")
    print(f"❌ Misclassified samples saved to: {MISCLASSIFIED_CSV}")

    # Pairplot of training features
    sns.pairplot(df_train.sample(min(200, len(df_train)), random_state=42), hue="label", diag_kind="kde", palette={0: "red", 1: "blue"})
    plt.tight_layout()
    plt.savefig(PAIRPLOT_PNG)
    plt.close()
    print(f"📷 Feature pairplot saved to: {PAIRPLOT_PNG}")

    # Feature Importance Plot (tree-based models only)
    tree_models = ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]
    if model_name in tree_models:
        model = joblib.load(FEATURE_MODEL_PATH)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = x_train.columns
            top_n = 20
            sorted_idx = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[sorted_idx], align="center")
            plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx], fontsize=10)
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances ({model_name})")
            plt.tight_layout()
            plt.savefig(TOP_IMPORTANCE_PNG)
            plt.close()
            print(f"📊 Feature importance saved to: {TOP_IMPORTANCE_PNG}")
        else:
            print(f"⚠️ {model_name} does not support `feature_importances_`.")
    else:
        print(f"⏭ Skipping feature importance — {model_name} is not tree-based.")

    print("[✅] Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
