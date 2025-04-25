import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TRAIN_FEATURES, TEST_FEATURES,
    FEATURE_MODEL_PATH, FEATURE_PRED_CSV, FEATURE_CM_PNG
)

def run_model_training():
    print("[INFO] Loading feature data...")
    df_train = pd.read_csv(TRAIN_FEATURES)
    df_test = pd.read_csv(TEST_FEATURES)
    
    image_paths = df_test["Image_Path"] if "Image_Path" in df_test.columns else np.arange(len(df_test))
    
    x_train = df_train.drop(columns=["label", "Image_Path"], errors="ignore")
    y_train = df_train["label"]
    x_test = df_test.drop(columns=["label", "Image_Path"], errors="ignore")
    y_test = df_test["label"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }

    best_model = None
    best_model_name = ""
    best_accuracy = 0.0
    best_y_pred = None
    best_y_proba = None
    metrics = []

    print("[INFO] Training and evaluating models...")

    for name, model in models.items():
        print(f"\n▶️ Training: {name}")
        if name in ["SVM", "MLP"]:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            y_proba = model.predict_proba(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True, zero_division=0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        print(f"{name} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        if acc > best_accuracy:
            best_model = model
            best_model_name = name
            best_accuracy = acc
            best_y_pred = y_pred
            best_y_proba = y_proba

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = FEATURE_PRED_CSV.replace("prob_predictions.csv", "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n📊 Model comparison metrics saved to: {metrics_csv}")

    # Save predictions + best model
    result_df = pd.DataFrame({
        "Image_Path": image_paths,
        "Actual": y_test,
        "Predicted": best_y_pred,
        "Prob_Fake": best_y_proba[:, 0],
        "Prob_Real": best_y_proba[:, 1]
    })
    result_df.to_csv(FEATURE_PRED_CSV, index=False)
    print(f"📄 Best model predictions saved to: {FEATURE_PRED_CSV}")
    
        # Save the best model
    joblib.dump(best_model, FEATURE_MODEL_PATH)
    print(f"💾 Best model ({best_model_name}) saved to: {FEATURE_MODEL_PATH}")

    # Save outputs in .npz format for evaluation script
    outputs_path = os.path.join(os.path.dirname(FEATURE_PRED_CSV), "feature_best_model_outputs.npz")
    np.savez(
        outputs_path,
        predictions=best_y_pred,
        probabilities=best_y_proba,
        labels=y_test.to_numpy(),
        model_name=best_model_name
    )
    print(f"🗂️ Model outputs saved to: {outputs_path}")

    print(f"✅ Accuracy: {best_accuracy:.4f}")




# === Main Execution ===
if __name__ == "__main__":
    run_model_training()
