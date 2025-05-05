import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap
import joblib
import os
from config import TEST_EMB, TEST_TEXT, MODEL_PATH, OUTPUT_DIR

# === Load test data and model ===
test_x = pd.read_csv(TEST_EMB)
test_y = pd.read_csv(TEST_TEXT)["truth_score"]

model = joblib.load(MODEL_PATH)

# === Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🔵 Starting Explainability Analysis...")

# === SHAP Analysis ===
print("\n[SHAP] Computing SHAP values...")
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(test_x)

# === SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, test_x, feature_names=[f"f{i}" for i in range(test_x.shape[1])], show=False)
plt.title("SHAP Summary Plot (Global Feature Importance)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"))
plt.close()
print("[SHAP] ✅ Saved SHAP Summary Plot!")

# === SHAP Dependence Plot (Optional - Feature 0)
plt.figure()
shap.dependence_plot(0, shap_values, test_x, feature_names=[f"f{i}" for i in range(test_x.shape[1])], show=False)
plt.title("SHAP Dependence Plot (Feature 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_feature0.png"))
plt.close()
print("[SHAP] ✅ Saved SHAP Dependence Plot (Feature 0)")

# === LIME Analysis ===
print("\n[LIME] Explaining a Sample Prediction...")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=test_x.values,
    feature_names=[f"f{i}" for i in range(test_x.shape[1])],
    mode="regression",
    discretize_continuous=True
)

sample_idx = 10  # <-- You can change the sample index to explain a different sample
sample = test_x.iloc[sample_idx]

exp = explainer_lime.explain_instance(
    data_row=sample.values,
    predict_fn=model.predict
)

lime_output_path = os.path.join(OUTPUT_DIR, f"lime_explanation_sample{sample_idx}.html")
exp.save_to_file(lime_output_path)
print(f"[LIME] ✅ Saved LIME explanation for sample {sample_idx} to {lime_output_path}!")

print("\n🎯 Explainability Analysis Completed! All outputs saved to:", OUTPUT_DIR)
