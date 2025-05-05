import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# === Paths ===
DATA_DIR = "/home/bna36/misinfo_detection/data/liar_dataset"
OUTPUT_DIR = "/home/bna36/misinfo_detection/text_pipeline/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load your true and predicted scores
true_df = pd.read_csv(os.path.join(DATA_DIR, "test_cleaned.csv"))
pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, "rf_test_predictions_comparison.csv"))

# === Extract true and predicted scores
y_true_scores = true_df['truth_score']
y_pred_scores = pred_df['Predicted Truth Score']

# === Compute ROC Curve and AUC
fpr, tpr, thresholds = roc_curve((y_true_scores >= 0.5).astype(int), y_pred_scores)
roc_auc = auc(fpr, tpr)

# === Find best threshold (Youden's J statistic)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
print(f"🔍 Best Threshold (Youden's J): {optimal_threshold:.4f}")

# === Binarize using optimal threshold
y_true_bin = (y_true_scores >= 0.5).astype(int)
y_pred_bin = (y_pred_scores >= optimal_threshold).astype(int)

# === Force binary confusion matrix
cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# === Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Optimized Confusion Matrix (Threshold = {optimal_threshold:.2f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tuned_confusion_matrix.png"))
plt.close()
print("✅ Saved confusion matrix with optimized threshold")

# === Save ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.axvline(x=fpr[optimal_idx], color='r', linestyle='--', label=f"Best Threshold = {optimal_threshold:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Optimized Threshold)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tuned_roc_curve.png"))
plt.close()
print("✅ Saved ROC curve with optimized threshold")

# === Print Evaluation Summary
print(f"\n📊 Optimized Threshold Evaluation Summary")
print(f"Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.4f}")
print(f"TPR (Recall): {tp / (tp + fn):.4f}")
print(f"FPR: {fp / (fp + tn):.4f}")
print(f"FNR: {fn / (tp + fn):.4f}")
print("\nClassification Report:")
print(classification_report(y_true_bin, y_pred_bin, target_names=["Fake", "Real"]))
