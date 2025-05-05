import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, classification_report
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_PRED_CSV, FEATURE_CM_PNG

# === Load Predictions ===
df = pd.read_csv(FEATURE_PRED_CSV)

# Extract
y_true = df["Actual"]
y_pred = df["Predicted"]
y_proba_real = df["Prob_Real"]  # Needed for ROC

# === Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate metrics
tpr = tp / (tp + fn) if (tp + fn) else 0  # Recall or Sensitivity
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0

print(f"🧾 Confusion Matrix:\n{cm}")
print(f"✅ True Positive Rate (TPR / Recall): {tpr:.4f}")
print(f"✅ False Positive Rate (FPR): {fpr:.4f}")
print(f"✅ False Negative Rate (FNR): {fnr:.4f}")

# === Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"], zero_division=0))

# === Save Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Feature-Based Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig(FEATURE_CM_PNG)
plt.close()
print(f"📊 Confusion matrix plot saved to: {FEATURE_CM_PNG}")

# === ROC Curve
fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_proba_real)
roc_auc = auc(fpr_curve, tpr_curve)

plt.figure(figsize=(6,5))
plt.plot(fpr_curve, tpr_curve, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Feature-Based Classifier ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = FEATURE_PRED_CSV.replace("prob_predictions.csv", "roc_curve.png")
plt.savefig(roc_path)
plt.close()
print(f"📈 ROC curve saved to: {roc_path}")
