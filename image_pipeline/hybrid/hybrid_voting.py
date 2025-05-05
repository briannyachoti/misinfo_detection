import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_PRED_CSV, EFFNET_PRED_CSV, HYBRID_PRED_CSV, HYBRID_LOG

# Output paths
HYBRID_CM_PNG = HYBRID_PRED_CSV.replace("hybrid_predictions.csv", "hybrid_confusion_matrix.png")
HYBRID_ROC_PNG = HYBRID_PRED_CSV.replace("hybrid_predictions.csv", "hybrid_roc_curve.png")

# === Load predictions
best_df = pd.read_csv(FEATURE_PRED_CSV)
effnet_df = pd.read_csv(EFFNET_PRED_CSV)

best_df["Image_Path"] = best_df["Image_Path"].astype(str)
effnet_df["Image_Path"] = effnet_df["Image_Path"].astype(str)

# Rename for clarity
best_df = best_df.rename(columns={
    "Predicted": "Best_Pred",
    "Prob_Fake": "Best_Prob_Fake",
    "Prob_Real": "Best_Prob_Real",
    "Actual": "Actual_Best"
})

# Merge
merged = pd.merge(best_df, effnet_df, on="Image_Path", suffixes=("_Best", "_EffNet"))

# === Hybrid Voting
hybrid_preds = []
for _, row in merged.iterrows():
    if row["Best_Pred"] == row["EffNet_Pred"]:
        hybrid_preds.append(row["Best_Pred"])
    else:
        best_conf = row["Best_Prob_Real"] if row["Best_Pred"] == 1 else row["Best_Prob_Fake"]
        effnet_conf = row["EffNet_Prob_Real"] if row["EffNet_Pred"] == 1 else row["EffNet_Prob_Fake"]
        hybrid_preds.append(row["Best_Pred"] if best_conf > effnet_conf else row["EffNet_Pred"])

merged["Hybrid_Pred"] = hybrid_preds

# === Evaluate
y_true = merged["Actual_Best"]
y_pred = merged["Hybrid_Pred"]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
tpr = tp / (tp + fn) if (tp + fn) else 0  # Recall or Sensitivity
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (tp + fn) if (tp + fn) else 0

print(f"✅ Hybrid Voting Accuracy: {accuracy:.4f}")
print(f"✅ True Positive Rate (TPR): {tpr:.4f}")
print(f"✅ False Positive Rate (FPR): {fpr:.4f}")
print(f"✅ False Negative Rate (FNR): {fnr:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

# === Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid Model Confusion Matrix")
plt.tight_layout()
plt.savefig(HYBRID_CM_PNG)
plt.close()
print(f"📄 Confusion matrix saved to {HYBRID_CM_PNG}")

# === ROC Curve
hybrid_probs = []
for _, row in merged.iterrows():
    if row["Best_Pred"] == row["EffNet_Pred"]:
        hybrid_probs.append(row["Best_Prob_Real"])
    else:
        best_conf = row["Best_Prob_Real"] if row["Best_Pred"] == 1 else 1 - row["Best_Prob_Real"]
        effnet_conf = row["EffNet_Prob_Real"] if row["EffNet_Pred"] == 1 else 1 - row["EffNet_Prob_Fake"]
        hybrid_probs.append(row["Best_Prob_Real"] if best_conf > effnet_conf else row["EffNet_Prob_Real"])

fpr_curve, tpr_curve, thresholds = roc_curve(y_true, hybrid_probs)
roc_auc = auc(fpr_curve, tpr_curve)

plt.figure(figsize=(6,5))
plt.plot(fpr_curve, tpr_curve, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Hybrid Model")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(HYBRID_ROC_PNG)
plt.close()
print(f"📈 ROC curve saved to {HYBRID_ROC_PNG}")

# === Save summary
with open(HYBRID_LOG, "w") as f:
    f.write("Hybrid Voting Full Evaluation\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"True Positive Rate (TPR): {tpr:.4f}\n")
    f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
    f.write(f"False Negative Rate (FNR): {fnr:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
print(f"📝 Full evaluation summary saved to {HYBRID_LOG}")



from sklearn.metrics import accuracy_score

# Accuracy of individual models
acc_feature = accuracy_score(merged["Actual_Best"], merged["Best_Pred"])
acc_effnet = accuracy_score(merged["Actual_Best"], merged["EffNet_Pred"])

print(f"\n🎯 Feature-Based Accuracy: {acc_feature:.4f}")
print(f"🎯 EfficientNet Accuracy: {acc_effnet:.4f}")
print(f"🎯 Hybrid Voting Accuracy: {accuracy:.4f}")



disagreements = merged[merged["Best_Pred"] != merged["EffNet_Pred"]]
correct_best = disagreements["Best_Pred"] == disagreements["Actual_Best"]
correct_effnet = disagreements["EffNet_Pred"] == disagreements["Actual_Best"]
correct_hybrid = disagreements["Hybrid_Pred"] == disagreements["Actual_Best"]

print(f"\n🔍 Total disagreements: {len(disagreements)}")
print(f"✔️ Best correct: {correct_best.sum()} | EffNet correct: {correct_effnet.sum()} | Hybrid correct: {correct_hybrid.sum()}")