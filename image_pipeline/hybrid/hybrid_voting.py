import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    FEATURE_PRED_CSV,
    EFFNET_PRED_CSV,
    HYBRID_PRED_CSV,
    HYBRID_LOG
)

HYBRID_CM_PNG = HYBRID_PRED_CSV.replace("hybrid_predictions.csv", "hybrid_confusion_matrix.png")

# === Load predictions ===
best_df = pd.read_csv(FEATURE_PRED_CSV)
effnet_df = pd.read_csv(EFFNET_PRED_CSV)

# Ensure 'Image_Path' is the same type
best_df["Image_Path"] = best_df["Image_Path"].astype(str)
effnet_df["Image_Path"] = effnet_df["Image_Path"].astype(str)

# Rename for clarity
best_df = best_df.rename(columns={
    "Predicted": "Best_Pred",
    "Prob_Fake": "Best_Prob_Fake",
    "Prob_Real": "Best_Prob_Real",
    "Actual": "Actual_Best"
})

# Merge on Image Path
merged = pd.merge(best_df, effnet_df, on="Image_Path", suffixes=("_Best", "_EffNet"))

# === Hybrid Voting Logic ===
hybrid_preds = []
for _, row in merged.iterrows():
    if row["Best_Pred"] == row["EffNet_Pred"]:
        hybrid_preds.append(row["Best_Pred"])
    else:
        best_conf = row["Best_Prob_Real"] if row["Best_Pred"] == 1 else row["Best_Prob_Fake"]
        effnet_conf = row["EffNet_Prob_Real"] if row["EffNet_Pred"] == 1 else row["EffNet_Prob_Fake"]
        hybrid_preds.append(row["Best_Pred"] if best_conf > effnet_conf else row["EffNet_Pred"])

merged["Hybrid_Pred"] = hybrid_preds

# === Evaluate Hybrid Voting ===
y_true = merged["Actual_Best"]
acc = accuracy_score(y_true, hybrid_preds)
cm = confusion_matrix(y_true, hybrid_preds)
report = classification_report(y_true, hybrid_preds, labels=[0, 1], target_names=["Fake", "Real"], zero_division=0)

print(f"\n🔀 Hybrid Voting Accuracy: {acc:.4f}")
print("\nClassification Report:\n", report)

# Save results
merged.to_csv(HYBRID_PRED_CSV, index=False)
print(f"✅ Saved hybrid predictions to {HYBRID_PRED_CSV}")

# Save confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid Model Confusion Matrix")
plt.tight_layout()
plt.savefig(HYBRID_CM_PNG)
plt.close()
print(f"✅ Saved hybrid confusion matrix to {HYBRID_CM_PNG}")

# Save summary log
with open(HYBRID_LOG, "w") as f:
    f.write("Hybrid Voting Summary\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
print(f"📝 Voting summary saved to {HYBRID_LOG}")
