import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# bring in your config paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import HYBRID_PRED_CSV

# load the merged predictions
df = pd.read_csv(HYBRID_PRED_CSV)
y_true = df["Actual_Best"]

# Feature-based ROC
fpr_feat, tpr_feat, _ = roc_curve(y_true, df["Best_Prob_Real"])
auc_feat = auc(fpr_feat, tpr_feat)

# EfficientNet ROC
fpr_eff, tpr_eff, _ = roc_curve(y_true, df["EffNet_Prob_Real"])
auc_eff = auc(fpr_eff, tpr_eff)

# Hybrid ROC: reconstruct which probability was used
hybrid_probs = []
for _, row in df.iterrows():
    if row["Best_Pred"] == row["EffNet_Pred"]:
        # they agreed—average (or pick one)
        hybrid_probs.append((row["Best_Prob_Real"] + row["EffNet_Prob_Real"]) / 2)
    else:
        # pick the winning model's Real‐prob
        best_conf = row["Best_Prob_Real"] if row["Best_Pred"] == 1 else row["Best_Prob_Fake"]
        eff_conf  = row["EffNet_Prob_Real"] if row["EffNet_Pred"] == 1 else row["EffNet_Prob_Fake"]
        hybrid_probs.append(row["Best_Prob_Real"] if best_conf > eff_conf else row["EffNet_Prob_Real"])
hyybrid_probs = np.array(hybrid_probs)

fpr_hyb, tpr_hyb, _ = roc_curve(y_true, hybrid_probs)
auc_hyb = auc(fpr_hyb, tpr_hyb)

# Plot comparison
plt.figure(figsize=(8,6))
plt.plot(fpr_feat, tpr_feat, label=f"Feature (AUC={auc_feat:.2f})")
plt.plot(fpr_eff,  tpr_eff,  label=f"EffNet  (AUC={auc_eff:.2f})")
plt.plot(fpr_hyb,  tpr_hyb,  label=f"Hybrid  (AUC={auc_hyb:.2f})")
plt.plot([0,1],[0,1],"k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# save it next to your hybrid CSV
save_dir  = os.path.dirname(HYBRID_PRED_CSV)
save_path = os.path.join(save_dir, "roc_comparison_full.png")
plt.savefig(save_path)
plt.close()

print(f"✅ ROC comparison plot saved to {save_path}")
