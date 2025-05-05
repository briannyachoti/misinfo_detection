import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# add project root to path so config.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    FEATURE_PRED_CSV,
    EFFNET_PRED_CSV,
    HYBRID_PRED_CSV,
    HYBRID_LOG,
    HYBRID_CM_PNG
)

def main():
    # === Load & rename predictions ===
    best_df = pd.read_csv(FEATURE_PRED_CSV)
    effnet_df = pd.read_csv(EFFNET_PRED_CSV)

    # unify types for merge
    best_df["Image_Path"]   = best_df["Image_Path"].astype(str)
    effnet_df["Image_Path"] = effnet_df["Image_Path"].astype(str)

    # rename best-model cols
    best_df = best_df.rename(columns={
        "Predicted": "Best_Pred",
        "Prob_Fake": "Best_Prob_Fake",
        "Prob_Real": "Best_Prob_Real",
        "Actual":    "Actual_Best"
    })

    # rename effnet-model cols
    effnet_df = effnet_df.rename(columns={
        "Predicted": "EffNet_Pred",
        "Prob_Fake": "EffNet_Prob_Fake",
        "Prob_Real": "EffNet_Prob_Real",
        "Actual":    "Actual_EffNet"
    })

    # --- DEBUG: raw model accuracies & counts ---
    best_acc   = accuracy_score(best_df["Actual_Best"], best_df["Best_Pred"])
    effnet_acc = accuracy_score(effnet_df["Actual_EffNet"], effnet_df["EffNet_Pred"])
    print(f"[DEBUG] Best-model accuracy:   {best_acc:.4f}")
    print(f"[DEBUG] EffNet-model accuracy: {effnet_acc:.4f}")

    print(f"[DEBUG] Rows in best_df:   {len(best_df)}")
    print(f"[DEBUG] Rows in effnet_df: {len(effnet_df)}")

    # outer merge to catch any missing rows
    merged_full = pd.merge(
        best_df, effnet_df,
        on="Image_Path",
        how="outer",
        indicator=True
    )
    print("[DEBUG] Merge indicator counts:")
    print(merged_full["_merge"].value_counts(), "\n")

    # filter to only “both” for hybrid logic
    merged = merged_full[merged_full["_merge"] == "both"].drop(columns=["_merge"])
    print(f"[DEBUG] Rows in merged (both): {len(merged)}")

    # sanity check: are actuals consistent?
    mismatches = merged[merged["Actual_Best"] != merged["Actual_EffNet"]]
    print(f"[DEBUG] rows with Actual_Best != Actual_EffNet: {len(mismatches)}\n")

    # === Hybrid Voting Logic ===
    hybrid_preds = []
    for _, row in merged.iterrows():
        if row["Best_Pred"] == row["EffNet_Pred"]:
            hybrid_preds.append(row["Best_Pred"])
        else:
            # pick higher-confidence prediction
            best_conf   = (
                row["Best_Prob_Real"] if row["Best_Pred"] == 1
                else row["Best_Prob_Fake"]
            )
            effnet_conf = (
                row["EffNet_Prob_Real"] if row["EffNet_Pred"] == 1
                else row["EffNet_Prob_Fake"]
            )
            hybrid_preds.append(
                row["Best_Pred"] if best_conf > effnet_conf else row["EffNet_Pred"]
            )

    merged["Hybrid_Pred"] = hybrid_preds

    # --- DEBUG: hybrid performance & agreement stats ---
    hybrid_acc = accuracy_score(merged["Actual_Best"], merged["Hybrid_Pred"])
    print(f"[DEBUG] Hybrid accuracy: {hybrid_acc:.4f}")

    agree = merged["Best_Pred"] == merged["EffNet_Pred"]
    print(f"[DEBUG] Agreement count: {agree.sum()}, Disagreement count: {(~agree).sum()}\n")

    print("[DEBUG] Sample disagreements:")
    print( merged[~agree].head()[[
        "Image_Path", "Actual_Best",
        "Best_Pred", "Best_Prob_Real", "Best_Prob_Fake",
        "EffNet_Pred", "EffNet_Prob_Real", "EffNet_Prob_Fake",
        "Hybrid_Pred"
    ]], "\n")

    # === Print final hybrid accuracy & classification report ===
    report_text = classification_report(
        merged["Actual_Best"], merged["Hybrid_Pred"],
        labels=[0, 1], target_names=["Fake", "Real"]
    )
    print(f"🔀 Hybrid Voting Accuracy: {hybrid_acc:.4f}")
    print("\nClassification Report:\n", report_text)

    # === Save merged & predictions ===
    merged.to_csv(HYBRID_PRED_CSV, index=False)
    print(f"✅ Saved hybrid predictions to {HYBRID_PRED_CSV}")

    # === Save confusion matrix plot ===
    cm = confusion_matrix(merged["Actual_Best"], merged["Hybrid_Pred"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hybrid Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(HYBRID_CM_PNG)
    plt.close()
    print(f"✅ Saved hybrid confusion matrix to {HYBRID_CM_PNG}")

    # === Save summary log ===
    with open(HYBRID_LOG, "w") as f:
        f.write("Hybrid Voting Summary\n")
        f.write(f"Accuracy: {hybrid_acc:.4f}\n\n")
        f.write(report_text)
    print(f"📝 Voting summary saved to {HYBRID_LOG}")

if __name__ == "__main__":
    main()
