import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from config import DATA_DIR, OUTPUT_DIR

# === File Paths ===
comparison_path = os.path.join(OUTPUT_DIR, "rf_test_predictions_comparison.csv")
test_cleaned_path = os.path.join(DATA_DIR, "test_cleaned.csv")
embedding_path = os.path.join(DATA_DIR, "test_roberta_embeddings.csv")

# === Load Data ===
comparison_df = pd.read_csv(comparison_path)
test_df = pd.read_csv(test_cleaned_path)
embeddings_df = pd.read_csv(embedding_path)
merged = pd.concat([test_df.reset_index(drop=True), comparison_df], axis=1)

# === Residual Plot ===
plt.figure(figsize=(6, 5))
residuals = merged["Actual Truth Score"] - merged["Predicted Truth Score"]
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_histogram.png"))
plt.close()

# === Error vs. Statement Length ===
merged["Statement Length"] = merged["statement"].str.split().str.len()
plt.figure(figsize=(6, 5))
sns.scatterplot(x="Statement Length", y="Absolute Error", data=merged, alpha=0.6)
plt.title("Error vs. Statement Length")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_statement_length.png"))
plt.close()

# === MAE by Speaker ===
if "speaker" in merged.columns:
    speaker_error = merged.groupby("speaker")["Absolute Error"].mean().sort_values(ascending=False).head(20)
    speaker_error.to_csv(os.path.join(OUTPUT_DIR, "mae_by_speaker.csv"))
    speaker_error.plot(kind="bar", figsize=(10, 4), title="Top 20 Speakers by MAE")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mae_by_speaker_top20.png"))
    plt.close()

# === MAE by Subject ===
if "subject" in merged.columns:
    subject_error = merged.groupby("subject")["Absolute Error"].mean().sort_values(ascending=False).head(20)
    subject_error.to_csv(os.path.join(OUTPUT_DIR, "mae_by_subject.csv"))
    subject_error.plot(kind="bar", figsize=(10, 4), title="Top 20 Subjects by MAE")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mae_by_subject_top20.png"))
    plt.close()

# === 3D UMAP Visualization ===
reducer_3d = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer_3d.fit_transform(embeddings_df)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                c=merged["Actual Truth Score"], cmap="coolwarm", alpha=0.6)
plt.colorbar(sc, ax=ax, shrink=0.5)
ax.set_title("3D UMAP of RoBERTa Embeddings")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_3d.png"))
plt.close()

# === t-SNE 2D Plot ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_proj = tsne.fit_transform(embeddings_df)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=tsne_proj[:, 0], y=tsne_proj[:, 1],
                hue=merged["Actual Truth Score"], palette="coolwarm", alpha=0.7)
plt.title("t-SNE of RoBERTa Embeddings")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_2d.png"))
plt.close()

# === MAE by Truth Score Bin ===
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
merged["Truth Bin"] = pd.cut(merged["Actual Truth Score"], bins=bins, labels=labels)

mae_by_bin = merged.groupby("Truth Bin")["Absolute Error"].mean()
mae_by_bin.to_csv(os.path.join(OUTPUT_DIR, "mae_by_truth_score_bin.csv"))

plt.figure(figsize=(7, 5))
mae_by_bin.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("MAE by Truth Score Range")
plt.xlabel("Truth Score Bin")
plt.ylabel("Mean Absolute Error")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mae_by_truth_score_bin.png"))
plt.close()

print("✅ Visualizations and speaker/subject error analysis completed.")
