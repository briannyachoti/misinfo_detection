import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
DATA_DIR = "/home/bna36/misinfo_detection/data/liar_dataset"
OUTPUT_DIR = "/home/bna36/misinfo_detection/text_pipeline/exploration_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load and Combine Splits ===
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_cleaned.csv"))
valid_df = pd.read_csv(os.path.join(DATA_DIR, "valid_cleaned.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_cleaned.csv"))

combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

print("\n====== COMBINED DATASET ======")
print("Shape:", combined_df.shape)
print("Columns:", combined_df.columns.tolist())
print("Missing values:\n", combined_df.isnull().sum())

# === Helper to show count + percent
def make_autopct(values):
    def inner(pct):
        val = int(round(pct/100 * np.sum(values)))
        return f"{pct:.1f}%\n({val})"
    return inner

# === Add statement length
combined_df["statement_length"] = combined_df["statement"].str.split().str.len()

# === Truth Score Distribution
plt.figure(figsize=(6, 4))
sns.histplot(combined_df["truth_score"], bins=10, kde=True)
plt.title("Truth Score Distribution (Combined)")
plt.xlabel("Truth Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_truth_score_distribution.png"))
plt.close()

# === Statement Length Distribution
plt.figure(figsize=(6, 4))
sns.histplot(combined_df["statement_length"], bins=30, kde=True, color="teal")
plt.title("Statement Length Distribution (Combined)")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_statement_length_distribution.png"))
plt.close()

# === Top 20 Speakers
if "speaker" in combined_df.columns:
    top_speakers = combined_df["speaker"].value_counts().head(20)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_speakers.values, y=top_speakers.index)
    plt.title("Top 20 Speakers (Combined)")
    plt.xlabel("Number of Statements")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_top20_speakers.png"))
    plt.close()

# === Top 20 Subjects
if "subject" in combined_df.columns:
    top_subjects = combined_df["subject"].value_counts().head(20)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_subjects.values, y=top_subjects.index)
    plt.title("Top 20 Subjects (Combined)")
    plt.xlabel("Number of Statements")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_top20_subjects.png"))
    plt.close()

# === Binning logic
bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01]
labels = ['Pants on Fire', 'False', 'Barely True', 'Half True', 'Mostly True', 'True']
combined_df['truth_bin'] = pd.cut(combined_df['truth_score'], bins=bins, labels=labels, right=False)

# === Combined Pie Chart
bin_counts = combined_df['truth_bin'].value_counts().reindex(labels).fillna(0).astype(int)
plt.figure(figsize=(8, 8))
plt.pie(bin_counts, labels=bin_counts.index, autopct=make_autopct(bin_counts.values), startangle=140)
plt.title('Truth Score Distribution (Combined)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_truth_score_pie_chart.png"))
plt.close()

# === Per-split Summary and Pie Charts
split_data = {
    "train": train_df,
    "valid": valid_df,
    "test": test_df
}
summary_rows = []

for split_name, df in split_data.items():
    df['truth_bin'] = pd.cut(df['truth_score'], bins=bins, labels=labels, right=False)
    counts = df['truth_bin'].value_counts().reindex(labels).fillna(0).astype(int)
    percentages = (counts / counts.sum() * 100).round(1)

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct=make_autopct(counts.values), startangle=140)
    plt.title(f"{split_name.capitalize()} Truth Score Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{split_name}_truth_score_pie_chart.png"))
    plt.close()

    for label in labels:
        summary_rows.append({
            "Split": split_name,
            "Label": label,
            "Count": counts[label],
            "Percentage": percentages[label]
        })

# === Save Summary Table
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "truth_score_label_distribution_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"📄 Label summary saved to: {summary_csv}")
