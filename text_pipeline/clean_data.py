import pandas as pd
import re
import os
from sklearn.utils import resample

# === File paths
base_dir = "/home/bna36/misinfo_detection/data"
data_dir = os.path.join(base_dir, "liar_dataset")

train_path = os.path.join(data_dir, "train.tsv")
test_path = os.path.join(data_dir, "test.tsv")
valid_path = os.path.join(data_dir, "valid.tsv")

# === Column names
columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", 
           "barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "context"]

# === Load data
train_df = pd.read_csv(train_path, sep="\t", names=columns)
test_df = pd.read_csv(test_path, sep="\t", names=columns)
valid_df = pd.read_csv(valid_path, sep="\t", names=columns)

# === Truth Score Mapping
truth_mapping = {
    "true": 1.0,
    "mostly-true": 0.8,
    "half-true": 0.6,
    "barely-true": 0.4,
    "false": 0.2,
    "pants-fire": 0.0
}

# === Score Computation
def compute_truth_score(label):
    return truth_mapping.get(label, 0.5)

# === Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https\S+", "[URL]", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Preprocess Splits
for df in [train_df, test_df, valid_df]:
    df["truth_score"] = df["label"].apply(compute_truth_score)
    df["statement"] = df["statement"].astype(str).apply(clean_text)

# === Keep only relevant columns
selected_columns = ["statement", "label", "truth_score", "subject", "speaker", "party"]
train_df = train_df[selected_columns].dropna()
test_df = test_df[selected_columns].dropna()
valid_df = valid_df[selected_columns].dropna()

# === BALANCE TRAINING DATA ONLY
def balance_dataframe(df, label_col="label"):
    max_count = df[label_col].value_counts().max()
    resampled_dfs = []
    for label in df[label_col].unique():
        df_label = df[df[label_col] == label]
        df_upsampled = resample(df_label, replace=True, n_samples=max_count, random_state=42)
        resampled_dfs.append(df_upsampled)
    return pd.concat(resampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = balance_dataframe(train_df)

# === Save cleaned (and balanced) files
train_df.to_csv(os.path.join(data_dir, "train_cleaned.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test_cleaned.csv"), index=False)
valid_df.to_csv(os.path.join(data_dir, "valid_cleaned.csv"), index=False)

print("✅ Cleaned and balanced training data. Saved to:", data_dir)
