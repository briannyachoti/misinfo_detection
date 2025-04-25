import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from config import DATA_DIR, EMBED_DIR, TRAIN_TEXT, VALID_TEXT, TEST_TEXT

# Get Roberta tokenizer and model
def get_roberta_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Create text text input and generate embedding
def get_embedding(row, tokenizer, model, device):
    text = f"Statement: {row['statement']} Speaker: {row['speaker']} Subject: {row['subject']}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Generate and save embeddings for one split
def generate_split_embeddings(split_name, input_csv_path, output_csv_path, tokenizer, model, device):
    if not os.path.exists(input_csv_path):
        print(f"[SKIP] {input_csv_path} not found.")
        return

    df = pd.read_csv(input_csv_path)
    df = df.dropna(subset=["statement", "speaker", "subject"])

    print(f"[INFO] Generating embeddings for '{split_name}' ({len(df)} samples)...")

    embeddings = np.array([
        get_embedding(row, tokenizer, model, device)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {split_name}")
    ])

    pd.DataFrame(embeddings).to_csv(output_csv_path, index=False)
    print(f"[DONE] Saved '{split_name}' embeddings to: {output_csv_path}")

# Main function
def generate_roberta_embeddings():
    tokenizer, model, device = get_roberta_model()

    splits = {
        "train": (TRAIN_TEXT, os.path.join(EMBED_DIR, "train_roberta_embeddings.csv")),
        "valid": (VALID_TEXT, os.path.join(EMBED_DIR, "valid_roberta_embeddings.csv")),
        "test": (TEST_TEXT, os.path.join(EMBED_DIR, "test_roberta_embeddings.csv"))
    }

    for split_name, (input_path, output_path) in splits.items():
        generate_split_embeddings(split_name, input_path, output_path, tokenizer, model, device)

# Entry point
if __name__ == "__main__":
    print(f"[START] RoBERTa Embedding Generation using config paths...")
    generate_roberta_embeddings()
    print("[COMPLETE] Embedding generation finished.")
