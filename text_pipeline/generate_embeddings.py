import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def get_roberta_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Function to generate embeddings from statement only
def get_embedding(row, tokenizer, model, device):
    text = row["statement"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


# Process each split
def generate_split_embeddings(split, data_dir, embed_dir, tokenizer, model, device):
    file_path = os.path.join(data_dir, f"{split}_cleaned.csv")
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Skipping.")
        return

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["statement"])  # ensure 'statement' exists

    print(f"[INFO] Generating embeddings for {split} ({len(df)} samples)...")
    embeddings = np.array([
        get_embedding(row, tokenizer, model, device)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {split}")
    ])

    os.makedirs(embed_dir, exist_ok=True)
    output_path = os.path.join(embed_dir, f"{split}_roberta_embeddings.csv")
    pd.DataFrame(embeddings).to_csv(output_path, index=False)
    print(f"[✓] Saved {split} embeddings to: {output_path}")

def generate_embeddings(data_dir, embed_dir, splits=["train", "test", "valid"]):
    tokenizer, model, device = get_roberta_model()
    for split in splits:
        generate_split_embeddings(split, data_dir, embed_dir, tokenizer, model, device)

# Entry point
if __name__ == "__main__":
    data_dir = "/home/bna36/misinfo_detection/data/liar_dataset"
    embed_dir = data_dir  # Save directly into liar_dataset folder

    print(f"[START] Generating statement-only RoBERTa embeddings in: {embed_dir}")
    generate_embeddings(data_dir, embed_dir)
    print("[DONE] Embedding generation complete.")
