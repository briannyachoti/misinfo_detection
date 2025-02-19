import os
import pandas as pd
import json

# Define dataset path
dataset_path = "FakeNewsNet/Data"

# Function to load news articles
def load_news_data(source, category):
    data_dir = os.path.join(dataset_path, source, category)
    articles = []

    for news_id in os.listdir(data_dir):
        news_file = os.path.join(data_dir, news_id, "news content.json")
        if os.path.exists(news_file):
            with open(news_file, "r", encoding="utf-8") as f:
                news_data = json.load(f)
                news_data["news_id"] = news_id
                articles.append(news_data)

    return pd.DataFrame(articles)

# Load Fake and Real news from PolitiFact and GossipCop
politifact_fake = load_news_data("politifact", "fake")
politifact_real = load_news_data("politifact", "real")
gossipcop_fake = load_news_data("gossipcop", "fake")
gossipcop_real = load_news_data("gossipcop", "real")

# Combine datasets
df_fake = pd.concat([politifact_fake, gossipcop_fake])
df_real = pd.concat([politifact_real, gossipcop_real])

# Add labels
df_fake["label"] = 0  # Fake news
df_real["label"] = 1  # Real news

# Merge into one dataset
df = pd.concat([df_fake, df_real]).reset_index(drop=True)

# Display dataset info
print(df.head())
print(df.info())
