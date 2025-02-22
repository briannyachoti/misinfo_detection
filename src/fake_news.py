import os
import pandas as pd

def load_fakenews_dataset(folder_path="../data/fakenews_dataset"):
    
    """
    Loads all CSV files from the  folder, adds source & label columns, and combines them.
    
    """
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    combined_df = []

    for file in data_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        # Extract source (gossipcop or politifact) and label (real or fake) from the filename
        if "gossipcop" in file.lower():
            source = "gossipcop"
        elif "politifact" in file.lower():
            source = "politifact"
        else:
            source = "unknown"
        
        label = "real" if "real" in file.lower() else "fake"

        # Add columns to the DataFrame
        df["source"] = source
        df["label"] = label
        
        combined_df.append(df)

    # Concatenate all dataframes
    return pd.concat(combined_df, ignore_index=True)
