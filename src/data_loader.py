import pandas as pd

def load_data(dir="../data/liar_dataset"):
    
    """
    Load the LIAR dataset (train, test, valid).
    
    parameters:
    dir(str): Path to the directory containing the dataset files.
    
    Returns:
    tuple: (train_df, test_df, valid_df)
    
    """
    
    # Column names of the LIAR dataset
    columns = [
    "id", "label", "statement", "subject", "speaker", "speaker_job", "state",
    "party", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    
    # File paths
    train_path = f"{dir}/train.tsv"
    valid_path = f"{dir}/valid.tsv"
    test_path = f"{dir}/test.tsv"

    # Load the dataset
    train_df = pd.read_csv(train_path, sep='\t', names=columns, header=None)
    valid_df = pd.read_csv(valid_path, sep='\t', names=columns, header=None)
    test_df = pd.read_csv(test_path, sep='\t', names=columns, header=None)
    
    return train_df, test_df, valid_df