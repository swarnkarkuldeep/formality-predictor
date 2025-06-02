import pandas as pd

def load_labeled_data(train_path="data/training_labeled.csv", eval_path="data/eval_labeled.csv"):
    """
    Load labeled training and evaluation datasets from CSV files.
    
    Args:
        train_path (str): Path to training data CSV file. Defaults to "data/training_labeled.csv".
        eval_path (str): Path to evaluation data CSV file. Defaults to "data/eval_labeled.csv".
        
    Returns:
        tuple: A tuple containing (training_data, evaluation_data) as pandas DataFrames
    """
    return (
        pd.read_csv(train_path),
        pd.read_csv(eval_path)
    )

def load_reddit_comments(path="data/reddit_comments.csv"):
    """
    Load Reddit comments data from a CSV file, keeping only subreddit and body columns.
    
    Args:
        path (str): Path to Reddit comments CSV file. Defaults to "data/reddit_comments.csv".
        
    Returns:
        DataFrame: Pandas DataFrame containing only 'subreddit' and 'body' columns
    """
    return pd.read_csv(path)[["subreddit", "body"]]