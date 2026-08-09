import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def preprocess_news(data_dir, split_name):
    """Parses news.tsv into a clean DataFrame."""
    news_path = os.path.join(data_dir, split_name, "news.tsv")
    
    if not os.path.exists(news_path):
        print(f"Warning: {news_path} not found.")
        return None
    
    columns = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    df_news = pd.read_csv(news_path, sep="\t", header=None, names=columns)
    
    # Select only required columns
    df_news = df_news[["news_id", "category", "subcategory", "title", "abstract"]]
    df_news["abstract"] = df_news["abstract"].fillna("")
    df_news["title_word_count"] = df_news["title"].apply(lambda x: len(str(x).split()))
    
    return df_news

def preprocess_behaviors_to_interactions(data_dir, split_name):
    """
    Parses behaviors.tsv, explodes impressions to one row per (user, news, time) interaction.
    """
    behaviors_path = os.path.join(data_dir, split_name, "behaviors.tsv")
    
    if not os.path.exists(behaviors_path):
        print(f"Warning: {behaviors_path} not found.")
        return None
        
    columns = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    df_behaviors = pd.read_csv(behaviors_path, sep="\t", header=None, names=columns)
    
    # We want to keep the split identity (train/dev) implicitly via timestamps, but we can also store the split name.
    # Convert timestamp to datetime
    df_behaviors["timestamp"] = pd.to_datetime(df_behaviors["timestamp"], format="%m/%d/%Y %I:%M:%S %p")
    
    # Explode impressions: "N123-1 N456-0"
    records = []
    
    # We iterate over rows. Iterrows is slow, but MIND-small behaviors is ~150k rows. 
    # For a real pipeline, we'd use PySpark or efficient pandas explode. Let's do pandas explode.
    
    # First, split the impressions string into a list of "newsid-label" strings
    df_behaviors["impressions_list"] = df_behaviors["impressions"].str.split(" ")
    
    # Explode the list so each impression candidate gets its own row
    df_exploded = df_behaviors.explode("impressions_list").reset_index(drop=True)
    
    # Drop rows where impressions_list is null or empty
    df_exploded = df_exploded.dropna(subset=["impressions_list"])
    
    # Extract news_id and clicked from the "newsid-label" string
    def extract_news_label(item):
        if pd.isna(item):
            return None, None
        parts = item.split("-")
        if len(parts) == 2:
            return parts[0], int(parts[1])
        return None, None
        
    extracted = df_exploded["impressions_list"].apply(extract_news_label)
    df_exploded["news_id"] = [x[0] for x in extracted]
    df_exploded["clicked"] = [x[1] for x in extracted]
    
    df_exploded = df_exploded.dropna(subset=["news_id", "clicked"])
    df_exploded["clicked"] = df_exploded["clicked"].astype(int)
    
    # Keep only relevant columns for the interactions log
    df_interactions = df_exploded[["impression_id", "user_id", "timestamp", "news_id", "clicked", "history"]].copy()
    
    # History is also needed for candidate generation, we'll keep it here or process it separately.
    df_interactions["split"] = split_name
    
    return df_interactions

def run_preprocessing():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    processed_data_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process news
    news_train = preprocess_news(raw_data_dir, "MINDsmall_train")
    news_dev = preprocess_news(raw_data_dir, "MINDsmall_dev")
    
    if news_train is not None and news_dev is not None:
        # Combine news and drop duplicates
        df_news = pd.concat([news_train, news_dev]).drop_duplicates(subset=["news_id"])
        news_output_path = os.path.join(processed_data_dir, "news.parquet")
        df_news.to_parquet(news_output_path, index=False)
        print(f"Saved preprocessed news to {news_output_path}")
    
    # Process interactions
    inter_train = preprocess_behaviors_to_interactions(raw_data_dir, "MINDsmall_train")
    inter_dev = preprocess_behaviors_to_interactions(raw_data_dir, "MINDsmall_dev")
    
    if inter_train is not None and inter_dev is not None:
        df_interactions = pd.concat([inter_train, inter_dev])
        # Sort by timestamp to simulate real log stream
        df_interactions = df_interactions.sort_values("timestamp")
        
        inter_output_path = os.path.join(processed_data_dir, "interactions.parquet")
        df_interactions.to_parquet(inter_output_path, index=False)
        print(f"Saved {len(df_interactions)} preprocessed interactions to {inter_output_path}")

if __name__ == "__main__":
    run_preprocessing()
