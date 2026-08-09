import os
import sys
import shutil

def download_mind():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    
    try:
        import kagglehub
        print("Downloading MIND dataset from Kaggle via kagglehub...")
        # Download the dataset
        path = kagglehub.dataset_download("arashnic/mind-news-dataset")
        print(f"Dataset downloaded to: {path}")
        
        os.makedirs(raw_data_dir, exist_ok=True)
        
        print(f"Copying files to {raw_data_dir}...")
        # Recursively copy the required directories and files
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(raw_data_dir, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d)
                else:
                    print(f"Directory {d} already exists, skipping...")
            else:
                shutil.copy2(s, d)
        
        print("Download and setup complete.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\n--- MANUAL DOWNLOAD INSTRUCTIONS ---")
        print("1. Go to https://msnews.github.io/")
        print("2. Download the MIND-small dataset (MINDsmall_train.zip and MINDsmall_dev.zip).")
        print("3. Extract the contents into:")
        print(f"   - {os.path.join(raw_data_dir, 'MINDsmall_train')}")
        print(f"   - {os.path.join(raw_data_dir, 'MINDsmall_dev')}")
        print("   The required files in each directory are: news.tsv and behaviors.tsv")
        print("4. Proceed with the next steps.")
        sys.exit(1)

if __name__ == "__main__":
    download_mind()
