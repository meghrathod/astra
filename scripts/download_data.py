import os
import shutil
import kagglehub

def download_dataset():
    """Downloads the Kaggle dataset using kagglehub."""
    
    dataset = "jsrojas/ip-network-traffic-flows-labeled-with-87-apps"
    download_dir = "data"
    
    print(f"Downloading dataset '{dataset}'...")
    
    # Download latest version using kagglehub
    path = kagglehub.dataset_download(dataset)
    
    print(f"Dataset downloaded to: {path}")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Move files from kagglehub download location to data directory
    # kagglehub downloads to a cache location, so we'll copy the files
    if os.path.exists(path):
        # Get all files in the downloaded path
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(download_dir, item)
            
            if os.path.isfile(src):
                print(f"Copying {item} to {download_dir}/")
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                print(f"Copying directory {item} to {download_dir}/")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
    
    print(f"Dataset files available in '{download_dir}/'")
    print("Download completed successfully.")

if __name__ == "__main__":
    download_dataset()
