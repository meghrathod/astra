import os
import json
import subprocess

def download_dataset():
    """Downloads the Kaggle dataset."""

    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        print("Kaggle API credentials not found in environment variables.")
        print("Please set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return

    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    with open(kaggle_json_path, "w") as f:
        json.dump({"username": kaggle_username, "key": kaggle_key}, f)

    os.chmod(kaggle_json_path, 0o600)
    print("Kaggle API credentials saved.")

    dataset = "jsrojas/ip-network-traffic-flows-labeled-with-87-apps"
    download_dir = "data"

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"Downloading dataset '{dataset}' to '{download_dir}'...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", download_dir, "--unzip"], check=True)
    print("Dataset downloaded and unzipped successfully.")

if __name__ == "__main__":
    download_dataset()
