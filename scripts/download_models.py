import os
from huggingface_hub import snapshot_download, login

# Define the models to download
models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
    "microsoft/Phi-4-mini-instruct",
    "ministral/Ministral-3b-instruct"
]

# Define the base directory
base_dir = "models"

# Create the directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

def download_model(model_id):
    try:
        print(f"Downloading {model_id}...")
        # Create a local directory name from the model ID
        local_dir = os.path.join(base_dir, model_id.split("/")[-1])

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Download actual files
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")
    except Exception as e:
        print(f"Failed to download {model_id}: {e}")

if __name__ == "__main__":
    for model in models:
        download_model(model)
