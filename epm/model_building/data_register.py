from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "Vignesh-vigu/Engine-Predictive-Maintenance-MLOps"
repo_type = "dataset"

token = os.getenv("HF_TOKEN_EPM")
if not token:
    raise RuntimeError("❌ HF_TOKEN_EPM is NOT set in environment")

api = HfApi(token=token)

# Check dataset repo
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"⚠️ Dataset '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print("✅ Dataset repo created.")

# Upload data
api.upload_folder(
    folder_path="epm/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 Dataset uploaded successfully.")
