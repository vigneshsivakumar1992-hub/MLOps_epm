from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "Vignesh-vigu/Engine-Predictive-Maintenance-MLOps"
repo_type = "space"

token = os.getenv("HF_TOKEN_EPM")
if not token:
    raise RuntimeError("HF_TOKEN_EPM not set")

api = HfApi(token=token)

# Ensure Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print("⚠️ Space not found. Creating Space...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False
    )
    print("✅ Space created.")

# Upload frontend
api.upload_folder(
    folder_path="epm/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("🚀 Deployment uploaded successfully.")
