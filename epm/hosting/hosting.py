from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN_EPM"))
api.upload_folder(
    folder_path="epm/deployment",     # the local folder containing your files
    repo_id="Vignesh-vigu/Engine-Predictive-Maintenance-MLOps",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
