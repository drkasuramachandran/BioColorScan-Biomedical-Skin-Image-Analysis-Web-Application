"""
Upload the three model files to a Hugging Face MODEL repo (run once, locally).

    pip install huggingface_hub
    huggingface-cli login
    python tools/upload_models.py  your-username/skin-imaging-models  path/to/folder_with_weights
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

FILES = ["my_5layer_cnn.h5", "unet_best.pth", "densenet_best.pth"]

repo_id, folder = sys.argv[1], Path(sys.argv[2])
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

for name in FILES:
    path = folder / name
    if not path.exists() or path.stat().st_size < 1024:
        print(f"SKIP {name}: not found or looks like an LFS pointer")
        continue
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id, repo_type="model")
    print("uploaded", name)
