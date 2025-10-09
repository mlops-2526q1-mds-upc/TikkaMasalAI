import sys
from huggingface_hub import snapshot_download

local_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/raw/food101"

snapshot_download(
    repo_id="ethz/food101",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # ensures actual files, not symlinks
)
