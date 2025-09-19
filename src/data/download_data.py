from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ethz/food101",
    repo_type="dataset",
    local_dir="./data/raw/food101",
    local_dir_use_symlinks=False  # ensures actual files, not symlinks
)