import pytest
import os
from src.data.download_data import snapshot_download

@pytest.fixture
def data_dir():
    """Fixture to provide the path to the raw data directory."""
    return "./data/raw/food101"

def test_data_download(data_dir):
    """Test that the data is downloaded correctly."""
    # Ensure the directory exists
    assert os.path.exists(data_dir), "Data directory does not exist."

    # Check for some expected files
    expected_files = [
        "train-00000-of-00008.parquet",
        "validation-00000-of-00003.parquet",
    ]
    for file in expected_files:
        file_path = os.path.join(data_dir, "data", file)
        assert os.path.exists(file_path), f"Expected file {file} is missing."

def test_snapshot_download(data_dir):
    """Test the snapshot_download function."""
    # Call the snapshot_download function
    snapshot_download(
        repo_id="ethz/food101",
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
    )

    # Verify that the data directory is populated
    assert os.listdir(data_dir), "Data directory is empty after download."