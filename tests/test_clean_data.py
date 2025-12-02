import os

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import CategoryMismatchTrainTest, MixedNulls
from loguru import logger
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Deepchecks data validation tests are run locally only (CI has no dataset).",
)


@pytest.fixture
def clean_data_dir():
    """Fixture to provide the path to the clean data directory."""
    return "./data/raw/food101/data"


def test_clean_data_download(clean_data_dir):
    """Test that the clean data is downloaded correctly."""
    # Ensure the directory exists
    assert os.path.exists(clean_data_dir), "Clean data directory does not exist."
    expected_files = [
        "train-00000-of-00008.parquet",
        "validation-00000-of-00003.parquet",
    ]
    for file in expected_files:
        assert os.path.exists(os.path.join(clean_data_dir, file)), f"{file} missing"


def test_clean_data_validation(clean_data_dir):
    """Test the clean data using Deepchecks."""
    # Load the clean data
    train_data_path = os.path.join(clean_data_dir, "train-00000-of-00008.parquet")
    validation_data_path = os.path.join(clean_data_dir, "validation-00000-of-00003.parquet")

    assert os.path.exists(train_data_path), "Train data file is missing."
    assert os.path.exists(validation_data_path), "Validation data file is missing."

    # Load data using Pandas
    train_df = pd.read_parquet(train_data_path)
    validation_df = pd.read_parquet(validation_data_path)

    # Preprocess data
    train_df["image"] = train_df["image"].astype(str)
    validation_df["image"] = validation_df["image"].astype(str)

    # Explicitly set which columns are categorical
    categorical_features = ["label"] if "label" in train_df.columns else []

    # Create Deepchecks datasets with explicit categorical features
    train_dataset = Dataset(train_df, label=None, cat_features=categorical_features)
    validation_dataset = Dataset(validation_df, label=None, cat_features=categorical_features)

    # Run Deepchecks checks
    mixed_nulls_check = MixedNulls().run(train_dataset)
    category_mismatch_check = CategoryMismatchTrainTest().run(train_dataset, validation_dataset)

    # Extract conditions safely (new API version compatible)
    def get_failed_conditions(check_result):
        conditions = getattr(check_result, "conditions_results", [])
        return [c for c in conditions if getattr(c, "category", "") == "FAIL"]

    failed_mixed = get_failed_conditions(mixed_nulls_check)
    failed_mismatch = get_failed_conditions(category_mismatch_check)

    # Log detailed info
    if failed_mixed:
        logger.warning(f"MixedNulls failed conditions: {failed_mixed}")
    if failed_mismatch:
        logger.warning(f"CategoryMismatch failed conditions: {failed_mismatch}")

    # Assert no failed conditions
    assert not failed_mixed, "MixedNulls check failed."
    assert not failed_mismatch, "CategoryMismatchTrainTest check failed."
