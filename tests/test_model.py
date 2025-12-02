import io
import os

from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Model integration tests are run locally only (CI avoids heavy model downloads).",
)

from src.models.prithiv_ml_food101 import PrithivMlFood101
from src.models.resnet18 import Resnet18
from src.models.vgg16 import VGG16


@pytest.fixture(scope="module")
def prithiv_pipe():
    return PrithivMlFood101()


@pytest.fixture(scope="module")
def resnet_pipe():
    return Resnet18()


@pytest.fixture(scope="module")
def vgg_pipe():
    return VGG16()


@pytest.fixture
def valid_image_bytes():
    """Generate a valid image in memory for testing."""
    image = Image.new("RGB", (224, 224), color="white")  # Create a white image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()  # Return bytes instead of tensor


@pytest.mark.parametrize(
    "expected_label",
    [0, 1],
)
def test_model_predictions(prithiv_pipe, resnet_pipe, vgg_pipe, valid_image_bytes, expected_label):
    """
    Test the predictions of all models for given inputs.
    """
    for model in [prithiv_pipe, resnet_pipe, vgg_pipe]:
        predicted = model.classify(valid_image_bytes)
        assert isinstance(predicted, int), f"{model.__class__.__name__} did not return an integer."
        assert predicted >= 0, f"{model.__class__.__name__} returned a negative integer."
