import streamlit as st
from PIL import Image
import io
import sys
import os
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.models.model_discovery import discover_models as _discover_models
from src.labels import LABELS


def discover_models_with_streamlit_warnings():
    """Wrapper for discover_models that shows Streamlit warnings for failed imports."""
    import importlib
    import inspect
    from pathlib import Path
    from src.models.food_classification_model import FoodClassificationModel

    models_dir = Path(__file__).parent.parent / "src" / "models"
    available_models = {}

    # Iterate through all Python files in the models directory
    for py_file in models_dir.glob("*.py"):
        if (
            py_file.name.startswith("__")
            or py_file.name == "food_classification_model.py"
            or py_file.name == "model_discovery.py"
        ):
            continue

        try:
            # Import the module dynamically
            module_name = f"src.models.{py_file.stem}"
            module = importlib.import_module(module_name)

            # Find all classes in the module that inherit from FoodClassificationModel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, FoodClassificationModel)
                    and obj != FoodClassificationModel
                    and obj.__module__ == module_name
                ):
                    # Use the centralized display name function
                    from src.models.model_discovery import _create_display_name

                    display_name = _create_display_name(name)

                    available_models[display_name] = {
                        "class": obj,
                        "module": module_name,
                        "class_name": name,
                    }

        except Exception as e:
            st.warning(f"Could not load model from {py_file.name}: {str(e)}")
            continue

    return available_models


@st.cache_resource
def load_model(model_info):
    """Load and cache the selected model with proper error handling."""
    import tempfile
    import shutil
    from pathlib import Path

    model_class = model_info["class"]
    model_name = model_info["class_name"]

    # Set up custom cache directory to avoid permission issues
    custom_cache = Path(tempfile.gettempdir()) / "tikka_masalai_cache"
    custom_cache.mkdir(exist_ok=True)

    # Set HuggingFace cache directory (use HF_HOME instead of deprecated TRANSFORMERS_CACHE)
    os.environ["HF_HOME"] = str(custom_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(
        custom_cache
    )  # Keep for backward compatibility

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            st.info(
                f"Loading {model_name} model... (Attempt {attempt + 1}/{max_retries})"
            )

            # Try to load the model - handle different model initialization patterns
            if "prithiv" in model_name.lower():
                # PrithivML model with specific initialization
                model = model_class()
            elif "resnet" in model_name.lower():
                # ResNet model - check if it needs specific paths
                try:
                    model = model_class()
                except TypeError:
                    # Try with default parameters if it requires them
                    model = model_class(
                        preprocessor_path="microsoft/resnet-18",
                        model_path="microsoft/resnet-18",
                    )
            elif "vgg" in model_name.lower():
                # VGG model with default parameters
                model = model_class()
            else:
                # Generic model initialization
                try:
                    model = model_class()
                except TypeError:
                    # Skip models that require specific parameters we don't know about
                    raise RuntimeError(
                        f"Model {model_name} requires specific initialization parameters"
                    )

            st.success(f"{model_name} model loaded successfully!")
            return model

        except PermissionError as e:
            if "cache" in str(e).lower():
                st.warning(
                    f"Cache permission issue (attempt {attempt + 1}). Retrying..."
                )

                # Try to clear cache and retry
                try:
                    if custom_cache.exists():
                        shutil.rmtree(custom_cache, ignore_errors=True)
                        custom_cache.mkdir(exist_ok=True)
                except:
                    pass

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"❌ Persistent cache permission error: {str(e)}")
                    st.info(
                        "💡 This is likely a temporary HF Spaces issue. Please refresh the page in a few minutes."
                    )
                    return None
            else:
                st.error(f"❌ Permission error: {str(e)}")
                return None

        except Exception as e:
            error_msg = str(e)
            if "lock" in error_msg.lower() or "permission" in error_msg.lower():
                st.warning(
                    f"Model download conflict detected (attempt {attempt + 1}). Retrying..."
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

            st.error(f"❌ Error loading {model_name} model: {error_msg}")
            if attempt == max_retries - 1:
                st.info("💡 Possible solutions:")
                st.info("1. Refresh the page and try again")
                st.info(
                    "2. Wait a few minutes for any concurrent downloads to complete"
                )
                st.info("3. Check if HuggingFace services are available")
                st.info("4. Try a different model")
            return None

    return None


def predict_food(model, image_bytes):
    """Make a prediction on the uploaded image."""
    try:
        # Get prediction index
        prediction_idx = model.classify(image_bytes)

        # Get the label name
        if 0 <= prediction_idx < len(LABELS):
            prediction_label = LABELS[prediction_idx]
            return prediction_idx, prediction_label
        else:
            return None, "Unknown"
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, "Error"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="TikkaMasalAI Food Classifier", page_icon="🍽️", layout="centered"
    )

    st.title("🍽️ TikkaMasalAI Food Classifier")
    st.markdown("Upload an image of food and let our AI identify what it is!")

    # Discover available models
    try:
        available_models = discover_models_with_streamlit_warnings()
    except Exception as e:
        st.error(f"❌ Error discovering models: {str(e)}")
        st.info("Make sure the src/models directory contains valid model files.")
        return

    if not available_models:
        st.error("❌ No compatible models found in the src/models directory!")
        st.info("Make sure there are models that inherit from FoodClassificationModel.")
        return

    # Model selection in sidebar
    with st.sidebar:
        st.header("🤖 Model Selection")
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(available_models.keys()),
            help="Select which AI model to use for food classification",
        )

        selected_model_info = available_models[selected_model_name]

        # Show model information
        st.info(f"**Selected:** {selected_model_name}")
        st.write(f"**Class:** `{selected_model_info['class_name']}`")
        st.write(f"**Module:** `{selected_model_info['module']}`")

    # Show app status
    status_container = st.container()

    # Load model with better UX
    with status_container:
        model_status = st.empty()
        progress_bar = st.progress(0)

        model_status.info("🔄 Initializing AI model...")
        progress_bar.progress(25)

        model = load_model(selected_model_info)
        progress_bar.progress(100)

        if model is None:
            model_status.error("❌ Failed to load the model.")
            st.error("### 🚨 Model Loading Failed")
            st.markdown(
                f"""
            **Failed to load:** {selected_model_name}
            
            **Possible causes:**
            - Model-specific initialization requirements
            - Missing dependencies for this model
            - Temporary HuggingFace services issue
            - Model cache conflicts in HF Spaces
            - Network connectivity problems
            
            **Solutions:**
            1. **Try a different model** from the sidebar
            2. **Refresh the page** and try again
            3. **Wait 2-3 minutes** for any background downloads to complete
            4. If the issue persists, the model will automatically retry
            """
            )

            # Add a retry button
            if st.button("🔄 Retry Loading Model"):
                st.experimental_rerun()

            return

        model_status.success(f"✅ {selected_model_name} loaded and ready!")
        progress_bar.empty()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=["png", "jpg", "jpeg"],
        help="Upload an image of food to classify",
    )

    if uploaded_file is not None:
        # Read image bytes
        image_bytes = uploaded_file.read()

        # Display the uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Your uploaded image", width="stretch")

        with col2:
            st.subheader("🔍 Prediction Results")

            # Make prediction
            with st.spinner("Analyzing your image..."):
                prediction_idx, prediction_label = predict_food(model, image_bytes)

            if prediction_idx is not None:
                # Display results
                st.success("Classification complete!")

                # Format the label for display
                display_label = prediction_label.replace("_", " ").title()

                st.markdown(f"### 🏷️ **{display_label}**")
                st.markdown(f"**Class Index:** {prediction_idx}")

                # Show confidence bar (placeholder since the model doesn't return probabilities)
                st.markdown("**Prediction Details:**")
                st.info(f"The AI model identified this image as **{display_label}**")

                # Show additional info
                with st.expander("ℹ️ About this classification"):
                    st.write(f"- **Model:** {selected_model_name}")
                    st.write(f"- **Classes:** {len(LABELS)} different food types")
                    st.write(f"- **Raw label:** `{prediction_label}`")
                    st.write(f"- **Index:** {prediction_idx}")
            else:
                st.error("Failed to classify the image. Please try another image.")

    # Sidebar with information
    with st.sidebar:
        st.header("📋 About")
        st.write(
            f"""
        This app uses the **{selected_model_name}** model to classify food images into one of 101 different food categories.
        """
        )

        st.header("🎯 How to use")
        st.write(
            """
        1. Choose a model from the dropdown above
        2. Upload an image of food using the file uploader
        3. Wait for the AI to analyze your image
        4. View the classification results
        """
        )

        st.header("🍕 Supported Foods")
        st.write(
            f"The model can recognize **{len(LABELS)}** different types of food including:"
        )

        # Show a sample of labels
        sample_labels = [label.replace("_", " ").title() for label in LABELS[:10]]
        for label in sample_labels:
            st.write(f"• {label}")
        st.write(f"... and {len(LABELS) - 10} more!")

        st.header("🔧 Technical Details")
        st.write(
            f"""
        - **Selected Model:** {selected_model_name}
        - **Available Models:** {len(available_models)}
        - **Dataset:** Food-101
        - **Framework:** PyTorch + Transformers
        - **Performance:** Varies by model
        """
        )


if __name__ == "__main__":
    main()
