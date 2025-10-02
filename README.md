# Tikka MasalAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An MLOps project for food classification using computer vision techniques.

## 🤖 Models

### Model Evaluation

Evaluate the available models by running the evaluation script:

```bash
uv run src/eval/eval.py
```

This command evaluates three distinct models using MLflow tracking. You can view the experiment results on [DagsHub](https://dagshub.com/HubertWojcik10/TikkaMasalAI/experiments):

#### 1. **Food-101-93M** (Benchmark Model)
- **Source**: [HuggingFace Model](https://huggingface.co/prithivMLmods/Food-101-93M)
- **Performance**: ~90% accuracy on Food-101 dataset
- **Purpose**: Pre-trained benchmark model and potential deployment candidate

#### 2. **ResNet-18** (Base Model)
- **Source**: [Microsoft ResNet-18](https://huggingface.co/microsoft/resnet-18)
- **Training**: Pre-trained on ImageNet-1k dataset
- **Purpose**: Base model for fine-tuning on Food-101 dataset

#### 3. **VGG-16** (Alternative Base Model)
- **Source**: [PyTorch VGG-16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)
- **Training**: Pre-trained on ImageNet-1k dataset
- **Purpose**: Alternative base model for Food-101 fine-tuning

#### Viewing Results

Launch the MLflow UI to inspect evaluation results:

```bash
uv run mlflow ui
```

Then navigate to the displayed URL (typically [http://127.0.0.1:5000](http://127.0.0.1:5000)) to view the interactive dashboard.

### Adding New Models
- To add new models you can create a new class with the name of the model under `src/models`.
- To make existing scripts and code work with the model make sure that it inherits from [this abstract base model class](src/models/food_classification_model.py), requiring the model class to have a classify function that takes in an image as bytes and returns an integer indicating the id of the label.
- Examples can be found in the `src/models` directory.

## Setup
### uv
1. This project uses uv for Python dependency management. You can install it [here](https://docs.astral.sh/uv/getting-started/installation/).
2. Verify installation: `uv --version`
3. Create and activate a virtual environment (Python 3.10): `uv venv`.
4. Activate the environment (zsh/macOS): `source .venv/bin/activate` (If you don't have Python 3.10 installed, uv can install it: `uv python install 3.10`).
5. Install dependencies
Sync project dependencies defined in *pyproject.toml* (uses the existing *uv.lock* if present): `uv sync`.

### dvc
1. Configure the access keys to the dvc remote by running the following two commands. Replace YOUR_ACCESS_KEY and YOUR_SECRET_ACCESS_KEY with the actual keys. You can get them from Hubert.
```bash
uv run dvc remote modify origin --local access_key_id YOUR_ACCESS_KEY
uv run dvc remote modify origin --local secret_access_key YOUR_SECRET_ACCESS_KEY
```
2. Pull data with DVC: Pull the data from the configured remote: `dvc pull`.

## 🐳 Docker Setup

This project has **two different Docker setups** for different purposes:

### 🔧 **Local Development Docker**
**File:** `streamlit/Dockerfile`  
**Purpose:** Build and test locally from project root  
**Build Command:** `make docker-build` (from project root)

**Usage:**
```bash
make docker-build    # Build from project root
make docker-run      # Run locally on port 7860
make docker-test     # Build + run
```

### 🚀 **HF Spaces Deployment Docker**
**File:** `Food101/Dockerfile`  
**Purpose:** Deploy to Hugging Face Spaces  
**Build Command:** Automatic on HF Spaces push

**Usage:**
```bash
make deploy-hf       # Deploy to HF Spaces
```

### 📁 File Structure Differences

#### **Local Project Structure:**
```
TikkaMasalAI/
├── src/             # Source code
├── streamlit/       # Development files
│   ├── Dockerfile   # Local Docker config
│   ├── app.py       # Streamlit app
│   └── requirements.txt
└── Food101/         # HF Spaces deployment
    ├── Dockerfile   # HF Docker config
    ├── app.py       # (copied from streamlit/)
    ├── requirements.txt # (copied from streamlit/)
    └── src/         # (copied from ../src/)
```

#### **HF Spaces Structure:**
```
YourSpace/
├── Dockerfile       # HF-specific Docker config
├── app.py           # Streamlit app (root level)
├── requirements.txt # Dependencies (root level)
└── src/             # Source code
```

### 🔄 Deployment Workflow

1. **Develop locally** using `streamlit/` files
2. **Test locally** with `make docker-test`
3. **Deploy to HF** with `make deploy-hf`
   - Copies `src/` → `Food101/src/`
   - Copies `streamlit/app.py` → `Food101/app.py`
   - Copies `streamlit/requirements.txt` → `Food101/requirements.txt`
   - **Keeps existing `Food101/Dockerfile`** (HF-specific)

### ⚠️ Important Notes

- **Never copy `streamlit/Dockerfile` to `Food101/`** - they're different!
- The HF Dockerfile expects files in the root directory
- The local Dockerfile expects files in subdirectories
- Deployment scripts handle this automatically

### 🐛 Troubleshooting

#### **Local build fails:**
- Check you're running `make docker-build` from project root
- Ensure `streamlit/Dockerfile` has correct paths

#### **HF Spaces build fails:**
- Check `Food101/Dockerfile` uses root-level paths
- Ensure `app.py` and `requirements.txt` are in `Food101/`
- Don't copy local Dockerfile to HF Spaces

#### **Quick Fix Commands:**
```bash
# Fix local Docker setup
make docker-build

# Fix HF deployment
make deploy-hf
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── scripts            <- Deployment and utility scripts
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------