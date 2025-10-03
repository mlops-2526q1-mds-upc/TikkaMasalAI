# Tikka MasalAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An MLOps project for food classification using computer vision techniques.

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

### (Optional) Deploy Streamlit App on HF spaces
- We have a streamlit app deployed on HuggingFace Spaces which lets us run inference on all the models we have created.
- You can access the Space here: https://huggingface.co/spaces/AdrianHagen/Food101-Streamlit
- In order to contribute to this, follow these steps:
    1. Clone the Space repository by running: `git clone https://huggingface.co/spaces/AdrianHagen/Food101-Streamlit`
    2. Make changes to the source code in the `src` directory or modify the files in the `Food101-Streamlit` directory. You can check how these changes look before deploying them by running: `make streamlit`.
    3. Deploy your changes by running `make deploy-hf`.


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
├── models             <- (Deprecated) Local model storage; models now load from the Hub at runtime
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