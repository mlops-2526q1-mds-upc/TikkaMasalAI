# Tikka MasalAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOPS project FIB

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