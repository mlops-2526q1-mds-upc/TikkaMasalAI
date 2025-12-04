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
2. Pull data with DVC: Pull the data from the configured remote using the project environment: `uv run dvc pull`.

## Makefile quick reference
New to the project? The Makefile bundles common tasks so you don’t have to remember long commands.

- Show all available commands and short descriptions (default):

```bash
make
# or
make help
```

Core environment and hygiene:
- `make create_environment` – Create a uv virtualenv for Python 3.10 and print activation hints
- `make requirements` – Install project dependencies via uv (uses pyproject.toml/uv.lock)
- `make clean` – Remove Python bytecode and __pycache__ folders
- `make lint` – Check formatting and lint with Ruff (no changes)
- `make format` – Auto-fix lint issues and format with Ruff
- `make test` – Run the test suite with pytest

Project workflows:
- `make train-resnet18` – Fine-tune ResNet-18 on Food-101; tries local ImageFolder at `data/raw/food101`, falls back to Hugging Face dataset if not present
- `make eval` – Run the unified evaluation script (`uv run src/eval/eval.py`) with MLflow tracking

Documentation:
- `make docs-build` – Build the docs site with MkDocs (outputs to `docs/site`)
- `make docs-serve` – Serve docs locally at http://127.0.0.1:8001 with live reload
- `make docs` – Build docs, start the server, and open your browser automatically; stop with CTRL+C

Containerization and deployment:
- Docker & Compose usage: see docs/docs/development/containers.md
- Deployment strategy (images, tags, registry, rollout): see docs/docs/development/deployment.md

Typical first run on macOS (zsh):
```bash
# 1) Create and activate env
make create_environment
source .venv/bin/activate

# 2) Install deps
make requirements

# 3) (Optional) Pull data with DVC – see steps above

# 4) Verify setup
make lint
make test
```

Notes:
- Run make targets from the project root.
- Targets that call `python` (e.g., `make test`) expect your virtual environment to be activated.
- Docs use MkDocs config at `docs/mkdocs.yml`.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

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

### Fine-tuning ResNet-18

Run a short fine-tuning job on Food-101:

```bash
uv run -m src.train.finetune_resnet18 --epochs 2 --train_samples 1000 --eval_samples 200 --output_dir models/resnet18-food101-2e-1k
```

You can also use configuration files (check `/configs`):
```bash
uv run -m src.train.finetune_resnet18 --config configs/training_quick.yaml

# Override config parameters
uv run -m src.train.finetune_resnet18 --config configs/training_quick.yaml --epochs 5
```

- To evaluate afterward:

```bash
uv run -m src.eval.eval --resnet_model_path models/resnet18-food101-2e-1k
```

To evaluate with shap:
```bash
uv run -m src.eval.shap_qa --model resnet18 --model-path models/resnet18-food101-2e-1k
```

Replace the path with any other trained model directory as needed.

### Adding New Models
- To add new models you can create a new class with the name of the model under `src/models`.
- To make existing scripts and code work with the model make sure that it inherits from [this abstract base model class](src/models/food_classification_model.py), requiring the model class to have a classify function that takes in an image as bytes and returns an integer indicating the id of the label.
- Examples can be found in the `src/models` directory.