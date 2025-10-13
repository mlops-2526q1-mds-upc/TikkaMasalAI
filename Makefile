#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Tikka MasalAI
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Fine-tune ResNet-18 on Food-101 (local imagefolder or HF dataset)
.PHONY: train-resnet18
train-resnet18:
	python src/train/finetune_resnet18.py --data_dir data/raw/food101 || \
	python src/train/finetune_resnet18.py

## Evaluate models using the unified evaluation script (MLflow tracking)
.PHONY: eval
eval:
	uv run src/eval/eval.py

## Build the documentation site (MkDocs)
.PHONY: docs-build
docs-build:
	uv run mkdocs build --strict -f docs/mkdocs.yml

## Serve documentation locally with live reload
.PHONY: docs-serve
docs-serve:
	uv run mkdocs serve -f docs/mkdocs.yml -a 127.0.0.1:8000

.PHONY: docs
docs:
	@echo "Building docs..."
	$(MAKE) docs-build
	@echo "Starting docs server..."
	# Run the server in the background
	uv run mkdocs serve -f docs/mkdocs.yml -a 127.0.0.1:8000 & \
	SERVER_PID=$$!; \
	sleep 1; \
	if command -v open >/dev/null 2>&1; then \
	  echo "Opening browser at http://127.0.0.1:8000"; \
	  open http://127.0.0.1:8000; \
	fi; \
	echo "Docs server running with PID $$SERVER_PID (press CTRL+C to stop)"; \
	wait $$SERVER_PID



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
