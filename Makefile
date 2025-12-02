#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Tikka MasalAI
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Default target
.DEFAULT_GOAL := help

# Mark all targets as phony (order doesn't matter)
.PHONY: help \
	requirements create_environment clean lint format \
	test test-backend code-coverage \
	build-frontend-docker run-frontend-docker frontend-docker push-frontend-docker \
	build-backend-docker run-backend-docker backend-docker push-backend-docker \
	compose-up compose-down compose-logs \
	local-up local-down local-logs \
	test-local-api \
	test-deployed-api \
	train-resnet18 eval \
	docs-build docs-serve docs

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements: ## Install Python dependencies
	uv sync

create_environment: ## Set up Python interpreter environment
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint: ## Lint using ruff (use `make format` to do formatting)
	ruff format --check
	ruff check

format: ## Format source code with ruff
	ruff check --fix
	ruff format

#################################################################################
# TESTING AND CODE COVERAGE                                                     #
#################################################################################

test: ## Run tests
	uv run pytest -q

test-backend: ## Test only the backend
	uv run pytest -q tests/backend

code-coverage: ## Open code coverage report
	open reports/coverage/index.html

#################################################################################
# Docker Containers                                                             #
#################################################################################

build-frontend-docker: ## Build frontend docker container
	docker build -f src/frontend/Dockerfile -t tikka-frontend .

run-frontend-docker: ## Run frontend docker container
	docker run --rm -p 8501:8501 tikka-frontend

frontend-docker: build-frontend-docker run-frontend-docker ## Build and run frontend docker container

push-frontend-docker: ## Push frontend docker container to GitHub Container Registry
	docker build --platform=linux/arm64 -f src/frontend/Dockerfile -t ghcr.io/mlops-2526q1-mds-upc/tikka-frontend:latest --push .

build-backend-docker: ## Build backend docker container
	docker build -f src/backend/Dockerfile -t tikka-backend .

run-backend-docker: ## Run backend docker container
	docker run --rm -p 8000:8000 tikka-backend

backend-docker: build-backend-docker run-backend-docker ## Build and run backend docker container

push-backend-docker: ## Push backend docker container to GitHub Container Registry
	docker build --platform=linux/arm64 \
		--secret id=gcp_credentials,src=$(HOME)/.config/gcloud/application_default_credentials.json \
		--build-arg GOOGLE_CLOUD_PROJECT=academic-torch-476716-h3 \
		-f src/backend/Dockerfile \
		-t ghcr.io/mlops-2526q1-mds-upc/tikka-backend:latest \
		--push .

#################################################################################
# Docker Compose                                                                #
#################################################################################

compose-up: ## Start stack (registry images)
	docker compose -f docker-compose.yml up -d

compose-down: ## Stop stack (registry images)
	docker compose -f docker-compose.yml down

compose-logs: ## Tail logs (registry images)
	docker compose -f docker-compose.yml logs -f --tail=200

local-up: ## Start local stack (build backend/frontend locally)
	docker compose -f docker-compose-local.yml up --build

local-down: ## Stop local stack
	docker compose -f docker-compose-local.yml down

local-logs: ## Tail logs for local stack
	docker compose -f docker-compose-local.yml logs -f --tail=200

#################################################################################
# API Testing	                                                                #
#################################################################################

test-local-api: ## Run Bru tests against local API
	cd tikkamasalai-requests && \
	bru run --env-file environments/local.bru

test-deployed-api: ## Run Bru tests against deployed API
	cd tikkamasalai-requests && \
	bru run --env-file environments/production.bru

#################################################################################
# MODEL TRAINING AND EVAL                                                       #
#################################################################################

train-resnet18: ## Fine-tune ResNet-18 on Food-101 (local imagefolder or HF dataset)
	python src/train/finetune_resnet18.py --data_dir data/raw/food101 || \
	python src/train/finetune_resnet18.py

eval: ## Evaluate models using the unified evaluation script (MLflow tracking)
	uv run src/eval/eval.py

#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

api-docs: ## Build the api documentation. Make sure to have node.js installed (https://nodejs.org/en/download).
	uv run src/backend/export_schema.py
	npx --yes @redocly/cli build-docs src/backend/openapi.json -o docs/docs/api.html

docs-build: ## Build the documentation site (MkDocs)
	$(MAKE) api-docs
	uv run mkdocs build --strict -f docs/mkdocs.yml

.PHONY: docs-kill-port
docs-kill-port: ## Kills any process running on port 8001
	@echo "Checking for existing process on port 8001..."
	# lsof -t -i :8001 gets the PID of the process on that port
	# xargs kill -9 pipes the PID to a force-kill command
	# 2>/dev/null || true suppresses errors if no process is found
	@lsof -t -i :8001 | xargs kill -9 2>/dev/null || true
	@sleep 0.5 # Give the OS a moment to release the port

docs-serve: docs-kill-port ## Serve documentation locally with live reload (uses 8001)
	uv run mkdocs serve -f docs/mkdocs.yml -a 127.0.0.1:8001

docs: docs-build docs-kill-port ## Build then serve docs locally (opens browser on 8001)
	@echo "Starting new docs server..."
	uv run mkdocs serve -f docs/mkdocs.yml -a 127.0.0.1:8001 & \
	SERVER_PID=$$!; \
	sleep 1; \
	if command -v open >/dev/null 2>&1; then \
		echo "Opening browser at http://127.0.0.1:8001"; \
		open http://127.0.0.1:8001; \
	fi; \
	echo "Docs server running with PID $$SERVER_PID (press CTRL+C to stop)"; \
	wait $$SERVER_PID

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "%-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
