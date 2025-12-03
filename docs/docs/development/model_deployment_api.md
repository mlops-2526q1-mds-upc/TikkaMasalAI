# Model Deployment: API

## Overview
- Goal: Expose the food classification model as an HTTP API.
- Framework: FastAPI (`src/backend/app.py`) with CORS and Prometheus instrumentation.
- Endpoints:
  - `GET /` health banner
  - `GET /health` lightweight health check
  - `POST /predict` image classification (multipart file `image`)
  - `POST /predict/explain` attention heatmap visualization for the uploaded image
- Model: `SiglipForImageClassification` with `AutoImageProcessor` loaded from `models/prithiv`.

## System Design

### Architecture
- **Service**: A single FastAPI app (`src/backend/app.py`) that mounts routers from `src/backend/routers/`.
- **Routers**:
  - `predict.py` (prefix `/predict`):
    - `POST /predict`: Classify uploaded image (multipart file `image`).
    - `POST /predict/explain`: Return attention heatmap and prediction metadata.
    - Notes: Model + processor lazy-loaded via `@lru_cache`; device auto-selected (MPS→CUDA→CPU).
  - `dashboard.py`:
    - `GET /dashboard`: HTML metrics dashboard with charts, fed by Prometheus `/metrics`.
  - `llm.py` (prefix `/llm`):
    - `POST /llm/generate`: Generate text from prompt (JSON: `prompt`, `temperature`, `max_tokens`).
    - `GET /llm/health`: Check Ollama availability; returns available models and configured model.
    - Config: `OLLAMA_HOST` (default `http://localhost:11434`), model `gemma3:270m`.
- **Model Loading**: `@lru_cache` caches `(model, processor)` on first call; selects device: Apple MPS > CUDA > CPU.
- **Observability**: `prometheus_fastapi_instrumentator` adds metrics exposure; CORS configured for local dev and `tikkamasalai.tech`.
- **OpenAPI**: Schema can be exported via `src/backend/export_schema.py` to `src/backend/openapi.json`.

### Deployment Targets
- **Cloud Provider**: As a cloud provider, we decided on GCP, where we are hosting a virtual machine that runs and exposes both our backend and frontend. The reason we decided on using GCP is their rather generous free credit amount of 270€ over three months, which together with a wide range of readily available VMs and the ability to easily add storage to existing VMs was perfect for our use case.
- **Containerization**: Compose files: `docker-compose.yml` and `docker-compose-local.yml` orchestrate `backend`, `frontend`, and `ollama` images; health checks ensure readiness; ports `8000` (API) and `8501` (frontend) are exposed.

Note: Detailed containerization, orchestration, CI/CD, and networking are documented in [Model Deployment: Component Delivery](./model_deployment_component_delivery.md). This API document focuses on the ML system design and API surface (Milestone 4).

#### GCP VM Specification
- **Location**: `europe-west1-c` (region: europe-west1), this gives us faster response times compared to US locations (although US locations are a bit cheaper).
- **Machine type**: `c4a-standard-2` (2 vCPUs, 8 GB RAM), sufficient for runnign our image classification model and a small LLM. The computational requirements of the frontend are not significant. 
- **CPU platform/Architecture**: Google Axion, Arm64. We chose an ARM architecture because most of the team develops on macOS (Apple Silicon), and at selection time we did not use Docker Buildx to produce multi-architecture images. Using an Arm VM ensured images ran consistently with our local environment without additional build complexity. 
- **OS image**: `debian-12-bookworm-arm64-v20251014`
- **Boot disk**: 20 GB, Hyperdisk Balanced (read/write), daily schedule configured. An additional boot disk was required to enhance the standard memory of the VM (10GB). Otherwise the VM storage was insufficient for storing the both our image classification model and the LLM.

### Security & Compliance
- **Data Handling**: Images uploaded via multipart form; processed transiently in-memory; no persistence by default.
- **Access Control**: Public API (no authentication) accessible at `https://api.tikkamasalai.tech`.
- **CORS**: Origins allowed for local and production domains.
- **Secrets**: Follow `CONTRIBUTING.md` guidance — never commit credentials; use env vars or secret stores.
 - **Justification**: The system is a demonstration project to showcase an MLOps lifecycle. Given its educational scope and non-sensitive inputs, we prioritized simplicity and accessibility over strict security controls. If usage expands beyond demo contexts, we could evaluate adding HTTPS enforcement details, rate limiting, and authentication.

## API Design

### Endpoints
See [API Documentation](../api.html).

## Deployment

### Container Image
- **Dockerfile**: `src/backend/Dockerfile` uses `python:3.12-slim`, installs `uv`, syncs backend deps from `pyproject.toml`, copies `src`, downloads model assets via `src/backend/download_model.py` with GCP credentials mounted as a secret. Exposes port `8000`. Entrypoint runs `uvicorn` serving `src.backend.app:app`.
- For Compose, registry, CI, env wiring, healthchecks, and Nginx/TLS, see [Model Deployment: Component Delivery](./model_deployment_component_delivery.md).

### Model Artifacts Storage (GCS)
- **Bucket**: `tikkamasalai-models`
- **Location**: EU multi‑region (`eu`), location type: Multi‑region
- **Usage in this project**:
  - The backend image downloads model files during build via `src/backend/download_model.py`.
  - Credentials provided as a build secret (`GOOGLE_APPLICATION_CREDENTIALS`) in `src/backend/Dockerfile` and `docker-compose-local.yml`.
  - Project ID passed via `GOOGLE_CLOUD_PROJECT`; files are synced to `./models`.
  - See `src/backend/download_model.py` for implementation.

The decision to use GCS here was due to the fact that we have already set up GCP for our VM, which made renting an additional storage space for models very straight forward.

### Cloud Rollout
This section intentionally defers the full rollout mechanics (multi-arch builds, GHCR, VM deploy script, Nginx/TLS) to [Model Deployment: Component Delivery](./model_deployment_component_delivery.md) to keep Milestone 4 focused on API design and system architecture.

## Testing the ML Component
- **Unit Tests (Pytest)**: CI runs code tests via GitHub Actions (`.github/workflows/tests.yml`) using `uv sync` and `make test`.
  - Relevant files: `tests/test_model.py`, `tests/test_data.py`, `tests/test_clean_data.py`.
  - Trigger: on `push` to `main` or `code-api-tests`, and on pull requests.
- **API Tests (Bruno, Deployed)**: The `verify-deployed-api` job in `deploy.yml` runs Bruno collections against the production API after deployment was successful. It can also be run manually by executing `make test-deployed-api`.
  - Collection: `tikkamasalai-requests/` with environments under `tikkamasalai-requests/environments/` for `local` and `production`.
  - Key requests: 
    - Availability: `tikkamasalai-requests/availability.bru`
    - Health: `tikkamasalai-requests/health.bru`
    - Predict: `tikkamasalai-requests/predict.bru`
    - Predict + Explain: `tikkamasalai-requests/predict-explain.bru`
    - LLM health/generate: `tikkamasalai-requests/llm-health.bru`, `tikkamasalai-requests/llm-generate.bru`
  - Usage: The workflow installs Bruno CLI (`@usebruno/cli`) and executes `make test-deployed-api` targeting the `production` environment.
    
    ```zsh
    # Run deployed API tests (production)
    make test-deployed-api
    ```
 
For CI validation of documentation freshness (OpenAPI regeneration and MkDocs build), see the CI/CD section in [Model Deployment: Component Delivery](./model_deployment_component_delivery.md).

## Adoption & Developer Experience
- **Ease of Use**: Simple multipart upload; predictable JSON responses; OpenAPI available.
- **Versioning**: The project currently uses unversioned routes (e.g., `/predict`). Versioning the API is a future-friendly proposal and could be adopted later.