# Model Deployment: Component Delivery

## Overview
- Goal: Deliver a portable, reproducible ML component ready for deployment.
- Scope: Container build, orchestration via Compose on VM, CI/CD for builds, deploys, and tests.
- Component surfaces: HTTP API (`/predict`, `/predict/explain`), Streamlit UI, and optional LLM helpers.
 - See also: API design, routers, and model loading in [Model Deployment: API](./model_deployment_api.md) (Milestone 4).

## Container and Orchestration
- **Images**: Backend and frontend are built as containers.
  - Backend: `python:3.12-slim`, packages via `uv`, model assets downloaded during build with GCP credentials provided as a build secret.
  - Frontend: Streamlit app containerized with pinned dependencies.
  - Ollama: `ollama/ollama:latest`, healthcheck via `ollama list`, persistent volume `ollama_data:/root/.ollama`; init pulls `gemma3:270m`.
- **Multi-arch**: CI uses Docker Buildx with QEMU to produce `linux/amd64` and `linux/arm64` images ensuring portability across Intel and ARM environments.
- **Registry**: Images pushed to GHCR:
  - `ghcr.io/mlops-2526q1-mds-upc/tikka-backend`
  - `ghcr.io/mlops-2526q1-mds-upc/tikka-frontend`
- **Orchestration**: `docker-compose.yml` (prod) and `docker-compose-local.yml` (dev) define services:
  - Services: `backend`, `frontend`, `ollama` (LLM), and health checks.
  - Restart policy: `unless-stopped` with HTTP health checks for readiness.
  - Env wiring: `OLLAMA_HOST`, `APP_DEBUG`; for builds, `GOOGLE_CLOUD_PROJECT` and GCP credentials are provided as secrets.
- **Networking**: Nginx reverse proxy on the VM fronts `:443` and routes `api.tikkamasalai.tech` to `backend:8000`; TLS via Certbot.

### Guarantees Across Environments
- **Portability**: Multi-arch builds, pinned dependencies, externalized config via env vars, and GHCR images ensure consistent runtime on both local (Apple Silicon) and cloud (ARM/AMD64) hosts.
- **Modularity**: Separate containers for backend, frontend, and LLM; clear boundaries via Compose services and env; routers modularized under `src/backend/routers/`.
- **Consistency**: CI builds from the same manifests (`Dockerfile`, `pyproject.toml`), health checks and restart policies applied uniformly.
- **Efficiency**: Slim base images, `uv` for fast, deterministic dependency resolution, model download during build to avoid runtime stalls.

## CI/CD for ML (Automation & Collaboration)
- **Build & Deploy**: `.github/workflows/deploy.yml`
  - Triggers on: Pushes to `main`.
  - Steps: Setup QEMU/Buildx → auth GHCR/GCP → build backend/frontend (multi-arch) with model download via secret → push images → SSH into VM to run `run.sh` which pulls new images and restarts services → post-deploy API verification job.
  - Post-deploy tests: A `verify-deployed-api` job waits for `/health` to be ready and runs Bruno tests (`make test-deployed-api`) against production.
  - Tags: `latest` and short SHA for traceability.
- **Tests (Code + API)**: `.github/workflows/tests.yml`
  - Unit tests: Run with `uv sync` + `make test` on push/PR.
  - Bruno API tests (deployed): Moved to `deploy.yml` and run after deployment completes.
- **Docs Validation**: `.github/workflows/docs.yml`
  - Regenerates OpenAPI via `make api-docs` and fails if `docs/docs/api.html` is out-of-date; builds MkDocs site; uploads site artifact for PR previews.
- **Collaboration**: GitHub Actions status checks enforce build/test/docs freshness; GHCR provides shared, versioned artifacts; compose files document shared deployment semantics.

For the API surface, architecture, endpoint documentation, and ML system design decisions, see [Model Deployment: API](./model_deployment_api.md).

### Deployment Details (VM)
- **Deploy script**: VM `~/run.sh` executes stop → cleanup → prune → pull → up on the `tikkamasalai` folder. It removes known containers (`ollama`, `tikka-backend`, `tikka-frontend`, `ollama-init`, `nginx`, `certbot`), runs `docker system prune -a -f`, then `docker compose pull && docker compose up -d`.
- **Compose file**: `~/tikkamasalai/docker-compose.yml` defines services and volumes (see below).
- **Nginx config**: `~/tikkamasalai/nginx/conf.d/default.conf` mounted into the `nginx` container at `/etc/nginx/conf.d/`. Certbot volumes: `./certbot/www` → `/var/www/certbot`, `./certbot/conf` → `/etc/letsencrypt`.
- **TLS**: Certs under `/etc/letsencrypt/live/tikkamasalai.tech-0001/` used for both app and API server blocks.

### Compose Services (Production)
- `ollama`:
  - Image: `ollama/ollama:latest`
  - Env: `OLLAMA_HOST=0.0.0.0`
  - Volume: `ollama_data:/root/.ollama`
  - Healthcheck: `ollama list`
  - Restart: `unless-stopped`
- `backend`:
  - Image: `ghcr.io/mlops-2526q1-mds-upc/tikka-backend:latest`
  - Env: `OLLAMA_HOST=http://ollama:11434`, `APP_DEBUG=${APP_DEBUG:-false}`
  - Depends on: `ollama` healthy
  - Healthcheck: GET `http://127.0.0.1:8000/health`
  - Restart: `unless-stopped`
- `frontend`:
  - Image: `ghcr.io/mlops-2526q1-mds-upc/tikka-frontend:latest`
  - Ports: `0.0.0.0:8501:8501`
  - Depends on: `backend` healthy
  - Restart: `unless-stopped`
- `ollama-init`:
  - Image: `ollama/ollama:latest`
  - Env: `OLLAMA_HOST=http://ollama:11434`
  - Depends on: `ollama` healthy
  - Entrypoint: `ollama pull gemma3:270m`
  - Restart: `no`
- `nginx`:
  - Image: `nginx:latest`
  - Ports: `80:80`, `443:443`
  - Volumes: `./nginx/conf.d/:/etc/nginx/conf.d/`, `./certbot/www:/var/www/certbot`, `./certbot/conf:/etc/letsencrypt`
  - Depends on: `frontend`
- `certbot`:
  - Image: `certbot/certbot`
  - Volumes: `./certbot/www:/var/www/certbot`, `./certbot/conf:/etc/letsencrypt`
  - Command: renew loop every 12h
- Volumes:
  - `ollama_data: { driver: local }`

### Nginx Server Blocks
- HTTP `:80`: serves `/.well-known/acme-challenge/` from `/var/www/certbot`, redirects all other traffic to HTTPS.
- HTTPS app: domains `tikkamasalai.tech`, `www.tikkamasalai.tech`; proxies `/` to `frontend:8501` with WebSocket headers.
- HTTPS API: domain `api.tikkamasalai.tech`; proxies `/` to `backend:8000` with standard proxy headers.

### Environment Variables Summary
- Backend: `OLLAMA_HOST` (default `http://ollama:11434`), `APP_DEBUG` (default `false`).
- Frontend: none explicitly in compose (configured via application defaults).
- Ollama: `OLLAMA_HOST=0.0.0.0` (service) and `OLLAMA_HOST=http://ollama:11434` (init).
 - Build-time: `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` are passed as secrets for model artifact download during backend image build.

## ML Pipeline Automation
- **Dependency management**: `uv` ensures reproducible Python environments in CI and locally.
- **Model artifacts**: Downloaded during backend image build from GCS (`tikkamasalai-models`) using a service account key passed as a build secret; avoids manual copying.
- **Deployment script**: VM `run.sh` standardizes pull/restart for services, prunes unused images, and validates health checks.

Testing the API endpoints (unit + deployed Bruno tests) is described in [Model Deployment: API](./model_deployment_api.md) and is part of Milestone 4.

## Delivery Checklist
- Container images built (multi-arch) and pushed to GHCR.
- Compose files updated with image tags and envs.
- VM Nginx + Certbot configured for TLS and reverse proxy.
- CI green: unit tests, deployed API tests, docs freshness.
- OpenAPI exported (`docs/docs/api.html`) and linked in docs.