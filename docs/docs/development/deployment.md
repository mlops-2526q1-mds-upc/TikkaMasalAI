# Deployment Strategy

This project deploys a small, containerized stack: backend (FastAPI), frontend (Streamlit), and an Ollama service for LLM helpers. You can run locally with Docker Compose or push images to GHCR for remote environments.

## Prerequisites
- Docker and Docker Compose installed
- Node.js ≥ 20 (for `make api-docs` / Redoc build)
- Bruno CLI (`bru`) on PATH for API tests (optional)
- Python 3.10+ and `uv` for local tooling

## System Architecture & API
- Framework: FastAPI app (`src/backend/app.py`) with CORS and Prometheus metrics.
- Routers under `src/backend/routers/`:
    - `predict.py` (prefix `/predict`):
        - `POST /predict`: Classify uploaded image (multipart `image`).
        - `POST /predict/explain`: Return attention heatmap + prediction metadata.
        - Model + processor lazy-loaded via `@lru_cache`; device auto-selected (MPS → CUDA → CPU).
    - `dashboard.py`: `GET /dashboard` HTML metrics dashboard fed by Prometheus `/metrics`.
    - `llm.py` (prefix `/llm`):
        - `POST /llm/generate`: Generate text from prompt (JSON: `prompt`, `temperature`, `max_tokens`).
        - `GET /llm/health`: Check Ollama availability; returns available models and configured model.
        - Config: `OLLAMA_HOST` (default `http://localhost:11434`), model `gemma3:270m`.
- OpenAPI: Exportable via `src/backend/export_schema.py` to `src/backend/openapi.json` and published to docs.
- Observability: `prometheus_fastapi_instrumentator` exposes metrics; CORS configured for local dev and `tikkamasalai.tech`.

## Images and Tagging
For per-service Dockerfiles, Compose service definitions, registry images, and CI/CD details, see Containers → [Compose & CI/CD](./containers.md).

## Model Artifacts (GCS)
- Bucket: `tikkamasalai-models` (location: `eu`, multi‑region).
- Artifacts are downloaded during backend image build via `src/backend/download_model.py`.
- Build secrets provide GCP credentials (`GOOGLE_APPLICATION_CREDENTIALS`); `GOOGLE_CLOUD_PROJECT` is passed at build time.
- Files are synced to `./models` in the image. This avoids runtime stalls and ensures portable builds.

## Environments
Environment-specific Compose usage and image sources are documented in Containers → [Compose & CI/CD](./containers.md). In brief:
- Development: `docker-compose-local.yml` builds backend/frontend locally and mounts frontend secrets.
- Staging/Production: `docker-compose.yml` uses prebuilt GHCR images.

### Secrets and Configuration
- Frontend secrets: mount `./src/frontend/.streamlit/secrets.toml` into `/app/.streamlit/secrets.toml` (read-only) for local; use platform secret stores for remote.
- Backend runtime variables:
    - `APP_DEBUG=false` for production-like runs.
    - `OLLAMA_HOST` must point to internal Ollama service (e.g., `http://ollama:11434` under Compose).

## Cloud Deployment (VM on GCP)
- Provider: GCP VM hosting backend and frontend; Ollama runs as a service.
- VM spec:
    - Location: `europe-west1-c` (region `europe-west1`).
    - Machine type: `c4a-standard-2` (2 vCPUs, 8 GB RAM).
    - CPU platform/Architecture: Google Axion, Arm64 (chosen to align with Apple Silicon dev machines and early non-multi-arch builds).
    - OS image: `debian-12-bookworm-arm64-v20251014`.
    - Boot disk: 20 GB Hyperdisk Balanced with daily schedule; additional disk attached to store both the image classification model and LLM.
- Networking & TLS: Nginx reverse proxy fronts `:443`; Certbot manages TLS; routes `tikkamasalai.tech` to frontend and `api.tikkamasalai.tech` to backend.

## Rollout with Docker Compose (Server)
On a remote host with Docker and Compose installed:
1) Copy `docker-compose.yml` (and optionally `docker-compose.override.yml`).
2) Configure environment (`.env`) and secrets.
3) Start:
     docker compose up -d
4) Verify health:
     - Backend: `curl http://<host>:8000/health`
     - Frontend: open `http://<host>:8501`

## Health Checks and Monitoring
Compose healthchecks and startup ordering are detailed in [Containers](./containers.md). For quick verification after rollout, use `make compose-logs` (or `docker compose logs -f`).

## Testing After Deployment
- Quick checks:
    - Backend health: `curl http://<host>:8000/health` → 200
    - Frontend: available at `http://<host>:8501`
- Full guidance: Development → [Testing](./testing.md)
    - Unit tests: `make test`
    - API tests (Bruno): `make test-local-api`, `make test-deployed-api`

## Security & Compliance
- Data handling: Images uploaded via multipart form; processed transiently in-memory; no persistence by default.
- Access control: Public API at `https://api.tikkamasalai.tech` (demo scope).
- CORS: Origins allowed for local and production domains.
- Secrets: Follow `CONTRIBUTING.md`; never commit credentials. Use env vars or secret stores.
- Future hardening: Consider HTTPS enforcement details (already via Nginx), rate limiting, and auth if usage expands.

## Adoption & Developer Experience
- Ease of use: Simple multipart upload; predictable JSON responses; OpenAPI published via `make api-docs`.
- Versioning: Routes are currently unversioned (e.g., `/predict`). Versioning the API is a potential future enhancement.

## See Also
- Containers & Compose details: [Containers](./containers.md)
- CI/CD, multi-arch builds, registry, and VM rollout: [Component Delivery](./containers.md)