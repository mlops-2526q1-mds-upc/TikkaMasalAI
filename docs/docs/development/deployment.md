# Deployment strategy

This project deploys as a small stack of containers: ollama, backend (FastAPI), and frontend (Streamlit). You can run locally with Compose or push images to a registry (GHCR) for remote environments.

## Images and tagging

- Frontend Dockerfile: `src/frontend/Dockerfile`
- Backend Dockerfile: `src/backend/Dockerfile`
- Local builds (no push):
    - `make build-backend-docker`
    - `make build-frontend-docker`
- Run locally (single service):
    - `make run-backend-docker` (binds 8000)
    - `make run-frontend-docker` (binds 8501)

#### Registry
- GHCR namespace: ghcr.io/mlops-2526q1-mds-upc
- Push latest multi-arch (arm64) images:
    - make push-backend-docker
    - make push-frontend-docker

## Model Storage with Google Cloud Storage

The backend Docker image downloads models from Google Cloud Storage (GCS) during the build process. This ensures models are baked into the image for easy distribution.

#### Building Backend with Models

**Prerequisites:**
1. Authenticate with GCP: `gcloud auth application-default login`
2. GitHub Personal Access Token with `write:packages` scope for pushing to GHCR

**Build and push backend image:**
```bash
# Build with models from GCS
docker build -t ghcr.io/mlops-2526q1-mds-upc/tikka-backend:latest \
  --build-arg GOOGLE_CLOUD_PROJECT=academic-torch-476716-h3 \
  --secret id=gcp_credentials,src=$HOME/.config/gcloud/application_default_credentials.json \
  -f src/backend/Dockerfile .

# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Push to registry
docker push ghcr.io/mlops-2526q1-mds-upc/tikka-backend:latest
```

**Notes:**
- GCP credentials are used only during build and are NOT stored in the image
- Models are downloaded from `gs://tikkamasalai-models` bucket
- The pushed image contains models but no credentials
- Teammates can pull the image without GCP access

#### For Team Members

**To pull and use the pre-built image (no GCP needed):**
```bash
docker pull ghcr.io/mlops-2526q1-mds-upc/tikka-backend:latest
docker compose up -d
```

**To build the image yourself (requires GCP access):**
1. Get GCP credentials from team lead
2. Run `gcloud auth application-default login`
3. Follow build steps above

## Environments
- Development: docker-compose-local.yml builds images locally and mounts secrets for the frontend.
- Staging/Production: docker-compose.yml uses prebuilt images from GHCR. 

#### Secrets and configuration
- Frontend secrets.toml can be mounted in Compose. For remote deployments, provide secrets via platform-specific secret stores or inject a mounted file.
- Backend runtime variables:
  - APP_DEBUG=false for production
  - OLLAMA_HOST must point to the internal ollama service (http://ollama:11434 when using Compose)

## Rollout with Docker Compose (server)

On a remote host with Docker and Compose installed:
1) Copy docker-compose.yml (and optionally docker-compose.override.yml)
2) Configure environment (.env) and secrets
3) Start:
   docker compose up -d
4) Verify health:
   - Backend: curl http://<host>:8000/health
   - Frontend: open http://<host>:8501

## Health checks and monitoring

- Compose defines healthchecks for ollama and backend; frontend depends on a healthy backend.
- Logs: use make compose-logs (or docker compose logs -f) to stream logs.

## Testing after deployment

#### Quick checks
- **Backend Health**: curl http://<host>:8000/health should return 200
- **Frontend**: open http://<host>:8501

#### Unit tests
- Run locally before shipping: make test (see Development → Testing for details)

#### API tests (Bruno)
- The request collection lives in tikkamasalai-requests/
- For a deployed environment, configure tikkamasalai-requests/environments/production.bru with your base URL and secrets, then run:
    - make test-deployed-api
- For local stacks:
    - make test-local-api

See Development → Testing for more details and CLI examples.