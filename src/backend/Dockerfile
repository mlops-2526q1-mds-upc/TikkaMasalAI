FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install uv

COPY pyproject.toml /app/pyproject.toml

RUN uv sync --only-group backend

COPY src ./src

# Download models from GCP
# Expects GOOGLE_APPLICATION_CREDENTIALS to be mounted or credentials passed at build time
ARG GOOGLE_CLOUD_PROJECT=academic-torch-476716-h3
ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}

RUN --mount=type=secret,id=gcp_credentials,target=/tmp/gcp-key.json \
    GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json \
    uv run python src/backend/download_model.py

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["uv", "run", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]