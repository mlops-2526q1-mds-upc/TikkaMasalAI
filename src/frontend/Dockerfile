FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install uv

# Copy dependency manifest (from project root) for better caching
COPY pyproject.toml /app/pyproject.toml

# Install deps (uses the root pyproject)
RUN uv sync --only-group frontend

# Copy only the frontend code
COPY src/frontend ./app/frontend

EXPOSE 8501

# No need for PYTHONPATH if we run directly from /app/frontend
CMD ["uv", "run", "streamlit", "run", "app/frontend/main.py"]
