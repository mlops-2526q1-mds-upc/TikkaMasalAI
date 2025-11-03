FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install uv

COPY pyproject.toml /app/pyproject.toml

RUN uv sync --only-group backend

COPY src ./src
COPY models ./models

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["uv", "run", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]