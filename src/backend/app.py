import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict

from src.backend.routers import dashboard, llm, predict

# Debug toggle via env var
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

# Basic logging config (uvicorn sets its own handlers; this complements them)
_log_level = logging.DEBUG if APP_DEBUG else logging.INFO
logging.basicConfig(level=_log_level)

app = FastAPI(title="TikkaMasalAI Backend", debug=APP_DEBUG)

origins = [
    "http://localhost:3000",
    "http://tikkamasalai.tech",
    "https://tikkamasalai.tech",
    "http://localhost:8501",  # Streamlit
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(predict.router)
app.include_router(llm.router)
app.include_router(dashboard.router)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})


@app.get("/", response_model=StatusResponse)
def read_root() -> StatusResponse:
    return {"status": "TikkaMasalAI Backend is running."}


@app.get("/health", response_model=StatusResponse)
def health_check() -> StatusResponse:
    """Lightweight health endpoint for container orchestration.

    Returns 200 OK when the app is up and routers are registered.
    """
    return {"status": "ok"}
