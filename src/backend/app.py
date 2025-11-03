import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.routers import llm, predict

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

# Include routers
app.include_router(predict.router)
app.include_router(llm.router)


@app.get("/")
def read_root():
    return {"status": "TikkaMasalAI Backend is running."}


@app.get("/health")
def health_check():
    """Lightweight health endpoint for container orchestration.

    Returns 200 OK when the app is up and routers are registered.
    """
    return {"status": "ok"}
