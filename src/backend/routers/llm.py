"""LLM router for text generation using Ollama."""

import os
from typing import Any

from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/llm", tags=["llm"])

# Ollama API endpoint
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "gemma3:270m"


class LLMRequest(BaseModel):
    """Request model for LLM generation."""

    prompt: str = Field(examples=["Write a haiku about masala dosa."])
    temperature: float = Field(default=0.7, ge=0, le=2, examples=[0.7])
    max_tokens: int = Field(default=500, ge=1, le=4096, examples=[200])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Write a haiku about masala dosa.",
                "temperature": 0.7,
                "max_tokens": 200,
            }
        }
    )


class LLMResponse(BaseModel):
    """Response model for LLM generation."""

    response: str = Field(
        examples=["Golden crisp dosa,\nSpiced potato dreams within,\nCurry-scented dawn."]
    )
    model: str = Field(examples=["gemma3:270m"])
    done: bool = Field(default=False, examples=[True])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Golden crisp dosa,\nSpiced potato dreams within,\nCurry-scented dawn.",
                "model": "gemma3:270m",
                "done": True,
            }
        }
    )


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Ollama service is not running. Please start Ollama first."])


class LLMHealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    ollama_url: str = Field(examples=["http://localhost:11434"])
    available_models: list[str] = Field(examples=[["gemma3:270m", "llama3.1:8b"]])
    configured_model: str = Field(examples=["gemma3:270m"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "ollama_url": "http://localhost:11434",
                "available_models": ["gemma3:270m", "llama3.1:8b"],
                "configured_model": "gemma3:270m",
            }
        }
    )


@router.post(
    "/generate",
    summary="Generate text from prompt",
    description=(
        "Generate text with the configured Ollama model.\n\n"
        "POST a JSON body with `prompt`, optional `temperature` and `max_tokens`.\n"
        "Returns the generated text and the model used."
    ),
    response_model=LLMResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Ollama not running or model missing"},
        504: {"model": ErrorResponse, "description": "Ollama timeout"},
        500: {"model": ErrorResponse, "description": "Unexpected error"},
    },
)
async def generate_text(request: LLMRequest) -> dict[str, Any]:
    """Generate text using Ollama LLM.

    Args:
        request: LLMRequest containing prompt and generation parameters.

    Returns:
        Dictionary containing the generated text and model name.

    Raises:
        HTTPException: If Ollama service is not available or generation fails.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Make request to Ollama API
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{MODEL_NAME}' not found. Please pull the model first using: ollama pull {MODEL_NAME}",
                )

            response.raise_for_status()
            result = response.json()

            return {
                "response": result.get("response", ""),
                "model": result.get("model", MODEL_NAME),
                "done": result.get("done", False),
            }

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is not running. Please start Ollama first.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The prompt may be too long or the model is taking too long to respond.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get(
    "/health",
    summary="Ollama service health",
    description=(
        "Check whether the local Ollama service is reachable and return a list of"
        " available models plus the configured model name.\n\n"
        "Returns a JSON object with `status`, `ollama_url`, `available_models` and `configured_model`."
    ),
    response_model=LLMHealthResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Ollama not running"},
        500: {"model": ErrorResponse, "description": "Unexpected error"},
    },
)
async def check_ollama_health() -> dict[str, Any]:
    """Check if Ollama service is running and accessible.

    Returns:
        Dictionary containing health status and available models.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            return {
                "status": "healthy",
                "ollama_url": OLLAMA_BASE_URL,
                "available_models": [m.get("name") for m in models],
                "configured_model": MODEL_NAME,
            }

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is not running. Please start Ollama first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}",
        )
