"""LLM router for text generation using Ollama."""

import os
from typing import Any

from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel

router = APIRouter(prefix="/llm", tags=["llm"])

# Ollama API endpoint
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "gemma3:270m"


class LLMRequest(BaseModel):
    """Request model for LLM generation."""

    prompt: str
    temperature: float = 0.7
    max_tokens: int = 500


class LLMResponse(BaseModel):
    """Response model for LLM generation."""

    response: str
    model: str


@router.post(
    "/generate",
    summary="Generate text from prompt",
    description=(
        "Generate text with the configured Ollama model.\n\n"
        "POST a JSON body with `prompt`, optional `temperature` and `max_tokens`.\n"
        "Returns the generated text and the model used."
    ),
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
