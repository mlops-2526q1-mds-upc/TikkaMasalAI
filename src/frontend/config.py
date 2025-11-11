import os

import streamlit as st

DEV_BASE_URL = "http://127.0.0.1:8000/"
PROD_BASE_URL = "http://34.79.200.186:8000/"

URL = PROD_BASE_URL

DEFAULT_API_URL = os.path.join(URL, "predict")
DEFAULT_LLM_URL = os.path.join(URL, "llm/generate")
DEFAULT_EXPLAIN_URL = os.path.join(URL, "predict/explain")


def get_api_url() -> str:
    """Return the API endpoint URL used by the frontend.

    Priority of sources (highest first):
    1. Query parameter "api_url" (useful for local testing)
    2. Streamlit secrets["api_url"] (for deployments)
    3. Project default (`DEFAULT_API_URL`)

    Returns:
        str: Fully qualified API URL to call for predictions.
    """
    # Prioritize secrets so deployments can configure without code changes.
    try:
        api_url = st.secrets.get("api_url", DEFAULT_API_URL)
    except Exception:
        api_url = DEFAULT_API_URL
    # Allow quick overrides via query param for local testing.
    api_override = st.query_params.get("api_url")
    if isinstance(api_override, list) and api_override:
        api_url = api_override[-1]
    elif isinstance(api_override, str) and api_override:
        api_url = api_override
    return api_url


def get_llm_url() -> str:
    """Return the LLM endpoint URL used by the frontend.

    Priority of sources (highest first):
    1. Query parameter "llm_url"
    2. Streamlit secrets["llm_url"]
    3. Project default (`DEFAULT_LLM_URL`)

    Returns:
        str: Fully qualified LLM URL to call for text generation.
    """
    try:
        llm_url = st.secrets.get("llm_url", DEFAULT_LLM_URL)
    except Exception:
        llm_url = DEFAULT_LLM_URL
    llm_override = st.query_params.get("llm_url")
    if isinstance(llm_override, list) and llm_override:
        llm_url = llm_override[-1]
    elif isinstance(llm_override, str) and llm_override:
        llm_url = llm_override
    return llm_url


def get_explain_url() -> str:
    """Return the explanation endpoint URL used by the frontend.

    Priority of sources (highest first):
    1. Query parameter "explain_url"
    2. Streamlit secrets["explain_url"]
    3. Project default (`DEFAULT_EXPLAIN_URL`)

    Returns:
        str: Fully qualified URL to call for heatmaps/overlays.
    """
    try:
        explain_url = st.secrets.get("explain_url", DEFAULT_EXPLAIN_URL)
    except Exception:
        explain_url = DEFAULT_EXPLAIN_URL
    explain_override = st.query_params.get("explain_url")
    if isinstance(explain_override, list) and explain_override:
        explain_url = explain_override[-1]
    elif isinstance(explain_override, str) and explain_override:
        explain_url = explain_override
    return explain_url
