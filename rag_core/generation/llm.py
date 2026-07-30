"""
Gemini LLM via LlamaIndex GoogleGenAI wrapper.
"""

from functools import lru_cache

from llama_index.llms.google_genai import GoogleGenAI

from rag_core.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE


@lru_cache(maxsize=1)
def get_llm() -> GoogleGenAI:
    """
    LlamaIndex LLM wrapper around Gemini.

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
    Default model: gemini-3-flash-preview (override with GEMINI_MODEL).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
            "Export it before running generation."
        )
    return GoogleGenAI(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=LLM_TEMPERATURE,
    )