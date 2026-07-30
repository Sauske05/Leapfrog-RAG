"""
Gemini LLM via LlamaIndex GoogleGenAI wrapper.
"""

from llama_index.llms.google_genai import GoogleGenAI

from rag_core.runtime import get_runtime


def get_llm() -> GoogleGenAI:
    """
    LlamaIndex LLM wrapper around Gemini.

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
    Default model: gemini-3-flash-preview (override with GEMINI_MODEL).
    """
    return get_runtime().llm