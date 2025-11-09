# backend/gemini_client.py
"""
Simple Gemini API wrapper.

This module tries to use the `google-generativeai` package if installed.
It provides:
  - GeminiClient.generate_text(prompt, **kwargs)
  - GeminiClient.embed_texts(list_of_texts)

Notes:
  - Make sure GEMINI_API_KEY is set in your environment (or .env).
  - If the `google-generativeai` package is not installed or no API key is provided,
    the client will raise a clear error explaining how to fix it.
  - You can extend this wrapper (streaming support, chat, retry/backoff, etc.)
    as needed for your project.
"""

import os
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    _HAS_GENAI = False

from config import settings


class GeminiClientError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-bison-001"):
        """
        Initialize the Gemini client wrapper.

        :param api_key: Gemini API key. If None, tries to read from settings.GEMINI_API_KEY.
        :param model: Default model to call (Gemini text model name). Keep as-is or override.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model

        if not self.api_key:
            raise GeminiClientError(
                "Gemini API key not found. Set GEMINI_API_KEY in your environment or .env."
            )

        if not _HAS_GENAI:
            raise GeminiClientError(
                "google-generativeai package not installed. Install it with:\n\n"
                "    pip install google-generativeai\n\n"
                "Or, if you prefer, implement an alternative wrapper here."
            )

        # configure client (google-generativeai uses genai.configure)
        try:
            genai.configure(api_key=self.api_key)
        except Exception as exc:
            raise GeminiClientError(f"Failed to configure google-generativeai client: {exc}")

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from Gemini.

        Returns the raw response dict from the google.generativeai client.

        :param prompt: The input prompt (string).
        :param temperature: sampling temperature.
        :param max_output_tokens: maximum tokens to generate.
        :param top_k: optional top_k sampling.
        :param top_p: optional top_p sampling.
        :param safety_settings: optional safety settings per Google API.
        :param extra_kwargs: forwarded to underlying client call.
        """
        try:
            # The google-generativeai library exposes a generate_text helper.
            # Using the "model" identifier (e.g., "models/text-bison-001" or "models/chat-bison-001").
            response = genai.generate_text(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_k=top_k,
                top_p=top_p,
                safety_settings=safety_settings,
                **extra_kwargs,
            )
            return response
        except Exception as exc:
            # surface a friendly error
            raise GeminiClientError(f"Gemini text generation failed: {exc}")

    def embed_texts(self, texts: List[str], model: str = "embed-text-v1") -> List[List[float]]:
        """
        Create embeddings for a list of texts.

        Note: Depending on the google-generativeai version and your account, embedding
        APIs and model names may differ. If your account uses a different path, update the model param.

        :param texts: list of strings to embed
        :param model: embedding model name (default "embed-text-v1")
        :return: list of embedding vectors (list of floats)
        """
        try:
            # The exact call for embeddings may differ by version.
            # google-generativeai typically provides an embeddings API via genai.generate_embeddings or similar.
            if hasattr(genai, "embeddings"):
                # newer client style: genai.embeddings.create(...)
                resp = genai.embeddings.create(model=model, input=texts)
                # expected format: resp.data is a list of {embedding: [...]}
                embeddings = [item["embedding"] for item in resp["data"]]
                return embeddings
            elif hasattr(genai, "generate_embeddings"):
                # fall back if method name differs
                resp = genai.generate_embeddings(model=model, input=texts)
                embeddings = [item["embedding"] for item in resp["data"]]
                return embeddings
            else:
                # Try a generic generate_text-based fallback (not ideal)
                raise GeminiClientError(
                    "Installed google-generativeai client does not expose a known embeddings API. "
                    "Check package version or implement an embedding fallback (e.g., sentence-transformers)."
                )
        except Exception as exc:
            raise GeminiClientError(f"Gemini embedding request failed: {exc}")


# Helper factory to get a configured client easily
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
