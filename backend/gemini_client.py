# backend/gemini_client.py
"""
Robust Gemini API wrapper (based on google-generativeai).

Features / behavior:
- Reads GEMINI_API_KEY from config.settings (or environment via settings).
- Uses google-generativeai if installed. Handles multiple client method shapes.
- generate_text(...) returns {"content": str, "raw": <raw_response>} for easy consumption.
- embed_texts(list_of_texts, model=...) returns List[List[float]].
- Raises GeminiClientError with clear instructions when dependencies or keys are missing.
"""

from typing import List, Dict, Any, Optional
import os

from config import settings

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    _HAS_GENAI = False


class GeminiClientError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-bison-001"):
        """
        Initialize the Gemini client wrapper.

        :param api_key: Gemini API key. If None, tries to read from settings.GEMINI_API_KEY.
        :param model: Default model to call (Gemini text model name).
        """
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise GeminiClientError(
                "Gemini API key not found. Set GEMINI_API_KEY in your environment or .env (project root)."
            )

        if not _HAS_GENAI:
            raise GeminiClientError(
                "google-generativeai package not installed. Install it with:\n\n"
                "    pip install google-generativeai\n\n"
                "Or implement an alternative adapter in backend/gemini_client.py."
            )

        # configure client (google-generativeai uses genai.configure)
        try:
            # Some versions require genai.configure(api_key=...), some may accept environment-based auth.
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
        Generate text from Gemini and return standardized dict:
            {"content": <string>, "raw": <raw_response_object>}
        The shape of `raw` depends on the installed google-generativeai version.

        :param prompt: The input prompt (string).
        :param temperature: sampling temperature.
        :param max_output_tokens: maximum tokens to generate.
        :param extra_kwargs: forwarded to underlying client call.
        """
        try:
            # Preferred modern helper: genai.generate_text(...)
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    safety_settings=safety_settings,
                    **extra_kwargs,
                )
                # try to extract text in several possible shapes
                content = None
                # 1) resp may be an object with .text
                if hasattr(resp, "text"):
                    content = resp.text
                # 2) resp may be dict-like with 'content' or 'candidates' or 'output'
                elif isinstance(resp, dict):
                    # common fields
                    if "content" in resp and isinstance(resp["content"], str):
                        content = resp["content"]
                    elif "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
                        first = resp["candidates"][0]
                        if isinstance(first, dict) and "content" in first:
                            content = first["content"]
                        elif isinstance(first, str):
                            content = first
                    elif "output" in resp:
                        # some responses nest output
                        output = resp["output"]
                        if isinstance(output, str):
                            content = output
                        elif isinstance(output, dict) and "content" in output:
                            content = output["content"]
                # 3) fallback: try str(resp)
                if content is None:
                    content = str(resp)

                return {"content": content, "raw": resp}
            # Older / alternative API: genai.text.generate(...)
            elif hasattr(genai, "text") and hasattr(genai.text, "generate"):
                resp = genai.text.generate(
                    model=self.model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    **extra_kwargs,
                )
                # common shape: resp.result or resp.output[0].content
                content = None
                if hasattr(resp, "result") and hasattr(resp.result, "output"):
                    try:
                        content = resp.result.output[0].content
                    except Exception:
                        content = str(resp)
                elif isinstance(resp, dict) and "candidates" in resp:
                    c = resp["candidates"][0]
                    content = c.get("content") if isinstance(c, dict) else str(c)
                else:
                    content = str(resp)
                return {"content": content, "raw": resp}
            else:
                raise GeminiClientError("Installed google-generativeai client does not expose a supported text-generation API.")
        except Exception as exc:
            raise GeminiClientError(f"Gemini text generation failed: {exc}")

    def embed_texts(self, texts: List[str], model: str = "embed-text-v1") -> List[List[float]]:
        """
        Create embeddings for a list of texts and return a list of vectors.
        Handles a few variants of the google-generativeai client API.
        """
        if not isinstance(texts, list):
            raise GeminiClientError("embed_texts expects a list of strings.")

        try:
            # Modern style: genai.embeddings.create(...)
            if hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
                resp = genai.embeddings.create(model=model, input=texts)
                # resp.data is typically a list of dicts like {"embedding": [...]}
                embeddings = []
                # support dict-like or object-like resp
                data = None
                if isinstance(resp, dict) and "data" in resp:
                    data = resp["data"]
                elif hasattr(resp, "data"):
                    data = getattr(resp, "data")
                else:
                    data = None

                if data is None:
                    # try alternative shapes
                    raise GeminiClientError("Unexpected embeddings response shape from google-generativeai.")

                for item in data:
                    # item may be dict or object; try dict access then attribute access
                    if isinstance(item, dict):
                        vec = item.get("embedding") or item.get("embedding_vector") or item.get("vector")
                    else:
                        vec = getattr(item, "embedding", None) or getattr(item, "vector", None)
                    if vec is None:
                        raise GeminiClientError("Embedding item missing 'embedding' field in response.")
                    embeddings.append([float(x) for x in vec])
                return embeddings

            # Alternate older API: genai.generate_embeddings(...)
            elif hasattr(genai, "generate_embeddings"):
                resp = genai.generate_embeddings(model=model, input=texts)
                # try dict/object shapes
                data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
                embeddings = []
                for item in data:
                    vec = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                    embeddings.append([float(x) for x in vec])
                return embeddings

            # If nothing is available, raise informative error
            raise GeminiClientError(
                "Installed google-generativeai client does not expose an embeddings API compatible with this wrapper. "
                "Check package version or adapt backend/gemini_client.py for your client."
            )
        except GeminiClientError:
            raise
        except Exception as exc:
            raise GeminiClientError(f"Gemini embedding request failed: {exc}")


# singleton factory
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
