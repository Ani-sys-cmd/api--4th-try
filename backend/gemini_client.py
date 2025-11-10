"""
Hardcoded-local Gemini client wrapper (local-dev convenience).

- Paste your local GEMINI API key in HARDCODED_GEMINI_API_KEY below.
- This wrapper tolerates SDK signature differences and prefers safer call forms
  (client.generate_text -> models.generate_text -> models.generate_content).
- It will try to discover a usable model via client.models.list() and cache it.
- It retries calls removing unsupported kwargs if an SDK raises an unexpected-kw TypeError.
- Returns {"content": str, "raw": <raw_response>, "provider": <sdk_name>}
- Intended for local development only. Rotate/revoke key if it was exposed.
"""

from typing import Any, Dict, List, Optional, Sequence, Callable
import os
import re
import logging

# ---------- EDIT THIS (local-only) ----------
HARDCODED_GEMINI_API_KEY = "AIzaSyDAgrB9j3gWi6QawMeA3dmEkMrYpWLZb40"
# --------------------------------------------

logger = logging.getLogger("backend.gemini_client")
if os.getenv("GEMINI_CLIENT_DEBUG", "").lower() in ("1", "true", "yes", "on"):
    logger.setLevel(logging.DEBUG)

# Attempt to load same settings style as rest of app (but prefer the hardcoded key)
try:
    from config import settings  # type: ignore
except Exception:
    try:
        from backend.config import settings  # type: ignore
    except Exception:
        settings = None  # type: ignore

# Try modern google.genai
_HAS_GOOGLE_GENAI = False
_GOOGLE_GENAI = None
try:
    import google.genai as google_genai  # type: ignore
    _GOOGLE_GENAI = google_genai
    _HAS_GOOGLE_GENAI = True
except Exception:
    _HAS_GOOGLE_GENAI = False
    _GOOGLE_GENAI = None

# Try legacy google.generativeai as fallback
_HAS_GENAI_LEGACY = False
_GENAI_LEGACY = None
try:
    if not _HAS_GOOGLE_GENAI:
        import google.generativeai as genai_legacy  # type: ignore
        _GENAI_LEGACY = genai_legacy
        _HAS_GENAI_LEGACY = True
except Exception:
    _HAS_GENAI_LEGACY = False
    _GENAI_LEGACY = None


class GeminiClientError(RuntimeError):
    pass


def _get_hardcoded_key() -> Optional[str]:
    # priority: hardcoded constant > settings > env
    if HARDCODED_GEMINI_API_KEY and HARDCODED_GEMINI_API_KEY.strip() and HARDCODED_GEMINI_API_KEY != "YOUR_LOCAL_KEY_GOES_HERE":
        return HARDCODED_GEMINI_API_KEY.strip()
    try:
        if settings:
            v = getattr(settings, "GEMINI_API_KEY", None)
            if v:
                return v
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _extract_content(resp: Any) -> str:
    """Heuristics to extract text from response objects/dicts."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        # common simple keys
        for k in ("content", "text", "output", "message"):
            if k in resp and isinstance(resp[k], str):
                return resp[k]
        # list-like outputs
        for k in ("candidates", "outputs", "choices", "responses"):
            v = resp.get(k)
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict):
                    for fk in ("content", "text"):
                        if fk in first and isinstance(first[fk], str):
                            return first[fk]
                elif isinstance(first, str):
                    return first
        return str(resp)
    # object-like (clients often produce objects)
    try:
        if hasattr(resp, "candidates") and getattr(resp, "candidates"):
            first = getattr(resp, "candidates")[0]
            if hasattr(first, "content"):
                return str(first.content)
        if hasattr(resp, "outputs") and getattr(resp, "outputs"):
            out0 = getattr(resp, "outputs")[0]
            # object may have .content or .text
            if hasattr(out0, "content"):
                return str(out0.content)
            if hasattr(out0, "text"):
                return str(out0.text)
    except Exception:
        pass
    try:
        return str(resp)
    except Exception:
        return ""


# Keys we will forward to SDK calls when present (expanded to include common variants)
_FORWARD_KEYS = (
    "temperature",
    "max_tokens",
    "max_new_tokens",
    "max_output_tokens",
    "top_p",
    "top_k",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "candidate_count",
)


def _gather_forward_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict of recognized generation kwargs from the caller kwargs."""
    out: Dict[str, Any] = {}
    for k in _FORWARD_KEYS:
        if k in kwargs:
            out[k] = kwargs[k]
    return out


def _resolve_prompt_from_args(prompt: Optional[str], args: Sequence[Any], kwargs: Dict[str, Any]) -> str:
    """
    Resolve a textual prompt from various possible caller conventions.
    Always return a string (empty string if none found).
    """
    # 1. explicit prompt or positional first arg
    if prompt:
        return prompt
    if args:
        first = args[0]
        if isinstance(first, str):
            return first
    # 2. common kw names
    for key in ("prompt", "input", "text", "message", "contents", "content"):
        if key in kwargs:
            val = kwargs.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return "\n".join(val)
    # 3. chat-style messages
    msgs = kwargs.get("messages") or kwargs.get("conversation") or kwargs.get("messages_list")
    if isinstance(msgs, str):
        return msgs
    if isinstance(msgs, list) and msgs:
        parts: List[str] = []
        for m in msgs:
            if isinstance(m, str):
                parts.append(m)
            elif isinstance(m, dict):
                if "content" in m and isinstance(m["content"], str):
                    parts.append(m["content"])
                elif "text" in m and isinstance(m["text"], str):
                    parts.append(m["text"])
                else:
                    for v in m.values():
                        if isinstance(v, str):
                            parts.append(v)
                            break
        if parts:
            return "\n".join(parts)
    # 4. contents as list of dicts -> extract text fields
    contents = kwargs.get("contents")
    if isinstance(contents, list) and contents:
        parts = []
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                if "text" in c and isinstance(c["text"], str):
                    parts.append(c["text"])
                elif "content" in c and isinstance(c["content"], str):
                    parts.append(c["content"])
        if parts:
            return "\n".join(parts)
    return ""


# helper to attempt a call, removing unexpected-kw args reported in TypeError messages
def call_with_kw_retry(fn: Callable, /, *args, **kwargs):
    """
    Call fn(*args, **kwargs). If a TypeError arises complaining about an unexpected
    keyword argument 'X', remove X from kwargs and retry. Repeat until success
    or no unexpected kw can be inferred. Log removed kwargs at DEBUG level.
    """
    last_exc = None
    current_kwargs = dict(kwargs)
    unexpected_kw_pattern = re.compile(r"unexpected keyword argument '([^']+)'")
    removed_keys: List[str] = []
    while True:
        try:
            if removed_keys:
                logger.debug("Retrying %s after removing unsupported kwargs: %s", getattr(fn, '__name__', str(fn)), removed_keys)
            return fn(*args, **current_kwargs)
        except TypeError as e:
            last_exc = e
            msg = str(e)
            m = unexpected_kw_pattern.search(msg)
            if not m:
                # cannot parse unexpected kw, re-raise
                logger.debug("TypeError during call to %s and couldn't parse unexpected kw: %s", getattr(fn, '__name__', str(fn)), msg)
                raise
            bad_key = m.group(1)
            # If the detected bad_key exists in our current kwargs, drop it and retry
            if bad_key in current_kwargs:
                current_kwargs.pop(bad_key)
                removed_keys.append(bad_key)
                logger.debug("Removed unsupported kw '%s' for %s and will retry (remaining kwargs: %s)", bad_key, getattr(fn, '__name__', str(fn)), list(current_kwargs.keys()))
                continue
            else:
                # The bad key wasn't supplied by us — can't recover
                logger.debug("Unexpected kw '%s' reported but was not in current kwargs; aborting.", bad_key)
                raise
    if last_exc:
        raise last_exc


def _choose_model_from_client(client: Any) -> Optional[str]:
    """
    Try to discover a usable model id from client.models.list() if available.
    Returns first model id/name found or None.
    """
    try:
        if hasattr(client, "models") and hasattr(client.models, "list"):
            resp = client.models.list()
            # resp might be a dict or object; try to extract model list
            models_list = None
            if isinstance(resp, dict):
                models_list = resp.get("models") or resp.get("model") or resp.get("modelInfos")
            else:
                models_list = getattr(resp, "models", None) or getattr(resp, "model", None)
            if isinstance(models_list, list) and models_list:
                # pick the first model identifier-like field
                first = models_list[0]
                if isinstance(first, dict):
                    for key in ("name", "id", "model", "modelId"):
                        if key in first and isinstance(first[key], str):
                            logger.debug("Discovered model via models.list(): %s", first[key])
                            return first[key]
                elif isinstance(first, str):
                    logger.debug("Discovered model via models.list(): %s", first)
                    return first
    except Exception as e:
        logger.debug("Could not list models to auto-discover model: %s", e)
    return None


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or _get_hardcoded_key()
        if not self.api_key:
            raise GeminiClientError(
                "No API key available. Set HARDCODED_GEMINI_API_KEY in this file or GEMINI_API_KEY in env/.env."
            )
        # prefer explicit env override; otherwise don't force a possibly-unsupported default
        self.default_model = default_model or os.getenv("GEMINI_MODEL") or None
        # cache chosen model if discovered
        self._discovered_model: Optional[str] = None

        # configure legacy if present (best-effort)
        if _HAS_GENAI_LEGACY and _GENAI_LEGACY is not None:
            try:
                if hasattr(_GENAI_LEGACY, "configure"):
                    _GENAI_LEGACY.configure(api_key=self.api_key)
            except Exception:
                pass

    def _get_or_discover_model(self, client_obj: Any) -> Optional[str]:
        # priority: explicit env/default > cached discovery > attempt discovery
        if self.default_model:
            return self.default_model
        if self._discovered_model:
            return self._discovered_model
        # try to discover via client.models.list()
        m = _choose_model_from_client(client_obj)
        if m:
            self._discovered_model = m
            return m
        return None

    def generate_text(self, prompt: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Robust generate_text accepting flexible caller signatures.

        - Resolves prompt from many patterns
        - Forwards a conservative set of kwargs (see _FORWARD_KEYS)
        - If an SDK rejects a forwarded kw, it is removed and the call is retried.
        - Prefers client-level generate_text (safer) -> models.generate_text -> models.generate_content
        """
        last_exc: Optional[Exception] = None

        resolved_prompt = _resolve_prompt_from_args(prompt, args, kwargs)
        sdk_forward_kwargs = _gather_forward_kwargs(kwargs)

        if sdk_forward_kwargs:
            logger.debug("Will attempt to forward generation kwargs: %s", sdk_forward_kwargs)

        # 1) Try modern google.genai minimal calls (PREFER client.generate_text first)
        if _HAS_GOOGLE_GENAI and _GOOGLE_GENAI is not None:
            try:
                try:
                    client = _GOOGLE_GENAI.Client(api_key=self.api_key) if hasattr(_GOOGLE_GENAI, "Client") else _GOOGLE_GENAI
                except TypeError:
                    client = _GOOGLE_GENAI.Client()
                except Exception:
                    client = _GOOGLE_GENAI

                chosen_model = self._get_or_discover_model(client)

                # (A) Prefer client-level generate_text if available (safer across API versions)
                try:
                    if hasattr(client, "generate_text"):
                        call_kwargs = dict(sdk_forward_kwargs)  # conservative options
                        if chosen_model:
                            call_kwargs["model"] = chosen_model
                        try:
                            resp = call_with_kw_retry(client.generate_text, prompt=resolved_prompt, **call_kwargs)
                        except TypeError:
                            # fallback to positional
                            resp = call_with_kw_retry(client.generate_text, resolved_prompt, **call_kwargs)
                        return {"content": _extract_content(resp), "raw": resp, "provider": "google.genai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("client.generate_text attempt failed: %s", e)

                # (B) Try models.generate_text(model=..., input=prompt) if present
                try:
                    if hasattr(client, "models") and hasattr(client.models, "generate_text"):
                        call_kwargs = {}
                        if chosen_model:
                            call_kwargs["model"] = chosen_model
                        call_kwargs["input"] = resolved_prompt
                        call_kwargs.update(sdk_forward_kwargs)
                        resp = call_with_kw_retry(client.models.generate_text, **call_kwargs)
                        return {"content": _extract_content(resp), "raw": resp, "provider": "google.genai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("client.models.generate_text attempt failed: %s", e)

                # (C) Finally try models.generate_content(model=..., contents=[...])
                try:
                    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                        # Many SDK variants accept a list of strings for contents.
                        contents = [resolved_prompt]
                        call_kwargs = {}
                        if chosen_model:
                            call_kwargs["model"] = chosen_model
                        call_kwargs["contents"] = contents
                        call_kwargs.update(sdk_forward_kwargs)
                        resp = call_with_kw_retry(client.models.generate_content, **call_kwargs)
                        return {"content": _extract_content(resp), "raw": resp, "provider": "google.genai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("client.models.generate_content attempt failed: %s", e)

            except Exception as e_outer:
                last_exc = e_outer
                logger.debug("Error while trying google.genai paths: %s", e_outer)

        # 2) Try legacy google.generativeai minimal calls
        if _HAS_GENAI_LEGACY and _GENAI_LEGACY is not None:
            try:
                try:
                    chosen_model = self.default_model or None
                    if hasattr(_GENAI_LEGACY, "generate_text"):
                        call_kwargs = {"prompt": resolved_prompt}
                        if chosen_model:
                            call_kwargs["model"] = chosen_model
                        call_kwargs.update(sdk_forward_kwargs)
                        resp = call_with_kw_retry(_GENAI_LEGACY.generate_text, **call_kwargs)
                        return {"content": _extract_content(resp), "raw": resp, "provider": "google.generativeai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("legacy generate_text attempt failed: %s", e)

                try:
                    if hasattr(_GENAI_LEGACY, "text") and hasattr(_GENAI_LEGACY.text, "generate"):
                        call_kwargs = {"input": resolved_prompt}
                        if self.default_model:
                            call_kwargs["model"] = self.default_model
                        call_kwargs.update(sdk_forward_kwargs)
                        resp = call_with_kw_retry(_GENAI_LEGACY.text.generate, **call_kwargs)
                        return {"content": _extract_content(resp), "raw": resp, "provider": "google.generativeai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("legacy text.generate attempt failed: %s", e)

                try:
                    if hasattr(_GENAI_LEGACY, "GenerativeModel"):
                        gm = _GENAI_LEGACY.GenerativeModel(self.default_model or "")
                        if hasattr(gm, "generate_text"):
                            resp = gm.generate_text(resolved_prompt)
                            return {"content": _extract_content(resp), "raw": resp, "provider": "google.generativeai"}
                except Exception as e:
                    last_exc = e
                    logger.debug("legacy GenerativeModel attempt failed: %s", e)

            except Exception as e_outer:
                last_exc = e_outer
                logger.debug("Error while trying legacy genai paths: %s", e_outer)

        msg = "No supported GenAI text-generation signature succeeded."
        if last_exc:
            msg += f" Last error: {last_exc}"
        msg += " For local dev either ensure google-genai is installed or adapt this wrapper."
        raise GeminiClientError(msg)

    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Minimal embeddings: tries modern and legacy minimal calls.
        Returns list of vectors.
        """
        if not isinstance(texts, list):
            raise GeminiClientError("embed_texts expects a list of strings.")
        chosen = model or os.getenv("GEMINI_EMBED_MODEL") or "embed-text-v1"
        last_exc: Optional[Exception] = None

        if _HAS_GOOGLE_GENAI and _GOOGLE_GENAI is not None:
            try:
                client = _GOOGLE_GENAI.Client(api_key=self.api_key) if hasattr(_GOOGLE_GENAI, "Client") else _GOOGLE_GENAI
                if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
                    resp = client.embeddings.create(model=chosen, input=texts)
                    data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
                    if not data:
                        raise GeminiClientError("Unexpected embeddings response shape from google.genai client.")
                    out: List[List[float]] = []
                    for item in data:
                        vec = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                        if vec is None:
                            raise GeminiClientError("Missing embedding field in response item.")
                        out.append([float(x) for x in vec])
                    return out
            except Exception as e:
                last_exc = e

        if _HAS_GENAI_LEGACY and _GENAI_LEGACY is not None:
            try:
                if hasattr(_GENAI_LEGACY, "embeddings") and hasattr(_GENAI_LEGACY.embeddings, "create"):
                    resp = _GENAI_LEGACY.embeddings.create(model=chosen, input=texts)
                    data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
                    if not data:
                        raise GeminiClientError("Unexpected embeddings response shape from legacy client.")
                    out: List[List[float]] = []
                    for item in data:
                        vec = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                        out.append([float(x) for x in vec])
                    return out
            except Exception as e:
                last_exc = e

        msg = "No supported embeddings API succeeded."
        if last_exc:
            msg += f" Last error: {last_exc}"
        raise GeminiClientError(msg)


# singleton
_singleton_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _singleton_client
    if _singleton_client is None:
        _singleton_client = GeminiClient()
    return _singleton_client
