"""
Enrich a synthesized OpenAPI stub using the Gemini LLM (via gemini_client).

This module builds a clear prompt containing:
  - the basic OpenAPI stub produced by the scanner
  - a short instruction describing what to enrich (parameters, request/response examples, status codes, auth)

It calls Gemini to produce a JSON OpenAPI document (or a structured patch). If Gemini fails or
returns unparsable text, the original stub is returned and saved.

Functions:
  - synthesize_openapi(scan_report: dict, job_id: str) -> dict
"""

from __future__ import annotations
import json
import re
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import logging
from backend.config import settings
from backend.gemini_client import get_gemini_client, GeminiClientError

logger = logging.getLogger("backend.openapi_synthesizer")
if os.getenv("OPENAPI_SYNTH_DEBUG", "").lower() in ("1", "true", "yes", "on"):
    logger.setLevel(logging.DEBUG)


PROMPT_TEMPLATE = """
You are an expert engineer that converts minimal OpenAPI stubs into detailed OpenAPI v3.0 JSON documents.

Input:
1) A minimal OpenAPI stub (JSON) produced by a static code scanner below.
2) A short project context and any discovered frontend calls.

Task:
- Produce a valid JSON object representing an enriched OpenAPI v3 document.
- For each path/method, add:
  - a short summary
  - parameters (path/query/header) where obvious
  - a requestBody schema if body-like endpoints are detected (e.g., POST/PUT)
  - example request and response bodies (JSON) where possible
  - common HTTP response codes (200, 400, 401, 404, 500) with brief descriptions
  - mark any endpoints that likely require authentication with "security" hint
- When you are uncertain, keep types generic (e.g., { "type": "string" }) and add a helpful description.
- Return **only** the enriched OpenAPI JSON object (no commentary). If you cannot produce a valid JSON, return the original stub.

Scanner context (JSON):
{scanner_json}

Project notes / frontend calls:
{frontend_calls}

Produce the enriched OpenAPI JSON now.
"""


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    """Write JSON to path (create parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _safe_serialize_for_prompt(obj: Any, max_chars: int = 50_000) -> str:
    """
    Serialize JSON for inclusion in a text prompt.
    Limit length to avoid huge prompts; add a marker if truncated.
    """
    s = json.dumps(obj, indent=2, ensure_ascii=False)
    if len(s) > max_chars:
        # Try to cut at a newline boundary for nicer context
        truncated = s[:max_chars]
        nl = truncated.rfind("\n")
        if nl > int(max_chars * 0.6):
            truncated = truncated[:nl]
        logger.debug("Prompt context truncated: original=%d chars, kept=%d chars", len(s), len(truncated))
        return truncated + "\n\n...TRUNCATED (original JSON too large) ..."
    return s


def _strip_code_fence(s: str) -> str:
    """
    Remove markdown code fences and leading/trailing whitespace.
    Also tries to extract the first JSON object/array block found in the string.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()

    # Remove triple-backtick fences and optional language token.
    # e.g. ```json\n{...}\n```  -> {...}
    fence_re = re.compile(r"^```[ \t]*[a-zA-Z0-9_-]*\n", flags=re.MULTILINE)
    s = fence_re.sub("", s)  # remove leading ```lang\n if present
    s = re.sub(r"\n```$", "", s)  # remove trailing ```
    # Remove single backticks if surrounding
    if s.startswith("`") and s.endswith("`"):
        s = s.strip("`").strip()

    # Extract the first JSON object/array if present
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
    if json_match:
        candidate = json_match.group(1).strip()
        return candidate
    return s


def _extract_text_from_response(response: Any) -> str:
    """
    Handle multiple possible response shapes from Gemini / google-generativeai clients.
    Return the best textual content found or raise ValueError if not found.
    """
    # If it's already a string
    if isinstance(response, str):
        return response

    # If it's a mapping/dict, try common fields (also handle the wrapper output)
    if isinstance(response, dict):
        # wrapper may return {"content": "...", "raw": ..., "provider": "..."}
        if "content" in response and isinstance(response["content"], str) and response["content"].strip():
            return response["content"]
        # try other common top-level fields
        for key in ("text", "output", "message", "response"):
            v = response.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # nested choices/candidates/outputs
        candidates = response.get("candidates") or response.get("outputs") or response.get("choices")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                for key in ("content", "text", "output"):
                    v = first.get(key)
                    if isinstance(v, str) and v.strip():
                        return v
                if "message" in first and isinstance(first["message"], dict):
                    m = first["message"]
                    for k in ("content", "text"):
                        vv = m.get(k)
                        if isinstance(vv, str) and vv.strip():
                            return vv
            elif isinstance(first, str):
                return first

        # sometimes the wrapped 'raw' field contains the useful payload
        raw = response.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw
        try:
            # last resort: stringify (may include JSON)
            s = json.dumps(response) if not isinstance(response, str) else str(response)
            if s and s.strip():
                return s
        except Exception:
            pass

    # object-like (SDK objects)
    try:
        # tries common attribute shapes
        if hasattr(response, "candidates") and getattr(response, "candidates"):
            first = getattr(response, "candidates")[0]
            if hasattr(first, "content"):
                return str(first.content)
        if hasattr(response, "outputs") and getattr(response, "outputs"):
            out0 = getattr(response, "outputs")[0]
            if hasattr(out0, "content"):
                return str(out0.content)
            if hasattr(out0, "text"):
                return str(out0.text)
    except Exception:
        pass

    # Fallback: cast to string
    try:
        s = str(response)
        if s and s.strip():
            return s
    except Exception:
        pass

    raise ValueError("No textual output could be extracted from the LLM response.")


def synthesize_openapi(scan_report: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enrich the scan_report['sample_openapi'] using Gemini LLM.
    :param scan_report: dict returned by scanner.project_scanner.scan_project
    :param job_id: optional job id used for saving output
    :returns: enriched openapi dict (fallback to original stub if enrichment fails)
    """
    job_id = job_id or scan_report.get("job_id", "local")
    base_stub = scan_report.get("sample_openapi", {}) or {}
    frontend_calls = scan_report.get("frontend_calls", []) or []

    out_path = Path(settings.SCANS_DIR) / f"openapi_enriched_{job_id}.json"
    stub_save_path = Path(settings.SCANS_DIR) / f"openapi_stub_{job_id}.json"
    report_save_path = Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json"

    # Always save the original stub to help debugging
    try:
        _save_json(base_stub, stub_save_path)
    except Exception:
        logger.exception("Failed to save base stub for job %s", job_id)

    # Build prompt safely (avoid .format to keep braces literal)
    try:
        scanner_json_str = _safe_serialize_for_prompt(base_stub)
        frontend_calls_str = _safe_serialize_for_prompt(frontend_calls)
        prompt = PROMPT_TEMPLATE.replace("{scanner_json}", scanner_json_str).replace("{frontend_calls}", frontend_calls_str)
        logger.debug("Built prompt for job %s (len=%d)", job_id, len(prompt))
    except Exception as exc:
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({"status": "prompt_build_failed", "error": str(exc), "traceback": traceback.format_exc()})
        try:
            _save_json(scan_report, report_save_path)
        except Exception:
            logger.exception("Failed to save scan_report when prompt build failed for job %s", job_id)
        logger.exception("Prompt construction failed for job %s", job_id)
        return base_stub

    # Acquire Gemini client
    try:
        client = get_gemini_client()
    except GeminiClientError as exc:
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({"status": "gemini_unavailable", "error": str(exc), "traceback": traceback.format_exc()})
        try:
            _save_json(scan_report, report_save_path)
        except Exception:
            logger.exception("Failed to save scan_report when gemini client unavailable for job %s", job_id)
        return base_stub

    # Call the LLM
    try:
        # SAFER: only pass prompt + max_output_tokens. Avoid temperature by default.
        response = client.generate_text(
            prompt=prompt,
            max_output_tokens=1200,
        )

        # Keep a small snippet of raw response for diagnostics (not full raw to avoid huge logs)
        try:
            raw_snippet = ""
            if isinstance(response, dict):
                raw_candidate = response.get("raw") or response.get("content") or response
                raw_snippet = str(raw_candidate)[:2000]
            else:
                raw_snippet = str(response)[:2000]
        except Exception:
            raw_snippet = ""

        text_output = _extract_text_from_response(response)
        if not text_output or not text_output.strip():
            raise ValueError("Gemini returned empty text output.")

        cleaned = _strip_code_fence(text_output)

        # Attempt to parse JSON
        try:
            enriched = json.loads(cleaned)
        except Exception:
            # try to locate the first JSON substring and parse that
            json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
            if json_match:
                candidate = json_match.group(1)
                try:
                    enriched = json.loads(candidate)
                except Exception as exc_inner:
                    logger.debug("Failed JSON parse for job %s. cleaned head: %s", job_id, cleaned[:1000])
                    raise ValueError(f"Failed to parse JSON from LLM output: {exc_inner}\nOutput head: {cleaned[:1000]}")
            else:
                logger.debug("No JSON found in LLM output for job %s. cleaned head: %s", job_id, cleaned[:1000])
                raise ValueError("No JSON object found in LLM output.")

        # Validate enriched JSON shape
        if not isinstance(enriched, dict):
            raise ValueError("Enriched output is not a JSON object (expected dict).")

        if "openapi" not in enriched and "paths" not in enriched:
            # If enriched contains 'paths', merge with base stub
            if "paths" in enriched:
                merged = dict(base_stub)
                merged_paths = dict(merged.get("paths", {}))
                merged_paths.update(enriched.get("paths", {}))
                merged["paths"] = merged_paths
                enriched = merged
            else:
                raise ValueError("Enriched output does not contain 'openapi' or 'paths' keys.")

        # Save enriched OpenAPI
        try:
            _save_json(enriched, out_path)
        except Exception:
            logger.exception("Failed to save enriched OpenAPI for job %s", job_id)

        # Update scan_report metadata
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({
            "status": "enriched",
            "enriched_openapi_path": str(out_path),
            "llm_raw_snippet": raw_snippet[:2000]
        })
        try:
            _save_json(scan_report, report_save_path)
        except Exception:
            logger.exception("Failed to save scan_report after enrichment for job %s", job_id)

        return enriched

    except Exception as exc:
        # Annotate scan_report with error and limited debug info, then return base stub
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({
            "status": "enrichment_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        # include a tiny snippet of the LLM response if we have one (helpful for debugging)
        try:
            if 'raw_snippet' in locals() and raw_snippet:
                scan_report["synthesizer"]["llm_raw_snippet"] = raw_snippet[:2000]
        except Exception:
            pass

        try:
            _save_json(scan_report, report_save_path)
        except Exception:
            logger.exception("Failed to save scan_report after enrichment failure for job %s", job_id)

        logger.exception("OpenAPI synthesis failed for job %s", job_id)
        return base_stub
