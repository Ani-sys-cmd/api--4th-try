# backend/openapi_synthesizer.py
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

import json
from pathlib import Path
from typing import Dict, Any, Optional

from config import settings
from gemini_client import get_gemini_client, GeminiClientError


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

def _save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def synthesize_openapi(scan_report: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enrich the scan_report['sample_openapi'] using Gemini LLM.
    :param scan_report: dict returned by scanner.project_scanner.scan_project
    :param job_id: optional job id used for saving output
    :returns: enriched openapi dict (fallback to original stub if enrichment fails)
    """
    job_id = job_id or scan_report.get("job_id", "local")
    base_stub = scan_report.get("sample_openapi", {})
    frontend_calls = scan_report.get("frontend_calls", [])

    # Prepare prompt with scanner JSON inline
    prompt = PROMPT_TEMPLATE.format(
        scanner_json=json.dumps(base_stub, indent=2),
        frontend_calls=json.dumps(frontend_calls, indent=2)
    )

    # Default output path
    out_path = Path(settings.SCANS_DIR) / f"openapi_enriched_{job_id}.json"

    # Save the base stub for record
    try:
        _save_json(base_stub, out_path.with_name(f"openapi_stub_{job_id}.json"))
    except Exception:
        # non-fatal
        pass

    # Call Gemini
    try:
        client = get_gemini_client()
    except GeminiClientError as exc:
        # Gemini not available — return base stub and record note
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"]["status"] = "gemini_unavailable"
        scan_report["synthesizer"]["error"] = str(exc)
        # Save report
        try:
            _save_json(scan_report, Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json")
        except Exception:
            pass
        return base_stub

    try:
        response = client.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=1200  # allow enough space for JSON output
        )
        # The exact shape of response depends on google-generativeai version.
        # Try several common extraction patterns.
        text_output = None
        if isinstance(response, dict):
            # Attempt to extract text content from possible fields
            # Newer clients may return {'candidates': [...]} or {'content': '...'}
            if "content" in response:
                text_output = response["content"]
            elif "candidates" in response and isinstance(response["candidates"], list) and len(response["candidates"]) > 0:
                # candidate may be dict or string
                first = response["candidates"][0]
                if isinstance(first, dict) and "content" in first:
                    text_output = first["content"]
                elif isinstance(first, str):
                    text_output = first
            elif "output" in response and isinstance(response["output"], str):
                text_output = response["output"]
            else:
                # try str cast as last resort
                text_output = str(response)
        else:
            text_output = str(response)

        if not text_output:
            raise ValueError("Gemini returned no textual output.")

        # Some LLMs wrap JSON in triple backticks or markdown; strip common wrappers
        cleaned = _strip_code_fence(text_output)

        # Attempt to load JSON
        enriched = json.loads(cleaned)
        # Basic validation: must be a dict with 'openapi' or 'paths'
        if not isinstance(enriched, dict) or ("paths" not in enriched and "openapi" not in enriched):
            # fallback: if returned a patch dict with 'paths', try to merge
            if isinstance(enriched, dict) and "paths" in enriched:
                merged = dict(base_stub)
                merged_paths = dict(merged.get("paths", {}))
                merged_paths.update(enriched.get("paths", {}))
                merged["paths"] = merged_paths
                enriched = merged
            else:
                raise ValueError("Enriched output did not contain OpenAPI structure.")

        # Save enriched OpenAPI to disk
        _save_json(enriched, out_path)

        # Update scan_report metadata
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({
            "status": "enriched",
            "enriched_openapi_path": str(out_path),
        })
        try:
            _save_json(scan_report, Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json")
        except Exception:
            pass

        return enriched

    except Exception as exc:
        # On any failure, return base stub and annotate the scan_report
        scan_report.setdefault("synthesizer", {})
        scan_report["synthesizer"].update({
            "status": "enrichment_failed",
            "error": str(exc),
        })
        try:
            _save_json(scan_report, Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json")
        except Exception:
            pass
        return base_stub


def _strip_code_fence(s: str) -> str:
    """
    Remove markdown code fences and leading/trailing whitespace.
    """
    s = s.strip()
    # remove triple backticks and optional language hint
    if s.startswith("```"):
        # find the next ```
        parts = s.split("```")
        # parts[0] is before first fence (likely empty), parts[1] is content if only one fence used
        if len(parts) >= 2:
            # content may be parts[1] if single fenced block
            inner = "```".join(parts[1:]).strip()
            # if there's another trailing fence at the end, remove it
            if inner.endswith("```"):
                inner = inner[:-3].strip()
            return inner
    # also remove single-line fences like `json` or surrounding backticks
    if s.startswith("`") and s.endswith("`"):
        return s.strip("`").strip()
    return s
