# backend/self_heal/healer.py
"""
Self-healing helper that inspects last test run, constructs a compact context,
asks Gemini for repair suggestions, and applies safe, reversible edits to
the generated Postman collection (or pytest tests) as a first-step automated repair.

Important:
- Repairs here are intentionally conservative and reversible (we never delete tests;
  we write healed copies alongside originals).
- For complex fixes, this module returns suggested patches for a human to review.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.config import settings
from backend.gemini_client import get_gemini_client, GeminiClientError

# utility to locate job file
def _job_file(job_id: str) -> Path:
    return Path(settings.SCANS_DIR) / f"job_{job_id}.json"

def _read_job(job_id: str) -> Dict[str, Any]:
    jf = _job_file(job_id)
    if not jf.exists():
        raise FileNotFoundError("Job record not found.")
    return json.loads(jf.read_text(encoding="utf-8"))

def _write_job(job_id: str, data: Dict[str, Any]):
    jf = _job_file(job_id)
    jf.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_last_test_report(job_id: str) -> Optional[Dict[str, Any]]:
    """Try to read last test run summary file produced by test_runner."""
    candidate = Path(settings.TESTS_DIR) / f"test_run_{job_id}.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None
    # fallback to job record
    job = _read_job(job_id)
    return job.get("last_test_summary")


def _load_collection(job_id: str) -> Optional[Dict[str, Any]]:
    """Load collection.json from job tests dir if present."""
    coll_path = Path(settings.TESTS_DIR) / job_id / "collection.json"
    if coll_path.exists():
        try:
            return json.loads(coll_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_healed_collection(job_id: str, healed: Dict[str, Any]) -> Path:
    healed_dir = Path(settings.TESTS_DIR) / job_id / "healed"
    healed_dir.mkdir(parents=True, exist_ok=True)
    out = healed_dir / "collection.healed.json"
    out.write_text(json.dumps(healed, indent=2, default=str), encoding="utf-8")
    return out


def _simple_heuristic_analyze_failures(test_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract simple failure signals:
      - HTTP status codes (401 -> likely auth)
      - 404 -> path mismatch
      - 500/502 -> server error (suggest retries or payload changes)
      - connection refused / timeout (service not up)
    Returns a compact analysis dict.
    """
    analysis = {"signals": [], "examples": []}
    if not test_summary:
        return analysis

    # try to find exit_code and stdout / stderr
    ec = test_summary.get("exit_code")
    if ec is not None:
        analysis["exit_code"] = ec

    # look for http snippets in stdout or report
    stdout = test_summary.get("stdout_snippet", "") or test_summary.get("log_snippet", "") or ""
    stderr = test_summary.get("stderr_snippet", "") or ""
    combined = (stdout + "\n" + stderr).lower()

    if "401" in combined or "unauthorized" in combined or "authentication" in combined:
        analysis["signals"].append("auth_failure")
    if "404" in combined or "not found" in combined:
        analysis["signals"].append("not_found")
    if "timeout" in combined or "connection refused" in combined or "failed to connect" in combined:
        analysis["signals"].append("connection_issue")
    if "500" in combined or "internal server error" in combined:
        analysis["signals"].append("server_error")

    # capture short examples lines for LLM context
    lines = [ln.strip() for ln in (combined.splitlines()) if ln.strip()]
    analysis["examples"] = lines[-10:] if lines else []
    return analysis


REPAIR_PROMPT_TEMPLATE = """
You are an expert API testing assistant. The user has an existing Postman collection (JSON) and recent test-run logs.
Analyze the failures signals and propose small, conservative, reversible edits to make the tests pass or become more robust.

Input:
1) Failure analysis: {analysis_json}
2) Example failure lines (if any): {examples}
3) Collection (truncated if large): {collection_json}

Tasks:
- Produce a JSON array of repair actions. Each action should be one of:
  - add_header: {{ "type":"add_header", "header": {"key": "...", "value":"...", "apply_to": "all" or index list} }}
  - set_path_param: {{ "type":"set_path_param", "path": "/users/{id}", "example_values": ["123"] }}
  - add_json_body_example: {{ "type":"add_json_body_example", "path": "/path", "method":"POST", "example": {...} }}
  - suggest_manual: {{ "type":"suggest_manual", "message":"explain what to check" }}
- Keep actions conservative: do not remove tests or change assertions.
- When unsure, prefer suggest_manual with a short explanation.

Return ONLY valid JSON (an array of action objects).
"""

def propose_repairs_via_gemini(analysis: Dict[str, Any], collection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Call Gemini with a compact prompt to propose repair actions."""
    try:
        client = get_gemini_client()
    except GeminiClientError as exc:
        # fallback: simple heuristic proposals without Gemini
        proposals = []
        if "auth_failure" in analysis.get("signals", []):
            proposals.append({"type": "add_header", "header": {"key": "Authorization", "value": "Bearer <TOKEN>", "apply_to": "all"}})
        if "not_found" in analysis.get("signals", []):
            proposals.append({"type": "set_path_param", "path": "/", "example_values": ["1"]})
        if not proposals:
            proposals.append({"type": "suggest_manual", "message": "No automated repairs available; inspect server logs and endpoints."})
        return proposals

    prompt = REPAIR_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)[:2000],
        examples=json.dumps(analysis.get("examples", [])[:20], indent=2)[:1000],
        collection_json=json.dumps(collection if collection else {})[:8000],
    )

    try:
        resp = client.generate_text(prompt=prompt, temperature=0.0, max_output_tokens=800)
        # try to extract text
        text = None
        if isinstance(resp, dict):
            # look for common fields
            if "content" in resp:
                text = resp["content"]
            elif "candidates" in resp and isinstance(resp["candidates"], list) and len(resp["candidates"]) > 0:
                first = resp["candidates"][0]
                text = first.get("content") if isinstance(first, dict) else (first if isinstance(first, str) else None)
            elif "output" in resp:
                text = resp["output"]
            else:
                text = str(resp)
        else:
            text = str(resp)

        # strip fences and attempt to parse JSON
        text = text.strip()
        if text.startswith("```"):
            # naive strip
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()

        proposals = json.loads(text)
        if not isinstance(proposals, list):
            raise ValueError("Gemini did not return a JSON array of actions.")
        return proposals
    except Exception:
        # On any failure, fallback to simple heuristics
        proposals = []
        if "auth_failure" in analysis.get("signals", []):
            proposals.append({"type": "add_header", "header": {"key": "Authorization", "value": "Bearer <TOKEN>", "apply_to": "all"}})
        if "not_found" in analysis.get("signals", []):
            proposals.append({"type": "set_path_param", "path": "/", "example_values": ["1"]})
        if not proposals:
            proposals.append({"type": "suggest_manual", "message": "No automated repairs available; please inspect the server and logs."})
        return proposals


def apply_repairs_to_collection(collection: Dict[str, Any], repairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply conservative repairs to a Postman-style collection.
    Only implements a few action types:
      - add_header: add header to request objects (applied to all requests or selected indices)
      - set_path_param: replace templated segments or append example query params
      - add_json_body_example: set the request.body.raw to the provided example
    Returns a modified copy; original collection is not mutated.
    """
    import copy
    healed = copy.deepcopy(collection)
    items = healed.get("item", [])

    for action in repairs:
        t = action.get("type")
        if t == "add_header":
            header = action.get("header", {})
            key = header.get("key")
            val = header.get("value", "")
            apply_to = header.get("apply_to", "all")
            if not key:
                continue
            if apply_to == "all":
                for it in items:
                    req = it.get("request", {})
                    headers = req.get("header", []) or []
                    # avoid duplicates
                    if not any(h.get("key", "").lower() == key.lower() for h in headers):
                        headers.append({"key": key, "value": val})
                        req["header"] = headers
            elif isinstance(apply_to, list):
                for idx in apply_to:
                    if 0 <= idx < len(items):
                        req = items[idx].get("request", {})
                        headers = req.get("header", []) or []
                        if not any(h.get("key", "").lower() == key.lower() for h in headers):
                            headers.append({"key": key, "value": val})
                            req["header"] = headers

        elif t == "set_path_param":
            # attempt to substitute placeholders or append example query param
            path = action.get("path")
            values = action.get("example_values", ["1"])
            if not path:
                continue
            for it in items:
                req = it.get("request", {})
                url = req.get("url", {})
                raw = url.get("raw") if isinstance(url, dict) else url
                if not raw:
                    continue
                # simple replacement: replace {id} style placeholders
                for v in values:
                    if "{" in raw and "}" in raw:
                        # replace first placeholder occurrence for demonstration
                        new_raw = raw
                        new_raw = re_replace_first_placeholder(new_raw, v)
                        url["raw"] = new_raw
                        # update path segments if present
                        if isinstance(url.get("path"), list):
                            # naive: replace segments that look like {id}
                            url["path"] = [seg if "{" not in seg else v for seg in url["path"]]
                    else:
                        # append as query param ?example=...
                        if "?" in raw:
                            url["raw"] = f"{raw}&example={v}"
                        else:
                            url["raw"] = f"{raw}?example={v}"
                req["url"] = url

        elif t == "add_json_body_example":
            p = action.get("path")
            method = (action.get("method") or "").upper()
            example = action.get("example", {})
            for it in items:
                name = it.get("name", "")
                if p and p in name and (not method or method in name):
                    req = it.get("request", {})
                    body = req.get("body", {})
                    body["mode"] = "raw"
                    try:
                        body["raw"] = json.dumps(example)
                    except Exception:
                        body["raw"] = str(example)
                    req["body"] = body

        elif t == "suggest_manual":
            # no op: suggestions are stored separately
            continue

    healed["item"] = items
    return healed


def re_encode_job_patch(job: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """
    Store the patch (repair metadata) into the job record under `self_heal` key.
    """
    job.setdefault("self_heal", {})
    job["self_heal"].setdefault("patches", [])
    job["self_heal"]["patches"].append(patch)
    job["self_heal"]["last_healed_at"] = datetime.utcnow().isoformat()


def re_replace_first_placeholder(raw: str, value: str) -> str:
    """
    Replace the first {...} placeholder in raw with the given value.
    Simple helper using string operations.
    """
    start = raw.find("{")
    end = raw.find("}", start + 1)
    if start != -1 and end != -1 and end > start:
        return raw[:start] + value + raw[end + 1 :]
    return raw


def heal_job(job_id: str) -> Dict[str, Any]:
    """
    Main entrypoint for healing a job.
    Returns a dict with results including applied repairs and paths to healed artifacts.
    """
    job = _read_job(job_id)
    summary = _read_last_test_report(job_id)
    analysis = _simple_heuristic_analyze_failures(summary)

    collection = _load_collection(job_id)
    if not collection:
        # nothing we can auto-heal; suggest manual action
        patch = {"type": "suggest_manual", "message": "No collection found to heal; ensure tests were generated first."}
        re_encode_job_patch(job, patch)
        _write_job(job_id, job)
        return {"job_id": job_id, "healed": False, "reason": "no_collection", "patch": patch}

    # propose repairs
    repairs = propose_repairs_via_gemini(analysis, collection)

    # apply repairs conservatively
    healed_collection = apply_repairs_to_collection(collection, repairs)

    # write healed artifacts
    healed_path = _write_healed_collection(job_id, healed_collection)

    # store patch metadata in job record
    patch_record = {
        "proposed_at": datetime.utcnow().isoformat(),
        "repairs": repairs,
        "healed_collection_path": str(healed_path),
    }
    re_encode_job_patch(job, patch_record)
    # update job status
    job.update({"status": "healed", "healed_at": datetime.utcnow().isoformat(), "healed_artifact": str(healed_path)})
    _write_job(job_id, job)

    return {"job_id": job_id, "healed": True, "healed_collection": str(healed_path), "repairs": repairs}

