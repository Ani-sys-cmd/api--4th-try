# backend/self_heal/repair_strategies.py
"""
Conservative, reversible repair strategies for Postman-style collections.

This module implements small, well-scoped transformation functions that can be applied
to a Postman collection (dict) to produce a healed copy. Each function returns:
    healed_collection_dict, patch_metadata_dict

Patch metadata includes the action performed and a short description so it can be recorded
in the job's self_heal patches list.

Strategies implemented:
  - add_auth_header(collection, header_key="Authorization", header_value="Bearer <TOKEN>", apply_to="all" | [indices])
  - set_path_params(collection, path_template="/users/{id}", example_values=["123"], match_method=None)
  - add_json_body_example(collection, path_substring, method="POST", example_obj)
  - bump_timeouts(collection, factor=2.0)
  - mark_manual_suggestion(message) -> returns patch metadata (no collection change)

All changes are conservative: originals are not mutated (deep-copy is used by callers).
"""

from copy import deepcopy
from typing import Dict, Any, List, Tuple, Union
from datetime import datetime


def _now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat()


def add_auth_header(collection: Dict[str, Any],
                    header_key: str = "Authorization",
                    header_value: str = "Bearer <TOKEN>",
                    apply_to: Union[str, List[int]] = "all") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Add an Authorization (or other) header to requests.
    - apply_to: "all" or list of item indices to modify.
    Returns (new_collection, patch_meta).
    """
    healed = deepcopy(collection)
    items = healed.get("item", [])
    modified_indices = []

    if apply_to == "all":
        target_indices = range(len(items))
    elif isinstance(apply_to, list):
        target_indices = [i for i in apply_to if 0 <= i < len(items)]
    else:
        target_indices = []

    for i in target_indices:
        it = items[i]
        req = it.setdefault("request", {})
        headers = req.get("header") or []
        # ensure headers is list of dicts {key, value}
        if not isinstance(headers, list):
            headers = []
        # avoid duplicate header keys
        lower_keys = {h.get("key", "").lower() for h in headers if isinstance(h, dict)}
        if header_key.lower() not in lower_keys:
            headers.append({"key": header_key, "value": header_value})
            req["header"] = headers
            modified_indices.append(i)

    patch = {
        "type": "add_header",
        "header": {"key": header_key, "value": header_value, "apply_to": "all" if apply_to == "all" else list(target_indices)},
        "modified_indices": modified_indices,
        "applied_at": _now_iso(),
        "note": "Added Authorization header placeholder to requests."
    }
    return healed, patch


def set_path_params(collection: Dict[str, Any],
                    path_template: str,
                    example_values: List[str] = None,
                    match_method: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Replace first placeholder like {id} in matching request URLs with example values.
    - path_template: substring to match against request name or URL raw
    - example_values: list of example strings to apply (applies first by default)
    - match_method: optional filter e.g., "GET" or "POST" to limit which requests to change
    Returns (new_collection, patch_meta).
    """
    if example_values is None:
        example_values = ["1"]

    healed = deepcopy(collection)
    items = healed.get("item", [])
    applied = []

    for idx, it in enumerate(items):
        name = it.get("name", "") or ""
        req = it.get("request", {}) or {}
        url = req.get("url", {})
        raw = url.get("raw") if isinstance(url, dict) else url
        method = (req.get("method") or "").upper()

        # filter by match_method if provided
        if match_method and match_method.upper() != method:
            continue

        # quick match by name or raw url containing template
        if (path_template in name) or (isinstance(raw, str) and "{" in raw and "}" in raw):
            new_raw = raw
            for val in example_values:
                # replace first placeholder occurrence
                if isinstance(new_raw, str) and "{" in new_raw and "}" in new_raw:
                    start = new_raw.find("{")
                    end = new_raw.find("}", start+1)
                    if start != -1 and end != -1:
                        new_raw = new_raw[:start] + str(val) + new_raw[end+1:]
                else:
                    # append as query param if no placeholder
                    if isinstance(new_raw, str):
                        if "?" in new_raw:
                            new_raw = f"{new_raw}&example={val}"
                        else:
                            new_raw = f"{new_raw}?example={val}"
            if new_raw != raw:
                if isinstance(url, dict):
                    # update both raw and path segments if present
                    url["raw"] = new_raw
                    if isinstance(url.get("path"), list):
                        # naive mapping: replace segments that look like {param}
                        url["path"] = [seg if "{" not in seg else (example_values[0] if example_values else seg) for seg in url.get("path")]
                    req["url"] = url
                else:
                    req["url"] = new_raw
                it["request"] = req
                applied.append({"index": idx, "old_raw": raw, "new_raw": new_raw})

    patch = {
        "type": "set_path_param",
        "template": path_template,
        "example_values": example_values,
        "matches": applied,
        "applied_at": _now_iso(),
        "note": "Replaced path placeholders or appended example query params."
    }
    return healed, patch


def add_json_body_example(collection: Dict[str, Any],
                          path_substring: str,
                          method: str = "POST",
                          example_obj: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    For requests whose name or URL contains `path_substring` and match `method`,
    set a JSON body example in request.body.raw.
    Returns (new_collection, patch_meta).
    """
    if example_obj is None:
        example_obj = {"example": "value"}

    healed = deepcopy(collection)
    items = healed.get("item", [])
    applied = []

    for idx, it in enumerate(items):
        name = (it.get("name") or "").lower()
        req = it.get("request", {}) or {}
        m = (req.get("method") or "").upper()
        url = req.get("url", {}) or {}
        raw = url.get("raw") if isinstance(url, dict) else url

        if method and m != method.upper():
            continue
        if path_substring.lower() in name or (isinstance(raw, str) and path_substring.lower() in raw.lower()):
            body = req.get("body", {}) or {}
            body["mode"] = "raw"
            try:
                import json as _json
                body["raw"] = _json.dumps(example_obj, indent=2)
            except Exception:
                body["raw"] = str(example_obj)
            req["body"] = body
            it["request"] = req
            applied.append({"index": idx, "method": m, "path_contains": path_substring})

    patch = {
        "type": "add_json_body_example",
        "path_substring": path_substring,
        "method": method,
        "example_preview": example_obj,
        "matches": applied,
        "applied_at": _now_iso(),
        "note": "Added JSON body examples to matching requests."
    }
    return healed, patch


def bump_timeouts(collection: Dict[str, Any], factor: float = 2.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Multiply timeout settings in request-level or collection-level settings by `factor`, if present.
    This helps when tests fail due to slow local services.
    Conservative: only touch numeric fields named 'timeout' or 'requestTimeout'.
    """
    healed = deepcopy(collection)
    items = healed.get("item", [])
    modified = []

    # check top-level event/setting (Postman collection variants differ)
    settings_obj = healed.get("event") or healed.get("settings") or {}
    if isinstance(settings_obj, dict):
        for key in list(settings_obj.keys()):
            if key.lower() in {"timeout", "requesttimeout", "timeoutms"}:
                try:
                    old = float(settings_obj[key])
                    settings_obj[key] = old * factor
                    modified.append({"where": "collection_settings", "key": key, "old": old, "new": settings_obj[key]})
                except Exception:
                    continue
        healed.update({"event": settings_obj})

    # per-request timeouts (if present)
    for idx, it in enumerate(items):
        req = it.get("request", {}) or {}
        timeout_keys = [k for k in req.keys() if k.lower() in {"timeout", "requesttimeout", "timeoutms"}]
        for k in timeout_keys:
            try:
                old = float(req[k])
                req[k] = old * factor
                modified.append({"where": f"item_{idx}", "key": k, "old": old, "new": req[k]})
                it["request"] = req
            except Exception:
                continue

    patch = {
        "type": "bump_timeouts",
        "factor": factor,
        "modified": modified,
        "applied_at": _now_iso(),
        "note": "Increased request/collection timeouts to reduce false failures due to slowness."
    }
    return healed, patch


def mark_manual_suggestion(message: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Produce a suggested manual patch (no collection change). Useful when automated repairs are unsafe.
    Returns (original_collection_placeholder, patch_meta).
    """
    patch = {
        "type": "suggest_manual",
        "message": message,
        "generated_at": _now_iso(),
    }
    # return empty change (caller should record patch)
    return {}, patch
