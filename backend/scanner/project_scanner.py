# backend/scanner/project_scanner.py
"""
Simple project scanner for demo/proof-of-concept.

Features:
- Accepts either a local path (extracted upload) or a git URL.
- If git URL is provided, uses `git clone` via subprocess into settings.UPLOADS_DIR/<job_id>_clone.
- Walks files and performs lightweight regex-based detection of:
  - Express / Node backend routes (app.get, router.post, etc.)
  - FastAPI / Flask route decorators (@app.get, @app.route)
  - Frontend API uses (fetch(...), axios.get/post, fetch(`/api/...`))
- Produces a scan report dict that includes:
  - job_id, scanned_path, detected_frameworks, backend_endpoints, frontend_calls, sample_openapi (stub)
"""

import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import settings

# regexes for basic detection
EXPRESS_ROUTE_RE = re.compile(r"\b(app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
FLASK_ROUTE_RE = re.compile(r"@(?:app|bp)\.route\(\s*['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?", re.IGNORECASE)
FASTAPI_ROUTE_RE = re.compile(r"@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
FETCH_RE = re.compile(r"\bfetch\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
AXIOS_RE = re.compile(r"\baxios\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
IMPORT_REACT = re.compile(r"from\s+['\"]react['\"]|require\(['\"]react['\"]\)", re.IGNORECASE)
IMPORT_EXPRESS = re.compile(r"from\s+['\"]express['\"]|require\(['\"]express['\"]\)", re.IGNORECASE)
IMPORT_FASTAPI = re.compile(r"from\s+fastapi\s+import|import\s+fastapi", re.IGNORECASE)
IMPORT_FLASK = re.compile(r"from\s+flask\s+import|import\s+flask", re.IGNORECASE)

# useful file extensions to scan
JS_EXTS = {".js", ".jsx", ".ts", ".tsx"}
PY_EXTS = {".py"}


def _is_url_like(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://") or s.endswith(".git") or s.startswith("git@")


def _git_clone(git_url: str, dest_dir: Path) -> Path:
    """Clone the repo into dest_dir using `git clone` via subprocess. Returns path to clone."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", git_url, str(dest_dir)], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        # Clean up partial clone
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {exc.stderr.decode('utf-8', errors='ignore')}")
    return dest_dir


def _read_file_safe(path: Path) -> Optional[str]:
    try:
        # attempt utf-8, fall back to latin-1
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None


def _scan_file_for_patterns(path: Path, report: Dict[str, Any]):
    content = _read_file_safe(path)
    if not content:
        return

    suffix = path.suffix.lower()
    # detect frameworks / imports
    if suffix in JS_EXTS:
        if IMPORT_REACT.search(content):
            report["detected_frameworks"].add("react")
        if IMPORT_EXPRESS.search(content):
            report["detected_frameworks"].add("express")
        # find express-like routes
        for m in EXPRESS_ROUTE_RE.finditer(content):
            method = m.group(2).upper()
            route = m.group(3)
            report["backend_endpoints"].append({"method": method, "path": route, "source_file": str(path)})
        # frontend calls
        for m in FETCH_RE.finditer(content):
            url = m.group(1)
            report["frontend_calls"].append({"type": "fetch", "url": url, "source_file": str(path)})
        for m in AXIOS_RE.finditer(content):
            method = m.group(1).upper()
            url = m.group(2)
            report["frontend_calls"].append({"type": "axios", "method": method, "url": url, "source_file": str(path)})

    if suffix in PY_EXTS:
        if IMPORT_FASTAPI.search(content):
            report["detected_frameworks"].add("fastapi")
        if IMPORT_FLASK.search(content):
            report["detected_frameworks"].add("flask")
        # find fastapi/flask routes
        for m in FASTAPI_ROUTE_RE.finditer(content):
            method = m.group(1).upper()
            route = m.group(2)
            report["backend_endpoints"].append({"method": method, "path": route, "source_file": str(path)})
        for m in FLASK_ROUTE_RE.finditer(content):
            route = m.group(1)
            methods = m.group(2) or "GET"
            report["backend_endpoints"].append({"method": methods, "path": route, "source_file": str(path)})


def _synthesize_openapi_stub(endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a minimal OpenAPI-like stub from discovered endpoints.
    This is a baseline that later LLM/RAG can enrich.
    """
    openapi = {
        "openapi": "3.0.0",
        "info": {"title": "synthesized-api", "version": "0.1.0"},
        "paths": {},
    }
    for ep in endpoints:
        path = ep.get("path", "/unknown")
        method = ep.get("method", "GET").lower()
        if path not in openapi["paths"]:
            openapi["paths"][path] = {}
        openapi["paths"][path][method] = {
            "summary": f"Discovered endpoint from {ep.get('source_file')}",
            "responses": {"200": {"description": "OK"}},
        }
    return openapi


def scan_project(path_or_url: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Primary scanner entrypoint. Returns a report dict.
    :param path_or_url: local path (str) or git url
    :param job_id: optional job id for naming clones
    """
    job_id = job_id or "local"
    uploads_dir = Path(settings.UPLOADS_DIR)
    report = {
        "job_id": job_id,
        "scanned_at": None,
        "scanned_path": None,
        "detected_frameworks": set(),
        "backend_endpoints": [],
        "frontend_calls": [],
        "notes": [],
    }

    # Determine if we need to git clone
    target_path = None
    if _is_url_like(path_or_url):
        # clone into uploads/<job_id>_clone
        clone_folder = uploads_dir / f"{job_id}_clone"
        report["notes"].append(f"Detected git url; cloning into {clone_folder}")
        clone_folder_str = str(clone_folder)
        try:
            _git_clone(path_or_url, clone_folder)
            target_path = clone_folder
        except Exception as exc:
            raise RuntimeError(f"Failed to clone repo: {exc}")
    else:
        # assume local path
        p = Path(path_or_url)
        if not p.exists():
            raise FileNotFoundError(f"Path '{path_or_url}' does not exist.")
        target_path = p

    report["scanned_path"] = str(target_path)

    # Walk files
    file_count = 0
    for root, dirs, files in os.walk(target_path):
        for fname in files:
            file_count += 1
            path = Path(root) / fname
            try:
                _scan_file_for_patterns(path, report)
            except Exception as exc:
                # non-fatal; record and continue
                report["notes"].append(f"Failed to scan {path}: {exc}")

    report["scanned_at"] = __import__("datetime").datetime.utcnow().isoformat()
    report["detected_frameworks"] = sorted(list(report["detected_frameworks"]))
    report["file_count"] = file_count

    # synthesize a basic OpenAPI stub from discovered backend_endpoints
    report["sample_openapi"] = _synthesize_openapi_stub(report["backend_endpoints"])

    # convert lists to serializable form (they are already lists)
    return report
