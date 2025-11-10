# backend/scanner/project_scanner.py
"""
Improved project scanner for local demo repos.

Enhancements:
- Detects common entry filenames: server.js, app.js, index.js, main.py, app.py, index.ts
- Reads package.json to find "main" or "scripts.start" hints
- Explicitly scans common subfolders: backend/, server/, src/, app/
- Keeps regex-based detection for Express/FastAPI/Flask/fetch/axios
- Records helpful notes when nothing was found so generate-tests can fallback
"""

import os
import re
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import settings

# regexes for basic detection
EXPRESS_ROUTE_RE = re.compile(
    r"\b(app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE
)
FLASK_ROUTE_RE = re.compile(
    r"@(?:app|bp)\.route\(\s*['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?", re.IGNORECASE
)
FASTAPI_ROUTE_RE = re.compile(
    r"@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE
)
FETCH_RE = re.compile(r"\bfetch\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
AXIOS_RE = re.compile(r"\baxios\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
IMPORT_REACT = re.compile(r"from\s+['\"]react['\"]|require\(['\"]react['\"]\)", re.IGNORECASE)
IMPORT_EXPRESS = re.compile(r"from\s+['\"]express['\"]|require\(['\"]express['\"]\)", re.IGNORECASE)
IMPORT_FASTAPI = re.compile(r"from\s+fastapi\s+import|import\s+fastapi", re.IGNORECASE)
IMPORT_FLASK = re.compile(r"from\s+flask\s+import|import\s+flask", re.IGNORECASE)

# file extension groups
JS_EXTS = {".js", ".jsx", ".ts", ".tsx"}
PY_EXTS = {".py"}
COMMON_ENTRY_FILES = {"server.js", "app.js", "index.js", "main.py", "app.py", "index.ts"}


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


def _guess_entry_files(root: Path) -> List[Path]:
    """
    Search for common entry files and package.json hints.
    Returns candidate paths prioritized by likely server entry.
    """
    candidates: List[Path] = []

    # check package.json for main or scripts
    pkg = root / "package.json"
    if pkg.exists():
        try:
            parsed = json.loads(pkg.read_text(encoding="utf-8"))
            main = parsed.get("main")
            scripts = parsed.get("scripts", {})
            start = scripts.get("start")
            if main:
                main_path = (root / main).resolve()
                if main_path.exists():
                    candidates.append(main_path)
            if start:
                # attempt to find a JS file referenced by start (basic heuristics)
                m = re.search(r"node\s+([^\s]+)", start)
                if m:
                    p = (root / m.group(1)).resolve()
                    if p.exists():
                        candidates.append(p)
        except Exception:
            pass

    # look for common entry filenames in root and common subfolders
    for folder in [root, root / "backend", root / "server", root / "src", root / "app"]:
        if not folder.exists():
            continue
        for fname in COMMON_ENTRY_FILES:
            p = folder / fname
            if p.exists():
                candidates.append(p.resolve())

    # dedupe while preserving order
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            unique.append(c)
    return unique


def scan_project(path_or_url: str, job_id: Optional[str] = None) -> Dict[str, Any]:
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
        clone_folder = uploads_dir / f"{job_id}_clone"
        report["notes"].append(f"Detected git url; cloning into {clone_folder}")
        try:
            _git_clone(path_or_url, clone_folder)
            target_path = clone_folder
        except Exception as exc:
            raise RuntimeError(f"Failed to clone repo: {exc}")
    else:
        p = Path(path_or_url)
        if not p.exists():
            raise FileNotFoundError(f"Path '{path_or_url}' does not exist.")
        # if path points to an extracted folder that contains a single top-level folder, prefer that
        if p.is_dir():
            children = [c for c in p.iterdir() if c.is_dir()]
            if len(children) == 1 and not any(c.name.startswith(".") for c in children):
                candidate = children[0]
                report["notes"].append(f"Using nested project root: {candidate}")
                p = candidate
        target_path = p

    report["scanned_path"] = str(target_path)

    # Try to guess entry files and add note
    entry_candidates = _guess_entry_files(Path(target_path))
    if entry_candidates:
        report["notes"].append(f"Entry file candidates: {', '.join(str(x.name) for x in entry_candidates[:5])}")

    # Walk and scan files
    file_count = 0
    # explicit directories to ensure we scan common layout
    scan_roots = [Path(target_path)]
    for sub in ["backend", "server", "src", "app"]:
        p = Path(target_path) / sub
        if p.exists():
            scan_roots.append(p)

    scanned_paths = set()
    for root in scan_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # skip node_modules and .venv for speed
            if "node_modules" in dirpath or ".venv" in dirpath or "__pycache__" in dirpath:
                continue
            for fname in filenames:
                file_count += 1
                path = Path(dirpath) / fname
                # avoid scanning same file twice
                key = str(path.resolve())
                if key in scanned_paths:
                    continue
                scanned_paths.add(key)
                try:
                    _scan_file_for_patterns(path, report)
                except Exception as exc:
                    report["notes"].append(f"Failed to scan {path}: {exc}")

    report["scanned_at"] = __import__("datetime").datetime.utcnow().isoformat()
    report["detected_frameworks"] = sorted(list(report["detected_frameworks"]))
    report["file_count"] = file_count

    # Synthesize openapi stub
    report["sample_openapi"] = _synthesize_openapi_stub(report["backend_endpoints"])

    # If nothing found, add helpful notes so generate-tests can do fallback
    if not report["backend_endpoints"] and not report["frontend_calls"]:
        report["notes"].append("No backend endpoints or frontend calls detected. Consider checking nested folders or use a simple server file (server.js/app.js/index.js) in project root or backend/ folder.")

    return report

