# backend/test_executor/test_runner.py
"""
Unified test runner used by routes and workers.

- Prefer running Postman collections with Newman via newman_adapter.run_collection.
- If pytest-style tests are present, run pytest and capture output.
- Writes a summary JSON into settings.TESTS_DIR/test_run_{job_id}.json.

This file is intentionally small and local-first.
"""

import json
import subprocess
import shlex
import time
from pathlib import Path
from typing import Optional, Dict, Any

from config import settings
from utils.logger import get_logger
from .newman_adapter import run_collection, NewmanError

logger = get_logger(__name__)


def _write_summary(job_id: str, data: Dict[str, Any]) -> Path:
    out_dir = Path(settings.TESTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_run_{job_id}.json"
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out_path


def run_newman_for_job(collection_path: str, job_id: str, environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a Postman collection via the newman_adapter and produce a friendly summary.
    """
    start = time.time()
    try:
        result = run_collection(collection_path, job_id, environment_path=environment)
        summary = {
            "job_id": job_id,
            "runner": "newman",
            "exit_code": result.get("exit_code"),
            "report_path": result.get("report_path"),
            "summary": result.get("summary"),
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "duration": result.get("duration_seconds", time.time() - start),
            "ran_at": result.get("ran_at"),
        }
    except NewmanError as exc:
        logger.error("Newman run failed: %s", exc)
        summary = {"job_id": job_id, "runner": "newman", "error": str(exc), "exit_code": -1, "duration": time.time() - start}
    except Exception as exc:
        logger.exception("Unexpected error running newman")
        summary = {"job_id": job_id, "runner": "newman", "error": str(exc), "exit_code": -2, "duration": time.time() - start}

    _write_summary(job_id, summary)
    return summary


def run_pytest_for_dir(tests_dir: str, job_id: str, additional_args: Optional[str] = None) -> Dict[str, Any]:
    """
    Run pytest over a directory and capture the output.
    """
    start = time.time()
    out_dir = Path(settings.TESTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"pytest_{job_id}.log"

    cmd = ["pytest", str(tests_dir), "-q", "--disable-warnings", "--maxfail=1"]
    if additional_args:
        cmd += shlex.split(additional_args)

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=1800)
        exit_code = proc.returncode
    except FileNotFoundError:
        msg = "pytest not installed or not found on PATH"
        logger.error(msg)
        summary = {"job_id": job_id, "runner": "pytest", "error": msg, "exit_code": -1, "duration": time.time() - start}
        _write_summary(job_id, summary)
        return summary
    except subprocess.TimeoutExpired as exc:
        logger.error("pytest timed out: %s", exc)
        summary = {"job_id": job_id, "runner": "pytest", "error": "timeout", "exit_code": -1, "duration": time.time() - start}
        _write_summary(job_id, summary)
        return summary
    except Exception as exc:
        logger.exception("Unexpected pytest error")
        summary = {"job_id": job_id, "runner": "pytest", "error": str(exc), "exit_code": -2, "duration": time.time() - start}
        _write_summary(job_id, summary)
        return summary

    # read a snippet of the log
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        text = ""

    summary = {
        "job_id": job_id,
        "runner": "pytest",
        "exit_code": exit_code,
        "log_path": str(log_path),
        "log_snippet": text[-4000:],
        "duration": time.time() - start,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_summary(job_id, summary)
    return summary


def run_tests_auto(path_or_dir: str, job_id: str) -> Dict[str, Any]:
    """
    Auto-detect and run tests:
    - If a Postman collection JSON exists: run newman
    - Else if pytest tests present: run pytest
    - Else: return error dict
    """
    p = Path(path_or_dir)
    if not p.exists():
        summary = {"job_id": job_id, "error": f"path not found: {path_or_dir}"}
        _write_summary(job_id, summary)
        return summary

    # detect collection.json
    if p.is_file() and p.suffix.lower() == ".json" and "collection" in p.name.lower():
        return run_newman_for_job(str(p), job_id)
    if p.is_dir():
        # check for collection in dir
        coll = next(p.glob("**/*collection*.json"), None)
        if coll:
            return run_newman_for_job(str(coll), job_id)
        # check for pytest tests
        tests = list(p.glob("**/test_*.py"))
        if tests:
            return run_pytest_for_dir(str(p), job_id)
    # nothing found
    summary = {"job_id": job_id, "error": "No test artifacts found (collection.json or pytest test_*.py)", "path": str(p)}
    _write_summary(job_id, summary)
    return summary
