# backend/test_executor/newman_adapter.py
"""
Newman CLI adapter.

Provides:
- run_collection(collection_path, job_id, environment_path=None, data_file=None, timeout=300)
  -> returns dict summary and path to newman JSON report.

Notes:
- Requires `newman` to be installed and available on PATH:
    npm install -g newman
- The module calls newman via subprocess and expects the JSON reporter to be generated to TESTS_DIR.
"""

from pathlib import Path
import subprocess
import shlex
import json
import time
from typing import Optional, Dict, Any

from config import settings


class NewmanError(RuntimeError):
    pass


def _ensure_newman_available():
    """Check `newman --version` to ensure newman is installed."""
    try:
        proc = subprocess.run(["newman", "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            raise NewmanError(f"Newman returned non-zero exit on --version: {proc.stderr.strip()}")
        # ok
        return True
    except FileNotFoundError:
        raise NewmanError("Newman CLI not found. Install it with: npm install -g newman")
    except subprocess.TimeoutExpired:
        raise NewmanError("Timed out while checking newman installation.")
    except Exception as exc:
        raise NewmanError(f"Failed to verify newman installation: {exc}")


def _build_newman_cmd(collection_path: str, report_path: str, environment_path: Optional[str] = None,
                     data_file: Optional[str] = None, reporters: Optional[list] = None) -> list:
    reporters = reporters or ["json"]
    cmd = ["newman", "run", str(collection_path), "--reporters", ",".join(reporters)]
    # reporter export path for json
    if "json" in reporters:
        cmd += ["--reporter-json-export", str(report_path)]
    if environment_path:
        cmd += ["--environment", str(environment_path)]
    if data_file:
        cmd += ["--iteration-data", str(data_file)]
    # be explicit about timeouts if provided later via env/newman options (could extend)
    return cmd


def run_collection(collection_path: str, job_id: str, environment_path: Optional[str] = None,
                   data_file: Optional[str] = None, timeout: int = 600, reporters: Optional[list] = None,
                   capture_output: bool = True) -> Dict[str, Any]:
    """
    Run a Postman collection via newman and return a structured summary.

    Returns dict:
    {
      "job_id": job_id,
      "exit_code": int,
      "report_path": "<path or None>",
      "summary": { parsed newman run summary if available },
      "stdout": "...",
      "stderr": "...",
      "ran_at": "<iso timestamp>",
      "duration_seconds": float
    }
    """
    start = time.time()
    tests_dir = Path(settings.TESTS_DIR)
    tests_dir.mkdir(parents=True, exist_ok=True)
    report_path = tests_dir / f"newman_report_{job_id}.json"

    # ensure newman is available
    _ensure_newman_available()

    # build command
    cmd = _build_newman_cmd(collection_path, report_path, environment_path=environment_path, data_file=data_file, reporters=reporters)
    # run the process
    try:
        proc = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # mark failure due to timeout
        return {
            "job_id": job_id,
            "exit_code": -1,
            "report_path": None,
            "summary": {"error": "timeout", "timeout_seconds": timeout},
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or str(exc),
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": time.time() - start
        }
    except FileNotFoundError:
        raise NewmanError("Newman CLI not found. Install with: npm install -g newman")
    except Exception as exc:
        raise NewmanError(f"Newman run failed: {exc}")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    exit_code = proc.returncode

    parsed_summary = None
    if report_path.exists():
        try:
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
            # extract high-level summary if present
            parsed_summary = parsed.get("run", {})
        except Exception:
            parsed_summary = {"error": "failed_to_parse_report", "path": str(report_path)}

    # assemble friendly summary
    result = {
        "job_id": job_id,
        "exit_code": exit_code,
        "report_path": str(report_path) if report_path.exists() else None,
        "summary": parsed_summary,
        "stdout": stdout[:8000],
        "stderr": stderr[:2000],
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": time.time() - start
    }
    return result


# Optional convenience: parse run summary into key metrics
def parse_newman_metrics(parsed_run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given parsed 'run' object from Newman JSON, extract metrics:
      - total_requests
      - assertions_total
      - assertions_failed
      - failures (request-level)
      - total_response_time_ms (sum of response times)
    """
    if not parsed_run:
        return {}

    metrics = {}
    stats = parsed_run.get("stats", {})
    metrics["total_requests"] = stats.get("requests", {}).get("total", 0)
    metrics["request_failed"] = stats.get("requests", {}).get("failed", 0)
    metrics["assertions_total"] = stats.get("assertions", {}).get("total", 0)
    metrics["assertions_failed"] = stats.get("assertions", {}).get("failed", 0)
    # compute failures list
    failures = []
    executions = parsed_run.get("executions", [])
    total_resp_time = 0.0
    for ex in executions:
        response = ex.get("response")
        if response:
            rt = response.get("responseTime", 0)
            try:
                total_resp_time += float(rt)
            except Exception:
                pass
        assertions = ex.get("assertions", [])
        for a in assertions:
            if not a.get("passed", True):
                failures.append({
                    "item": ex.get("item", {}).get("name"),
                    "assertion": a.get("assertion"),
                    "error": a.get("error"),
                })
    metrics["failures"] = failures
    metrics["total_response_time_ms"] = total_resp_time
    return metrics
