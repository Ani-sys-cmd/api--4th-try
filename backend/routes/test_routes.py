# backend/routes/test_routes.py
"""
Test-related routes: generate tests, run tests, fetch test results,
and lightweight RL metrics endpoints.

This version:
 - attempts safe imports (both top-level and backend-prefixed) to avoid ModuleNotFoundError
 - preserves generate-tests, run-tests, test-result behavior
 - adds rl-metrics GET (placeholder) and POST (dev helper)
 - provides save_rl_metrics helper for background tasks to call
 - logs stacktraces and updates job records on errors
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import traceback
import os

# imports that should work when uvicorn is run as `backend.main:app`
# but also try alternative import paths to avoid ModuleNotFoundError.
try:
    # many other backend modules use `from config import settings`
    from config import settings
except Exception:
    # fallback to backend.config if running from a different cwd
    from backend.config import settings  # type: ignore

# logger helper - try both import styles used in repo
try:
    from utils.logger import get_logger
except Exception:
    try:
        from backend.utils.logger import get_logger  # type: ignore
    except Exception:
        # minimal fallback logger
        import logging
        logging.basicConfig(level=logging.INFO)
        def get_logger(name):
            return logging.getLogger(name)

logger = get_logger(__name__)

# Attempt to import synthesizer (try both module paths)
synthesize_openapi = None
for synth_path in ("openapi_synthesizer", "backend.openapi_synthesizer"):
    try:
        mod = __import__(synth_path, fromlist=["synthesize_openapi"])
        synth_fn = getattr(mod, "synthesize_openapi", None)
        if callable(synth_fn):
            synthesize_openapi = synth_fn
            logger.info("Loaded synthesize_openapi from %s", synth_path)
            break
    except Exception:
        logger.debug("Could not import %s (will skip LLM synthesis if missing).", synth_path)

# Attempt to import test runner (try both module paths)
execute_tests_for_job = None
for runner_path in ("test_executor.test_runner", "backend.test_executor.test_runner"):
    try:
        mod = __import__(runner_path, fromlist=["execute_tests_for_job"])
        fn = getattr(mod, "execute_tests_for_job", None)
        if callable(fn):
            execute_tests_for_job = fn
            logger.info("Loaded execute_tests_for_job from %s", runner_path)
            break
    except Exception:
        logger.debug("Could not import %s (test runner may be missing).", runner_path)

router = APIRouter(tags=["tests"])


# -------------------------
# Helpers for job persistence
# -------------------------
def _job_file_path(job_id: str) -> Path:
    return Path(settings.SCANS_DIR) / f"job_{job_id}.json"


def _read_job(job_id: str) -> Dict[str, Any]:
    job_file = _job_file_path(job_id)
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        return json.loads(job_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read job file: {exc}")


def _write_job(job_id: str, data: Dict[str, Any]):
    job_file = _job_file_path(job_id)
    job_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


# -------------------------
# RL metrics helpers
# -------------------------
def save_rl_metrics(job_id: str, metrics: Dict[str, Any]):
    """
    Persist RL metrics for a job in two places:
      - inside the job JSON under "rl_metrics"
      - as a standalone file under settings.TESTS_DIR (rl_metrics_<job_id>.json)
    Safe to call from background tasks.
    """
    tests_dir = Path(settings.TESTS_DIR)
    try:
        tests_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to ensure tests dir exists")

    # Update job file (if exists)
    try:
        job = {}
        job_file = _job_file_path(job_id)
        if job_file.exists():
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except Exception:
                job = {"job_id": job_id}
        else:
            job = {"job_id": job_id}

        job["rl_metrics"] = metrics
        _write_job(job_id, job)
    except Exception:
        logger.exception("[save_rl_metrics] failed to update job file for %s", job_id)

    # Write standalone metrics file
    try:
        metrics_file = tests_dir / f"rl_metrics_{job_id}.json"
        metrics_file.write_text(json.dumps(metrics, default=str, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("[save_rl_metrics] failed to write metrics file for %s", job_id)


# -------------------------
# RL metrics endpoints
# -------------------------
@router.get("/rl-metrics/{job_id}")
def rl_metrics(job_id: str):
    """
    Return RL training/eval metrics for a job. If metrics not available yet,
    return an empty-but-200 placeholder so the frontend does not 404.
    """
    job = _read_job(job_id)

    metrics = job.get("rl_metrics")
    if metrics and isinstance(metrics, dict):
        return {
            "job_id": job_id,
            "timesteps": metrics.get("timesteps", []),
            "rewards": metrics.get("rewards", []),
            "loss": metrics.get("loss", []),
            "status": job.get("status", "unknown"),
            **{k: v for k, v in metrics.items() if k not in ("timesteps", "rewards", "loss")},
        }

    metrics_file = Path(settings.TESTS_DIR) / f"rl_metrics_{job_id}.json"
    if metrics_file.exists():
        try:
            m = json.loads(metrics_file.read_text(encoding="utf-8"))
            return {
                "job_id": job_id,
                "timesteps": m.get("timesteps", []),
                "rewards": m.get("rewards", []),
                "loss": m.get("loss", []),
                "status": job.get("status", "unknown"),
                **{k: v for k, v in m.items() if k not in ("timesteps", "rewards", "loss")},
            }
        except Exception:
            logger.exception("[rl_metrics] failed to read metrics file for %s", job_id)

    return {
        "job_id": job_id,
        "timesteps": [],
        "rewards": [],
        "loss": [],
        "status": job.get("status", "unknown"),
    }


@router.post("/rl-metrics/{job_id}", status_code=201)
def post_rl_metrics(job_id: str, metrics: Dict[str, Any]):
    """
    Dev helper: POST metrics to persist them for testing charts.
    """
    try:
        save_rl_metrics(job_id, metrics)
        return {"job_id": job_id, "saved": True}
    except Exception as exc:
        logger.exception("Failed to save metrics for %s", job_id)
        raise HTTPException(status_code=500, detail=f"Failed to save metrics: {exc}")


# -------------------------
# Existing endpoints (generate-tests, run-tests, test-result)
# -------------------------
@router.post("/generate-tests/{job_id}")
def generate_tests(job_id: str):
    """
    Enrich the OpenAPI stub (if available) using LLM and produce initial test artifacts.
    Produces:
      - enriched openapi JSON in tests dir
      - minimal Postman collection.json placeholder
    """
    job = _read_job(job_id)

    # locate scan report or sample_openapi
    scan_report_path = job.get("scan_report_path")
    sample_stub = None
    if scan_report_path:
        rpt = Path(scan_report_path)
        if rpt.exists():
            try:
                scan_report = json.loads(rpt.read_text(encoding="utf-8"))
                sample_stub = scan_report.get("sample_openapi")
            except Exception:
                logger.exception("Failed to parse existing scan_report for %s", job_id)
                sample_stub = None

    # if no scan_report found, try to run scanner quickly (best-effort)
    if not sample_stub and "extracted_path" in job:
        try:
            from backend.scanner.project_scanner import scan_project  # local import to avoid top-level failures
            scan_report = scan_project(job["extracted_path"], job_id=job_id)
            sample_stub = scan_report.get("sample_openapi")
            # write scan report to disk
            report_path = Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json"
            report_path.write_text(json.dumps(scan_report, indent=2, default=str), encoding="utf-8")
            job.update({"scan_report_path": str(report_path)})
            _write_job(job_id, job)
        except Exception:
            logger.exception("Scanner quick-run failed for job %s", job_id)
            sample_stub = None

    if not sample_stub:
        # fallback: create a trivial stub
        sample_stub = {"openapi": "3.0.0", "info": {"title": "fallback", "version": "0.1.0"}, "paths": {}}

    # Call synthesizer to enrich (if available)
    enriched = sample_stub
    if synthesize_openapi:
        try:
            enriched = synthesize_openapi({"sample_openapi": sample_stub, "frontend_calls": job.get("frontend_calls", [])}, job_id=job_id)
        except Exception as exc:
            logger.exception("Synthesizer error for job %s: %s", job_id, exc)
            job.setdefault("synthesizer", {})
            job["synthesizer"].update({"status": "error", "error": str(exc)})
            _write_job(job_id, job)

    # Save enriched OpenAPI to tests dir
    tests_job_dir = Path(settings.TESTS_DIR) / job_id
    tests_job_dir.mkdir(parents=True, exist_ok=True)
    openapi_path = tests_job_dir / "openapi_enriched.json"
    openapi_path.write_text(json.dumps(enriched, indent=2, default=str), encoding="utf-8")

    # Produce a minimal Postman collection.json as a placeholder
    collection = {
        "info": {"name": f"auto-generated-collection-{job_id}", "_postman_id": job_id, "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "item": []
    }
    for path, methods in enriched.get("paths", {}).items():
        for method, meta in methods.items():
            item = {
                "name": f"{method.upper()} {path}",
                "request": {
                    "method": method.upper(),
                    "header": [],
                    "body": {"mode": "raw", "raw": ""},
                    "url": {"raw": path, "path": path.strip("/").split("/")}
                },
                "response": []
            }
            collection["item"].append(item)

    collection_path = tests_job_dir / "collection.json"
    collection_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    # update job record
    job.update({
        "status": "tests_generated",
        "tests_dir": str(tests_job_dir),
        "openapi_artifact": str(openapi_path),
        "collection_path": str(collection_path),
        "tests_generated_at": datetime.utcnow().isoformat(),
    })
    _write_job(job_id, job)

    return {"job_id": job_id, "status": "tests_generated", "tests_dir": str(tests_job_dir)}


@router.post("/run-tests/{job_id}")
def run_tests(job_id: str, background_tasks: BackgroundTasks):
    """
    Run tests for a given job in background.
    Uses tests folder produced by generate-tests (or any detected tests).
    """
    job = _read_job(job_id)

    tests_dir = job.get("tests_dir") or job.get("collection_path") or job.get("saved_path")
    if not tests_dir:
        raise HTTPException(status_code=400, detail="No tests available for this job. Run /generate-tests/{job_id} first.")

    if execute_tests_for_job is None:
        # provide a helpful error message and record to job file
        err = "Test runner not available. Ensure backend.test_executor.test_runner exists."
        logger.error(err)
        job.update({"status": "tests_error", "tests_error": err, "tests_started_at": datetime.utcnow().isoformat()})
        _write_job(job_id, job)
        raise HTTPException(status_code=500, detail=err)

    # update job status
    job.update({"status": "tests_running", "tests_started_at": datetime.utcnow().isoformat()})
    _write_job(job_id, job)

    def _runner(path: str, jid: str):
        try:
            summary = execute_tests_for_job(path, jid)
            # write summary into job
            current = _read_job(jid)
            current.update({
                "status": "tests_completed" if summary.get("exit_code", 1) == 0 else "tests_failed",
                "last_test_summary": summary,
                "tests_completed_at": datetime.utcnow().isoformat(),
            })
            _write_job(jid, current)
        except Exception as exc:
            logger.exception("[run_tests._runner] tests failed for job %s", jid)
            try:
                cur = _read_job(jid)
                cur.update({
                    "status": "tests_error",
                    "tests_error": str(exc),
                    "tests_completed_at": datetime.utcnow().isoformat(),
                })
                _write_job(jid, cur)
            except Exception:
                logger.exception("Failed to write job error record for %s", jid)

    background_tasks.add_task(_runner, tests_dir if isinstance(tests_dir, str) else str(tests_dir), job_id)
    return {"job_id": job_id, "status": "tests_running", "message": "Tests started in background. Poll /api/test-result/{job_id}."}


@router.get("/test-result/{job_id}")
def test_result(job_id: str):
    """
    Fetch the latest test summary for the job (if any).
    """
    job = _read_job(job_id)
    summary = job.get("last_test_summary")
    if not summary:
        # try to read saved test run report file
        test_report_file = Path(settings.TESTS_DIR) / f"test_run_{job_id}.json"
        if test_report_file.exists():
            try:
                summary = json.loads(test_report_file.read_text(encoding="utf-8"))
            except Exception:
                summary = {"error": "failed to read test run report file."}
        else:
            return {"job_id": job_id, "status": job.get("status"), "message": "No test run summary available yet."}
    return {"job_id": job_id, "status": job.get("status"), "summary": summary}
