# backend/routes/test_routes.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
import os

from config import settings

# Import synthesizer and test runner
try:
    from openapi_synthesizer import synthesize_openapi
except Exception:
    synthesize_openapi = None

try:
    from test_executor.test_runner import execute_tests_for_job
except Exception:
    execute_tests_for_job = None

router = APIRouter(tags=["tests"])


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


@router.post("/generate-tests/{job_id}")
def generate_tests(job_id: str):
    """
    Enrich the OpenAPI stub (if available) using LLM and produce initial test artifacts.
    This will:
      - load the scan report (if present) to get the sample_openapi
      - call synthesize_openapi to produce enriched OpenAPI (if gemini available)
      - produce a minimal collection.json (Postman style) or pytest folder as placeholder
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
                sample_stub = None

    # if no scan_report found, try job record extracted path to synthesize from scratch
    if not sample_stub and "extracted_path" in job:
        # try to run scanner quickly (best-effort)
        try:
            from scanner.project_scanner import scan_project
            scan_report = scan_project(job["extracted_path"], job_id=job_id)
            sample_stub = scan_report.get("sample_openapi")
            # write scan report to disk
            report_path = Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json"
            report_path.write_text(json.dumps(scan_report, indent=2, default=str), encoding="utf-8")
            job.update({"scan_report_path": str(report_path)})
            _write_job(job_id, job)
        except Exception:
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
            # non-fatal: continue with base stub
            job.setdefault("synthesizer", {}) 
            job["synthesizer"].update({"status": "error", "error": str(exc)})
            _write_job(job_id, job)

    # Save enriched OpenAPI to tests dir as a baseline artifact
    tests_job_dir = Path(settings.TESTS_DIR) / job_id
    tests_job_dir.mkdir(parents=True, exist_ok=True)
    openapi_path = tests_job_dir / "openapi_enriched.json"
    openapi_path.write_text(json.dumps(enriched, indent=2, default=str), encoding="utf-8")

    # Produce a minimal Postman collection.json as a placeholder (very basic)
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
    It will use the tests folder produced by generate-tests (or any detected tests).
    """
    job = _read_job(job_id)
    tests_dir = job.get("tests_dir") or job.get("collection_path") or job.get("saved_path")
    if not tests_dir:
        raise HTTPException(status_code=400, detail="No tests available for this job. Run /generate-tests/{job_id} first.")

    # ensure test runner present
    if execute_tests_for_job is None:
        raise HTTPException(status_code=500, detail="Test runner not available. Ensure test_executor.test_runner exists.")

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
            cur = _read_job(jid)
            cur.update({
                "status": "tests_error",
                "tests_error": str(exc),
                "tests_completed_at": datetime.utcnow().isoformat(),
            })
            _write_job(jid, cur)

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
