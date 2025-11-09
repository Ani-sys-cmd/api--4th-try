# backend/routes/scan_routes.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

from config import settings
from storage.file_manager import file_manager

# scanner implementation is expected in scanner.project_scanner
try:
    from scanner.project_scanner import scan_project  # function to run the static analysis
except Exception:
    # postpone import error until runtime; provide helpful message in endpoint
    scan_project = None

router = APIRouter(tags=["scan"])


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


@router.get("/scan-status/{job_id}")
def scan_status(job_id: str):
    """
    Return the current job record (status, paths, messages).
    """
    job = _read_job(job_id)
    return {"job_id": job_id, "status": job.get("status", "unknown"), "record": job}


@router.post("/start-scan/{job_id}")
def start_scan(job_id: str, background_tasks: BackgroundTasks):
    """
    Start static scanning for the job. This will:
      - If an uploaded archive exists: use the extracted path.
      - If a git_url exists: scanning code should clone and process.
    Scanning runs in the background and updates the job JSON record with progress.
    """
    job = _read_job(job_id)

    # ensure scan_project is available
    if scan_project is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Scanner module not ready. Ensure backend/scanner/project_scanner.py exists "
                "and defines `scan_project(extracted_path: str, job_id: str)`."
            ),
        )

    # Determine path to scan
    extracted_path = job.get("extracted_path")
    git_url = job.get("git_url")
    if not extracted_path and not git_url:
        raise HTTPException(
            status_code=400,
            detail="No extracted_path or git_url available for scanning. Upload or provide git_url first."
        )

    # update job record to scanning
    job.update({
        "status": "scanning",
        "scan_started_at": datetime.utcnow().isoformat(),
    })
    _write_job(job_id, job)

    # background task wrapper (write progress updates inside)
    def _run_scan(path_or_url: str):
        try:
            # scan_project should return a report dict (or raise on error)
            report = scan_project(path_or_url, job_id=job_id)
            # write report to disk
            report_path = Path(settings.SCANS_DIR) / f"scan_report_{job_id}.json"
            report_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")

            # update job record
            updated = _read_job(job_id)
            updated.update({
                "status": "scanned",
                "scan_completed_at": datetime.utcnow().isoformat(),
                "scan_report_path": str(report_path),
            })
            _write_job(job_id, updated)
        except Exception as exc:
            # record failure
            updated = _read_job(job_id)
            updated.update({
                "status": "scan_failed",
                "scan_error": str(exc),
                "scan_failed_at": datetime.utcnow().isoformat(),
            })
            _write_job(job_id, updated)

    # choose the best input for scanner: extracted path if available, else git_url
    scan_input = extracted_path if extracted_path else git_url
    background_tasks.add_task(_run_scan, scan_input)

    return {"job_id": job_id, "status": "scanning", "message": "Scan started in background. Poll /api/scan-status/{job_id}."}


@router.get("/scan-report/{job_id}")
def fetch_scan_report(job_id: str):
    """
    Retrieve the scan report JSON (produced by the scanner).
    """
    job = _read_job(job_id)
    report_path = job.get("scan_report_path")
    if not report_path:
        raise HTTPException(status_code=404, detail="Scan report not available yet.")
    report_file = Path(report_path)
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Scan report file missing.")
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        return {"job_id": job_id, "report": report}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read scan report: {exc}")
