# backend/routes/upload_routes.py
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from uuid import uuid4
from pathlib import Path
import json
from datetime import datetime

from backend.storage.file_manager import file_manager
from backend.config import settings

router = APIRouter(tags=["upload"])

# Simple job persistence (file-based) for upload stage
def _write_job_record(job_id: str, record: dict):
    jobs_dir = Path(settings.SCANS_DIR)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"job_{job_id}.json"
    job_file.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    return job_file


def _background_extract_and_mark(archive_path: Path, job_id: str):
    """
    Background worker to extract archive and update job metadata.
    Keep this lightweight — heavy scanning/training workers will be separate.
    """
    try:
        extracted = file_manager.extract_archive(archive_path)
        job_file = Path(settings.SCANS_DIR) / f"job_{job_id}.json"
        if job_file.exists():
            record = json.loads(job_file.read_text(encoding="utf-8"))
        else:
            record = {}
        record.update({
            "status": "extracted",
            "extracted_path": str(extracted),
            "extracted_at": datetime.utcnow().isoformat(),
        })
        job_file.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        # update job with error status
        job_file = Path(settings.SCANS_DIR) / f"job_{job_id}.json"
        record = {"status": "extract_failed", "error": str(exc), "timestamp": datetime.utcnow().isoformat()}
        job_file.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")


@router.post("/upload-project")
async def upload_project(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    git_url: str | None = Form(None),
    project_name: str | None = Form(None),
):
    """
    Upload a project archive (zip/tar/gz) OR provide a git_url.
    If file is provided it will be saved and extracted in the background.
    Returns a job_id which can be polled by /api/scan-status/{job_id} (to be implemented).
    """
    if file is None and not git_url:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a git_url.")

    job_id = uuid4().hex[:12]
    timestamp = datetime.utcnow().isoformat()

    record = {
        "job_id": job_id,
        "project_name": project_name or (file.filename if file else git_url),
        "created_at": timestamp,
        "status": "uploaded" if file else "cloned",
        "source": "upload" if file else "git",
        "file_name": file.filename if file else None,
        "git_url": git_url,
    }

    # Save initial job record
    _write_job_record(job_id, record)

    # If file uploaded: save and extract in background
    if file:
        try:
            saved_path = file_manager.save_upload(file)
        except HTTPException as hexc:
            # propagate file validation errors
            raise hexc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

        # update record with saved_path
        record.update({"saved_path": str(saved_path)})
        _write_job_record(job_id, record)

        # schedule extraction in background
        background_tasks.add_task(_background_extract_and_mark, saved_path, job_id)

    else:
        # For git_url: simple placeholder behavior (actual git clone implementation occurs in scanner)
        record.update({"note": "git clone scheduled in scanner step; scanner will clone and process the repo."})
        _write_job_record(job_id, record)

    return {"job_id": job_id, "status": record["status"], "message": "Upload accepted. Poll /api/scan-status/{job_id} for updates."}

