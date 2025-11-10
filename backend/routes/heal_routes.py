# backend/routes/heal_routes.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

from backend.config import settings

# healer import (may raise if module missing)
try:
    from backend.self_heal.healer import heal_job
except Exception:
    heal_job = None

router = APIRouter(tags=["heal"])


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


@router.post("/heal/{job_id}")
def trigger_heal(job_id: str, background_tasks: BackgroundTasks):
    """
    Trigger self-healing for the given job.
    Healing runs in the background and updates the job record with patches/healed artifacts.
    """
    if heal_job is None:
        raise HTTPException(
            status_code=500,
            detail="Healer module not available. Ensure `backend/self_heal/healer.py` exists and is importable."
        )

    # ensure job exists
    job = _read_job(job_id)

    # update job status to indicate healing started
    job.update({"status": "healing", "healing_started_at": datetime.utcnow().isoformat()})
    _write_job(job_id, job)

    def _run_heal(jid: str):
        try:
            result = heal_job(jid)
            # update job record with healer result (heal_job already writes job but we ensure status here)
            current = _read_job(jid)
            current.update({
                "last_heal_result": result,
                "status": "healed" if result.get("healed") else "heal_suggested",
                "healing_completed_at": datetime.utcnow().isoformat(),
            })
            _write_job(jid, current)
        except Exception as exc:
            current = _read_job(jid)
            current.update({
                "status": "heal_failed",
                "heal_error": str(exc),
                "healing_completed_at": datetime.utcnow().isoformat(),
            })
            _write_job(jid, current)

    background_tasks.add_task(_run_heal, job_id)
    return {"job_id": job_id, "status": "healing", "message": "Healing started in background. Poll /api/heal-status/{job_id}."}


@router.get("/heal-status/{job_id}")
def heal_status(job_id: str):
    """
    Return job record including self-heal metadata (patches, healed_artifact, last_heal_result).
    """
    job = _read_job(job_id)
    heal_meta = job.get("self_heal", {})
    last_result = job.get("last_heal_result")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "self_heal": heal_meta,
        "last_heal_result": last_result,
    }

