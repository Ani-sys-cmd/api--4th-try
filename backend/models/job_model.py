# backend/models/job_model.py
"""
SQLAlchemy model for Job records.

This model stores the lifecycle metadata for an upload/scan/test/heal job.
Includes convenient helper functions for common operations (create, read, update).
Designed for a local SQLite DB but works with other SQL backends configured in config.Settings.DATABASE_URL.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import Base, SessionLocal  # uses the database.py module in backend
from config import settings


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)  # short UUID used in API
    project_name = Column(String(256), nullable=True)
    status = Column(String(64), nullable=True, index=True)
    source = Column(String(32), nullable=True)  # "upload" or "git"
    file_name = Column(String(512), nullable=True)

    saved_path = Column(String(1024), nullable=True)
    extracted_path = Column(String(1024), nullable=True)
    scan_report_path = Column(String(1024), nullable=True)

    tests_dir = Column(String(1024), nullable=True)
    openapi_artifact = Column(String(1024), nullable=True)
    collection_path = Column(String(1024), nullable=True)

    last_test_summary = Column(JSON, nullable=True)  # store JSON summary from runner
    self_heal = Column(JSON, nullable=True)  # record patches, history

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "project_name": self.project_name,
            "status": self.status,
            "source": self.source,
            "file_name": self.file_name,
            "saved_path": self.saved_path,
            "extracted_path": self.extracted_path,
            "scan_report_path": self.scan_report_path,
            "tests_dir": self.tests_dir,
            "openapi_artifact": self.openapi_artifact,
            "collection_path": self.collection_path,
            "last_test_summary": self.last_test_summary,
            "self_heal": self.self_heal,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# -----------------------
# Convenience CRUD ops
# -----------------------
def create_job(session: Optional[Session], job_payload: Dict[str, Any]) -> Job:
    """
    Create and persist a Job record.
    If session is None, a new SessionLocal is used for the duration.
    job_payload should include at least 'job_id', other fields optional.
    """
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        job = Job(
            job_id=job_payload.get("job_id"),
            project_name=job_payload.get("project_name"),
            status=job_payload.get("status"),
            source=job_payload.get("source"),
            file_name=job_payload.get("file_name"),
            saved_path=job_payload.get("saved_path"),
            extracted_path=job_payload.get("extracted_path"),
            scan_report_path=job_payload.get("scan_report_path"),
            tests_dir=job_payload.get("tests_dir"),
            openapi_artifact=job_payload.get("openapi_artifact"),
            collection_path=job_payload.get("collection_path"),
            last_test_summary=job_payload.get("last_test_summary"),
            self_heal=job_payload.get("self_heal"),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        return job
    except SQLAlchemyError as exc:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_job_by_job_id(session: Optional[Session], job_id: str) -> Optional[Job]:
    """
    Retrieve a Job by its public job_id string.
    """
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True
    try:
        job = session.query(Job).filter(Job.job_id == job_id).one_or_none()
        return job
    finally:
        if own_session:
            session.close()


def update_job_fields(session: Optional[Session], job_id: str, updates: Dict[str, Any]) -> Optional[Job]:
    """
    Update fields on a Job record. Returns the updated Job or None if not found.
    Commonly used to set saved_path, status, scan_report_path, tests_dir, etc.
    """
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        job = session.query(Job).filter(Job.job_id == job_id).one_or_none()
        if not job:
            return None
        # update whitelisted fields only for safety
        allowed = {
            "project_name", "status", "source", "file_name",
            "saved_path", "extracted_path", "scan_report_path",
            "tests_dir", "openapi_artifact", "collection_path",
            "last_test_summary", "self_heal"
        }
        for k, v in updates.items():
            if k in allowed:
                setattr(job, k, v)
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)
        return job
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
