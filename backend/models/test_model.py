# backend/models/test_model.py
"""
SQLAlchemy model for storing test execution results linked to a Job.

Each record corresponds to one test run (manual, automated, or RL-driven),
including the runner type, metrics (coverage, failures, duration),
and result artifacts (e.g., Newman JSON, log paths).

Provides helper functions for CRUD operations and summaries.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import Session, relationship
from sqlalchemy.exc import SQLAlchemyError

from backend.database import Base, SessionLocal
from backend.config import settings


class TestResult(Base):
    __tablename__ = "test_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), ForeignKey("jobs.job_id"), index=True, nullable=False)
    run_id = Column(String(64), index=True, nullable=False)  # unique id per run
    runner = Column(String(32), nullable=True)  # "newman" | "pytest" | "custom"
    status = Column(String(32), nullable=True)  # "success" | "failed" | "timeout"
    coverage = Column(Integer, nullable=True)
    failures = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)

    report_path = Column(String(1024), nullable=True)
    log_path = Column(String(1024), nullable=True)
    summary = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, onupdate=datetime.utcnow)

    job = relationship("Job", backref="test_results")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "runner": self.runner,
            "status": self.status,
            "coverage": self.coverage,
            "failures": self.failures,
            "duration": self.duration,
            "report_path": self.report_path,
            "log_path": self.log_path,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# -----------------------
# CRUD helpers
# -----------------------

def create_test_result(session: Optional[Session], data: Dict[str, Any]) -> TestResult:
    """Insert a new test result row."""
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        tr = TestResult(**data)
        session.add(tr)
        session.commit()
        session.refresh(tr)
        return tr
    except SQLAlchemyError as exc:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_results_by_job(session: Optional[Session], job_id: str) -> List[TestResult]:
    """Return all test results for a given job."""
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        results = session.query(TestResult).filter(TestResult.job_id == job_id).order_by(TestResult.created_at.desc()).all()
        return results
    finally:
        if own_session:
            session.close()


def get_latest_result(session: Optional[Session], job_id: str) -> Optional[TestResult]:
    """Return the most recent result for the job."""
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True
    try:
        result = (
            session.query(TestResult)
            .filter(TestResult.job_id == job_id)
            .order_by(TestResult.created_at.desc())
            .first()
        )
        return result
    finally:
        if own_session:
            session.close()


def update_result_status(session: Optional[Session], run_id: str, updates: Dict[str, Any]) -> Optional[TestResult]:
    """Update status/summary fields for a given run_id."""
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True
    try:
        tr = session.query(TestResult).filter(TestResult.run_id == run_id).one_or_none()
        if not tr:
            return None
        allowed = {"status", "coverage", "failures", "duration", "summary", "report_path", "log_path"}
        for k, v in updates.items():
            if k in allowed:
                setattr(tr, k, v)
        tr.updated_at = datetime.utcnow()
        session.add(tr)
        session.commit()
        session.refresh(tr)
        return tr
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()

