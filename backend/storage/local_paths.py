# backend/storage/local_paths.py
"""
Path utilities for local storage.

Centralized definitions for directory paths used across the backend.
Ensures consistency between modules and auto-creates directories if missing.

Used by:
 - file_manager (for uploaded project zips)
 - project_scanner (for scan outputs)
 - test_executor (for reports)
 - rl_engine (for models/policies)
"""

from pathlib import Path
from backend.config import settings


class LocalPaths:
    """Helper for constructing and ensuring key project directories."""

    @staticmethod
    def base() -> Path:
        return Path(settings.BASE_DIR).resolve()

    @staticmethod
    def uploads() -> Path:
        path = Path(settings.UPLOADS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def extracted() -> Path:
        path = Path(settings.EXTRACTED_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def scans() -> Path:
        path = Path(settings.SCANS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def tests() -> Path:
        path = Path(settings.TESTS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def models() -> Path:
        path = Path(settings.MODELS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def policies() -> Path:
        path = LocalPaths.models() / "policies"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def job_root(job_id: str) -> Path:
        """Return path for a specific job (under extracted or tests)."""
        path = LocalPaths.extracted() / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def scan_result(job_id: str) -> Path:
        """Return scan result path for a job."""
        path = LocalPaths.scans() / f"scan_{job_id}.json"
        return path

    @staticmethod
    def newman_report(job_id: str) -> Path:
        """Return path for a job's newman JSON report."""
        return LocalPaths.tests() / f"newman_report_{job_id}.json"

    @staticmethod
    def rl_metrics(job_id: str) -> Path:
        """Return path for the RL metrics log file."""
        path = LocalPaths.models() / "policies" / f"metrics_{job_id}.jsonl"
        return path

