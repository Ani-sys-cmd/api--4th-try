# backend/storage/file_manager.py
"""
Local file manager for uploads, scans, and test artifacts.

Handles:
  • saving uploaded project archives (e.g., .zip)
  • extracting and validating them
  • returning paths for scanning and analysis
"""

import shutil
import uuid
import zipfile
from fastapi import UploadFile, HTTPException
from pathlib import Path
from datetime import datetime
from backend.config import settings


class FileManager:
    """Manages local project upload, extraction, and storage."""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOADS_DIR)
        self.scan_dir = Path(settings.SCANS_DIR)
        self.test_dir = Path(settings.TESTS_DIR)
        self.allowed_exts = settings.ALLOWED_ARCHIVE_EXTS
        self.max_upload_size_mb = settings.MAX_UPLOAD_SIZE_MB

        for path in (self.upload_dir, self.scan_dir, self.test_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _validate_file(self, file: UploadFile):
        """Ensure file extension and size are valid."""
        suffix = Path(file.filename).suffix.lower()
        if suffix not in self.allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(self.allowed_exts)}"
            )

    def save_upload(self, file: UploadFile) -> Path:
        """Save uploaded project archive to /uploads and return the path."""
        self._validate_file(file)
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = self.upload_dir / f"{timestamp}_{unique_id}_{file.filename}"

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file.file.close()
        return dest_path

    def extract_archive(self, archive_path: Path) -> Path:
        """Extract uploaded ZIP/TAR archive into a unique folder under /uploads."""
        extract_dir = archive_path.parent / f"{archive_path.stem}_extracted"
        extract_dir.mkdir(exist_ok=True)

        suffix = archive_path.suffix.lower()
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif suffix in [".tar", ".gz"]:
                shutil.unpack_archive(str(archive_path), extract_dir)
            else:
                raise HTTPException(status_code=400, detail="Unsupported archive format.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

        return extract_dir

    def cleanup_upload(self, file_path: Path):
        """Delete an uploaded archive or extracted folder (optional cleanup)."""
        try:
            if file_path.exists():
                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    file_path.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {file_path}: {e}")


# Shared instance
file_manager = FileManager()

