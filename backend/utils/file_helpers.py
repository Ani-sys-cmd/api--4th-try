# backend/utils/file_helpers.py
"""
Utility helpers for working with files and directories.

Used by:
 - storage/file_manager.py (upload & extract)
 - project_scanner.py (source enumeration)
 - openapi_synthesizer.py (context loading)

Features:
 - safe_extract_zip(): prevents zip-slip vulnerability
 - clean_dir(): clears directories recursively
 - list_files(): yields file paths matching extensions
 - compute_sha1(): returns content hash for caching/dedup
"""

import os
import shutil
import zipfile
import hashlib
from pathlib import Path
from typing import List, Iterator, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------
# ZIP handling
# ----------------------------

def safe_extract_zip(zip_path: str, extract_to: str) -> List[str]:
    """
    Safely extract a ZIP archive to a target directory.
    Returns list of extracted files.
    Protects against Zip Slip (path traversal) attacks.
    """
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            filename = member.filename
            # security check: prevent directory traversal
            target = extract_path / filename
            if not str(target.resolve()).startswith(str(extract_path.resolve())):
                logger.warning(f"Skipped unsafe file: {filename}")
                continue
            # extract
            zf.extract(member, extract_path)
            extracted_files.append(str(target))
    logger.info(f"Extracted {len(extracted_files)} files to {extract_path}")
    return extracted_files


# ----------------------------
# Cleaning utilities
# ----------------------------

def clean_dir(path: str):
    """
    Remove directory and recreate it empty.
    """
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cleaned and recreated directory: {p}")


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ----------------------------
# File listing / hashing
# ----------------------------

def list_files(base_path: str, extensions: Optional[List[str]] = None) -> Iterator[Path]:
    """
    Walk directory tree and yield all file paths matching given extensions.
    If extensions=None, yield all files.
    """
    base = Path(base_path)
    if not base.exists():
        return
    for root, _, files in os.walk(base):
        for f in files:
            p = Path(root) / f
            if extensions:
                if any(f.lower().endswith(ext.lower()) for ext in extensions):
                    yield p
            else:
                yield p


def compute_sha1(file_path: str) -> str:
    """Return SHA1 hash of file contents."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()

