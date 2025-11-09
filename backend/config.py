# backend/config.py
"""
Lightweight config loader that does NOT depend on pydantic/BaseSettings.
Reads environment variables (and .env if present) and exposes a `settings` object.
This is intentionally simple and stable for a local dev/demo environment.
"""

from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import os

# Load .env from project root if present
dotenv_path = Path(__file__).resolve().parents[1] / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

def _path_from_env(var: str, default: Path) -> Path:
    val = os.getenv(var)
    if val:
        return Path(val)
    return default

class Settings:
    # App
    APP_NAME: str = os.getenv("APP_NAME", "Hybrid Agentic API Tester")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    # Storage paths (local)
    BASE_DATA_DIR: Path = _path_from_env("BASE_DATA_DIR", Path.cwd() / "data")
    UPLOADS_DIR: Path = _path_from_env("UPLOADS_DIR", Path.cwd() / "data" / "uploads")
    SCANS_DIR: Path = _path_from_env("SCANS_DIR", Path.cwd() / "data" / "scans")
    TESTS_DIR: Path = _path_from_env("TESTS_DIR", Path.cwd() / "data" / "tests")
    MODELS_DIR: Path = _path_from_env("MODELS_DIR", Path.cwd() / "models")

    # Gemini / LLM
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_PROJECT: Optional[str] = os.getenv("GEMINI_PROJECT")
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Database (use sqlite by default for local mode)
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{Path.cwd() / 'data' / 'app.db'}")

    # Frontend allowed origins (dev)
    FRONTEND_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Runner / tooling
    NEWMAN_CMD: str = os.getenv("NEWMAN_CMD", "newman")
    PYTEST_CMD: str = os.getenv("PYTEST_CMD", "pytest")

    # Misc
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    ALLOWED_ARCHIVE_EXTS: List[str] = [".zip", ".tar", ".gz"]

# single settings instance
settings = Settings()

# Ensure directories exist
for path in (settings.BASE_DATA_DIR, settings.UPLOADS_DIR, settings.SCANS_DIR, settings.TESTS_DIR, settings.MODELS_DIR):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
