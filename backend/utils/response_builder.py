# backend/utils/response_builder.py
"""
Unified response builder for all FastAPI routes.

Ensures consistent response format:
{
    "success": bool,
    "message": str,
    "data": { ... } | null,
    "error": { ... } | null,
    "timestamp": "<UTC ISO>",
    "job_id": "<optional>"
}
"""

from datetime import datetime
from typing import Any, Dict, Optional


def success_response(
    data: Any = None,
    message: str = "OK",
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return standardized success response.
    """
    return {
        "success": True,
        "message": message,
        "data": data,
        "error": None,
        "timestamp": datetime.utcnow().isoformat(),
        "job_id": job_id,
    }


def error_response(
    message: str = "Error",
    error: Any = None,
    status_code: int = 400,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return standardized error response.
    """
    if isinstance(error, Exception):
        error = str(error)
    return {
        "success": False,
        "message": message,
        "data": None,
        "error": {"detail": error, "status_code": status_code},
        "timestamp": datetime.utcnow().isoformat(),
        "job_id": job_id,
    }


def paginated_response(
    items: list,
    total: int,
    page: int,
    limit: int,
    message: str = "OK",
) -> Dict[str, Any]:
    """
    Return standardized paginated data response.
    """
    return {
        "success": True,
        "message": message,
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total // limit) + int(total % limit > 0),
        },
        "error": None,
        "timestamp": datetime.utcnow().isoformat(),
    }
