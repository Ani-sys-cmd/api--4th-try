# backend/utils/logger.py
"""
Centralized logger utility for the backend.

Features:
- Unified logging format across all modules (scanner, RL, heal, routes, etc.)
- Rotating file handler for local persistence under ./logs/
- Colorized console logs for readability during local runs
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.
    - Logs to both console and file (rotating)
    - Uses uniform format for timestamps and levels
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # prevent double handlers

    logger.setLevel(level)

    # ---- Formatter ----
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Console Handler (with color support if available) ----
    try:
        from colorama import Fore, Style, init as color_init

        color_init(autoreset=True)

        class ColorFormatter(logging.Formatter):
            COLORS = {
                "DEBUG": Fore.BLUE,
                "INFO": Fore.GREEN,
                "WARNING": Fore.YELLOW,
                "ERROR": Fore.RED,
                "CRITICAL": Fore.MAGENTA,
            }

            def format(self, record):
                base = super().format(record)
                color = self.COLORS.get(record.levelname, "")
                return f"{color}{base}{Style.RESET_ALL}"

        console_formatter = ColorFormatter(fmt=formatter._fmt, datefmt=formatter.datefmt)
    except Exception:
        console_formatter = formatter

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # ---- File Handler ----
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Example global logger
logger = get_logger("backend")

if __name__ == "__main__":
    # quick smoke test
    log = get_logger("test_logger")
    log.info("Logger initialized successfully ✅")
    log.warning("This is a test warning ⚠️")
    log.error("This is a test error ❌")
