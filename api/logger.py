"""
Simple file-based logger for API operations.
"""

import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOG_PATH = os.path.join(ARTIFACTS_DIR, "api.log")

def ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def write_log(msg: str):
    """Append a timestamped message to the API log."""
    ensure_artifacts_dir()
    ts = datetime.utcnow().isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def read_log():
    """Return the full log text."""
    if not os.path.exists(LOG_PATH):
        return ""
    with open(LOG_PATH, "r") as f:
        return f.read()
