import sqlite3
from ..config.settings import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_preferences (
    identifier TEXT PRIMARY KEY,
    preferences_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _ensure_memory_schema() -> None:
    """Create customer_memory.db's tables if they don't already exist. Safe to call every time."""
    settings.MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.MEMORY_DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def get_memory_connection() -> sqlite3.Connection:
    """Return a connection to customer_memory.db, ensuring schema exists first."""
    _ensure_memory_schema()
    conn = sqlite3.connect(settings.MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn