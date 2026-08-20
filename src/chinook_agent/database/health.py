# database/health.py — new file

from typing import TypedDict
from .connection import get_connection
from .memory_connection import get_memory_connection
from ..config.logging import get_logger

logger = get_logger(__name__)


class DatabaseHealth(TypedDict):
    healthy: bool
    chinook_db: dict
    memory_db: dict


def _check_database(get_conn_fn, expected_table: str) -> dict:
    """Try to connect and run a trivial query against one database. Returns a
    structured result instead of letting a raw exception propagate — callers
    (a /health endpoint, a startup check, a monitoring script) get a clear
    answer either way, not a stack trace."""
    try:
        conn = get_conn_fn()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) as row_count FROM sqlite_master "
                f"WHERE type='table' AND name=?",
                (expected_table,),
            ).fetchone()
            table_exists = row is not None and row["row_count"] > 0
            return {"reachable": True, "expected_table_found": table_exists}
        finally:
            conn.close()
    except Exception as exc:
        logger.error("database health check failed", exc_info=True)
        return {"reachable": False, "error": str(exc)}


def health_check() -> DatabaseHealth:
    """Verify both databases are reachable and have their expected schema.
    Never raises — always returns a structured result so this is safe to call
    from a health endpoint, a startup check, or a monitoring script."""
    chinook_result = _check_database(get_connection, "Customer")
    memory_result = _check_database(get_memory_connection, "customer_preferences")

    healthy = chinook_result.get("reachable") and chinook_result.get("expected_table_found") \
        and memory_result.get("reachable") and memory_result.get("expected_table_found")

    return {
        "healthy": bool(healthy),
        "chinook_db": chinook_result,
        "memory_db": memory_result,
    }