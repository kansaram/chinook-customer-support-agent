import json
from datetime import datetime, timezone
from .memory_connection import get_memory_connection


def save_preferences_list(identifier: str, preferences: list[str]) -> None:
    """Overwrite the stored preference list for this identifier with the current full list."""
    conn = get_memory_connection()
    try:
        conn.execute(
            """
            INSERT INTO customer_preferences (identifier, preferences_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(identifier) DO UPDATE SET
                preferences_json = excluded.preferences_json,
                updated_at = excluded.updated_at
            """,
            (identifier, json.dumps(preferences), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_preferences_list(identifier: str) -> list[str]:
    """Return the stored preference list for this identifier, or an empty list if none exist."""
    conn = get_memory_connection()
    try:
        row = conn.execute(
            "SELECT preferences_json FROM customer_preferences WHERE identifier = ?", (identifier,)
        ).fetchone()
        return json.loads(row["preferences_json"]) if row else []
    finally:
        conn.close()