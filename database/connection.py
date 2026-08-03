import sqlite3
import urllib.request
from config.settings import settings


def _db_is_seeded() -> bool:
    """Check the DB file exists AND actually has tables in it."""
    if not settings.DB_FILE_PATH.exists():
        return False

    # An empty/corrupt file can still "exist" but have 0 bytes or 0 tables
    if settings.DB_FILE_PATH.stat().st_size == 0:
        return False

    try:
        conn = sqlite3.connect(settings.DB_FILE_PATH)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            return count > 0
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        # File exists but isn't a valid SQLite DB (corrupted/partial download)
        return False


def _ensure_db_exists() -> None:
    """Build chinook.db locally if it's missing or empty (dev fallback; Docker builds it at image build time)."""
    if _db_is_seeded():
        return

    print("chinook.db missing or empty — downloading and building...")
    settings.DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove any stale/empty/corrupt file before rebuilding
    if settings.DB_FILE_PATH.exists():
        settings.DB_FILE_PATH.unlink()

    sql_path = settings.DB_FILE_PATH.parent / "Chinook_Sqlite.sql"
    urllib.request.urlretrieve(settings.CHINOOK_SQL_URL, sql_path)

    conn = sqlite3.connect(settings.DB_FILE_PATH)
    try:
        conn.executescript(sql_path.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()

    sql_path.unlink()  # clean up the .sql file, keep only the .db
    print(f"chinook.db built at {settings.DB_FILE_PATH}")


def get_connection() -> sqlite3.Connection:
    _ensure_db_exists()
    return sqlite3.connect(settings.DB_FILE_PATH)