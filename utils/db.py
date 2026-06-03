"""
utils/db.py — SQLite helpers for SMART-SHUFFLE
"""
import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
    print(f"✅ DB initialised at {DB_PATH}")


def insert_songs(songs: list[dict]) -> int:
    """
    Bulk-insert songs. Skips duplicates (INSERT OR IGNORE).
    Returns number of newly inserted rows.
    """
    if not songs:
        return 0
    with get_conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO songs
                (id, name, artist, album, year, duration_ms, spotify_uri)
            VALUES
                (:id, :name, :artist, :album, :year, :duration_ms, :spotify_uri)
            """,
            songs,
        )
        after = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    return after - before


def get_pending(limit: int = 0) -> list[sqlite3.Row]:
    """Songs with no processed_at and no extraction_error."""
    sql = """
        SELECT id, name, artist FROM songs
        WHERE processed_at IS NULL AND extraction_error IS NULL
        ORDER BY ROWID
    """
    if limit:
        sql += f" LIMIT {limit}"
    with get_conn() as conn:
        return conn.execute(sql).fetchall()


def mark_done(song_id: str, features: dict):
    """Write features row and stamp processed_at."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    cols = ", ".join(features.keys())
    placeholders = ", ".join(f":{k}" for k in features.keys())
    with get_conn() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO features (song_id, {cols}) VALUES (:song_id, {placeholders})",
            {"song_id": song_id, **features},
        )
        conn.execute(
            "UPDATE songs SET processed_at = ? WHERE id = ?", (ts, song_id)
        )


def mark_failed(song_id: str, error: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE songs SET extraction_error = ? WHERE id = ?",
            (error[:500], song_id),
        )


def status() -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM extraction_status").fetchone()
        return dict(row)