import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("PARKING_DB_PATH", "parking.db"))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_number TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                duration_minutes INTEGER,
                amount REAL,
                payment_status TEXT DEFAULT 'PENDING',
                payment_ref TEXT
            )
            """
        )
        conn.commit()


def create_entry(vehicle_number: str, vehicle_type: str, entry_time: datetime):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO parking_sessions (vehicle_number, vehicle_type, entry_time)
            VALUES (?, ?, ?)
            """,
            (vehicle_number, vehicle_type, entry_time.isoformat()),
        )
        conn.commit()


def get_open_session(vehicle_number: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM parking_sessions
            WHERE vehicle_number = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
            """,
            (vehicle_number,),
        ).fetchone()
    return row


def close_session(session_id: int, exit_time: datetime, duration_minutes: int, amount: float):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE parking_sessions
            SET exit_time = ?, duration_minutes = ?, amount = ?
            WHERE id = ?
            """,
            (exit_time.isoformat(), duration_minutes, amount, session_id),
        )
        conn.commit()


def mark_paid(session_id: int, payment_ref: str):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE parking_sessions
            SET payment_status = 'PAID', payment_ref = ?
            WHERE id = ?
            """,
            (payment_ref, session_id),
        )
        conn.commit()


def list_active_sessions():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, vehicle_number, vehicle_type, entry_time
            FROM parking_sessions
            WHERE exit_time IS NULL
            ORDER BY entry_time ASC
            """
        ).fetchall()
    return rows
