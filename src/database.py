import sqlite3
from pathlib import Path
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data" / "ai_sentinel.db"


@contextmanager
def get_connection():
    """Context manager that automatically commits transactions and closes the connection."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_ip TEXT,
                destination_ip TEXT,
                predicted_label TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                analyst_notes TEXT
            )
        """)
        connection.commit()


def get_all_alerts(limit=50):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM alerts
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_alert_by_id(alert_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
    return dict(row) if row else None


def update_alert(alert_id, status=None, severity=None, analyst_notes=None):
    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)

    if severity is not None:
        fields.append("severity = ?")
        values.append(severity)

    if analyst_notes is not None:
        fields.append("analyst_notes = ?")
        values.append(analyst_notes)

    if not fields:
        return False

    values.append(alert_id)
    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE alerts SET {', '.join(fields)} WHERE id = ?",
            values
        )
        connection.commit()
        return cursor.rowcount > 0


def clear_all_alerts():
    with get_connection() as connection:
        connection.execute("DELETE FROM alerts")
        connection.commit()


def get_alert_counts():
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM alerts"
        ).fetchone()[0]

        critical = connection.execute(
            "SELECT COUNT(*) FROM alerts WHERE severity = 'Critical'"
        ).fetchone()[0]

        unresolved = connection.execute(
            "SELECT COUNT(*) FROM alerts WHERE status = 'New'"
        ).fetchone()[0]

    return {
        "total": total,
        "critical": critical,
        "unresolved": unresolved
    }
