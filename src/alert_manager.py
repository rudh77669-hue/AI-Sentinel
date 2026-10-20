
from datetime import datetime, timezone

from src.database import get_connection


ALLOWED_SEVERITIES = {"Low", "Medium", "High", "Critical"}


def create_alert(
    predicted_label,
    severity,
    source_ip=None,
    destination_ip=None
):
    """Store a detection produced by a real prediction or authorized test."""

    if not isinstance(predicted_label, str) or not predicted_label.strip():
        raise ValueError("predicted_label must not be empty.")

    if severity not in ALLOWED_SEVERITIES:
        raise ValueError(
            f"severity must be one of {sorted(ALLOWED_SEVERITIES)}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO alerts (
                timestamp,
                source_ip,
                destination_ip,
                predicted_label,
                severity,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                source_ip,
                destination_ip,
                predicted_label.strip(),
                severity,
                "New"
            )
        )

        connection.commit()
        return cursor.lastrowid
