import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.flow_extractor import capture_flows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
FLOWS_PATH = REPORT_DIR / "live_flows.json"
STATUS_PATH = REPORT_DIR / "live_monitor_status.json"


def run_monitor(duration=30):
    if duration < 1:
        raise ValueError("Capture duration must be at least 1 second.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAI Sentinel — Diagnostic Live Monitor")
    print("Mode: DIAGNOSTIC ONLY")
    print("Predictions: Disabled")
    print("Automatic alerts: Disabled\n")

    flows = capture_flows(duration=duration)

    report = {
        "application": "AI Sentinel",
        "mode": "diagnostic_only",
        "monitoring_status": "completed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "capture_duration_seconds": duration,
        "unique_flows": len(flows),
        "predictions_run": False,
        "alerts_created": False,
        "flows": flows,
    }

    FLOWS_PATH.write_text(
        json.dumps(flows, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    STATUS_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nCapture completed.")
    print(f"Flows captured: {len(flows)}")
    print("Predictions run: No")
    print("Alerts created: No")
    print(f"Flow report: {FLOWS_PATH}")
    print(f"Monitor status: {STATUS_PATH}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AI Sentinel in diagnostic-only monitoring mode."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Capture duration in seconds (default: 30).",
    )
    args = parser.parse_args()
    run_monitor(duration=args.duration)
