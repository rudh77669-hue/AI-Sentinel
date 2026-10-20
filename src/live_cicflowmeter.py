import argparse
import time
from pathlib import Path

from cicflowmeter.sniffer import create_sniffer

from src.flow_extractor import get_wifi_interface

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORT_DIR / "live_cicflowmeter.csv"


def capture_cicflowmeter(duration=10):
    if duration < 1:
        raise ValueError("Duration must be at least 1 second.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    interface = get_wifi_interface()

    print("\nAI Sentinel — CICFlowMeter Live Capture")
    print(f"Interface: {interface.name}")
    print(f"Duration: {duration} seconds")
    print("Predictions: Disabled")
    print("Automatic alerts: Disabled")

    sniffer, session = create_sniffer(
        input_file=None,
        input_interface=interface.name,
        output_mode="csv",
        output=str(OUTPUT_PATH),
        verbose=False,
    )

    try:
        sniffer.start()
        print("Capture started. Generate ordinary network activity...")
        time.sleep(duration)
    finally:
        if sniffer.running:
            sniffer.stop()
        session.flush_flows()

    if not OUTPUT_PATH.exists():
        raise RuntimeError("CICFlowMeter did not create its CSV output.")

    print("\nCapture finished.")
    print(f"CSV output: {OUTPUT_PATH}")
    print("Predictions run: No")
    print("Alerts created: No")

    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture live traffic using CICFlowMeter."
    )
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()
    capture_cicflowmeter(args.duration)
