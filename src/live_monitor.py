import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.flow_extractor import capture_flows
from src.live_feature_extractor import convert_flow
from src.predict import predict_flow
from src.alert_manager import create_alert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
FLOWS_PATH = REPORT_DIR / "live_flows.json"
STATUS_PATH = REPORT_DIR / "live_monitor_status.json"


def run_monitor(duration=12, detect=True):
    if duration < 1:
        raise ValueError("Capture duration must be at least 1 second.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  AI Sentinel — Live Network Sniffer & Threat Detection")
    print("=" * 65)
    print(f"Capture duration: {duration} seconds")
    print(f"ML Classification: {'ENABLED' if detect else 'DISABLED'}")
    print(f"Automatic Alerts:  {'ENABLED' if detect else 'DISABLED'}\n")

    flows = capture_flows(duration=duration)

    print("\n" + "-" * 65)
    print("  ANALYZING CAPTURED NETWORK FLOWS WITH AI SENTINEL")
    print("-" * 65)

    # 1. Multi-flow correlation for Reconnaissance / PortScan detection
    # In network security, a PortScan is identified when a source IP probes
    # multiple distinct destination ports within a short window.
    port_probes = {}
    for flow in flows:
        sip = flow["source_ip"]
        dip = flow["destination_ip"]
        dport = flow["destination_port"]
        port_probes.setdefault((sip, dip), set()).add(dport)

    portscan_sources = {
        pair: ports for pair, ports in port_probes.items() if len(ports) >= 6
    }

    processed_flows = []
    attacks_found = 0
    created_portscan_alerts = set()

    for idx, flow in enumerate(flows, start=1):
        sip = flow["source_ip"]
        dip = flow["destination_ip"]
        src = f"{sip}:{flow['source_port']}"
        dst = f"{dip}:{flow['destination_port']}"
        proto = flow["protocol"]
        pkts = flow["total_packets"]

        if detect:
            try:
                features = convert_flow(flow)
                pred = predict_flow(features)
                is_attack = pred["is_attack"]
                conf = pred["confidence"]
                threat_name = "Normal Traffic"
                alert_id = None

                # Check if flow is part of a PortScan reconnaissance burst
                pair = (sip, dip)
                is_part_of_portscan = pair in portscan_sources and flow["destination_port"] in portscan_sources[pair]

                if is_part_of_portscan:
                    is_attack = True
                    conf = 1.0
                    scanned_count = len(portscan_sources[pair])
                    threat_name = f"PortScan ({scanned_count} ports)"

                    # Create a consolidated incident alert for this portscan source
                    if pair not in created_portscan_alerts:
                        attacks_found += 1
                        alert_id = create_alert(
                            predicted_label="PortScan",
                            severity="High",
                            source_ip=sip,
                            destination_ip=dip,
                        )
                        created_portscan_alerts.add(pair)
                elif is_attack:
                    attacks_found += 1
                    prob = pred["attack_probability"]
                    severity = "Critical" if prob >= 0.95 else "High"
                    alert_id = create_alert(
                        predicted_label="ATTACK / Flood",
                        severity=severity,
                        source_ip=sip,
                        destination_ip=dip,
                    )

                tag = "🚨 ATTACK" if is_attack else "✅ BENIGN"
                threat_info = f" ({threat_name})" if is_attack else ""
                alert_info = f" -> Alert #{alert_id} Created!" if alert_id else ""
                print(f"Flow {idx:02d} [{tag}] {src} -> {dst} ({proto}){threat_info} | Pkts: {pkts:4d} | Conf: {conf*100:5.1f}%{alert_info}")

                processed_flows.append({
                    **flow,
                    "prediction": pred,
                    "is_attack": is_attack,
                    "threat_name": threat_name,
                    "alert_created": alert_id is not None,
                    "alert_id": alert_id,
                })
            except Exception as e:
                print(f"Flow {idx:02d} [ERROR] {src} -> {dst}: {e}")
                processed_flows.append(flow)
        else:
            print(f"Flow {idx:02d} [LOGGED] {src} -> {dst} ({proto}) | Pkts: {pkts:4d}")
            processed_flows.append(flow)

    report = {
        "application": "AI Sentinel",
        "mode": "live_detection" if detect else "diagnostic_only",
        "monitoring_status": "completed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "capture_duration_seconds": duration,
        "unique_flows": len(flows),
        "predictions_run": detect,
        "attacks_detected": attacks_found,
        "flows": processed_flows,
    }

    FLOWS_PATH.write_text(
        json.dumps(flows, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    STATUS_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("  CAPTURE & DETECTION SUMMARY")
    print("=" * 65)
    print(f"Total Unique Flows:   {len(flows)}")
    if detect:
        print(f"Attacks Identified:   {attacks_found}")
        print(f"Benign Flows Passed:  {len(flows) - attacks_found}")
    print(f"Flows Saved:          {FLOWS_PATH}")
    print(f"Status Saved:         {STATUS_PATH}")
    print("=" * 65 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AI Sentinel in live monitoring and detection mode."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=12,
        help="Capture duration in seconds (default: 12).",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable ML prediction and run diagnostic capture only.",
    )
    args = parser.parse_args()
    run_monitor(duration=args.duration, detect=not args.no_detect)
