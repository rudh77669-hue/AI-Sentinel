
from pathlib import Path
import json
import math
import statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "reports" / "live_flows.json"

FEATURE_NAMES = [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_std(values):
    return statistics.pstdev(values) if len(values) > 1 else (
        0.0 if values else None
    )


def safe_sum(values):
    return sum(values) if values else 0


def timing_stats(timestamps):
    """Return inter-arrival statistics in microseconds."""
    intervals = [
        (b - a) * 1_000_000
        for a, b in zip(timestamps, timestamps[1:])
    ]
    return {
        "total": sum(intervals),
        "mean": safe_mean(intervals),
        "std": safe_std(intervals),
        "max": max(intervals) if intervals else None,
        "min": min(intervals) if intervals else None,
    }


def packet_length_stats(packets):
    lengths = [p["ip_length"] for p in packets]
    return {
        "max": max(lengths) if lengths else None,
        "min": min(lengths) if lengths else None,
        "mean": safe_mean(lengths),
        "std": safe_std(lengths),
    }


def convert_flow(flow):
    """Build the model-shaped feature dictionary.

    This is an initial candidate converter, not yet CICFlowMeter-
    equivalent. Unverified feature definitions remain None.
    """
    packets = flow["packet_records"]
    forward = [p for p in packets if p["direction"] == "forward"]
    backward = [p for p in packets if p["direction"] == "backward"]

    all_lengths = [p["ip_length"] for p in packets]
    fwd_lengths = [p["ip_length"] for p in forward]
    bwd_lengths = [p["ip_length"] for p in backward]

    timestamps = sorted(p["timestamp"] for p in packets)
    fwd_times = sorted(p["timestamp"] for p in forward)
    bwd_times = sorted(p["timestamp"] for p in backward)

    duration_us = max(
        0.0, (timestamps[-1] - timestamps[0]) * 1_000_000
    ) if timestamps else 0.0

    flow_iat = timing_stats(timestamps)
    fwd_iat = timing_stats(fwd_times)
    bwd_iat = timing_stats(bwd_times)

    fwd_packet_stats = packet_length_stats(forward)
    bwd_packet_stats = packet_length_stats(backward)

    packet_mean = safe_mean(all_lengths)
    packet_std = safe_std(all_lengths)
    packet_variance = (
        packet_std ** 2 if packet_std is not None else None
    )

    # These are packet-level flags, not counts of packets with
    # each flag. A packet can carry several flags simultaneously.
    flags = [p["tcp_flags"] for p in packets]

    def flag_count(mask):
        return sum(1 for value in flags if value & mask)

    fwd_header_length = sum(
        p["ip_header_length"] + p["transport_header_length"]
        for p in forward
    )
    bwd_header_length = sum(
        p["ip_header_length"] + p["transport_header_length"]
        for p in backward
    )

    duration_seconds = duration_us / 1_000_000
    total_packets = len(packets)
    total_bytes = sum(all_lengths)

    def rate(count):
        return count / duration_seconds if duration_seconds > 0 else 0.0

    # Initialise every expected column so schema and order are stable.
    result = {name: None for name in FEATURE_NAMES}

    result.update({
        "Destination Port": flow["destination_port"],
        "Flow Duration": duration_us,
        "Total Fwd Packets": len(forward),
        "Total Backward Packets": len(backward),
        "Total Length of Fwd Packets": sum(fwd_lengths),
        "Total Length of Bwd Packets": sum(bwd_lengths),

        "Fwd Packet Length Max": fwd_packet_stats["max"],
        "Fwd Packet Length Min": fwd_packet_stats["min"],
        "Fwd Packet Length Mean": fwd_packet_stats["mean"],
        "Fwd Packet Length Std": fwd_packet_stats["std"],

        "Bwd Packet Length Max": bwd_packet_stats["max"],
        "Bwd Packet Length Min": bwd_packet_stats["min"],
        "Bwd Packet Length Mean": bwd_packet_stats["mean"],
        "Bwd Packet Length Std": bwd_packet_stats["std"],

        "Flow Bytes/s": rate(total_bytes),
        "Flow Packets/s": rate(total_packets),
        "Flow IAT Mean": flow_iat["mean"],
        "Flow IAT Std": flow_iat["std"],
        "Flow IAT Max": flow_iat["max"],
        "Flow IAT Min": flow_iat["min"],

        "Fwd IAT Total": fwd_iat["total"],
        "Fwd IAT Mean": fwd_iat["mean"],
        "Fwd IAT Std": fwd_iat["std"],
        "Fwd IAT Max": fwd_iat["max"],
        "Fwd IAT Min": fwd_iat["min"],

        "Bwd IAT Total": bwd_iat["total"],
        "Bwd IAT Mean": bwd_iat["mean"],
        "Bwd IAT Std": bwd_iat["std"],
        "Bwd IAT Max": bwd_iat["max"],
        "Bwd IAT Min": bwd_iat["min"],

        "Fwd PSH Flags": sum(
            1 for p in forward if p["tcp_flags"] & 0x08
        ),
        "Bwd PSH Flags": sum(
            1 for p in backward if p["tcp_flags"] & 0x08
        ),
        "Fwd URG Flags": sum(
            1 for p in forward if p["tcp_flags"] & 0x20
        ),
        "Bwd URG Flags": sum(
            1 for p in backward if p["tcp_flags"] & 0x20
        ),

        "Fwd Header Length": fwd_header_length,
        "Bwd Header Length": bwd_header_length,
        "Fwd Packets/s": rate(len(forward)),
        "Bwd Packets/s": rate(len(backward)),

        "Min Packet Length": min(all_lengths) if all_lengths else None,
        "Max Packet Length": max(all_lengths) if all_lengths else None,
        "Packet Length Mean": packet_mean,
        "Packet Length Std": packet_std,
        "Packet Length Variance": packet_variance,

        "FIN Flag Count": flag_count(0x01),
        "SYN Flag Count": flag_count(0x02),
        "RST Flag Count": flag_count(0x04),
        "PSH Flag Count": flag_count(0x08),
        "ACK Flag Count": flag_count(0x10),
        "URG Flag Count": flag_count(0x20),
        "CWE Flag Count": flag_count(0x80),
        "ECE Flag Count": flag_count(0x40),

        "Down/Up Ratio": (
            len(backward) / len(forward) if forward else None
        ),
        "Average Packet Size": (
            safe_mean(all_lengths) if all_lengths else None
        ),
        "Avg Fwd Segment Size": safe_mean(fwd_lengths),
        "Avg Bwd Segment Size": safe_mean(bwd_lengths),

        # Initial approximation: packet header bytes in forward
        # direction. Exact CICFlowMeter semantics need validation.
        "Fwd Header Length.1": fwd_header_length,

        "Subflow Fwd Packets": len(forward),
        "Subflow Fwd Bytes": sum(fwd_lengths),
        "Subflow Bwd Packets": len(backward),
        "Subflow Bwd Bytes": sum(bwd_lengths),

        "Init_Win_bytes_forward": next(
            (p["tcp_window"] for p in forward
             if p["tcp_window"] is not None), None
        ),
        "Init_Win_bytes_backward": next(
            (p["tcp_window"] for p in backward
             if p["tcp_window"] is not None), None
        ),

        "act_data_pkt_fwd": sum(
            1 for p in forward if p["payload_length"] > 0
        ),
        "min_seg_size_forward": min(
            (
                p["transport_header_length"]
                for p in forward
                if p["transport_header_length"] > 0
            ),
            default=None,
        ),
    })

    # Deliberately leave bulk, active and idle statistics unfilled
    # until their CICFlowMeter definitions are implemented.
    return result


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        flows = json.load(file)

    print("Captured flows:", len(flows))
    print("Expected feature count:", len(FEATURE_NAMES))

    if len(FEATURE_NAMES) != 78 or len(set(FEATURE_NAMES)) != 78:
        raise RuntimeError("Feature schema is invalid.")

    converted = [convert_flow(flow) for flow in flows]

    for index, features in enumerate(converted):
        if list(features.keys()) != FEATURE_NAMES:
            raise RuntimeError(f"Feature order mismatch in flow {index}.")

        numeric_values = [
            value for value in features.values()
            if value is not None
        ]
        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in numeric_values
        ):
            raise RuntimeError(f"Invalid numeric value in flow {index}.")

    output_path = PROJECT_ROOT / "reports" / "candidate_features.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(converted, file, indent=2, allow_nan=False)

    missing_counts = {
        name: sum(row[name] is None for row in converted)
        for name in FEATURE_NAMES
        if any(row[name] is None for row in converted)
    }

    print("Converted flows:", len(converted))
    print("Schema/order validation: PASSED")
    print("Finite numeric values: PASSED")
    print("Features containing missing values:", missing_counts)
    print("Saved candidate features to:", output_path)
    print("These features are NOT yet approved for live predictions.")


if __name__ == "__main__":
    main()
