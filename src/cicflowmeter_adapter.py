from pathlib import Path
import json
import math
import pandas as pd
from src.predict import get_feature_names

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports" / "cicflowmeter_test.csv"
OUTPUT = ROOT / "reports" / "adapter_diagnostic.json"

# CICFlowMeter CSV column -> model feature name.
MAP = {
    "dst_port": "Destination Port",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_seg_size_min": "min_seg_size_forward",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

TIME_FIELDS = {
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
}

# These fields are known not to be trusted as equivalent yet.
REVIEW_REQUIRED = {
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Bwd Header Length",
    "CWE Flag Count",
    "min_seg_size_forward",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
}

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    live = pd.read_csv(INPUT)
    expected = get_feature_names()

    absent_sources = sorted(set(MAP) - set(live.columns))
    unmapped = sorted(set(expected) - set(MAP.values()) - {"Fwd Header Length.1"})
    extra_targets = sorted(set(MAP.values()) - set(expected))

    if absent_sources or unmapped or extra_targets:
        raise ValueError(
            f"Mapping incomplete: absent sources={absent_sources}, "
            f"unmapped model features={unmapped}, "
            f"unexpected targets={extra_targets}"
        )

    converted = []
    for _, row in live.iterrows():
        features = {name: None for name in expected}

        for source, target in MAP.items():
            value = pd.to_numeric(pd.Series([row[source]]), errors="coerce").iloc[0]
            if pd.isna(value):
                features[target] = None
            else:
                number = float(value)
                features[target] = number * 1_000_000 if target in TIME_FIELDS else number

        # The sampled CIC-IDS2017 training data had all six bulk
        # features at zero; suppress incompatible CICFlowMeter bulk values.
        for name in (
            "Fwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
        ):
            features[name] = 0.0

        # The training dataset contains this duplicate feature column.
        features["Fwd Header Length.1"] = features["Fwd Header Length"]
        converted.append(features)

    invalid = []
    for i, features in enumerate(converted):
        for name, value in features.items():
            if value is not None and not math.isfinite(value):
                invalid.append((i, name, value))

    report = {
        "mode": "diagnostic_only_no_predictions",
        "input": str(INPUT.relative_to(ROOT)),
        "flow_count": len(converted),
        "expected_feature_count": len(expected),
        "mapping_count": len(MAP),
        "time_conversion": "seconds_to_microseconds",
        "rate_conversion": "unchanged_per_second",
        "review_required_features": sorted(REVIEW_REQUIRED),
        "finite_numeric_check_passed": not invalid,
        "invalid_values": invalid,
        "schema_order_check_passed": all(
            list(features.keys()) == expected for features in converted
        ),
        "flows": [
            {
                "source_ip": str(live.iloc[i]["src_ip"]),
                "destination_ip": str(live.iloc[i]["dst_ip"]),
                "features": features,
            }
            for i, features in enumerate(converted)
        ],
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Diagnostic adapter complete.")
    print("Flows converted:", len(converted))
    print("Features per flow:", len(expected))
    print("Mapping coverage:", len(MAP), "/", len(expected))
    print("Schema/order check:", report["schema_order_check_passed"])
    print("Finite numeric check:", report["finite_numeric_check_passed"])
    print("Features requiring further review:", len(REVIEW_REQUIRED))
    print("Predictions run: NO")
    print("Alerts created: NO")
    print("Output:", OUTPUT)

if __name__ == "__main__":
    main()

