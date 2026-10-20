import pandas as pd
from pathlib import Path
from src.predict import get_feature_names

live = pd.read_csv("reports/cicflowmeter_test.csv")
expected = get_feature_names()

# CICFlowMeter field -> model field
mapping = {
    "flow_duration": "Flow Duration",
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
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

print("CICFlowMeter test flows:", len(live))
print("Expected model features:", len(expected))
print("Mapped time features:", len(mapping))

print("\nTIME UNIT CHECK (seconds -> microseconds)")
for source, target in mapping.items():
    if source not in live.columns:
        print(f"MISSING SOURCE: {source}")
        continue

    raw = pd.to_numeric(live[source], errors="coerce")
    converted = raw * 1_000_000

    print(
        f"{source:18} -> {target:22} "
        f"raw median={raw.median():.6f}, "
        f"converted median={converted.median():.2f}"
    )

print("\nRATE FEATURES — KEEP PER SECOND")
for name in ["flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s"]:
    if name in live.columns:
        print(f"{name}: median={pd.to_numeric(live[name], errors='coerce').median():.6f}")

print("\nFEATURE MAPPING COVERAGE")
print("Mapped time destinations present in model:",
      all(target in expected for target in mapping.values()))
print("Missing CICFlowMeter time columns:",
      [source for source in mapping if source not in live.columns])

print("\nThis is a diagnostic only; no predictions were run.")
