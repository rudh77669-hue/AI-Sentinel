from pathlib import Path
import pandas as pd

root = Path("data/raw")
cic_path = Path("reports/cicflowmeter_test.csv")

pairs = [
    ("Flow Duration", "flow_duration"),
    ("Flow Bytes/s", "flow_byts_s"),
    ("Flow Packets/s", "flow_pkts_s"),
    ("Total Fwd Packets", "tot_fwd_pkts"),
    ("Total Backward Packets", "tot_bwd_pkts"),
    ("Total Length of Fwd Packets", "totlen_fwd_pkts"),
    ("Total Length of Bwd Packets", "totlen_bwd_pkts"),
    ("Fwd Packet Length Mean", "fwd_pkt_len_mean"),
    ("Packet Length Mean", "pkt_len_mean"),
    ("Flow IAT Mean", "flow_iat_mean"),
    ("Fwd IAT Mean", "fwd_iat_mean"),
    ("Fwd Header Length", "fwd_header_len"),
    ("Init_Win_bytes_forward", "init_fwd_win_byts"),
    ("Active Mean", "active_mean"),
    ("Fwd Avg Bytes/Bulk", "fwd_byts_b_avg"),
]

files = sorted(root.rglob("*.csv"))
samples = []

for path in files:
    print("Sampling:", path.name)
    chunks = pd.read_csv(path, chunksize=100_000, low_memory=False)
    collected = 0

    for chunk in chunks:
        chunk.columns = [str(c).strip() for c in chunk.columns]
        keep = [name for name, _ in pairs if name in chunk.columns]
        sample = chunk[keep].apply(pd.to_numeric, errors="coerce")
        sample = sample.sample(
            n=min(500, len(sample)), random_state=42
        )
        samples.append(sample)
        collected += len(sample)

        if collected >= 500:
            break

training = pd.concat(samples, ignore_index=True)
live = pd.read_csv(cic_path)

print("\nTRAINING DATA: sampled rows =", len(training))
print("LIVE CAPTURE: flows =", len(live))
print("\nComparison uses raw values; units may differ.")
print("TRAINING columns are sampled from the original CSVs.")
print("LIVE columns are from CICFlowMeter output.\n")

for train_name, live_name in pairs:
    if train_name not in training or live_name not in live:
        continue

    t = pd.to_numeric(training[train_name], errors="coerce").dropna()
    l = pd.to_numeric(live[live_name], errors="coerce").dropna()

    if t.empty or l.empty:
        continue

    print(f"FEATURE: {train_name}")
    print(
        "  TRAIN median / p95 / min / max:",
        t.median(), t.quantile(0.95), t.min(), t.max()
    )
    print(
        "  LIVE   median / p95 / min / max:",
        l.median(), l.quantile(0.95), l.min(), l.max()
    )

print("\nComparison complete. No model files were changed.")
