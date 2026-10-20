
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

files = sorted(RAW_DIR.rglob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

all_counts = {}

for file_path in files:
    print(f"\nReading: {file_path.name}")
    counts = {}

    for chunk in pd.read_csv(
        file_path,
        chunksize=50_000,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in {file_path.name}")

        values = chunk["Label"].astype(str).str.strip().value_counts()

        for label, count in values.items():
            counts[label] = counts.get(label, 0) + int(count)

    all_counts[file_path.name] = counts

    print("  " + ", ".join(
        f"{label}: {count:,}"
        for label, count in sorted(
            counts.items(), key=lambda item: item[1], reverse=True
        )
    ))

summary = pd.DataFrame.from_dict(all_counts, orient="index").fillna(0).astype(int)
summary.index.name = "capture_file"

output_path = BASE_DIR / "reports" / "attack_family_distribution.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_path)

print("\nATTACK FAMILY COVERAGE MATRIX")
print(summary.to_string())
print(f"\nSaved report: {output_path}")
