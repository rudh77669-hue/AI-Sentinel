from pathlib import Path
from collections import Counter
import pandas as pd
from src.predict import get_feature_names

files = sorted(Path("data/raw").rglob("*.csv"))
expected = get_feature_names()
label_counts = Counter()
total_rows = 0

print("Expected model features:", len(expected))
print("Dataset files:", len(files))

for path in files:
    header = pd.read_csv(path, nrows=0)
    columns = [str(c).strip() for c in header.columns]

    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    feature_columns = [c for c in columns if c.lower() != "label"]
    missing = sorted(set(expected) - set(feature_columns))
    extra = sorted(set(feature_columns) - set(expected))

    print(f"\nFILE: {path.name}")
    print("Feature columns:", len(feature_columns))
    print("Missing expected features:", missing)
    print("Unexpected features:", extra)
    print("Duplicate names after trimming:", duplicates)

    if missing or extra or duplicates or len(feature_columns) != 78:
        print("SCHEMA CHECK: REVIEW REQUIRED")
    else:
        print("SCHEMA CHECK: PASSED")

    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        label_col = next(
            (c for c in chunk.columns if str(c).strip().lower() == "label"),
            None
        )
        if label_col is None:
            raise ValueError(f"Label column missing in {path.name}")

        labels = chunk[label_col].astype(str).str.strip()
        label_counts.update(labels.value_counts().to_dict())
        total_rows += len(chunk)

print("\n" + "=" * 55)
print("TOTAL DATA ROWS:", f"{total_rows:,}")
print("\nLABEL DISTRIBUTION ACROSS ALL FILES:")
for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print("\nDataset inspection complete. No model files were changed.")
