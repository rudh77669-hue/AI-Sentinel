
from pathlib import Path
from collections import Counter

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CHUNK_SIZE = 100_000


def main():
    csv_files = sorted(RAW_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {RAW_DIR}")
        return

    overall = Counter()

    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")
        file_counts = Counter()

        try:
            header = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig")
            label_col = next(
                (col for col in header.columns if col.strip().lower() == "label"),
                None
            )

            if label_col is None:
                print("ERROR: Label column not found.")
                continue

            for chunk in pd.read_csv(
                file_path,
                usecols=[label_col],
                chunksize=CHUNK_SIZE,
                encoding="utf-8-sig"
            ):
                labels = chunk[label_col].astype("string").str.strip()
                file_counts.update(labels.dropna().tolist())

            for label, count in file_counts.items():
                overall[label] += count

            print(f"Rows counted: {sum(file_counts.values()):,}")
            for label, count in sorted(file_counts.items()):
                print(f"  {label}: {count:,}")

        except Exception as error:
            print(f"ERROR reading {file_path.name}: {error}")

    print("\n" + "=" * 55)
    print("OVERALL LABEL DISTRIBUTION")
    print("=" * 55)

    for label, count in sorted(overall.items()):
        print(f"{label}: {count:,}")

    benign = sum(
        count for label, count in overall.items()
        if label.strip().upper() == "BENIGN"
    )
    attacks = sum(overall.values()) - benign

    print(f"\nTotal records: {sum(overall.values()):,}")
    print(f"Benign records: {benign:,}")
    print(f"Non-benign records: {attacks:,}")


if __name__ == "__main__":
    main()
