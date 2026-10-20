
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def main():
    csv_files = list(RAW_DIR.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found yet.")
        print(f"Expected location: {RAW_DIR}")
        print("Download and extract the dataset, then run this script again.")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    for file_path in csv_files:
        print("\n" + "=" * 65)
        print(f"File: {file_path.name}")
        print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

        try:
            # Read only the header and a small sample first.
            sample = pd.read_csv(
                file_path,
                nrows=1000,
                low_memory=False
            )

            # Remove accidental spaces from column names.
            sample.columns = sample.columns.str.strip()

            print(f"Columns: {len(sample.columns)}")
            print("\nColumn names:")
            print(sample.columns.tolist())

            label_column = next(
                (
                    col for col in sample.columns
                    if col.lower() == "label"
                ),
                None
            )

            if label_column:
                print("\nLabels in first 1,000 rows:")
                print(sample[label_column].value_counts(dropna=False))

            print("\nMissing values in sample:")
            missing = sample.isnull().sum()
            print(missing[missing > 0].sort_values(ascending=False).head(10))

        except Exception as error:
            print(f"Could not inspect this file: {error}")


if __name__ == "__main__":
    main()
