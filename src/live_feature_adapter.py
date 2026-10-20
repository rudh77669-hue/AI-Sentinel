from pathlib import Path

import numpy as np
import pandas as pd

from src.cicflowmeter_adapter import MAP, TIME_FIELDS
from src.predict import get_feature_names

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "reports" / "live_cicflowmeter.csv"
OUTPUT_PATH = ROOT / "reports" / "live_feature_preview.csv"


def convert_cicflowmeter_dataframe(df):
    missing_sources = sorted(set(MAP) - set(df.columns))
    if missing_sources:
        raise ValueError(f"Missing CICFlowMeter columns: {missing_sources}")

    expected = get_feature_names()
    converted_rows = []

    for _, row in df.iterrows():
        features = {}

        for source, target in MAP.items():
            value = pd.to_numeric(row[source], errors="coerce")
            features[target] = float(value) if pd.notna(value) else 0.0

        # The training feature list contains this duplicate header feature.
        features["Fwd Header Length.1"] = features.get(
            "Fwd Header Length", 0.0
        )

        # CICFlowMeter records these time measurements in seconds;
        # the trained CIC-IDS2017 features use microseconds.
        for name in TIME_FIELDS:
            if name in features:
                features[name] *= 1_000_000.0

        converted_rows.append([features.get(name, 0.0) for name in expected])

    result = pd.DataFrame(converted_rows, columns=expected)

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Converted features contain NaN or infinite values.")

    return result


def main():
    df = pd.read_csv(INPUT_PATH)
    features = convert_cicflowmeter_dataframe(df)
    features.to_csv(OUTPUT_PATH, index=False)

    print("Live input rows:", len(df))
    print("Converted feature rows:", len(features))
    print("Model feature columns:", len(features.columns))
    print("Feature order matches model:", list(features.columns) == get_feature_names())
    print("All values finite:", np.isfinite(features.to_numpy()).all())
    print("Preview saved:", OUTPUT_PATH)
    print("Predictions run: NO")
    print("Alerts created: NO")


if __name__ == "__main__":
    main()
