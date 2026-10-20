from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"
DATA_DIR = ROOT / "data" / "raw" / "MachineLearningCVE"
OUTPUT_PATH = ROOT / "reports" / "model_evaluation.json"

RANDOM_STATE = 42
MAX_ROWS_PER_CLASS = 10000


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    bundle = joblib.load(MODEL_PATH)

    # Support either a saved estimator or a dictionary model bundle.
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle.get("classifier"))
        if model is None:
            raise ValueError(
                "Model bundle has no 'model' or 'classifier' entry. "
                "Inspect the bundle before proceeding."
            )
    else:
        model = bundle

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    frames = []
    for path in csv_files:
        # Use the Wednesday dataset as held-out data when available.
        if "Wednesday" in path.name:
            frames.append(pd.read_csv(path, low_memory=False))
            print("Held-out dataset:", path.name)

    if not frames:
        raise FileNotFoundError(
            "No Wednesday held-out CSV found. "
            "No evaluation was performed."
        )

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.strip()

    if "Label" not in data.columns:
        raise ValueError("Dataset does not contain the Label column.")

    data["Label"] = data["Label"].astype(str).str.strip().str.upper()
    data["target"] = (data["Label"] != "BENIGN").astype(int)

    # Sample both classes without replacing the held-out dataset.
    sampled = pd.concat(
        [
            group.sample(
                n=min(len(group), MAX_ROWS_PER_CLASS),
                random_state=RANDOM_STATE,
            )
            for _, group in data.groupby("target")
        ],
        ignore_index=True,
    )

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise ValueError(
            "Model does not expose feature_names_in_. "
            "Use the saved training feature list to ensure exact compatibility."
        )

    missing = sorted(set(features) - set(sampled.columns))
    if missing:
        raise ValueError(f"Evaluation data is missing model features: {missing}")

    X = sampled[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = sampled["target"]

    predictions = model.predict(X)

    # Handle pipelines or estimators that encode classes differently.
    classes = list(getattr(model, "classes_", []))
    if set(classes) == {0, 1}:
        y_pred = np.asarray(predictions, dtype=int)
    else:
        y_pred = np.asarray([
            0 if str(value).strip().upper() == "BENIGN" else 1
            for value in predictions
        ])

    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])

    result = {
        "dataset": "Wednesday held-out CIC-IDS2017 data",
        "rows_evaluated": int(len(sampled)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
        "alerts_created": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Rows evaluated:", len(sampled))
    print("Accuracy:", round(result["accuracy"], 4))
    print("\nClassification report:")
    print(classification_report(
        y, y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))
    print("Confusion matrix (actual rows, predicted columns):")
    print(matrix)
    print("Saved report:", OUTPUT_PATH)
    print("Alerts created: NO")


if __name__ == "__main__":
    main()
