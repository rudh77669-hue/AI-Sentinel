from pathlib import Path
import json

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
MAX_ROWS_PER_CLASS = 10000
RANDOM_STATE = 42


def normalize_labels(series):
    labels = series.astype(str).str.strip().str.upper()
    return labels.map(lambda value: "BENIGN" if value == "BENIGN" else "ATTACK")


def main():
    print("Loading saved model bundle...")
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_names = list(bundle["feature_names"])

    print("Model:", type(model).__name__)
    print("Features:", len(feature_names))
    print("Imputer:", type(imputer).__name__)
    print("Model classes:", list(model.classes_))

    files = list(DATA_DIR.rglob("*Wednesday*"))
    files = [p for p in files if p.suffix.lower() == ".csv"]

    if not files:
        raise FileNotFoundError(
            f"No Wednesday CSV found under {DATA_DIR}"
        )

    collected = {"BENIGN": [], "ATTACK": []}
    rng = np.random.default_rng(RANDOM_STATE)

    print("Sampling held-out Wednesday data...")
    for path in files:
        print("Reading:", path.name)

        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            label_col = next(
                (c for c in chunk.columns if c.lower() == "label"),
                None,
            )
            if label_col is None:
                continue

            labels = normalize_labels(chunk[label_col])
            available = [
                name for name in feature_names if name in chunk.columns
            ]
            X = chunk.reindex(columns=feature_names)
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)

            for target in ("BENIGN", "ATTACK"):
                remaining = MAX_ROWS_PER_CLASS - sum(
                    len(part) for part in collected[target]
                )
                if remaining <= 0:
                    continue

                indices = np.flatnonzero(labels.to_numpy() == target)
                if len(indices) == 0:
                    continue

                take = min(remaining, len(indices), 1000)
                chosen = rng.choice(indices, size=take, replace=False)
                collected[target].append(X.iloc[chosen].copy())

            if all(
                sum(len(part) for part in collected[target])
                >= MAX_ROWS_PER_CLASS
                for target in collected
            ):
                break

        if all(
            sum(len(part) for part in collected[target])
            >= MAX_ROWS_PER_CLASS
            for target in collected
        ):
            break

    frames = []
    targets = []

    for target in ("BENIGN", "ATTACK"):
        parts = collected[target]
        if not parts:
            raise RuntimeError(f"No evaluation samples found for {target}")

        frame = pd.concat(parts, ignore_index=True)
        frame = frame.sample(
            n=min(len(frame), MAX_ROWS_PER_CLASS),
            random_state=RANDOM_STATE,
        )
        frames.append(frame)
        targets.extend([target] * len(frame))

    X_eval = pd.concat(frames, ignore_index=True)
    y_true = np.asarray(targets)

    # Match the preprocessing used by the training pipeline.
    X_eval = X_eval.reindex(columns=feature_names)
    X_ready = pd.DataFrame(imputer.transform(X_eval), columns=feature_names)

    y_pred = model.predict(X_ready)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["BENIGN", "ATTACK"],
    )

    result = {
        "evaluation": "Wednesday held-out sample",
        "samples": int(len(y_true)),
        "feature_count": len(feature_names),
        "preprocessing": "saved training imputer applied",
        "model_classes": [str(c) for c in model.classes_],
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": matrix.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nAccuracy:", round(accuracy, 4))
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=["BENIGN", "ATTACK"],
        zero_division=0,
    ))
    print("Confusion matrix (actual rows, predicted columns):")
    print(matrix)
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
