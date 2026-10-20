
from pathlib import Path
import gc
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

CHUNK_SIZE = 50_000
MAX_PER_CLASS_PER_FILE = 8_000
MAX_PER_CLASS_PER_CHUNK = 1_000
TEST_FILE_KEYWORD = "Wednesday-workingHours"
RANDOM_STATE = 42


def normalize_label(value):
    if pd.isna(value):
        return None

    if str(value).strip().upper() == "BENIGN":
        return "BENIGN"

    return "ATTACK"


def read_sample(file_path, feature_names=None):
    """Read a bounded sample from one CSV without loading it all into RAM."""
    reservoirs = {"BENIGN": None, "ATTACK": None}
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Reading: {file_path.name}")

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk.columns = chunk.columns.str.strip()

        label_col = next(
            (
                column for column in chunk.columns
                if column.strip().lower() == "label"
            ),
            None,
        )

        if label_col is None:
            raise ValueError(f"Label column missing in {file_path.name}")

        y = chunk[label_col].map(normalize_label)
        X = chunk.drop(columns=[label_col])
        X = X.loc[:, ~X.columns.duplicated()]

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if feature_names is not None:
            X = X.reindex(columns=feature_names)

        for class_name in ("BENIGN", "ATTACK"):
            positions = np.flatnonzero(y.to_numpy() == class_name)

            if len(positions) == 0:
                continue

            if len(positions) > MAX_PER_CLASS_PER_CHUNK:
                positions = rng.choice(
                    positions,
                    size=MAX_PER_CLASS_PER_CHUNK,
                    replace=False,
                )

            candidate_X = X.iloc[positions].reset_index(drop=True)
            candidate_y = y.iloc[positions].reset_index(drop=True)

            existing = reservoirs[class_name]

            if existing is None:
                combined_X = candidate_X
                combined_y = candidate_y
            else:
                old_X, old_y = existing
                combined_X = pd.concat(
                    [old_X, candidate_X], ignore_index=True
                )
                combined_y = pd.concat(
                    [old_y, candidate_y], ignore_index=True
                )

            if len(combined_y) > MAX_PER_CLASS_PER_FILE:
                selected = rng.choice(
                    len(combined_y),
                    size=MAX_PER_CLASS_PER_FILE,
                    replace=False,
                )
                combined_X = combined_X.iloc[selected].reset_index(drop=True)
                combined_y = combined_y.iloc[selected].reset_index(drop=True)

            reservoirs[class_name] = (combined_X, combined_y)

    X_parts = []
    y_parts = []

    for class_name in ("BENIGN", "ATTACK"):
        existing = reservoirs[class_name]

        if existing is not None:
            X_parts.append(existing[0])
            y_parts.append(existing[1])

    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=str)

    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    test_files = [
        path for path in csv_files
        if TEST_FILE_KEYWORD.lower() in path.name.lower()
    ]
    train_files = [
        path for path in csv_files if path not in test_files
    ]

    if not test_files or not train_files:
        raise RuntimeError(
            "Could not identify training files and the Wednesday test file."
        )

    print("TRAINING FILES")
    for path in train_files:
        print(f"  {path.name}")

    print("\nHELD-OUT TEST FILES")
    for path in test_files:
        print(f"  {path.name}")

    # Sample training files only. Wednesday is never read during training.
    X_train_parts = []
    y_train_parts = []
    feature_names = None

    print("\nCollecting training samples...")

    for path in train_files:
        X, y = read_sample(path, feature_names)

        if X.empty:
            continue

        if feature_names is None:
            feature_names = X.columns.tolist()

        X_train_parts.append(X)
        y_train_parts.append(y)

    if not X_train_parts:
        raise RuntimeError("No training samples were collected.")

    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)

    del X_train_parts, y_train_parts
    gc.collect()

    # Keep only features that have usable values in training.
    feature_names = [
        column for column in X_train.columns
        if X_train[column].notna().any()
    ]
    X_train = X_train[feature_names]

    if y_train.nunique() < 2:
        raise RuntimeError("Training data must contain BENIGN and ATTACK.")

    print("\nCollecting held-out Wednesday samples...")

    X_test_parts = []
    y_test_parts = []

    for path in test_files:
        X, y = read_sample(path, feature_names)

        if not X.empty:
            X_test_parts.append(X)
            y_test_parts.append(y)

    if not X_test_parts:
        raise RuntimeError("No test samples were collected.")

    X_test = pd.concat(X_test_parts, ignore_index=True)
    y_test = pd.concat(y_test_parts, ignore_index=True)

    del X_test_parts, y_test_parts
    gc.collect()

    if y_test.nunique() < 2:
        raise RuntimeError(
            "The held-out test data must contain both BENIGN and ATTACK."
        )

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nHeld-out test class distribution:")
    print(y_test.value_counts())

    print(f"\nTraining rows: {len(y_train):,}")
    print(f"Test rows: {len(y_test):,}")
    print(f"Features: {len(feature_names)}")

    # Learn imputation values from training data only.
    imputer = SimpleImputer(strategy="median")

    X_train_clean = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=feature_names,
    )

    del X_train
    gc.collect()

    X_test_clean = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_names,
    )

    del X_test
    gc.collect()

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train_clean, y_train)

    del X_train_clean
    gc.collect()

    print("\nEvaluating on held-out Wednesday data...")
    predictions = model.predict(X_test_clean)

    labels = ["BENIGN", "ATTACK"]

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "project": "AI Sentinel",
        "task": "Binary benign-versus-attack classification",
        "evaluation_method": (
            "Sampled Wednesday holdout; training uses only the other CSV files"
        ),
        "training_files": [path.name for path in train_files],
        "test_files": [path.name for path in test_files],
        "note": (
            "Metrics apply to sampled Wednesday records. They do not establish "
            "performance on every unseen attack or live network traffic."
        ),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_class_counts": {
            str(key): int(value)
            for key, value in y_train.value_counts().items()
        },
        "test_class_counts": {
            str(key): int(value)
            for key, value in y_test.value_counts().items()
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "attack_precision": float(
            precision_score(
                y_test, predictions, pos_label="ATTACK", zero_division=0
            )
        ),
        "attack_recall": float(
            recall_score(
                y_test, predictions, pos_label="ATTACK", zero_division=0
            )
        ),
        "attack_f1": float(
            f1_score(
                y_test, predictions, pos_label="ATTACK", zero_division=0
            )
        ),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "classification_report": report,
    }

    model_bundle = {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
    }

    model_path = MODEL_DIR / "classifier.joblib"
    report_path = REPORT_DIR / "evaluation.json"

    joblib.dump(model_bundle, model_path)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, zero_division=0))

    print(f"Accuracy:         {results['accuracy']:.4f}")
    print(f"Attack precision: {results['attack_precision']:.4f}")
    print(f"Attack recall:    {results['attack_recall']:.4f}")
    print(f"Attack F1-score:  {results['attack_f1']:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted; BENIGN, ATTACK):")
    print(results["confusion_matrix"])

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
