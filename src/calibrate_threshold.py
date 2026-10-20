
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

CHUNK_SIZE = 50_000
MAX_PER_CLASS_PER_FILE = 8_000
MAX_PER_CLASS_PER_CHUNK = 1_000
RANDOM_STATE = 42


def normalize_label(value):
    if pd.isna(value):
        return None
    if str(value).strip().upper() == "BENIGN":
        return "BENIGN"
    return "ATTACK"


def read_sample(path, rng):
    reservoirs = {"BENIGN": None, "ATTACK": None}

    print(f"Sampling {path.name}")

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk.columns = chunk.columns.str.strip()

        label_col = next(
            (c for c in chunk.columns if c.lower() == "label"),
            None,
        )
        if label_col is None:
            raise ValueError(f"Label column missing: {path.name}")

        y = chunk[label_col].map(normalize_label)
        X = chunk.drop(columns=[label_col])
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        for class_name in ("BENIGN", "ATTACK"):
            positions = np.flatnonzero(y.to_numpy() == class_name)

            if len(positions) == 0:
                continue

            if len(positions) > MAX_PER_CLASS_PER_CHUNK:
                positions = rng.choice(
                    positions,
                    MAX_PER_CLASS_PER_CHUNK,
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
                    MAX_PER_CLASS_PER_FILE,
                    replace=False,
                )
                combined_X = combined_X.iloc[selected].reset_index(drop=True)
                combined_y = combined_y.iloc[selected].reset_index(drop=True)

            reservoirs[class_name] = (combined_X, combined_y)

    X_parts, y_parts = [], []

    for class_name in ("BENIGN", "ATTACK"):
        result = reservoirs[class_name]
        if result is not None:
            X_parts.append(result[0])
            y_parts.append(result[1])

    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=str)

    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    files = sorted(
        p for p in RAW_DIR.rglob("*.csv")
        if "Wednesday-workingHours" not in p.name
    )

    if not files:
        raise FileNotFoundError("No non-Wednesday training CSVs found.")

    X_parts, y_parts = [], []

    for path in files:
        X, y = read_sample(path, rng)
        if not X.empty:
            X_parts.append(X)
            y_parts.append(y)

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)

    # Remove features that are entirely missing.
    feature_names = X.columns[X.notna().any()].tolist()
    X = X[feature_names]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining distribution:")
    print(y_train.value_counts().to_string())
    print("\nValidation distribution:")
    print(y_val.value_counts().to_string())

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining temporary validation model...")
    model.fit(X_train, y_train)

    attack_index = list(model.classes_).index("ATTACK")
    probabilities = model.predict_proba(X_val)[:, attack_index]
    actual_attack = (y_val.to_numpy() == "ATTACK")

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("Threshold | Precision | Recall | F1 | False-positive rate")

    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        predicted_attack = probabilities >= threshold

        precision = precision_score(
            actual_attack, predicted_attack, zero_division=0
        )
        recall = recall_score(
            actual_attack, predicted_attack, zero_division=0
        )
        f1 = f1_score(
            actual_attack, predicted_attack, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            actual_attack,
            predicted_attack,
            labels=[False, True],
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        print(
            f"{threshold:8.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:5.4f} | {fpr:.4%}"
        )

    print("\nWednesday's held-out benchmark was not used in this experiment.")
    print("Temporary model only; the existing saved model was not changed.")


if __name__ == "__main__":
    main()
