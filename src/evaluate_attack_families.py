
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"

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


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_names = bundle["feature_names"]

    files = list(RAW_DIR.rglob("Wednesday-workingHours.pcap_ISCX.csv"))
    if not files:
        raise FileNotFoundError("Wednesday test CSV not found.")

    file_path = files[0]
    rng = np.random.default_rng(RANDOM_STATE)

    # Keep original labels alongside binary labels.
    reservoirs = {"BENIGN": None, "ATTACK": None}

    print(f"Reading {file_path.name} in chunks...")

    for chunk in pd.read_csv(
        file_path,
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
            raise ValueError("Label column not found.")

        raw_labels = chunk[label_col].astype(str).str.strip().reset_index(drop=True)
        binary_labels = raw_labels.map(normalize_label)

        X = chunk.drop(columns=[label_col])
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=feature_names)

        for class_name in ("BENIGN", "ATTACK"):
            positions = np.flatnonzero(
                binary_labels.to_numpy() == class_name
            )

            if len(positions) == 0:
                continue

            if len(positions) > MAX_PER_CLASS_PER_CHUNK:
                positions = rng.choice(
                    positions,
                    size=MAX_PER_CLASS_PER_CHUNK,
                    replace=False,
                )

            candidate_X = X.iloc[positions].reset_index(drop=True)
            candidate_binary = binary_labels.iloc[positions].reset_index(drop=True)
            candidate_raw = raw_labels.iloc[positions].reset_index(drop=True)

            existing = reservoirs[class_name]

            if existing is None:
                combined = (candidate_X, candidate_binary, candidate_raw)
            else:
                old_X, old_binary, old_raw = existing
                combined = (
                    pd.concat([old_X, candidate_X], ignore_index=True),
                    pd.concat([old_binary, candidate_binary], ignore_index=True),
                    pd.concat([old_raw, candidate_raw], ignore_index=True),
                )

            if len(combined[1]) > MAX_PER_CLASS_PER_FILE:
                selected = rng.choice(
                    len(combined[1]),
                    size=MAX_PER_CLASS_PER_FILE,
                    replace=False,
                )
                combined = tuple(
                    part.iloc[selected].reset_index(drop=True)
                    for part in combined
                )

            reservoirs[class_name] = combined

    X_parts = []
    raw_parts = []

    for class_name in ("BENIGN", "ATTACK"):
        result = reservoirs[class_name]
        if result is not None:
            X_parts.append(result[0])
            raw_parts.append(result[2])

    if not X_parts:
        raise RuntimeError("No test samples collected.")

    X_test = pd.concat(X_parts, ignore_index=True)
    raw_test = pd.concat(raw_parts, ignore_index=True)

    X_clean = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_names,
    )
    predictions = model.predict(X_clean)
    
    probabilities = model.predict_proba(X_clean)
    attack_index = list(model.classes_).index("ATTACK")
    
    results = pd.DataFrame({
        "original_label": raw_test,
        "predicted": predictions,
        "attack_probability": probabilities[:, attack_index],
    })
    
    print("\nAttack probability quantiles by original label:")
    print(
        results.groupby("original_label")["attack_probability"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .round(4)
        .to_string()
    )
    
    print("\nDiagnostic threshold comparison (not final model selection):")
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
        predicted_attack = results["attack_probability"] >= threshold
        true_attack = results["original_label"] != "BENIGN"
    
        detected = (predicted_attack & true_attack).sum()
        missed = ((~predicted_attack) & true_attack).sum()
        false_alarms = (predicted_attack & ~true_attack).sum()
    
        print(
            f"Threshold={threshold:.2f} | "
            f"detected={detected} | missed={missed} | "
            f"false_alarms={false_alarms}"
        )
    
    results = pd.DataFrame({
        "original_label": raw_test,
        "predicted": predictions,
    })

    print("\nSampled original-label distribution:")
    print(results["original_label"].value_counts().to_string())

    print("\nPredictions by original label:")
    print(pd.crosstab(
        results["original_label"],
        results["predicted"],
        margins=True,
    ).to_string())

    print("\nDetection rate by original label:")
    for label, group in results.groupby("original_label"):
        if label == "BENIGN":
            false_positive_rate = (group["predicted"] == "ATTACK").mean()
            print(
                f"{label}: false-positive rate = "
                f"{false_positive_rate:.2%} "
                f"({(group['predicted'] == 'ATTACK').sum()}/{len(group)})"
            )
        else:
            detected = (group["predicted"] == "ATTACK").sum()
            print(
                f"{label}: detection rate = "
                f"{detected / len(group):.2%} "
                f"({detected}/{len(group)})"
            )


if __name__ == "__main__":
    main()
