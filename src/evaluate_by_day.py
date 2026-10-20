
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
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


def read_sample(file_path, rng):
    reservoirs = {"BENIGN": None, "ATTACK": None}

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
            raise ValueError(f"Label column missing in {file_path.name}")

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
                    [old_X, candidate_X],
                    ignore_index=True,
                )
                combined_y = pd.concat(
                    [old_y, candidate_y],
                    ignore_index=True,
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
    files = sorted(
        path
        for path in RAW_DIR.rglob("*.csv")
        if "Wednesday-workingHours" not in path.name
    )

    if len(files) < 2:
        raise RuntimeError(
            "At least two non-Wednesday CSV files are required."
        )

    print(f"Capture files for day-based validation: {len(files)}")
    print("Wednesday is reserved and will not be used.\n")

    fold_results = []

    for fold_number, test_file in enumerate(files, start=1):
        train_files = [path for path in files if path != test_file]

        print("=" * 78)
        print(f"FOLD {fold_number}/{len(files)}")
        print(f"Validation file: {test_file.name}")
        print(f"Training files: {len(train_files)}")

        X_train_parts, y_train_parts = [], []

        for path in train_files:
            print(f"  Sampling training: {path.name}")
            rng = np.random.default_rng(
                RANDOM_STATE + fold_number * 100 + len(X_train_parts)
            )
            X_part, y_part = read_sample(path, rng)

            if not X_part.empty:
                X_train_parts.append(X_part)
                y_train_parts.append(y_part)

        if not X_train_parts:
            print("Skipping fold: no training data.")
            continue

        X_train = pd.concat(X_train_parts, ignore_index=True)
        y_train = pd.concat(y_train_parts, ignore_index=True)

        del X_train_parts, y_train_parts
        gc.collect()

        # Use the feature schema from the training data.
        feature_names = X_train.columns[
            X_train.notna().any()
        ].tolist()
        X_train = X_train[feature_names]

        X_test, y_test = read_sample(
            test_file,
            np.random.default_rng(RANDOM_STATE + fold_number),
        )
        X_test = X_test.reindex(columns=feature_names)

        if X_test.empty or y_test.nunique() < 2 or y_train.nunique() < 2:
            print("Skipping fold: missing data or one of the classes.")
            continue

        print(f"Training rows: {len(y_train):,}")
        print(f"Validation rows: {len(y_test):,}")
        print(f"Features: {len(feature_names)}")
        print("Training class counts:")
        print(y_train.value_counts().to_dict())
        print("Validation class counts:")
        print(y_test.value_counts().to_dict())

        imputer = SimpleImputer(strategy="median")
        X_train_clean = imputer.fit_transform(X_train)
        X_test_clean = imputer.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        print("Training temporary model...")
        model.fit(X_train_clean, y_train)

        predictions = model.predict(X_test_clean)
        labels = ["BENIGN", "ATTACK"]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).ravel()

        precision = precision_score(
            y_test, predictions, pos_label="ATTACK", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label="ATTACK", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label="ATTACK", zero_division=0
        )
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        result = {
            "file": test_file.name,
            "train_rows": len(y_train),
            "validation_rows": len(y_test),
            "accuracy": accuracy_score(y_test, predictions),
            "attack_precision": precision,
            "attack_recall": recall,
            "attack_f1": f1,
            "false_positive_rate": fpr,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        fold_results.append(result)

        print("\nFOLD RESULTS")
        print(f"Accuracy:          {result['accuracy']:.4f}")
        print(f"Attack precision:  {precision:.4f}")
        print(f"Attack recall:     {recall:.4f}")
        print(f"Attack F1:         {f1:.4f}")
        print(f"False-positive rate: {fpr:.4%}")
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]")

        del X_train, X_test, X_train_clean, X_test_clean
        del y_train, y_test, model, imputer, predictions
        gc.collect()

    if not fold_results:
        raise RuntimeError("No validation folds completed successfully.")

    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "day_validation.csv"
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(report_path, index=False)

    print("\n" + "=" * 78)
    print("DAY-BASED VALIDATION SUMMARY")
    print(results_df[
        ["file", "attack_precision", "attack_recall", "attack_f1",
         "false_positive_rate"]
    ].to_string(index=False))

    print("\nMacro average across completed folds:")
    for column in (
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "false_positive_rate",
    ):
        print(f"{column}: {results_df[column].mean():.4f}")

    print(f"\nFull results saved to: {report_path}")
    print("Saved production model and Wednesday benchmark were not changed.")


if __name__ == "__main__":
    main()

