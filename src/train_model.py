"""
Train the AI Sentinel Network Threat Detection Classifier.

Samples balanced records across all CIC-IDS2017 capture files,
trains a Random Forest classifier with SimpleImputer on 78 tabular network
flow features, and evaluates on a held-out stratified test set.
"""

from pathlib import Path
import gc
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
DATA_DIR = BASE_DIR / "data"

CHUNK_SIZE = 50_000
MAX_PER_CLASS_PER_FILE = 5_000
MAX_PER_CLASS_PER_CHUNK = 1_000
TEST_SPLIT_RATIO = 0.20
RANDOM_STATE = 42


def clean_label(value):
    if pd.isna(value):
        return "BENIGN"
    s = str(value).strip().replace("\ufffd", "-")
    s = "".join(c for c in s if c.isprintable())
    return s if s else "BENIGN"


def normalize_binary(value):
    clean = clean_label(value).upper()
    return "BENIGN" if clean == "BENIGN" else "ATTACK"


def read_file_sample(file_path, feature_names=None):
    """Sample records from one CSV while preserving attack sub-types."""
    print(f"Reading: {file_path.name}")
    rng = np.random.default_rng(RANDOM_STATE)
    collected_X = []
    collected_y_binary = []
    collected_y_raw = []

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
            continue

        raw_labels = chunk[label_col].map(clean_label)
        binary_labels = raw_labels.map(normalize_binary)

        X = chunk.drop(columns=[label_col])
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if feature_names is not None:
            X = X.reindex(columns=feature_names)

        for category in ("BENIGN", "ATTACK"):
            indices = np.flatnonzero(binary_labels.to_numpy() == category)
            if len(indices) == 0:
                continue

            if len(indices) > MAX_PER_CLASS_PER_CHUNK:
                indices = rng.choice(
                    indices, size=MAX_PER_CLASS_PER_CHUNK, replace=False
                )

            collected_X.append(X.iloc[indices])
            collected_y_binary.append(binary_labels.iloc[indices])
            collected_y_raw.append(raw_labels.iloc[indices])

    if not collected_X:
        return pd.DataFrame(), pd.Series(dtype=str), pd.Series(dtype=str)

    df_X = pd.concat(collected_X, ignore_index=True)
    df_binary = pd.concat(collected_y_binary, ignore_index=True)
    df_raw = pd.concat(collected_y_raw, ignore_index=True)

    # Subsample per file to prevent single file dominance
    final_indices = []
    for category in ("BENIGN", "ATTACK"):
        cat_idx = np.flatnonzero(df_binary.to_numpy() == category)
        if len(cat_idx) > MAX_PER_CLASS_PER_FILE:
            cat_idx = rng.choice(cat_idx, size=MAX_PER_CLASS_PER_FILE, replace=False)
        final_indices.extend(cat_idx)

    return (
        df_X.iloc[final_indices].reset_index(drop=True),
        df_binary.iloc[final_indices].reset_index(drop=True),
        df_raw.iloc[final_indices].reset_index(drop=True),
    )


def extract_sample_flows(X_test, y_raw, feature_names):
    """Extract one real representative flow dictionary per attack family for testing & live simulation."""
    samples = {}
    y_raw_reset = y_raw.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)

    for label in y_raw_reset.unique():
        indices = np.flatnonzero(y_raw_reset.to_numpy() == label)
        if len(indices) > 0:
            first_idx = int(indices[0])
            row = X_test_reset.iloc[first_idx]
            flow_dict = {}
            for col in feature_names:
                val = row[col]
                flow_dict[col] = float(val) if pd.notna(val) else 0.0

            clean_name = str(label).strip()
            samples[clean_name] = {
                "source_label": clean_name,
                "is_attack": clean_name.upper() != "BENIGN",
                "default_severity": (
                    "Critical" if any(w in clean_name.upper() for w in ["DDOS", "HEARTBLEED"])
                    else ("High" if clean_name.upper() != "BENIGN" else "Low")
                ),
                "features": flow_dict,
            }
    return samples


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"Found {len(csv_files)} capture files.")

    all_X, all_y_bin, all_y_raw = [], [], []
    feature_names = None

    for path in csv_files:
        X, y_bin, y_raw = read_file_sample(path, feature_names)
        if X.empty:
            continue

        if feature_names is None:
            feature_names = [c for c in X.columns if X[c].notna().any()]

        all_X.append(X[feature_names])
        all_y_bin.append(y_bin)
        all_y_raw.append(y_raw)

    X_full = pd.concat(all_X, ignore_index=True)
    y_bin_full = pd.concat(all_y_bin, ignore_index=True)
    y_raw_full = pd.concat(all_y_raw, ignore_index=True)

    del all_X, all_y_bin, all_y_raw
    gc.collect()

    print(f"\nTotal sampled records: {len(X_full):,}")
    print("Binary class counts:")
    print(y_bin_full.value_counts().to_string())
    print("\nAttack types captured:")
    print(y_raw_full.value_counts().to_string())

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X_full,
        y_bin_full,
        y_raw_full,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_bin_full,
    )

    print(f"\nTraining set: {len(X_train):,} rows")
    print(f"Test set:     {len(X_test):,} rows")

    # Fit imputer on training data only
    imputer = SimpleImputer(strategy="median")
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

    # Train Random Forest Classifier
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_clean, y_train)

    # Predict on test set
    predictions = model.predict(X_test_clean)
    probabilities = model.predict_proba(X_test_clean)
    labels = ["BENIGN", "ATTACK"]

    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, pos_label="ATTACK", zero_division=0))
    rec = float(recall_score(y_test, predictions, pos_label="ATTACK", zero_division=0))
    f1 = float(f1_score(y_test, predictions, pos_label="ATTACK", zero_division=0))
    cm = confusion_matrix(y_test, predictions, labels=labels).tolist()

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Breakdown by specific attack category
    breakdown = {}
    test_results_df = pd.DataFrame({
        "raw_label": raw_test.values,
        "true_binary": y_test.values,
        "predicted": predictions,
    })

    print("\n" + "=" * 65)
    print("DETECTION RATE BY ATTACK TYPE")
    print("=" * 65)
    for attack_name, grp in test_results_df.groupby("raw_label"):
        clean_name = str(attack_name).strip()
        total = len(grp)
        if clean_name.upper() == "BENIGN":
            false_positives = int((grp["predicted"] == "ATTACK").sum())
            fpr = false_positives / total if total else 0.0
            breakdown[clean_name] = {
                "total": total,
                "false_positives": false_positives,
                "false_positive_rate": fpr,
            }
            print(f"  {clean_name:30s} : FP rate = {fpr:.2%} ({false_positives}/{total})")
        else:
            detected = int((grp["predicted"] == "ATTACK").sum())
            det_rate = detected / total if total else 0.0
            breakdown[clean_name] = {
                "total": total,
                "detected": detected,
                "detection_rate": det_rate,
            }
            print(f"  {clean_name:30s} : Detection rate = {det_rate:.2%} ({detected}/{total})")

    results_payload = {
        "project": "AI Sentinel",
        "task": "Network Threat Detection & Classification",
        "evaluation_method": "Stratified 80/20 train/test holdout across all 8 CIC-IDS2017 captures",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": len(feature_names),
        "features": feature_names,
        "accuracy": acc,
        "attack_precision": prec,
        "attack_recall": rec,
        "attack_f1": f1,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "attack_type_breakdown": breakdown,
    }

    # Save model bundle
    bundle_path = MODEL_DIR / "classifier.joblib"
    model_bundle = {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
    }
    joblib.dump(model_bundle, bundle_path)

    # Save reports
    eval_path = REPORT_DIR / "evaluation.json"
    model_eval_path = REPORT_DIR / "model_evaluation.json"

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    with open(model_eval_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Extract and save representative test flows for live simulation demo
    sample_flows = extract_sample_flows(X_test, raw_test, feature_names)
    sample_flows_path = DATA_DIR / "sample_flows.json"
    with open(sample_flows_path, "w", encoding="utf-8") as f:
        json.dump(sample_flows, f, indent=2)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Attack Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Attack Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"Attack F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nModel saved:       {bundle_path}")
    print(f"Evaluation report: {eval_path}")
    print(f"Sample flows:      {sample_flows_path}")


if __name__ == "__main__":
    main()
