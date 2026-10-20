from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

ROOT = Path(".")
DATA_DIR = ROOT / "data" / "raw" / "MachineLearningCVE"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "reports" / "model_comparison.json"
MAX_PER_CLASS = 10000
RNG = np.random.default_rng(42)

files = sorted(DATA_DIR.rglob("*Wednesday*.csv"))
if not files:
    raise FileNotFoundError("Wednesday holdout CSV not found")

# Sample across all chunks so early file ordering has less influence.
samples = {"BENIGN": [], "ATTACK": []}

for path in files:
    print("Sampling:", path.name)
    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        label_col = next(
            (c for c in chunk.columns if c.lower() == "label"), None
        )
        if label_col is None:
            continue

        labels = chunk[label_col].astype(str).str.strip().str.upper()
        targets = labels.map(
            lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
        )
        features = chunk.drop(columns=[label_col])
        features = features.loc[:, ~features.columns.duplicated()]
        features = features.apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)

        for target in ("BENIGN", "ATTACK"):
            idx = np.flatnonzero(targets.to_numpy() == target)
            if len(idx):
                take = min(1000, len(idx))
                chosen = RNG.choice(idx, size=take, replace=False)
                samples[target].append(
                    (features.iloc[chosen].copy(), take)
                )

parts = {}
for target in ("BENIGN", "ATTACK"):
    if not samples[target]:
        raise RuntimeError(f"No samples for class {target}")
    combined = pd.concat(
        [frame for frame, _ in samples[target]], ignore_index=True
    )
    n = min(MAX_PER_CLASS, len(combined))
    parts[target] = combined.sample(n=n, random_state=42)

results = []
for model_path in sorted(MODEL_DIR.glob("classifier*.joblib")):
    print("\nEvaluating:", model_path.name)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_names = list(bundle["feature_names"])

    X_parts = []
    y_parts = []

    for target in ("BENIGN", "ATTACK"):
        X = parts[target].reindex(columns=feature_names)
        X = X.replace([np.inf, -np.inf], np.nan)
        X_parts.append(X)
        y_parts.extend([target] * len(X))

    X_eval = pd.concat(X_parts, ignore_index=True)
    y_true = np.asarray(y_parts)

    transformed = imputer.transform(X_eval)
    X_ready = pd.DataFrame(transformed, columns=feature_names)
    y_pred = model.predict(X_ready)

    attack_recall = recall_score(
        y_true, y_pred, pos_label="ATTACK", zero_division=0
    )
    attack_precision = precision_score(
        y_true, y_pred, pos_label="ATTACK", zero_division=0
    )
    attack_f1 = f1_score(
        y_true, y_pred, pos_label="ATTACK", zero_division=0
    )

    result = {
        "model": model_path.name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "predicted_attack_count": int(np.sum(y_pred == "ATTACK")),
    }
    results.append(result)
    print(json.dumps(result, indent=2))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nSaved comparison:", OUT)
