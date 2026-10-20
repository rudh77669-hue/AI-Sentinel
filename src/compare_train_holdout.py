from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

bundle = joblib.load("models/classifier.joblib")
model = bundle["model"]
imputer = bundle["imputer"]
features = list(bundle["feature_names"])
data_dir = Path("data/raw/MachineLearningCVE")
rng = np.random.default_rng(42)

def sample_file(pattern, per_class=5000):
    paths = list(data_dir.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No file matched: {pattern}")

    collected = {"BENIGN": [], "ATTACK": []}

    for path in paths:
        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            label_col = next(
                (c for c in chunk.columns if c.lower() == "label"), None
            )
            if label_col is None:
                continue

            labels = chunk[label_col].astype(str).str.strip().str.upper()
            targets = np.where(labels == "BENIGN", "BENIGN", "ATTACK")
            X = chunk.drop(columns=[label_col])
            X = X.loc[:, ~X.columns.duplicated()]
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.reindex(columns=features)

            for cls in collected:
                ids = np.flatnonzero(targets == cls)
                if len(ids):
                    take = min(500, len(ids))
                    chosen = rng.choice(ids, take, replace=False)
                    collected[cls].append(X.iloc[chosen].copy())

    parts = {}
    for cls, frames in collected.items():
        if not frames:
            parts[cls] = pd.DataFrame(columns=features)
            continue
        combined = pd.concat(frames, ignore_index=True)
        n = min(per_class, len(combined))
        parts[cls] = combined.sample(n=n, random_state=42)

    nonempty = [parts[c] for c in ("BENIGN", "ATTACK") if len(parts[c])]
    X_eval = pd.concat(nonempty, ignore_index=True)
    y = np.concatenate([
        np.repeat(cls, len(parts[cls]))
        for cls in ("BENIGN", "ATTACK") if len(parts[cls])
    ])

    ready = pd.DataFrame(
        imputer.transform(X_eval), columns=features
    )
    pred = model.predict(ready)

    print(f"\n--- {pattern} ---")
    print("Samples:", len(y))
    print("Actual labels:", dict(zip(*np.unique(y, return_counts=True))))
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print(classification_report(
        y, pred, labels=["BENIGN", "ATTACK"], zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y, pred, labels=["BENIGN", "ATTACK"]))

sample_file("Tuesday-WorkingHours*.csv")
sample_file("Wednesday-workingHours*.csv")
