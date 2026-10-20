from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

bundle = joblib.load("models/classifier.joblib")
model = bundle["model"]
imputer = bundle["imputer"]
features = list(bundle["feature_names"])

path = next(
    Path("data/raw/MachineLearningCVE").rglob(
        "Wednesday-workingHours*.csv"
    )
)

rng = np.random.default_rng(42)
samples = {}
MAX_PER_LABEL = 2000

for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    label_col = next(
        (c for c in chunk.columns if c.lower() == "label"), None
    )
    if label_col is None:
        continue

    labels = chunk[label_col].astype(str).str.strip()
    X = chunk.drop(columns=[label_col])
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reindex(columns=features)

    for label in labels.unique():
        if label.upper() == "BENIGN":
            continue

        indices = np.flatnonzero(labels.to_numpy() == label)
        if len(indices) == 0:
            continue

        frame = X.iloc[indices]
        old = samples.get(label)
        combined = frame if old is None else pd.concat(
            [old, frame], ignore_index=True
        )

        if len(combined) > MAX_PER_LABEL:
            combined = combined.sample(
                n=MAX_PER_LABEL, random_state=42
            )

        samples[label] = combined

print("\n--- WEDNESDAY ATTACK-FAMILY RECALL ---")
print(f"{'Attack family':30} {'Samples':>8} {'Detected':>9} {'Recall':>9}")
print("-" * 60)

for label, X in sorted(samples.items()):
    ready = pd.DataFrame(
        imputer.transform(X), columns=features
    )
    pred = model.predict(ready)
    detected = int(np.sum(pred == "ATTACK"))
    recall = detected / len(X) if len(X) else 0
    print(f"{label[:30]:30} {len(X):8} {detected:9} {recall:8.2%}")

print("\nNo alerts were created; this is an offline evaluation only.")
