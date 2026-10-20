from pathlib import Path
import hashlib
import joblib
import pandas as pd

model_path = Path("models/classifier.joblib")
bundle = joblib.load(model_path)
expected = list(bundle["feature_names"])

csv_path = next(
    Path("data/raw/MachineLearningCVE").rglob(
        "Wednesday-workingHours*.csv"
    )
)

header = pd.read_csv(csv_path, nrows=0)
actual = [str(c).strip() for c in header.columns]
label = next((c for c in actual if c.lower() == "label"), None)
actual_features = [c for c in actual if c.lower() != "label"]

print("\n--- FEATURE ALIGNMENT ---")
print("Model feature count:", len(expected))
print("Dataset feature count:", len(actual_features))
print("Features missing from dataset:", sorted(set(expected) - set(actual_features)))
print("Features not expected by model:", sorted(set(actual_features) - set(expected)))
print("Duplicate model feature names:", len(expected) != len(set(expected)))

print("\n--- DATASET LABELS ---")
counts = {}
for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    if label not in chunk.columns:
        continue
    labels = chunk[label].astype(str).str.strip()
    for name, count in labels.value_counts().items():
        counts[name] = counts.get(name, 0) + int(count)

for name, count in sorted(counts.items()):
    print(f"{name}: {count:,}")

print("\n--- MODEL FILE FINGERPRINTS ---")
for path in sorted(Path("models").glob("classifier*.joblib")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: {digest[:20]}")
