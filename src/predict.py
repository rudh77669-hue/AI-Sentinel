
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    required = {"model", "imputer", "feature_names"}
    missing = required - set(bundle)

    if missing:
        raise ValueError(f"Missing model bundle keys: {sorted(missing)}")

    return bundle


def get_feature_names():
    return list(load_model()["feature_names"])


def predict_flow(flow_features):
    if not isinstance(flow_features, dict):
        raise TypeError("flow_features must be a dictionary.")

    bundle = load_model()
    names = list(bundle["feature_names"])

    missing = [name for name in names if name not in flow_features]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} features: {missing[:10]}"
        )

    import math
    import numpy as np
    import pandas as pd
    values = []
    for name in names:
        value = flow_features[name]

        if value is None:
            values.append(np.nan)
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Feature {name!r} must be numeric.") from None

        values.append(number if math.isfinite(number) else np.nan)

    X = pd.DataFrame([values], columns=names)
    try:
        X_imputed = bundle["imputer"].transform(X)
    except Exception:
        medians = getattr(bundle["imputer"], "statistics_", None)
        if medians is not None:
            X_imputed = X.fillna(pd.Series(medians, index=names)).to_numpy()
        else:
            X_imputed = X.fillna(0.0).to_numpy()

    X = pd.DataFrame(X_imputed, columns=names)

    model = bundle["model"]
    label = str(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    classes = [str(c) for c in model.classes_]

    return {
        "predicted_label": label,
        "is_attack": label == "ATTACK",
        "confidence": float(np.max(probabilities)),
        "attack_probability": (
            float(probabilities[classes.index("ATTACK")])
            if "ATTACK" in classes else 0.0
        ),
        "feature_count": len(names),
        "model": MODEL_PATH.name,
    }


if __name__ == "__main__":
    bundle = load_model()
    print("Model loaded successfully.")
    print("Classifier:", type(bundle["model"]).__name__)
    print("Required features:", len(get_feature_names()))
    print("Classes:", list(bundle["model"].classes_))
