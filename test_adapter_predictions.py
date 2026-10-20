import json
from pathlib import Path
from src.predict import predict_flow

report_path = Path("reports/adapter_diagnostic.json")
report = json.loads(report_path.read_text(encoding="utf-8"))

print("EXPERIMENTAL DRY RUN - NO ALERTS CREATED")
print("-" * 55)

for i, flow in enumerate(report["flows"], start=1):
    result = predict_flow(flow["features"])
    print(
        f"Flow {i}: "
        f"{flow['source_ip']} -> {flow['destination_ip']} | "
        f"Prediction: {result['predicted_label']} | "
        f"Confidence: {result['confidence']:.1%} | "
        f"Attack probability: {result['attack_probability']:.1%}"
    )
