from datetime import datetime, timezone
import json
import random
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.database import (
    initialize_database,
    get_all_alerts,
    get_alert_counts,
    get_alert_by_id,
    update_alert,
    clear_all_alerts,
)
from src.predict import load_model, predict_flow
from src.alert_manager import create_alert

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_FLOWS_PATH = PROJECT_ROOT / "data" / "sample_flows.json"

# Initialize database on startup
initialize_database()

MONITORING_MODE = "Ready / Simulation Active"


def get_sample_threats():
    if not SAMPLE_FLOWS_PATH.exists():
        return {}
    try:
        with open(SAMPLE_FLOWS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/status")
def system_status():
    try:
        bundle = load_model()
        model_loaded = bundle is not None
        feature_count = len(bundle["feature_names"]) if bundle else 0
        classes = [str(c) for c in bundle["classes"]] if bundle else []
    except Exception:
        model_loaded = False
        feature_count = 0
        classes = []

    return jsonify({
        "project": "AI Sentinel",
        "status": "Online",
        "model_loaded": model_loaded,
        "feature_count": feature_count,
        "classes": classes,
        "mode": MONITORING_MODE,
        "monitoring": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/alerts", methods=["GET"])
def alerts():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(get_all_alerts(limit=limit))


@app.route("/api/alert-stats", methods=["GET"])
def alert_stats():
    return jsonify(get_alert_counts())


@app.route("/api/alerts/<int:alert_id>", methods=["PATCH", "POST"])
def modify_alert(alert_id):
    existing = get_alert_by_id(alert_id)
    if not existing:
        return jsonify({"error": f"Alert with ID {alert_id} not found."}), 404

    data = request.get_json(silent=True) or {}
    status = data.get("status")
    severity = data.get("severity")
    notes = data.get("analyst_notes")

    updated = update_alert(
        alert_id,
        status=status,
        severity=severity,
        analyst_notes=notes,
    )

    if not updated:
        return jsonify({"error": "No valid fields provided for update."}), 400

    return jsonify({
        "success": True,
        "alert": get_alert_by_id(alert_id),
    })


@app.route("/api/clear-alerts", methods=["POST"])
def reset_alerts():
    clear_all_alerts()
    return jsonify({"success": True, "message": "All alerts have been cleared."})


@app.route("/api/sample-threats", methods=["GET"])
def list_sample_threats():
    samples = get_sample_threats()
    summary = [
        {
            "name": name,
            "is_attack": details.get("is_attack", True),
            "severity": details.get("default_severity", "High"),
        }
        for name, details in samples.items()
    ]
    return jsonify(summary)


@app.route("/api/simulate-threat", methods=["POST"])
def simulate_threat():
    samples = get_sample_threats()
    if not samples:
        return jsonify({"error": "No sample flows found. Please run model training first."}), 400

    data = request.get_json(silent=True) or {}
    threat_name = data.get("threat_name")

    if not threat_name or threat_name not in samples:
        # Default to a random attack flow
        attack_keys = [k for k, v in samples.items() if v.get("is_attack", True)]
        threat_name = random.choice(attack_keys) if attack_keys else list(samples.keys())[0]

    flow_info = samples[threat_name]
    features = flow_info["features"]

    # Run inference through ML pipeline
    prediction = predict_flow(features)

    # Generate synthetic IPs for investigation context
    src_ip = data.get(
        "source_ip",
        f"192.168.1.{random.randint(100, 250)}" if threat_name == "BENIGN"
        else f"198.51.100.{random.randint(10, 200)}"
    )
    dst_ip = data.get("destination_ip", "10.0.0.1")

    alert_id = None
    if prediction["is_attack"]:
        # Map probability/threat to security severity
        p = prediction["attack_probability"]
        if "DDOS" in threat_name.upper() or p >= 0.98:
            severity = "Critical"
        elif p >= 0.85:
            severity = "High"
        else:
            severity = "Medium"

        alert_id = create_alert(
            predicted_label=threat_name,
            severity=severity,
            source_ip=src_ip,
            destination_ip=dst_ip,
        )

    return jsonify({
        "success": True,
        "threat_name": threat_name,
        "prediction": prediction,
        "source_ip": src_ip,
        "destination_ip": dst_ip,
        "alert_created": alert_id is not None,
        "alert_id": alert_id,
    })


@app.route("/api/predict", methods=["POST"])
def custom_predict():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object with flow features."}), 400

    features = data.get("features", data)
    try:
        result = predict_flow(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
