
from src.predict import predict_flow
from src.alert_manager import create_alert


def analyze_flow(
    flow_features,
    source_ip=None,
    destination_ip=None,
    allow_alert_creation=False,
):
    """
    Analyze one network flow and optionally save an ATTACK prediction.

    Alert creation is disabled by default so synthetic tests cannot
    accidentally create records that look like real security incidents.
    """

    result = predict_flow(flow_features)

    alert_id = None

    if result["is_attack"] and allow_alert_creation:
        # Provisional severity: model probabilities are not calibrated
        # well enough to determine real-world incident severity.
        alert_id = create_alert(
            predicted_label="ATTACK",
            severity="Medium",
            source_ip=source_ip,
            destination_ip=destination_ip,
        )

    result["alert_created"] = alert_id is not None
    result["alert_id"] = alert_id

    return result
