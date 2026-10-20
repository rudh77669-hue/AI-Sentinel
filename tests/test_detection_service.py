
import unittest
from unittest.mock import patch

from src.detection_service import analyze_flow


class DetectionServiceTests(unittest.TestCase):

    def setUp(self):
        self.features = {"test_feature": 0.0}

    @patch("src.detection_service.create_alert", return_value=123)
    @patch("src.detection_service.predict_flow")
    def test_attack_creates_alert_when_enabled(
        self, mock_predict, mock_create_alert
    ):
        mock_predict.return_value = {
            "predicted_label": "ATTACK",
            "is_attack": True,
            "confidence": 0.8,
            "attack_probability": 0.8,
            "feature_count": 78,
            "model": "classifier.joblib",
        }

        result = analyze_flow(
            self.features,
            source_ip="192.0.2.10",
            destination_ip="192.0.2.20",
            allow_alert_creation=True,
        )

        self.assertTrue(result["alert_created"])
        self.assertEqual(result["alert_id"], 123)

        mock_create_alert.assert_called_once_with(
            predicted_label="ATTACK",
            severity="Medium",
            source_ip="192.0.2.10",
            destination_ip="192.0.2.20",
        )

    @patch("src.detection_service.create_alert")
    @patch("src.detection_service.predict_flow")
    def test_benign_prediction_never_creates_alert(
        self, mock_predict, mock_create_alert
    ):
        mock_predict.return_value = {
            "predicted_label": "BENIGN",
            "is_attack": False,
            "confidence": 0.95,
            "attack_probability": 0.05,
            "feature_count": 78,
            "model": "classifier.joblib",
        }

        result = analyze_flow(
            self.features,
            allow_alert_creation=True,
        )

        self.assertFalse(result["alert_created"])
        self.assertIsNone(result["alert_id"])
        mock_create_alert.assert_not_called()

    @patch("src.detection_service.create_alert")
    @patch("src.detection_service.predict_flow")
    def test_attack_does_not_create_alert_by_default(
        self, mock_predict, mock_create_alert
    ):
        mock_predict.return_value = {
            "predicted_label": "ATTACK",
            "is_attack": True,
            "confidence": 0.8,
            "attack_probability": 0.8,
            "feature_count": 78,
            "model": "classifier.joblib",
        }

        result = analyze_flow(self.features)

        self.assertTrue(result["is_attack"])
        self.assertFalse(result["alert_created"])
        self.assertIsNone(result["alert_id"])
        mock_create_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
