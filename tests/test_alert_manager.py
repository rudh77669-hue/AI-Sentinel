
import unittest

from src.database import initialize_database, get_all_alerts
from src.alert_manager import create_alert


class AlertManagerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_database()

    def test_alert_creation(self):
        alert_id = create_alert(
            predicted_label="TEST_ONLY",
            severity="High",
            source_ip="192.0.2.10",
            destination_ip="192.0.2.20"
        )

        self.assertIsInstance(alert_id, int)

        alerts = get_all_alerts()
        self.assertTrue(
            any(alert["id"] == alert_id for alert in alerts)
        )

    def test_invalid_severity(self):
        with self.assertRaises(ValueError):
            create_alert(
                predicted_label="TEST_ONLY",
                severity="Unknown"
            )


if __name__ == "__main__":
    unittest.main()
